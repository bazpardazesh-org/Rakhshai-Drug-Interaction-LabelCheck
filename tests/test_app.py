import json
from typing import Dict

import pytest
import respx
from fastapi.testclient import TestClient
from httpx import Response

from app.main import SERVICE_VERSION, app, METRICS


@pytest.fixture(autouse=True)
def reset_metrics():
    METRICS.clear()
    yield
    METRICS.clear()


def build_rxnorm_success() -> Dict:
    return {"idGroup": {"rxnormId": ["12345"]}}


def build_openfda_success() -> Dict:
    return {
        "results": [
            {
                "set_id": "abc",
                "effective_time": "20240101",
                "drug_interactions": [
                    "Co-administration of DrugA and DrugB is contraindicated due to increased exposure."
                ],
            }
        ]
    }


def build_openfda_no_evidence() -> Dict:
    return {
        "results": [
            {
                "set_id": "def",
                "effective_time": "20240101",
                "warnings": ["No significant drug interactions are known."],
            }
        ]
    }


def build_rxclass_response() -> Dict:
    return {
        "rxclassMinConceptList": {
            "rxclassMinConcept": [
                {"className": "DrugB", "classId": "N0000000001"},
            ]
        }
    }


def build_rxclass_members_response() -> Dict:
    return {
        "drugMemberGroup": {
            "drugMember": [
                {"minConcept": {"name": "DrugB", "rxcui": "12345", "tty": "IN"}}
            ]
        }
    }


def build_dailymed_spls_response(set_id: str = "set1") -> Dict:
    return {
        "data": [
            {"setid": set_id, "published_date": "Jan 01, 2024"},
        ]
    }


def build_dailymed_xml(text: str) -> str:
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
        "<document xmlns=\"urn:hl7-org:v3\">"
        "<component><structuredBody><component><section>"
        "<title>Drug Interactions</title>"
        "<text><paragraph>" + text + "</paragraph></text>"
        "</section></component></structuredBody></component>"
        "</document>"
    )


@respx.mock
def test_check_success():
    client = TestClient(app)
    respx.get("https://rxnav.nlm.nih.gov/REST/rxcui.json").mock(
        return_value=Response(200, json=build_rxnorm_success())
    )
    respx.get("https://rxnav.nlm.nih.gov/REST/approximateTerm.json").mock(
        return_value=Response(200, json={"approximateGroup": {"candidate": []}})
    )
    respx.get("https://api.fda.gov/drug/label.json").mock(
        return_value=Response(200, json=build_openfda_success())
    )
    respx.get("https://rxnav.nlm.nih.gov/REST/rxclass/class/byName.json").mock(
        return_value=Response(200, json=build_rxclass_response())
    )
    respx.get("https://rxnav.nlm.nih.gov/REST/rxclass/classMembers.json").mock(
        return_value=Response(200, json=build_rxclass_members_response())
    )

    response = client.post("/check", json={"drugs": ["DrugA", "DrugB"]})
    assert response.status_code == 200
    payload = response.json()
    assert payload["input_drugs"] == ["DrugA", "DrugB"]
    assert payload["interactions"]
    first = payload["interactions"][0]
    assert first["has_interaction"] is True
    assert first["evidence"][0]["confidence"] >= 0.5
    assert payload["sources_checked"]["openfda"]["queries"] >= 1
    assert payload["sources_checked"]["dailymed"]["queries"] == 0
    assert payload["note"] is None
    assert payload["version"]


@respx.mock
def test_rxnorm_not_found_returns_404():
    client = TestClient(app)
    respx.get("https://rxnav.nlm.nih.gov/REST/rxcui.json").mock(
        return_value=Response(200, json={"idGroup": {"rxnormId": []}})
    )
    respx.get("https://rxnav.nlm.nih.gov/REST/approximateTerm.json").mock(
        return_value=Response(200, json={"approximateGroup": {"candidate": []}})
    )

    response = client.post("/check", json={"drugs": ["Unknown", "DrugB"]})
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert detail["code"] == "rxnorm_not_found"
    assert "Unknown" in detail["names"]


@respx.mock
def test_dailymed_fallback_supplies_evidence():
    client = TestClient(app)
    respx.get("https://rxnav.nlm.nih.gov/REST/rxcui.json").mock(
        return_value=Response(200, json=build_rxnorm_success())
    )
    respx.get("https://rxnav.nlm.nih.gov/REST/approximateTerm.json").mock(
        return_value=Response(200, json={"approximateGroup": {"candidate": []}})
    )
    respx.get("https://api.fda.gov/drug/label.json").mock(
        return_value=Response(404, json={"error": "not found"})
    )
    respx.get("https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json").mock(
        return_value=Response(200, json=build_dailymed_spls_response())
    )
    respx.get(
        "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/set1.xml"
    ).mock(
        return_value=Response(
            200,
            text=build_dailymed_xml("DrugA must not be combined with DrugB due to risk."),
            headers={"Content-Type": "application/xml"},
        )
    )

    response = client.post("/check", json={"drugs": ["DrugA", "DrugB"]})
    assert response.status_code == 200
    payload = response.json()
    assert payload["interactions"][0]["evidence"]
    assert payload["sources_checked"]["dailymed"]["queries"] >= 1
    assert payload["note"] is None


@respx.mock
def test_no_evidence_returns_note():
    client = TestClient(app)
    respx.get("https://rxnav.nlm.nih.gov/REST/rxcui.json").mock(
        return_value=Response(200, json=build_rxnorm_success())
    )
    respx.get("https://rxnav.nlm.nih.gov/REST/approximateTerm.json").mock(
        return_value=Response(200, json={"approximateGroup": {"candidate": []}})
    )
    respx.get("https://api.fda.gov/drug/label.json").mock(
        return_value=Response(200, json=build_openfda_no_evidence())
    )
    respx.get("https://rxnav.nlm.nih.gov/REST/rxclass/class/byName.json").mock(
        return_value=Response(200, json=build_rxclass_response())
    )
    respx.get("https://rxnav.nlm.nih.gov/REST/rxclass/classMembers.json").mock(
        return_value=Response(200, json=build_rxclass_members_response())
    )

    response = client.post("/check", json={"drugs": ["DrugA", "DrugB"]})
    assert response.status_code == 200
    payload = response.json()
    assert payload["interactions"][0]["has_interaction"] is False
    assert payload["note"] == "No interaction evidence found in checked labels."


@respx.mock
def test_duplicate_names_validation_error():
    client = TestClient(app)
    response = client.post("/check", json={"drugs": ["DrugA", "druga"]})
    assert response.status_code == 422
    body = response.json()
    assert body["detail"][0]["type"] == "value_error"


def test_root_endpoint_matches_documentation():
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to Rakhshai‑Drug Interaction LabelCheck",
        "version": SERVICE_VERSION,
    }


def test_healthz_endpoint_reports_ok():
    client = TestClient(app)

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_endpoint_reports_ready():
    client = TestClient(app)

    response = client.get("/readyz")

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "version": SERVICE_VERSION}
