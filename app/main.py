"""FastAPI application exposing the drug interaction LabelCheck API."""
from __future__ import annotations

import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx
import structlog
from fastapi import Body, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict, field_validator
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from .interaction_checker import (
    METRICS,
    analyse_interactions,
    fetch_drug_interaction_sections,
    normalise_input_name,
    normalise_names_to_rxcui,
)

SERVICE_VERSION = "1.1.0"
MAX_DRUGS = 15
MAX_NAME_LENGTH = 80
ALLOWED_CHAR_PATTERN = r"^[\w\s\-/'()+.,]+$"

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
)
logger = structlog.get_logger("labelcheck")

REQUEST_LATENCY = Histogram(
    "labelcheck_request_latency_seconds",
    "Latency for /check requests",
)
REQUEST_ERRORS = Counter(
    "labelcheck_request_errors_total",
    "Number of failed /check requests",
    ["code"],
)

class InteractionCheckRequest(BaseModel):
    """Validated request payload for the interaction checker."""

    drugs: List[str] = Field(..., min_length=2, max_length=MAX_DRUGS)

    @field_validator("drugs")
    def validate_names(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("At least two drug names must be supplied.")
        clean_names = []
        for name in value:
            if not isinstance(name, str):
                raise ValueError("Drug names must be strings.")
            stripped = name.strip()
            if not stripped:
                continue
            if len(stripped) > MAX_NAME_LENGTH:
                raise ValueError(f"Drug name '{name}' exceeds {MAX_NAME_LENGTH} characters.")
            if not re.match(ALLOWED_CHAR_PATTERN, stripped):
                raise ValueError(
                    "Drug names may only contain letters, numbers, spaces and limited punctuation."
                )
            clean_names.append(stripped)
        if len(clean_names) < 2:
            raise ValueError("At least two valid drug names must be supplied.")
        if len(clean_names) != len(set(map(normalise_input_name, clean_names))):
            raise ValueError("Duplicate drug names are not allowed in a single request.")
        return clean_names

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


class Provenance(BaseModel):
    set_id: Optional[str]
    label_date: Optional[str]
    section_name: str
    url: Optional[str]


class Evidence(BaseModel):
    pair: List[str]
    sentence: str
    confidence: float = Field(ge=0.0, le=1.0)
    triggered: bool
    class_derived: bool
    negated: bool
    provenance: Provenance


class PairInteraction(BaseModel):
    pair: List[str]
    has_interaction: bool
    max_confidence: float = Field(ge=0.0, le=1.0)
    evidence: List[Evidence] = Field(default_factory=list)


class InteractionCheckResponse(BaseModel):
    input_drugs: List[str]
    interactions: List[PairInteraction]
    metrics: Dict[str, float]
    sources_checked: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    note: Optional[str] = None
    version: str = SERVICE_VERSION


allowed_origins = os.getenv("ALLOWED_ORIGINS", "").split(",") if os.getenv("ALLOWED_ORIGINS") else []

app = FastAPI(
    title="Rakhshai‑Drug Interaction LabelCheck",
    description="Check FDA labels for documented drug interaction signals.",
    version=SERVICE_VERSION,
)

if allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in allowed_origins if origin.strip()],
        allow_methods=["POST", "GET"],
        allow_headers=["*"],
    )


@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    correlation_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    request.state.correlation_id = correlation_id
    request.state.logger = logger.bind(request_id=correlation_id)
    start = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        duration = time.perf_counter() - start
        if request.url.path == "/check":
            REQUEST_LATENCY.observe(duration)
    response.headers["x-request-id"] = correlation_id
    return response


@app.get("/", summary="Service welcome message")
async def root() -> Dict[str, str]:
    return {"message": "Welcome to Rakhshai‑Drug Interaction LabelCheck", "version": SERVICE_VERSION}


@app.get("/healthz", summary="Liveness probe")
async def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz", summary="Readiness probe")
async def readyz() -> Dict[str, str]:
    # Could include dependency checks in future revisions
    return {"status": "ready", "version": SERVICE_VERSION}


@app.get("/metrics", include_in_schema=False)
async def metrics_endpoint() -> Response:
    payload = generate_latest()
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)


@app.post("/check", response_model=InteractionCheckResponse, summary="Check for drug interactions")
async def check_interactions(
    payload: InteractionCheckRequest = Body(..., description="List of drug names to check"),
    http_request: Request = None,
) -> InteractionCheckResponse:
    request_logger = getattr(http_request.state, "logger", logger)
    request_logger.info("interaction_check_start", request=payload.model_dump())
    METRICS["requests_total"] += 1

    try:
        rxcui_map = await normalise_names_to_rxcui(payload.drugs)
    except httpx.HTTPStatusError as exc:  # type: ignore[name-defined]
        code = exc.response.status_code
        REQUEST_ERRORS.labels(code=str(code)).inc()
        raise HTTPException(status_code=code, detail="RxNorm service returned an error.") from exc
    except httpx.RequestError as exc:  # type: ignore[name-defined]
        REQUEST_ERRORS.labels(code=str(status.HTTP_503_SERVICE_UNAVAILABLE)).inc()
        raise HTTPException(status_code=503, detail="Failed to reach RxNorm services.") from exc

    missing = [name for name in payload.drugs if name not in rxcui_map]
    if missing:
        REQUEST_ERRORS.labels(code=str(status.HTTP_404_NOT_FOUND)).inc()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "rxnorm_not_found",
                "message": "Unable to map the following drug names to an RxCUI.",
                "names": missing,
            },
        )

    try:
        sections, source_stats = await fetch_drug_interaction_sections(rxcui_map)
    except httpx.HTTPStatusError as exc:  # type: ignore[name-defined]
        code = exc.response.status_code
        REQUEST_ERRORS.labels(code=str(code)).inc()
        if code == status.HTTP_429_TOO_MANY_REQUESTS:
            raise HTTPException(status_code=code, detail="Rate limited by openFDA.") from exc
        raise HTTPException(status_code=code, detail="openFDA returned an error.") from exc
    except httpx.RequestError as exc:  # type: ignore[name-defined]
        REQUEST_ERRORS.labels(code=str(status.HTTP_503_SERVICE_UNAVAILABLE)).inc()
        raise HTTPException(status_code=503, detail="Failed to reach openFDA services.") from exc

    interactions = await analyse_interactions(payload.drugs, sections, rxcui_map=rxcui_map)
    has_any_evidence = any(item["evidence"] for item in interactions)
    note: Optional[str] = None
    if not has_any_evidence:
        note = "No interaction evidence found in checked labels."

    response = InteractionCheckResponse(
        input_drugs=payload.drugs,
        interactions=[PairInteraction(**item) for item in interactions],
        metrics=dict(METRICS),
        sources_checked=source_stats,
        note=note,
    )
    request_logger.info("interaction_check_complete", response=response.model_dump())
    return response


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
