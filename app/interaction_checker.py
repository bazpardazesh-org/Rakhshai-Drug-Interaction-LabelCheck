"""Core drug interaction logic for the LabelCheck service."""
from __future__ import annotations

import asyncio
import math
import random
import re
import unicodedata
from xml.etree import ElementTree as ET
from collections import defaultdict
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import httpx
from cachetools import TTLCache
import spacy
from spacy.language import Language
from spacy.pipeline import EntityRuler
import structlog

RXNORM_BASE_URL = "https://rxnav.nlm.nih.gov/REST"
OPENFDA_LABEL_URL = "https://api.fda.gov/drug/label.json"
DAILYMED_LABEL_URL = "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm"
DAILYMED_SPLS_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json"
DAILYMED_SPL_XML_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{setid}.xml"
RXCLASS_BASE_URL = "https://rxnav.nlm.nih.gov/REST/rxclass"

HTTP_TIMEOUT = httpx.Timeout(5.0, read=10.0)
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_HTTP_RETRIES = 4
OPENFDA_PAGE_SIZE = 50
MAX_OPENFDA_RECORDS = 200
DAILYMED_MAX_SPLS = 5

RXCUI_CACHE = TTLCache(maxsize=2048, ttl=60 * 60 * 24)
RXCLASS_CACHE = TTLCache(maxsize=1024, ttl=60 * 60 * 24)
_CACHE_LOCK = asyncio.Lock()
_HTTP_SEMAPHORE = asyncio.Semaphore(8)

TRIGGER_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"contraindicat(ed|ion)",
        r"avoid\b",
        r"should not be used",
        r"increases?\s+(auc|exposure)",
        r"cyp3a4\s+(inhibitor|inducer)",
        r"serotonergic\s+agents",
        r"black box",
        r"severe\s+interaction",
        r"dose\s+adjust",
    ]
]

CLASS_TRIGGER_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"strong\s+cyp3a(4)?\s+inhibitors",
        r"moderate\s+cyp3a(4)?\s+inhibitors",
        r"cyp3a(4)?\s+inducers",
        r"cyp2d6\s+inhibitors",
        r"cyp2c9\s+inhibitors",
        r"monoamine\s+oxidase\s+inhibitors",
    ]
]

TRIGGER_WEIGHT = 0.3
CLASS_WEIGHT = 0.2
NEGATION_PENALTY = 0.25
BASE_CONFIDENCE = 0.5
CLASS_BASE_CONFIDENCE = 0.35

OPENFDA_INTERACTION_FIELDS = [
    "drug_interactions",
    "drug_interactions_table",
    "drug_interactions_section",
]
OPENFDA_SUPPORTING_FIELDS = [
    "warnings",
    "warnings_and_cautions",
    "warnings_and_precautions",
    "boxed_warning",
    "boxed_warning_table",
    "precautions",
    "precautions_and_warnings",
    "precautions_table",
]
OPENFDA_SECTION_FIELDS = OPENFDA_INTERACTION_FIELDS + OPENFDA_SUPPORTING_FIELDS

logger = structlog.get_logger(__name__)

METRICS: Dict[str, float] = defaultdict(float)


def normalise_input_name(name: str) -> str:
    """Normalise an input drug name using Unicode NFKC and casefolding."""
    return unicodedata.normalize("NFKC", name).casefold().strip()


async def _fetch_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    retries: int = MAX_HTTP_RETRIES,
) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        async with _HTTP_SEMAPHORE:
            try:
                response = await client.get(url, params=params, timeout=HTTP_TIMEOUT)
            except (httpx.ConnectTimeout, httpx.ReadTimeout) as exc:  # pragma: no cover - network errors
                last_error = exc
            else:
                if response.status_code in RETRY_STATUS_CODES:
                    last_error = httpx.HTTPStatusError(
                        "retryable", request=response.request, response=response
                    )
                else:
                    response.raise_for_status()
                    return response.json()
        await asyncio.sleep((2**attempt) + random.random())
    if last_error:
        raise last_error
    raise RuntimeError("HTTP request failed without an exception")


async def _fetch_text(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    retries: int = MAX_HTTP_RETRIES,
) -> str:
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        async with _HTTP_SEMAPHORE:
            try:
                response = await client.get(
                    url, params=params, headers=headers, timeout=HTTP_TIMEOUT
                )
            except (httpx.ConnectTimeout, httpx.ReadTimeout) as exc:  # pragma: no cover - network errors
                last_error = exc
            else:
                if response.status_code in RETRY_STATUS_CODES:
                    last_error = httpx.HTTPStatusError(
                        "retryable", request=response.request, response=response
                    )
                else:
                    response.raise_for_status()
                    return response.text
        await asyncio.sleep((2**attempt) + random.random())
    if last_error:
        raise last_error
    raise RuntimeError("HTTP request failed without an exception")


async def _resolve_with_rxnorm(
    client: httpx.AsyncClient, name: str
) -> Optional[str]:
    params = {"name": name, "search": 1}
    url = f"{RXNORM_BASE_URL}/rxcui.json"
    try:
        data = await _fetch_json(client, url, params=params)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 429:
            METRICS["rxnorm_rate_limit"] += 1
        raise
    id_group = data.get("idGroup", {})
    rxcui_list = id_group.get("rxnormId") or []
    if rxcui_list:
        return rxcui_list[0]

    approx_url = f"{RXNORM_BASE_URL}/approximateTerm.json"
    approx_params = {"term": name, "maxEntries": 5}
    data = await _fetch_json(client, approx_url, params=approx_params)
    candidates = data.get("approximateGroup", {}).get("candidate", [])
    best_score = -math.inf
    best_id: Optional[str] = None
    for candidate in candidates:
        try:
            score = float(candidate.get("score", 0))
        except (TypeError, ValueError):
            score = 0.0
        rxcui = candidate.get("rxcui")
        if not rxcui:
            continue
        if score > best_score:
            best_score = score
            best_id = rxcui
    return best_id


async def normalise_names_to_rxcui(names: Iterable[str]) -> Dict[str, str]:
    results: Dict[str, str] = {}
    async with httpx.AsyncClient() as client:
        tasks = []
        for name in names:
            normalised = normalise_input_name(name)
            async with _CACHE_LOCK:
                cached = RXCUI_CACHE.get(normalised)
            if cached:
                METRICS["rxnorm_cache_hits"] += 1
                results[name] = cached
                continue
            tasks.append((name, normalised, asyncio.create_task(_resolve_with_rxnorm(client, name))))
        for name, normalised, task in tasks:
            try:
                rxcui = await task
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429:
                    raise
                continue
            if rxcui:
                async with _CACHE_LOCK:
                    RXCUI_CACHE[normalised] = rxcui
                results[name] = rxcui
    return results


def _build_openfda_queries(name: str, rxcui: str) -> List[str]:
    def quote(value: str) -> str:
        return value.replace("\"", "\\\"")

    terms = []
    if rxcui:
        terms.append(f'openfda.rxcui:"{quote(rxcui)}"')
    normalised_name = quote(name)
    for field in (
        "openfda.substance_name.exact",
        "openfda.generic_name.exact",
        "openfda.brand_name.exact",
    ):
        terms.append(f'{field}:"{normalised_name}"')
    base_query = " OR ".join(dict.fromkeys(terms))
    exists_clause = " OR ".join(f"_exists_:{field}" for field in OPENFDA_INTERACTION_FIELDS)
    queries = []
    if base_query:
        queries.append(f"({base_query}) AND ({exists_clause})")
        queries.append(f"({base_query})")
    return queries or ["_exists_:drug_interactions"]


def _normalise_snippet(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip().casefold()


def _iter_field_values(value: Any) -> Iterable[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [str(v) for v in value.values() if v]
    if isinstance(value, list):
        texts: List[str] = []
        for item in value:
            texts.extend(list(_iter_field_values(item)))
        return texts
    return [str(value)]


def _collect_openfda_snippets(
    label: Dict[str, Any],
    name: str,
    seen_snippets: Set[str],
) -> List[Dict[str, Any]]:
    set_id = label.get("set_id")
    effective_time = label.get("effective_time")
    provenance_base = {
        "set_id": set_id,
        "label_date": effective_time,
        "section_name": "drug_interactions",
        "url": f"{DAILYMED_LABEL_URL}?setid={set_id}" if set_id else None,
    }
    snippets: List[Dict[str, Any]] = []
    for section_name in OPENFDA_SECTION_FIELDS:
        if section_name not in label:
            continue
        for raw_text in _iter_field_values(label[section_name]):
            cleaned = raw_text.strip()
            if not cleaned:
                continue
            key = _normalise_snippet(cleaned)
            if key in seen_snippets:
                continue
            seen_snippets.add(key)
            snippets.append(
                {
                    "text": cleaned,
                    "provenance": {**provenance_base, "section_name": section_name},
                    "source_name": name,
                    "source": "openfda",
                }
            )
    return snippets


async def _fetch_openfda_sections(
    client: httpx.AsyncClient, name: str, rxcui: str
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    snippets: List[Dict[str, Any]] = []
    stats = {"queries": 0, "labels": 0, "hits": 0}
    seen_labels: Set[str] = set()
    seen_snippets: Set[str] = set()
    for query in _build_openfda_queries(name, rxcui):
        skip = 0
        while skip < MAX_OPENFDA_RECORDS and len(snippets) < MAX_OPENFDA_RECORDS:
            params = {
                "search": query,
                "limit": OPENFDA_PAGE_SIZE,
                "skip": skip,
                "sort": "effective_time:desc",
            }
            stats["queries"] += 1
            try:
                data = await _fetch_json(client, OPENFDA_LABEL_URL, params=params)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:
                    break
                if exc.response.status_code == 429:
                    raise
                break
            results = data.get("results") or []
            if not results:
                break
            for label in results:
                label_key = label.get("set_id") or label.get("id") or str(label)
                if label_key in seen_labels:
                    continue
                seen_labels.add(label_key)
                stats["labels"] += 1
                label_snippets = _collect_openfda_snippets(label, name, seen_snippets)
                if label_snippets:
                    stats["hits"] += 1
                    snippets.extend(label_snippets)
                if len(snippets) >= MAX_OPENFDA_RECORDS:
                    break
            if len(results) < OPENFDA_PAGE_SIZE:
                break
            skip += OPENFDA_PAGE_SIZE
        if len(snippets) >= MAX_OPENFDA_RECORDS:
            break
    return snippets, stats


def _flatten_xml_text(element: Optional["ET.Element"]) -> str:
    if element is None:
        return ""
    parts: List[str] = []

    def walk(node: "ET.Element") -> None:
        if node.text:
            parts.append(node.text)
        for child in list(node):
            walk(child)
            if child.tail:
                parts.append(child.tail)

    walk(element)
    return " ".join(part.strip() for part in parts if part.strip())


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


async def _fetch_dailymed_sections(
    client: httpx.AsyncClient,
    name: str,
    *,
    seen_snippets: Optional[Set[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    stats = {"queries": 0, "labels": 0, "hits": 0}
    snippets: List[Dict[str, Any]] = []
    seen = seen_snippets if seen_snippets is not None else set()
    params = {"drug_name": name, "pagesize": DAILYMED_MAX_SPLS}
    stats["queries"] += 1
    try:
        data = await _fetch_json(client, DAILYMED_SPLS_URL, params=params)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in {404, 204}:
            return snippets, stats
        if exc.response.status_code == 429:
            raise
        return snippets, stats
    spls = data.get("data") or []
    for item in spls[:DAILYMED_MAX_SPLS]:
        set_id = item.get("setid")
        if not set_id:
            continue
        stats["labels"] += 1
        try:
            xml_text = await _fetch_text(
                client,
                DAILYMED_SPL_XML_URL.format(setid=set_id),
                headers={"Accept": "application/xml"},
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                raise
            continue
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:  # pragma: no cover - depends on upstream data quality
            continue
        ns = {"hl7": "urn:hl7-org:v3"}
        for section in root.findall(".//hl7:section", ns):
            title_el = section.find("hl7:title", ns)
            title_text = _collapse_whitespace(_flatten_xml_text(title_el))
            if not title_text or "interaction" not in title_text.casefold():
                continue
            text_el = section.find("hl7:text", ns)
            section_text = _collapse_whitespace(_flatten_xml_text(text_el))
            if not section_text:
                continue
            key = _normalise_snippet(section_text)
            if key in seen:
                continue
            seen.add(key)
            snippets.append(
                {
                    "text": section_text,
                    "provenance": {
                        "set_id": set_id,
                        "label_date": item.get("published_date"),
                        "section_name": title_text,
                        "url": f"{DAILYMED_LABEL_URL}?setid={set_id}",
                    },
                    "source_name": name,
                    "source": "dailymed",
                }
            )
        if snippets:
            stats["hits"] += 1
    return snippets, stats


async def fetch_drug_interaction_sections(
    rxcui_map: Dict[str, str]
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, int]]]:
    sections: Dict[str, List[Dict[str, Any]]] = {}
    aggregate_stats = {
        "openfda": {"queries": 0, "labels": 0, "hits": 0},
        "dailymed": {"queries": 0, "labels": 0, "hits": 0},
    }
    headers = {"User-Agent": "Rakhshai-LabelCheck/1.1"}
    async with httpx.AsyncClient(headers=headers) as client:
        for name, rxcui in rxcui_map.items():
            try:
                openfda_snippets, openfda_stats = await _fetch_openfda_sections(client, name, rxcui)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429:
                    raise
                openfda_snippets, openfda_stats = [], {"queries": 0, "labels": 0, "hits": 0}
            for key in aggregate_stats["openfda"]:
                aggregate_stats["openfda"][key] += openfda_stats.get(key, 0)

            combined = list(openfda_snippets)
            daily_stats = {"queries": 0, "labels": 0, "hits": 0}
            if not combined:
                seen = {_normalise_snippet(item["text"]) for item in openfda_snippets}
                try:
                    daily_snippets, daily_stats = await _fetch_dailymed_sections(
                        client, name, seen_snippets=seen
                    )
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code == 429:
                        raise
                    daily_snippets, daily_stats = [], {"queries": 0, "labels": 0, "hits": 0}
                combined.extend(daily_snippets)
            for key in aggregate_stats["dailymed"]:
                aggregate_stats["dailymed"][key] += daily_stats.get(key, 0)

            sections[name] = combined
    return sections, aggregate_stats


@lru_cache(maxsize=1)
def load_nlp() -> Language:
    try:
        nlp = spacy.load("en_core_sci_sm")
    except Exception:  # pragma: no cover - model availability is environment-specific
        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            nlp = spacy.blank("en")
    if "entity_ruler" not in nlp.pipe_names:
        if "ner" in nlp.pipe_names:
            nlp.add_pipe("entity_ruler", before="ner")
        else:
            nlp.add_pipe("entity_ruler")
    if not (nlp.has_pipe("senter") or nlp.has_pipe("parser")):
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
    if "negex" not in nlp.pipe_names:
        try:
            import negspacy  # noqa: F401

            nlp.add_pipe("negex", config={"neg_termset": "en_clinical"})
        except (ImportError, ValueError):
            pass
    return nlp


def _ensure_drug_patterns(nlp: Language, names: Sequence[str]) -> None:
    ruler: EntityRuler = nlp.get_pipe("entity_ruler")  # type: ignore[assignment]
    if not hasattr(nlp, "_drug_patterns"):
        nlp._drug_patterns = set()  # type: ignore[attr-defined]
    patterns = []
    for name in names:
        normalised = normalise_input_name(name)
        if normalised in nlp._drug_patterns:  # type: ignore[attr-defined]
            continue
        patterns.append({"label": "DRUG", "pattern": name})
        if name.lower() != name.upper():
            patterns.append({"label": "DRUG", "pattern": name.lower()})
        nlp._drug_patterns.add(normalised)  # type: ignore[attr-defined]
    if patterns:
        ruler.add_patterns(patterns)


def _has_trigger(sentence: str) -> bool:
    return any(pattern.search(sentence) for pattern in TRIGGER_PATTERNS)


def _class_mentions(sentence: str) -> Set[str]:
    matches = set()
    for pattern in CLASS_TRIGGER_PATTERNS:
        match = pattern.search(sentence)
        if match:
            matches.add(match.group(0))
    return matches


RXCLASS_RELASOURCES = [
    "RXNORM",
    "MEDRT",
    "DAILYMED",
    "FDASPL",
    "VA",
    "ATC",
    "ATCPROD",
    "SNOMEDCT",
    "CDC",
    "FMTSME",
]


def _candidate_class_names(class_phrase: str) -> List[str]:
    phrase = _collapse_whitespace(class_phrase)
    if not phrase:
        return []
    candidates = [phrase]
    lower = phrase.lower()
    for prefix in ("strong ", "moderate ", "weak "):
        if lower.startswith(prefix):
            remainder = phrase[len(prefix) :]
            if remainder:
                candidates.append(remainder)
    if " inhibitors" in lower and "cyp" in lower:
        candidates.append(phrase.replace(" inhibitors", " inhibitor"))
    unique = list(dict.fromkeys(candidates))
    return unique


async def _resolve_class_members(
    client: httpx.AsyncClient, class_phrase: str
) -> Tuple[Set[str], Set[str]]:
    async with _CACHE_LOCK:
        cached = RXCLASS_CACHE.get(class_phrase)
    if cached is not None:
        name_list, rxcui_list = cached
        return set(name_list), set(rxcui_list)

    names: Set[str] = set()
    rxcuis: Set[str] = set()
    for candidate in _candidate_class_names(class_phrase):
        params = {"className": candidate}
        url = f"{RXCLASS_BASE_URL}/class/byName.json"
        try:
            data = await _fetch_json(client, url, params=params)
        except Exception:
            continue
        concepts = data.get("rxclassMinConceptList", {}).get("rxclassMinConcept", [])
        for concept in concepts:
            class_id = concept.get("classId")
            if not class_id:
                continue
            for source in RXCLASS_RELASOURCES:
                member_params = {
                    "classId": class_id,
                    "relaSource": source,
                    "trans": 0,
                    "ttys": "IN+PIN+SCD+SBD",
                }
                try:
                    members_data = await _fetch_json(
                        client, f"{RXCLASS_BASE_URL}/classMembers.json", params=member_params
                    )
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code == 400:
                        continue
                    raise
                except Exception:
                    continue
                member_group = members_data.get("drugMemberGroup", {}).get("drugMember", [])
                for member in member_group:
                    min_concept = member.get("minConcept", {})
                    member_name = min_concept.get("name")
                    member_rxcui = min_concept.get("rxcui")
                    if member_name:
                        names.add(member_name)
                    if member_rxcui:
                        rxcuis.add(str(member_rxcui))
            if names or rxcuis:
                break
        if names or rxcuis:
            break

    async with _CACHE_LOCK:
        RXCLASS_CACHE[class_phrase] = (tuple(sorted(names)), tuple(sorted(rxcuis)))
    return names, rxcuis


def _normalise_for_lookup(value: str) -> str:
    return normalise_input_name(value)


async def analyse_interactions(
    input_names: List[str],
    sections: Dict[str, List[Dict[str, Any]]],
    rxcui_map: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    nlp = load_nlp()
    _ensure_drug_patterns(nlp, input_names)
    interactions: List[Dict[str, Any]] = []
    seen_snippets: Set[Tuple[str, str, str]] = set()
    normalised_rxcui = {}
    if rxcui_map:
        normalised_rxcui = {
            _normalise_for_lookup(name): rxcui for name, rxcui in rxcui_map.items()
        }

    async with httpx.AsyncClient() as client:
        for idx, name_a in enumerate(input_names):
            for name_b in input_names[idx + 1 :]:
                evidence: List[Dict[str, Any]] = []
                for source in (name_a, name_b):
                    for entry in sections.get(source, []):
                        text = entry["text"]
                        provenance = entry["provenance"]
                        doc = nlp(text)
                        for sent in doc.sents:
                            sentence_text = sent.text.strip()
                            if not sentence_text:
                                continue
                            key = (source, sentence_text, name_a + "|" + name_b)
                            if key in seen_snippets:
                                continue
                            sentence_entities = {
                                _normalise_for_lookup(ent.text): ent for ent in sent.ents if ent.label_ == "DRUG"
                            }
                            class_patterns = _class_mentions(sentence_text)
                            class_member_names: Set[str] = set()
                            class_member_rxcuis: Set[str] = set()
                            for pattern in class_patterns:
                                member_names, member_rxcuis = await _resolve_class_members(client, pattern)
                                class_member_names.update(
                                    {_normalise_for_lookup(m) for m in member_names}
                                )
                                class_member_rxcuis.update(member_rxcuis)
                            name_a_norm = _normalise_for_lookup(name_a)
                            name_b_norm = _normalise_for_lookup(name_b)
                            has_a = (
                                name_a_norm in sentence_entities
                                or name_a_norm in class_member_names
                                or normalised_rxcui.get(name_a_norm) in class_member_rxcuis
                            )
                            has_b = (
                                name_b_norm in sentence_entities
                                or name_b_norm in class_member_names
                                or normalised_rxcui.get(name_b_norm) in class_member_rxcuis
                            )
                            if not (has_a and has_b):
                                continue
                            negated = False
                            for ent_norm, ent in sentence_entities.items():
                                if ent_norm in {name_a_norm, name_b_norm} and getattr(ent._, "negex", False):
                                    negated = True
                                    break
                            trigger = _has_trigger(sentence_text)
                            class_hit = bool(
                                class_patterns
                                and (class_member_names or class_member_rxcuis)
                            )
                            confidence = CLASS_BASE_CONFIDENCE if class_hit else BASE_CONFIDENCE
                            if trigger:
                                confidence += TRIGGER_WEIGHT
                            if class_hit:
                                confidence += CLASS_WEIGHT
                            if negated:
                                confidence = max(0.1, confidence - NEGATION_PENALTY)
                            confidence = min(confidence, 1.0)
                            evidence.append(
                                {
                                    "pair": [name_a, name_b],
                                    "sentence": sentence_text,
                                    "provenance": provenance,
                                    "triggered": trigger,
                                    "class_derived": class_hit,
                                    "negated": negated,
                                    "confidence": round(confidence, 2),
                                }
                            )
                            seen_snippets.add(key)
                has_interaction = any(item["confidence"] >= 0.5 for item in evidence)
                if not evidence:
                    evidence = []
                interactions.append(
                    {
                        "pair": [name_a, name_b],
                        "has_interaction": has_interaction,
                        "evidence": evidence,
                        "max_confidence": max((item["confidence"] for item in evidence), default=0.0),
                    }
                )
    return interactions
