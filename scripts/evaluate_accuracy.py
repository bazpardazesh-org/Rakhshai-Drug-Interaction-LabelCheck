"""Evaluate interaction detection accuracy with and without spaCy models."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import sys

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing_extensions import Literal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.interaction_checker import analyse_interactions, load_nlp


@dataclass(frozen=True)
class EvaluationCase:
    """Fixture describing a synthetic evaluation scenario."""

    name: str
    input_names: Sequence[str]
    sections: Dict[str, List[Dict[str, object]]]
    expected_pairs: Dict[Tuple[str, str], bool]


EVALUATION_CASES: Tuple[EvaluationCase, ...] = (
    EvaluationCase(
        name="direct_trigger",
        input_names=("Aspirin", "Ibuprofen"),
        sections={
            "Aspirin": [
                {
                    "text": "Aspirin should not be used with Ibuprofen due to increased bleeding risk.",
                    "provenance": {
                        "set_id": "synthetic-1",
                        "label_date": "20240101",
                        "section_name": "warnings",
                        "url": None,
                    },
                }
            ],
            "Ibuprofen": [],
        },
        expected_pairs={("Aspirin", "Ibuprofen"): True},
    ),
    EvaluationCase(
        name="no_evidence",
        input_names=("Metformin", "Lisinopril"),
        sections={
            "Metformin": [
                {
                    "text": "Metformin therapy is generally well tolerated when taken as monotherapy.",
                    "provenance": {
                        "set_id": "synthetic-2a",
                        "label_date": "20240101",
                        "section_name": "clinical_pharmacology",
                        "url": None,
                    },
                }
            ],
            "Lisinopril": [
                {
                    "text": "Lisinopril may be combined with thiazide diuretics without dosage adjustment.",
                    "provenance": {
                        "set_id": "synthetic-2b",
                        "label_date": "20240101",
                        "section_name": "dosage",
                        "url": None,
                    },
                }
            ],
        },
        expected_pairs={("Metformin", "Lisinopril"): False},
    ),
    EvaluationCase(
        name="class_trigger",
        input_names=("Simvastatin", "Clarithromycin"),
        sections={
            "Simvastatin": [
                {
                    "text": "Strong CYP3A4 inhibitors such as clarithromycin can markedly increase simvastatin exposure.",
                    "provenance": {
                        "set_id": "synthetic-3",
                        "label_date": "20240101",
                        "section_name": "drug_interactions",
                        "url": None,
                    },
                }
            ]
        },
        expected_pairs={("Simvastatin", "Clarithromycin"): True},
    ),
    EvaluationCase(
        name="separate_sentences",
        input_names=("Warfarin", "Vitamin K"),
        sections={
            "Warfarin": [
                {
                    "text": "Warfarin requires regular INR monitoring. Vitamin K supplementation is used for deficiency states.",
                    "provenance": {
                        "set_id": "synthetic-4",
                        "label_date": "20240101",
                        "section_name": "clinical_pharmacology",
                        "url": None,
                    },
                }
            ]
        },
        expected_pairs={("Warfarin", "Vitamin K"): False},
    ),
)


async def _run_case(case: EvaluationCase) -> Dict[Tuple[str, str], bool]:
    """Execute a single evaluation case and return model predictions."""

    interactions = await analyse_interactions(list(case.input_names), case.sections)
    predictions: Dict[Tuple[str, str], bool] = {}
    for interaction in interactions:
        pair = tuple(interaction["pair"])
        predictions[pair] = bool(interaction["has_interaction"])
    return predictions


async def _collect_predictions(cases: Iterable[EvaluationCase]) -> Tuple[List[bool], List[bool]]:
    """Gather gold labels and predictions for the provided cases."""

    gold: List[bool] = []
    predictions: List[bool] = []
    for case in cases:
        case_predictions = await _run_case(case)
        for pair in combinations(case.input_names, 2):
            if pair not in case.expected_pairs and pair[::-1] not in case.expected_pairs:
                continue
            key = pair if pair in case.expected_pairs else pair[::-1]
            gold.append(case.expected_pairs[key])
            predictions.append(case_predictions.get(key, False))
    return gold, predictions


def _metric_report(labels: List[bool], predictions: List[bool]) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
    }


@contextmanager
def force_blank_pipeline() -> Iterable[None]:
    """Force :func:`load_nlp` to fall back to a blank spaCy pipeline."""

    import spacy
    from unittest.mock import patch

    load_nlp.cache_clear()
    with patch.object(spacy, "load", side_effect=OSError("model unavailable")):
        yield
    load_nlp.cache_clear()


def evaluate(mode: Literal["spacy", "blank"]) -> Dict[str, float]:
    """Run the evaluation suite under the requested NLP configuration."""

    load_nlp.cache_clear()
    if mode == "blank":
        with force_blank_pipeline():
            labels, predictions = asyncio.run(_collect_predictions(EVALUATION_CASES))
    else:
        labels, predictions = asyncio.run(_collect_predictions(EVALUATION_CASES))
    return _metric_report(labels, predictions)


def main() -> None:
    spaCy_metrics = evaluate("spacy")
    blank_metrics = evaluate("blank")

    print("Evaluation results (4 synthetic interaction cases):")
    print("- With spaCy model:")
    for name, value in spaCy_metrics.items():
        print(f"    {name.title():<10}: {value:.2f}")
    print("- Blank pipeline fallback:")
    for name, value in blank_metrics.items():
        print(f"    {name.title():<10}: {value:.2f}")


if __name__ == "__main__":
    main()
