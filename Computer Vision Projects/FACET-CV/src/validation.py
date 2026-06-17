"""
Validation module for pilot study analysis in facial motor and speech behavior pipeline.

Generates confusion-style summaries comparing expected vs predicted screening
outcomes so pilot-session accuracy can be tracked across pipeline versions.

ValidationSummary accumulates per-session results and computes precision,
recall, and F1 score per indication type. PilotStudyValidator maps known
alteration types (e.g. deliberate_asymmetry, impaired_pataka_only) to their
expected indication labels and compares against pipeline predictions.

The alteration_mappings in PilotStudyValidator represent the ground truth
for pilot sessions where the participant deliberately performed a specific
disorder simulation. Correct detection means the pipeline produced exactly
the expected set of indication types with no false positives and no false
negatives.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path

from .utils import save_json


class ValidationSummary:
    """Accumulates per-session validation results and computes aggregate accuracy metrics."""

    def __init__(self):
        """Initialise empty summary; add results via add_result()."""
        self.results: List[Dict[str, Any]] = []
        self.ground_truth: Dict[str, List[str]] = {}
        self.predictions: Dict[str, List[str]] = {}

    def add_result(
        self,
        session_id: str,
        alteration_type: str,
        expected_indications: List[str],
        predicted_indications: List[str],
        confidence: float,
    ) -> None:
        """Record one session's expected vs predicted indications."""
        self.results.append(
            {
                "session_id": session_id,
                "alteration_type": alteration_type,
                "expected_indications": expected_indications,
                "predicted_indications": predicted_indications,
                "confidence": confidence,
                "matches": set(expected_indications) & set(predicted_indications),
                "false_positives": set(predicted_indications) - set(expected_indications),
                "false_negatives": set(expected_indications) - set(predicted_indications),
            }
        )
        self.ground_truth[session_id] = expected_indications
        self.predictions[session_id] = predicted_indications

    def compute_metrics(self) -> Dict[str, Any]:
        """Compute per-indication and overall precision / recall / F1."""
        if not self.results:
            return {"error": "No validation results available"}

        all_indications: set = set()
        for r in self.results:
            all_indications.update(r["expected_indications"])
            all_indications.update(r["predicted_indications"])

        indication_metrics: Dict[str, Dict[str, Any]] = {}

        for indication in all_indications:
            tp = sum(1 for r in self.results if indication in r["matches"])
            fp = sum(1 for r in self.results if indication in r["false_positives"])
            fn = sum(1 for r in self.results if indication in r["false_negatives"])
            tn = len(self.results) - tp - fp - fn

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            accuracy = (tp + tn) / len(self.results) if self.results else 0.0

            indication_metrics[indication] = {
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "true_negatives": tn,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "accuracy": accuracy,
            }

        overall_metrics = {
            "total_sessions": len(self.results),
            "total_indications": len(all_indications),
            "mean_precision": np.mean(
                [m["precision"] for m in indication_metrics.values()]
            ),
            "mean_recall": np.mean(
                [m["recall"] for m in indication_metrics.values()]
            ),
            "mean_f1": np.mean(
                [m["f1_score"] for m in indication_metrics.values()]
            ),
            "mean_confidence": np.mean([r["confidence"] for r in self.results]),
        }

        return {
            "indication_metrics": indication_metrics,
            "overall_metrics": overall_metrics,
        }

    def generate_confusion_table(self) -> pd.DataFrame:
        """Build a confusion matrix DataFrame (expected x predicted)."""
        if not self.results:
            return pd.DataFrame()

        all_indications = sorted(
            set(
                ind
                for r in self.results
                for ind in r["expected_indications"] + r["predicted_indications"]
            )
        )

        confusion_data: List[Dict[str, Any]] = []

        for expected in all_indications + ["none"]:
            row: Dict[str, Any] = {"expected": expected}
            for predicted in all_indications + ["none"]:
                count = 0
                for r in self.results:
                    exp_set = set(r["expected_indications"]) or {"none"}
                    pred_set = set(r["predicted_indications"]) or {"none"}
                    if expected in exp_set and predicted in pred_set:
                        count += 1
                row[f"pred_{predicted}"] = count
            confusion_data.append(row)

        return pd.DataFrame(confusion_data)

    def generate_summary_table(self) -> pd.DataFrame:
        """Build a per-session summary DataFrame."""
        summary_data: List[Dict[str, Any]] = []

        for r in self.results:
            summary_data.append(
                {
                    "session_id": r["session_id"],
                    "alteration_type": r["alteration_type"],
                    "expected": ", ".join(r["expected_indications"]) or "none",
                    "predicted": ", ".join(r["predicted_indications"]) or "none",
                    "matches": ", ".join(r["matches"]) or "none",
                    "false_positives": ", ".join(r["false_positives"]) or "none",
                    "false_negatives": ", ".join(r["false_negatives"]) or "none",
                    "confidence": r["confidence"],
                    "correct": (
                        len(r["false_positives"]) == 0
                        and len(r["false_negatives"]) == 0
                    ),
                }
            )

        return pd.DataFrame(summary_data)


class PilotStudyValidator:
    """Compares screening results against known alteration types for pilot studies."""

    def __init__(self, tasks_config: Dict[str, Any]):
        """Initialise validator from the loaded tasks YAML config."""
        self.tasks_config = tasks_config
        self.alteration_mappings = self._define_alteration_mappings()

    @staticmethod
    def _define_alteration_mappings() -> Dict[str, List[str]]:
        """Return the mapping from alteration descriptions to expected indication types."""
        return {
            "correct_execution": [],
            "deliberate_asymmetry": ["facial_paresis"],
            "deliberate_asymmetry_persistent": ["facial_paresis"],
            "incorrect_facial_task": ["buccofacial_apraxia"],
            "inconsistent_facial_task": ["buccofacial_apraxia"],
            "slow_articulation_all": ["dysarthria"],
            "impaired_articulation_all": ["dysarthria"],
            "impaired_pataka_only": ["speech_apraxia"],
            "complex_task_difficulty": ["speech_apraxia"],
            "consistent_phonological_errors": ["phonological_disorder"],
            "inconsistent_phonological_errors": ["speech_apraxia"],
            "mixed_asymmetry_and_apraxia": ["facial_paresis", "buccofacial_apraxia"],
        }

    def get_expected_indications(self, alteration_type: str) -> List[str]:
        """Look up the expected indication types for an alteration."""
        return self.alteration_mappings.get(alteration_type, [])

    def validate_session(
        self,
        session_id: str,
        alteration_type: str,
        screening_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compare a single session's predictions against its expected alteration."""
        expected = self.get_expected_indications(alteration_type)
        predicted = [
            ind["indication_type"]
            for ind in screening_results.get("indications", [])
        ]
        confidence = screening_results.get("confidence", {}).get("overall", 0.0)

        matches = set(expected) & set(predicted)
        false_positives = set(predicted) - set(expected)
        false_negatives = set(expected) - set(predicted)

        return {
            "session_id": session_id,
            "alteration_type": alteration_type,
            "expected_indications": expected,
            "predicted_indications": predicted,
            "confidence": confidence,
            "matches": list(matches),
            "false_positives": list(false_positives),
            "false_negatives": list(false_negatives),
            "is_correct": len(false_positives) == 0 and len(false_negatives) == 0,
        }

    def generate_validation_report(
        self,
        validation_results: List[Dict[str, Any]],
        output_path: Path,
    ) -> Dict[str, Any]:
        """Aggregate multiple session validations into a single report."""
        summary = ValidationSummary()

        for result in validation_results:
            summary.add_result(
                session_id=result["session_id"],
                alteration_type=result["alteration_type"],
                expected_indications=result["expected_indications"],
                predicted_indications=result["predicted_indications"],
                confidence=result["confidence"],
            )

        metrics = summary.compute_metrics()

        summary_table = summary.generate_summary_table()
        summary_table.to_csv(output_path.with_suffix(".csv"), index=False)

        confusion_table = summary.generate_confusion_table()
        confusion_table.to_csv(output_path.parent / "confusion_matrix.csv", index=False)

        report = {
            "metrics": metrics,
            "n_sessions": len(validation_results),
            "n_correct": sum(1 for r in validation_results if r["is_correct"]),
            "accuracy": (
                sum(1 for r in validation_results if r["is_correct"])
                / len(validation_results)
                if validation_results
                else 0.0
            ),
        }

        save_json(report, output_path.with_suffix(".json"))
        return report


def create_pilot_validator(tasks_config: Dict[str, Any]) -> PilotStudyValidator:
    """Factory: build a PilotStudyValidator from task configuration."""
    return PilotStudyValidator(tasks_config)
