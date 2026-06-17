"""
Metrics computation for the FACET-CV pipeline.

Aggregates frame-level features into repetition-level, task-level, and
session-level summary statistics.  These metrics are the primary inputs for
anomaly detection and clinical decision support.

The MetricsComputer class handles three levels of aggregation:
  1. Per-repetition: mean, std, min, max, range, and time-to-peak for every
     numeric feature, plus asymmetry summaries and detection rate.
  2. Per-task: cross-repetition mean, std, and coefficient of variation,
     plus consistency and dominant-side measures.
  3. Per-session: overall totals and global asymmetry/detection summaries.

Reference for DDK speech motor analysis context:
  Allison et al. (2022) Diadochokinesis in motor speech disorders.
  Am J Speech Lang Pathol 31(5):2239-2259.
  doi:10.1044/2022_AJSLP-21-00241
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

from .utils import get_numeric_feature_columns


class MetricsComputer:
    """Aggregates features into repetition, task, and session metrics."""

    _EXTRA_EXCLUDES = frozenset({"task_group", "task_id", "task_name"})

    def __init__(self, features_config: Dict[str, Any], tasks_config: Dict[str, Any]):
        """Initialise with feature and task configuration dicts.

        Args:
            features_config: Dict from features.yaml; used for aggregation
                method settings.
            tasks_config: Dict from tasks.yaml; reserved for future task-specific
                aggregation rules.
        """
        self.features_config = features_config
        self.tasks_config = tasks_config
        self.aggregation_config = features_config.get("aggregation_methods", {})

    def compute_repetition_metrics(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate frame-level features into per-repetition summary statistics.

        Groups measurement frames by (task_group, task_id, repetition) when task
        information is present, or by repetition alone when it is not.  For each
        group, calls _compute_single_repetition_metrics and collects the results.

        Returns an empty DataFrame when no measurement frames are found.
        """
        if "repetition" not in features_df.columns:
            features_df = features_df.copy()
            features_df["repetition"] = 0

        measurement_df = features_df[features_df["segment"] == "measurement"]
        if len(measurement_df) == 0:
            measurement_df = features_df

        has_task_info = (
            "task_group" in measurement_df.columns
            and measurement_df["task_group"].notna().any()
            and (measurement_df["task_group"] != "0").any()
        )

        metrics_list: List[Dict[str, Any]] = []

        if has_task_info:
            group_cols = ["task_group", "task_id", "repetition"]
            for (tg, tid, rep), rep_df in measurement_df.groupby(group_cols, dropna=False):
                if len(rep_df) == 0:
                    continue
                metrics_list.append(self._compute_single_repetition_metrics(rep_df, rep))
        else:
            for rep in measurement_df["repetition"].unique():
                rep_df = measurement_df[measurement_df["repetition"] == rep]
                if len(rep_df) == 0:
                    continue
                metrics_list.append(self._compute_single_repetition_metrics(rep_df, rep))

        if not metrics_list:
            return pd.DataFrame()
        return pd.DataFrame(metrics_list)

    def _compute_single_repetition_metrics(
        self, rep_df: pd.DataFrame, rep_id: int
    ) -> Dict[str, Any]:
        """Compute summary statistics for a single repetition group.

        Returns a dict containing:
          - repetition, n_frames, duration_sec, task_group, task_id, task_name
          - For each numeric feature: _mean, _std, _max, _min, _range,
            and optionally _time_to_peak (when time_rel_sec is present)
          - mean_asymmetry_ratio, max_asymmetry_ratio (from asymmetry_ratio_ columns)
          - mean_signed_asymmetry, dominant_side, and per-region signed_asymmetry_*
            (from asymmetry_ columns that are not ratio columns)
          - detection_rate (when detection_success is present)
        """
        if "timestamp_abs" in rep_df.columns and len(rep_df) > 0:
            _t = rep_df["timestamp_abs"].dropna()
            _duration = float(_t.max() - _t.min()) if len(_t) >= 2 else 0.0
        elif "time_rel_sec" in rep_df.columns and len(rep_df) > 0:
            _tr = rep_df["time_rel_sec"].dropna()
            _duration = float(_tr.max() - _tr.min()) if len(_tr) >= 2 else 0.0
        else:
            _duration = 0.0

        metrics: Dict[str, Any] = {
            "repetition": rep_id,
            "n_frames": len(rep_df),
            "duration_sec": _duration,
        }

        if "task_group" in rep_df.columns:
            tg = rep_df["task_group"].iloc[0] if len(rep_df) > 0 else None
            metrics["task_group"] = str(tg) if pd.notna(tg) else "0"
        else:
            metrics["task_group"] = "0"

        if "task_id" in rep_df.columns:
            tid = rep_df["task_id"].iloc[0] if len(rep_df) > 0 else None
            metrics["task_id"] = int(tid) if pd.notna(tid) else 0
        else:
            metrics["task_id"] = 0

        if "task_name" in rep_df.columns:
            tname = rep_df["task_name"].iloc[0] if len(rep_df) > 0 else None
            metrics["task_name"] = str(tname) if pd.notna(tname) else "(no task selected)"
        else:
            metrics["task_name"] = "(no task selected)"

        feature_cols = get_numeric_feature_columns(rep_df, self._EXTRA_EXCLUDES)

        for col in feature_cols:
            values = rep_df[col].dropna().values
            if len(values) > 0:
                metrics[f"{col}_mean"] = float(np.mean(values))
                metrics[f"{col}_std"] = float(np.std(values))
                metrics[f"{col}_max"] = float(np.max(values))
                metrics[f"{col}_min"] = float(np.min(values))
                metrics[f"{col}_range"] = float(np.max(values) - np.min(values))

                if "time_rel_sec" in rep_df.columns:
                    peak_idx = np.argmax(values)
                    metrics[f"{col}_time_to_peak"] = float(
                        rep_df["time_rel_sec"].iloc[peak_idx]
                    )

        asymmetry_cols = [c for c in rep_df.columns if c.startswith("asymmetry_ratio_")]
        if asymmetry_cols:
            asym_values = rep_df[asymmetry_cols].values.flatten()
            asym_values = asym_values[~np.isnan(asym_values)]
            if len(asym_values) > 0:
                metrics["mean_asymmetry_ratio"] = float(np.mean(np.abs(asym_values)))
                metrics["max_asymmetry_ratio"] = float(np.max(np.abs(asym_values)))

        signed_cols = [c for c in rep_df.columns
                       if c.startswith("asymmetry_") and not c.startswith("asymmetry_ratio_")]
        if signed_cols:
            signed_vals = rep_df[signed_cols].values.flatten()
            signed_vals = signed_vals[~np.isnan(signed_vals)]
            if len(signed_vals) > 0:
                mean_signed = float(np.mean(signed_vals))
                metrics["mean_signed_asymmetry"] = mean_signed
                if abs(mean_signed) < 0.01:
                    metrics["dominant_side"] = "symmetric"
                elif mean_signed > 0:
                    metrics["dominant_side"] = "right"
                else:
                    metrics["dominant_side"] = "left"

                for col in signed_cols:
                    region = col.replace("asymmetry_", "")
                    region_vals = rep_df[col].dropna().values
                    if len(region_vals) > 0:
                        region_mean = float(np.mean(region_vals))
                        metrics[f"signed_asymmetry_{region}"] = region_mean
                        if abs(region_mean) < 0.01:
                            metrics[f"dominant_side_{region}"] = "symmetric"
                        elif region_mean > 0:
                            metrics[f"dominant_side_{region}"] = "right"
                        else:
                            metrics[f"dominant_side_{region}"] = "left"

        if "detection_success" in rep_df.columns:
            metrics["detection_rate"] = float(rep_df["detection_success"].mean())

        return metrics

    def compute_task_metrics(self, repetition_metrics_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate repetition-level metrics into a single per-task summary row.

        For each numeric metric column, computes the mean, std, and coefficient
        of variation across repetitions.  Also computes total frame count,
        total duration, asymmetry consistency, and dominant side.

        Returns an empty DataFrame when repetition_metrics_df is empty.
        """
        if len(repetition_metrics_df) == 0:
            return pd.DataFrame()

        task_metrics: Dict[str, Any] = {}

        numeric_cols = repetition_metrics_df.select_dtypes(include=[np.number]).columns
        exclude_cols = {"repetition", "n_frames"}
        metric_cols = [c for c in numeric_cols if c not in exclude_cols]

        for col in metric_cols:
            values = repetition_metrics_df[col].dropna().values
            if len(values) > 0:
                task_metrics[f"{col}_across_reps_mean"] = float(np.mean(values))
                task_metrics[f"{col}_across_reps_std"] = float(np.std(values))
                task_metrics[f"{col}_cv"] = (
                    float(np.std(values) / np.abs(np.mean(values)))
                    if np.mean(values) != 0
                    else 0.0
                )

        task_metrics["n_repetitions"] = len(repetition_metrics_df)
        task_metrics["total_frames"] = int(repetition_metrics_df["n_frames"].sum())
        task_metrics["total_duration_sec"] = float(
            repetition_metrics_df["duration_sec"].sum()
        )

        if "mean_asymmetry_ratio_mean" in task_metrics:
            asym_vals = repetition_metrics_df["mean_asymmetry_ratio"].dropna().values
            if len(asym_vals) > 1:
                task_metrics["asymmetry_consistency"] = max(0.0, 1.0 - float(
                    np.std(asym_vals) / (np.mean(asym_vals) + 0.001)
                ))
            else:
                task_metrics["asymmetry_consistency"] = 1.0

        if "mean_signed_asymmetry" in repetition_metrics_df.columns:
            signed_vals = repetition_metrics_df["mean_signed_asymmetry"].dropna().values
            if len(signed_vals) > 0:
                task_mean_signed = float(np.mean(signed_vals))
                task_metrics["mean_signed_asymmetry"] = task_mean_signed
                if abs(task_mean_signed) < 0.01:
                    task_metrics["dominant_side"] = "symmetric"
                elif task_mean_signed > 0:
                    task_metrics["dominant_side"] = "right"
                else:
                    task_metrics["dominant_side"] = "left"

        return pd.DataFrame([task_metrics])

    def compute_session_metrics(
        self,
        task_metrics_df: pd.DataFrame,
        repetition_metrics_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Compute session-level aggregate statistics from task and repetition data.

        Returns a dict with total_repetitions, total_frames, total_duration_sec,
        and (when available) overall_mean_asymmetry, overall_max_asymmetry,
        asymmetry_across_tasks, overall_mean_signed_asymmetry, dominant_side,
        overall_detection_rate, and per-metric session mean values aggregated
        from task_metrics_df.
        """
        session_metrics: Dict[str, Any] = {
            "total_repetitions": (
                int(repetition_metrics_df["repetition"].nunique())
                if len(repetition_metrics_df) > 0
                else 0
            ),
            "total_frames": (
                int(repetition_metrics_df["n_frames"].sum())
                if len(repetition_metrics_df) > 0
                else 0
            ),
            "total_duration_sec": (
                float(repetition_metrics_df["duration_sec"].sum())
                if len(repetition_metrics_df) > 0
                else 0.0
            ),
        }

        if "mean_asymmetry_ratio" in repetition_metrics_df.columns:
            asym_values = repetition_metrics_df["mean_asymmetry_ratio"].dropna().values
            if len(asym_values) > 0:
                session_metrics["overall_mean_asymmetry"] = float(np.mean(asym_values))
                session_metrics["overall_max_asymmetry"] = float(np.max(asym_values))
                session_metrics["asymmetry_across_tasks"] = float(
                    np.mean(asym_values > 0.15)
                )

        if "mean_signed_asymmetry" in repetition_metrics_df.columns:
            signed_vals = repetition_metrics_df["mean_signed_asymmetry"].dropna().values
            if len(signed_vals) > 0:
                session_signed = float(np.mean(signed_vals))
                session_metrics["overall_mean_signed_asymmetry"] = session_signed
                if abs(session_signed) < 0.01:
                    session_metrics["dominant_side"] = "symmetric"
                elif session_signed > 0:
                    session_metrics["dominant_side"] = "right"
                else:
                    session_metrics["dominant_side"] = "left"

        if "detection_rate" in repetition_metrics_df.columns:
            session_metrics["overall_detection_rate"] = float(
                repetition_metrics_df["detection_rate"].mean()
            )

        mean_cols = [c for c in task_metrics_df.columns if c.endswith("_across_reps_mean")]
        for col in mean_cols:
            base_name = col.replace("_across_reps_mean", "")
            values = task_metrics_df[col].dropna().values
            if len(values) > 0:
                session_metrics[f"{base_name}_session_mean"] = float(np.mean(values))

        return session_metrics

    def compute_execution_correctness_score(
        self,
        repetition_metrics_df: pd.DataFrame,
        task_config: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Score how correctly the task was executed on a 0.0-1.0 scale.

        Combines up to three sub-scores: activation level (presence of movement),
        duration accuracy (closeness to expected task duration when task_config
        provides expected_duration_sec), and repetition consistency (inverse of
        the mean coefficient of variation across features).

        Returns 0.5 when no sub-scores can be computed.
        """
        if len(repetition_metrics_df) == 0:
            return 0.0

        scores: List[float] = []

        if "mean_activation_mean" in repetition_metrics_df.columns:
            activation = repetition_metrics_df["mean_activation_mean"].mean()
            scores.append(min(1.0, activation * 2) if activation > 0 else 0.0)

        if "duration_sec" in repetition_metrics_df.columns and task_config:
            expected_duration = task_config.get("expected_duration_sec", 2.0)
            actual_duration = repetition_metrics_df["duration_sec"].mean()
            duration_ratio = actual_duration / expected_duration if expected_duration > 0 else 1.0
            scores.append(1.0 - min(1.0, abs(1.0 - duration_ratio)))

        cv_cols = [c for c in repetition_metrics_df.columns if c.endswith("_cv")]
        if cv_cols:
            mean_cv = repetition_metrics_df[cv_cols].mean().mean()
            scores.append(max(0.0, 1.0 - mean_cv))

        return float(np.mean(scores)) if scores else 0.5

    def compute_articulation_score(
        self, repetition_metrics_df: pd.DataFrame, task_name: str
    ) -> float:
        """Compute an articulation quality score for speech tasks on a 0.0-1.0 scale.

        Combines activation level (presence of orofacial movement) and
        feature variability (lower within-repetition std implies more
        consistent articulation).  Returns 0.5 when no data is available.
        """
        if len(repetition_metrics_df) == 0:
            return 0.0

        scores: List[float] = []

        if "mean_activation_mean" in repetition_metrics_df.columns:
            activation = repetition_metrics_df["mean_activation_mean"].mean()
            scores.append(min(1.0, activation * 2))

        mean_cols = [
            c
            for c in repetition_metrics_df.columns
            if "_mean" in c and "asymmetry" not in c
        ]
        if mean_cols:
            std_cols = [
                c.replace("_mean", "_std")
                for c in mean_cols
                if c.replace("_mean", "_std") in repetition_metrics_df.columns
            ]
            if std_cols:
                variability = repetition_metrics_df[std_cols].mean().mean()
                scores.append(max(0.0, 1.0 - variability))

        return float(np.mean(scores)) if scores else 0.5

    def compute_error_consistency(self, repetition_metrics_df: pd.DataFrame) -> float:
        """Compute how consistent metric values are across repetitions on a 0-1 scale.

        A score of 1.0 means perfectly consistent (no variability across reps);
        lower scores indicate that feature values vary substantially between
        repetitions.  Returns 1.0 when fewer than two repetitions are available
        or when no _mean columns are found.
        """
        if len(repetition_metrics_df) < 2:
            return 1.0

        mean_cols = [
            c
            for c in repetition_metrics_df.columns
            if c.endswith("_mean") and "asymmetry" not in c
        ]
        if not mean_cols:
            return 1.0

        correlations: List[float] = []
        for col in mean_cols:
            values = repetition_metrics_df[col].dropna().values
            if len(values) > 1:
                cv = np.std(values) / (np.abs(np.mean(values)) + 0.001)
                correlations.append(1.0 - min(1.0, cv))

        return float(np.mean(correlations)) if correlations else 1.0


def create_metrics_computer(
    features_config: Dict[str, Any], tasks_config: Dict[str, Any]
) -> MetricsComputer:
    """Factory: build a MetricsComputer from configuration dicts."""
    return MetricsComputer(features_config, tasks_config)
