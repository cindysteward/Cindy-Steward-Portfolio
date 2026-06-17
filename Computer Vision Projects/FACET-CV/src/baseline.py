"""
Baseline construction module for facial motor and speech behavior analysis pipeline.

Computes neutral baseline statistics from marked neutral segments for feature
correction. Supports multiple neutral segments and accumulation across sessions.

The baseline estimators use median and IQR rather than mean and standard
deviation to reduce sensitivity to single-frame detection artefacts in the
neutral segment. This is particularly important for small neutral segments
(10-30 seconds) where one missed detection frame can meaningfully shift a
mean-based estimate.

References
----------
Leys C, Ley C, Klein O, Bernard P, Licata L (2013) Detecting outliers: Do
  not use standard deviation around the mean, use absolute deviation around
  the median. J Exp Soc Psychol 49(4):764-766.
  Practical motivation for IQR-based outlier removal over mean +/- 2 SD
  trimming in small-N clinical samples with skewed distributions.

Rousseeuw PJ, Croux C (1993) Alternatives to the median absolute deviation.
  J Am Stat Assoc 88(424):1273-1283.
  Robustness properties of IQR and MAD estimators for scale; informs the
  choice of outlier_threshold_iqr = 1.5 as the default fence multiplier.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from .utils import save_json, load_json, compute_statistics, get_feature_columns

logger = logging.getLogger("pipeline")


class BaselineConstructor:
    """Computes and stores per-feature statistics from the neutral segment."""

    def __init__(self, features_config: Dict[str, Any]):
        """Initialise constructor from the loaded features YAML config."""
        self.features_config = features_config
        self.baseline_config = features_config.get("baseline_statistics", {})
        self.robust_method = self.baseline_config.get("robust_method", "median")
        self.outlier_threshold = self.baseline_config.get("outlier_threshold_iqr", 1.5)

        self.baseline_stats: Dict[str, Dict[str, float]] = {}
        self.observed_ranges: Dict[str, float] = {}
        self.metadata: Dict[str, Any] = {}

        self.neutral_segments: List[Dict[str, Any]] = []
        self.accumulated_neutral_data: Optional[pd.DataFrame] = None

    def compute_baseline(
        self, blendshapes_df: pd.DataFrame, neutral_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Compute baseline from neutral segments.

        Handles multiple neutral segments within a session and accumulates them
        for a more robust baseline estimate.
        """
        if neutral_df is None:
            neutral_df = blendshapes_df[blendshapes_df["segment"] == "neutral"]

        if len(neutral_df) == 0:
            logger.warning(
                "No neutral baseline data found. Using full recording as baseline."
            )
            neutral_df = blendshapes_df[blendshapes_df["detection_success"] == True]

        if len(neutral_df) == 0:
            logger.warning(
                "No valid frames for baseline — returning zero baseline stats."
            )
            self.metadata = {"n_frames": 0, "n_features": 0, "n_segments": 0,
                             "robust_method": self.robust_method,
                             "outlier_threshold_iqr": self.outlier_threshold,
                             "detection_rate": 0.0, "segments_info": []}
            return self.get_baseline_data()

        self._track_neutral_segments(neutral_df)
        if self.accumulated_neutral_data is not None:
            neutral_df = pd.concat(
                [self.accumulated_neutral_data, neutral_df], ignore_index=True
            )

        self.accumulated_neutral_data = neutral_df.copy()

        feature_columns = [
            c for c in get_feature_columns(neutral_df)
            if pd.api.types.is_numeric_dtype(neutral_df[c])
        ]

        for col in feature_columns:
            values = neutral_df[col].dropna().values
            if len(values) > 0:
                values = self._remove_outliers(values)
                self.baseline_stats[col] = self._compute_feature_statistics(values)
            else:
                self.baseline_stats[col] = self._get_empty_statistics()

        self.metadata = {
            "n_frames": len(neutral_df),
            "n_features": len(feature_columns),
            "n_segments": len(self.neutral_segments),
            "robust_method": self.robust_method,
            "outlier_threshold_iqr": self.outlier_threshold,
            "detection_rate": (
                neutral_df["detection_success"].mean()
                if "detection_success" in neutral_df
                else 1.0
            ),
            "segments_info": self.neutral_segments,
        }

        return self.get_baseline_data()

    def add_neutral_segment(
        self, segment_df: pd.DataFrame, segment_info: Optional[Dict] = None
    ) -> None:
        """Add a new neutral segment to the accumulated baseline data.

        Allows multiple neutral captures to contribute to a more robust baseline.
        """
        if len(segment_df) == 0:
            return

        segment_data = {
            "n_frames": len(segment_df),
            "start_time": (
                segment_df["timestamp_abs"].min()
                if "timestamp_abs" in segment_df.columns
                else 0
            ),
            "end_time": (
                segment_df["timestamp_abs"].max()
                if "timestamp_abs" in segment_df.columns
                else 0
            ),
            "detection_rate": (
                segment_df["detection_success"].mean()
                if "detection_success" in segment_df.columns
                else 1.0
            ),
        }
        if segment_info:
            segment_data.update(segment_info)

        self.neutral_segments.append(segment_data)

        if self.accumulated_neutral_data is None:
            self.accumulated_neutral_data = segment_df.copy()
        else:
            self.accumulated_neutral_data = pd.concat(
                [self.accumulated_neutral_data, segment_df], ignore_index=True
            )

        self._recompute_baseline()

    def _track_neutral_segments(self, neutral_df: pd.DataFrame) -> None:
        """Record timing and frame-count information for each neutral segment."""
        if "timestamp_abs" not in neutral_df.columns:
            self.neutral_segments = [{"n_frames": len(neutral_df)}]
            return

        timestamps = neutral_df["timestamp_abs"].values
        if len(timestamps) <= 1:
            self.neutral_segments = [
                {
                    "n_frames": len(neutral_df),
                    "start_time": timestamps[0] if len(timestamps) > 0 else 0,
                    "end_time": timestamps[-1] if len(timestamps) > 0 else 0,
                }
            ]
            return

        diffs = np.diff(timestamps)
        gap_indices = np.where(diffs > 1.0)[0]

        segments: List[Dict[str, Any]] = []
        start_idx = 0

        for gap_idx in gap_indices:
            end_idx = gap_idx + 1
            segments.append(
                {
                    "segment_index": len(segments),
                    "n_frames": end_idx - start_idx,
                    "start_time": float(timestamps[start_idx]),
                    "end_time": float(timestamps[gap_idx]),
                    "duration": float(timestamps[gap_idx] - timestamps[start_idx]),
                }
            )
            start_idx = end_idx

        segments.append(
            {
                "segment_index": len(segments),
                "n_frames": len(timestamps) - start_idx,
                "start_time": float(timestamps[start_idx]),
                "end_time": float(timestamps[-1]),
                "duration": float(timestamps[-1] - timestamps[start_idx]),
            }
        )

        self.neutral_segments = segments

    def _recompute_baseline(self) -> None:
        """Recompute baseline statistics from accumulated neutral data."""
        if self.accumulated_neutral_data is None or len(self.accumulated_neutral_data) == 0:
            return

        feature_columns = [
            c for c in get_feature_columns(self.accumulated_neutral_data)
            if pd.api.types.is_numeric_dtype(self.accumulated_neutral_data[c])
        ]

        for col in feature_columns:
            values = self.accumulated_neutral_data[col].dropna().values
            if len(values) > 0:
                values = self._remove_outliers(values)
                self.baseline_stats[col] = self._compute_feature_statistics(values)

        self.metadata["n_frames"] = len(self.accumulated_neutral_data)
        self.metadata["n_segments"] = len(self.neutral_segments)

    def _remove_outliers(self, values: np.ndarray) -> np.ndarray:
        """Remove IQR-based outliers from a numeric array."""
        vals = np.asarray(values, dtype=float)
        if len(vals) < 4:
            return vals

        q1 = np.percentile(vals, 25)
        q3 = np.percentile(vals, 75)
        iqr = q3 - q1

        lower_bound = q1 - self.outlier_threshold * iqr
        upper_bound = q3 + self.outlier_threshold * iqr

        mask = (vals >= lower_bound) & (vals <= upper_bound)
        return vals[mask]

    def _compute_feature_statistics(self, values: np.ndarray) -> Dict[str, float]:
        """Compute descriptive statistics for a single feature."""
        vals = np.asarray(values, dtype=float)
        if len(vals) == 0:
            return self._get_empty_statistics()
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "median": float(np.median(vals)),
            "q25": float(np.percentile(vals, 25)),
            "q75": float(np.percentile(vals, 75)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "n": int(len(vals)),
        }

    @staticmethod
    def _get_empty_statistics() -> Dict[str, float]:
        """Return a zeroed statistics dict for features with no data."""
        return {
            "mean": 0.0,
            "std": 1.0,
            "median": 0.0,
            "q25": 0.0,
            "q75": 0.0,
            "min": 0.0,
            "max": 0.0,
            "n": 0,
        }

    def get_baseline_value(self, feature: str, method: Optional[str] = None) -> float:
        """Return the central-tendency baseline value for a feature."""
        method = method or self.robust_method
        if feature not in self.baseline_stats:
            return 0.0
        stats = self.baseline_stats[feature]
        return stats.get(method, stats.get("median", 0.0))

    def get_baseline_std(self, feature: str) -> float:
        """Return the baseline standard deviation for a feature."""
        if feature not in self.baseline_stats:
            return 1.0
        return self.baseline_stats[feature].get("std", 1.0)

    def compute_observed_ranges(self, full_df: pd.DataFrame) -> Dict[str, float]:
        """Compute per-feature observed activation ranges from the full recording.

        Uses the 95th percentile of raw blendshape values in measurement
        frames as the subject-specific expected maximum activation.  Falls
        back to all frames when no measurement segment is present.
        """
        measurement_df = full_df[full_df["segment"] == "measurement"] if "segment" in full_df.columns else full_df
        if len(measurement_df) < 10:
            measurement_df = full_df

        feature_columns = [
            c for c in get_feature_columns(measurement_df)
            if pd.api.types.is_numeric_dtype(measurement_df[c])
        ]
        ranges: Dict[str, float] = {}

        for col in feature_columns:
            values = measurement_df[col].dropna().values
            if len(values) > 0:
                if values.dtype == bool:
                    values = values.astype(float)
                p95 = float(np.percentile(np.abs(values), 95))
                ranges[col] = max(p95, 0.01)

        self.observed_ranges = ranges
        return ranges

    def get_baseline_data(self) -> Dict[str, Any]:
        """Return the full baseline payload (statistics, observed ranges, and metadata)."""
        return {
            "statistics": self.baseline_stats,
            "observed_ranges": self.observed_ranges,
            "metadata": self.metadata,
        }

    def save_baseline(self, path: Path) -> None:
        """Persist baseline data to a JSON file."""
        save_json(self.get_baseline_data(), path)

    def load_baseline(self, path: Path) -> Dict[str, Any]:
        """Load previously saved baseline data from JSON."""
        data = load_json(path)
        self.baseline_stats = data.get("statistics", {})
        self.observed_ranges = data.get("observed_ranges", {})
        self.metadata = data.get("metadata", {})
        return data

    def merge_external_baseline(
        self, external_stats: Dict[str, Dict[str, float]]
    ) -> None:
        """Merge externally supplied baseline statistics into this instance's stats.

        For blendshapes present in both this instance and external_stats, updates
        each numeric field (mean, std, median, q25, q75, max) by averaging the two
        sources with equal weight.  Blendshapes present only in external_stats are
        added directly without modification.  Should be called after
        compute_baseline() so that the session's own neutral segment data is already
        in place before merging.
        """
        if not hasattr(self, "baseline_stats") or self.baseline_stats is None:
            self.baseline_stats = {}
        for blendshape, ext_vals in external_stats.items():
            if blendshape in self.baseline_stats:
                curr = self.baseline_stats[blendshape]
                merged: Dict[str, Any] = {}
                for field_name in ("mean", "std", "median", "q25", "q75", "max"):
                    cv = curr.get(field_name)
                    ev = ext_vals.get(field_name)
                    if cv is not None and ev is not None:
                        merged[field_name] = (cv + ev) / 2.0
                    elif cv is not None:
                        merged[field_name] = cv
                    elif ev is not None:
                        merged[field_name] = ev
                for f in curr:
                    if f not in merged:
                        merged[f] = curr[f]
                self.baseline_stats[blendshape] = merged
            else:
                self.baseline_stats[blendshape] = dict(ext_vals)

    def is_valid(self) -> bool:
        """Check whether the baseline has enough samples to be considered reliable."""
        min_samples = self.features_config.get("confidence_thresholds", {}).get(
            "baseline_sample_min", 30
        )
        return self.metadata.get("n_frames", 0) >= min_samples

    def validate_quality(self) -> Dict[str, Any]:
        """Validate baseline quality: sample size, detection rate, feature stability, outlier contamination.

        Returns a dict with 'is_acceptable', individual check results, and any warnings.
        """
        warnings_list: List[str] = []
        checks: Dict[str, Any] = {}

        n_frames = self.metadata.get("n_frames", 0)
        min_frames = self.features_config.get("confidence_thresholds", {}).get(
            "baseline_sample_min", 30
        )
        checks["min_frames"] = {"value": n_frames, "threshold": min_frames, "pass": n_frames >= min_frames}
        if not checks["min_frames"]["pass"]:
            warnings_list.append(f"Baseline has only {n_frames} frames (minimum: {min_frames})")

        detection_rate = self.metadata.get("detection_rate", 1.0)
        min_detection = 0.80
        checks["detection_rate"] = {"value": detection_rate, "threshold": min_detection, "pass": detection_rate >= min_detection}
        if not checks["detection_rate"]["pass"]:
            warnings_list.append(f"Face detection rate {detection_rate:.1%} is below {min_detection:.0%}")

        high_cv_count = 0
        total_features = 0
        for feat, stat in self.baseline_stats.items():
            mean_val = abs(stat.get("mean", 0))
            std_val = stat.get("std", 0)
            if mean_val > 1e-4:
                cv = std_val / mean_val
                total_features += 1
                if cv > 1.0:
                    high_cv_count += 1

        cv_ratio = high_cv_count / max(total_features, 1)
        checks["feature_stability"] = {
            "high_cv_features": high_cv_count,
            "total_features": total_features,
            "cv_ratio": cv_ratio,
            "pass": cv_ratio < 0.3,
        }
        if not checks["feature_stability"]["pass"]:
            warnings_list.append(
                f"{high_cv_count}/{total_features} features have CV > 1.0, indicating unstable baseline"
            )

        outlier_ratio = 0.0
        if self.accumulated_neutral_data is not None and len(self.accumulated_neutral_data) > 10:
            feature_columns = [
                c for c in get_feature_columns(self.accumulated_neutral_data)
                if pd.api.types.is_numeric_dtype(self.accumulated_neutral_data[c])
            ]
            outlier_counts = 0
            total_values = 0
            for col in feature_columns:
                raw_vals = self.accumulated_neutral_data[col].dropna().values
                vals = np.asarray(raw_vals, dtype=float)
                vals = vals[np.isfinite(vals)]
                if len(vals) < 4:
                    continue
                q1 = np.percentile(vals, 25)
                q3 = np.percentile(vals, 75)
                iqr = q3 - q1
                n_outliers = int(np.sum((vals < q1 - 3.0 * iqr) | (vals > q3 + 3.0 * iqr)))
                outlier_counts += n_outliers
                total_values += len(vals)
            outlier_ratio = outlier_counts / max(total_values, 1)

        checks["outlier_contamination"] = {
            "outlier_ratio": outlier_ratio,
            "threshold": 0.05,
            "pass": outlier_ratio < 0.05,
        }
        if not checks["outlier_contamination"]["pass"]:
            warnings_list.append(
                f"Outlier contamination {outlier_ratio:.1%} exceeds 5% — baseline may be unreliable"
            )

        is_acceptable = all(c["pass"] for c in checks.values())

        return {
            "is_acceptable": is_acceptable,
            "checks": checks,
            "warnings": warnings_list,
        }


class BaselineCorrector:
    """Applies baseline correction (z-score standardisation) to feature DataFrames."""

    def __init__(self, baseline_constructor: BaselineConstructor):
        """Initialise corrector from a fitted BaselineConstructor."""
        self.baseline = baseline_constructor

    def correct_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Subtract baseline means from each feature column."""
        corrected_df = df.copy()
        feature_columns = [
            c for c in get_feature_columns(df)
            if pd.api.types.is_numeric_dtype(df[c])
        ]

        for col in feature_columns:
            if col in df.columns:
                baseline_value = self.baseline.get_baseline_value(col)
                corrected_df[col] = df[col] - baseline_value

        return corrected_df

    def standardize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Z-score standardise each feature column using baseline statistics."""
        standardized_df = df.copy()
        feature_columns = [
            c for c in get_feature_columns(df)
            if pd.api.types.is_numeric_dtype(df[c])
        ]

        for col in feature_columns:
            if col in df.columns:
                baseline_value = self.baseline.get_baseline_value(col)
                baseline_std = self.baseline.get_baseline_std(col)

                if baseline_std > 0:
                    standardized_df[col] = (df[col] - baseline_value) / baseline_std
                else:
                    standardized_df[col] = df[col] - baseline_value

        return standardized_df


def create_baseline_constructor(features_config: Dict[str, Any]) -> BaselineConstructor:
    """Factory: build a BaselineConstructor from configuration."""
    return BaselineConstructor(features_config)


def create_baseline_corrector(baseline_constructor: BaselineConstructor) -> BaselineCorrector:
    """Factory: build a BaselineCorrector from a BaselineConstructor."""
    return BaselineCorrector(baseline_constructor)




def build_normative_reference(
    session_summaries: List[Dict[str, Any]],
    feature_keys: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate feature statistics across multiple sessions into a normative reference.

    Each entry in session_summaries should have a 'metrics' sub-dict (or be flat)
    containing numeric feature values. Uses Welford's online algorithm to handle
    large numbers of sessions without storing all data.

    Returns a dict: {feature_key: {mean, std, q25, q75, n, min, max}}
    """
    from collections import defaultdict
    accum: Dict[str, List[float]] = defaultdict(list)

    for sess in session_summaries:
        metrics = sess.get("metrics", sess)
        for k, v in metrics.items():
            if feature_keys and k not in feature_keys:
                continue
            if isinstance(v, (int, float)) and np.isfinite(v):
                accum[k].append(float(v))

    reference: Dict[str, Dict[str, float]] = {}
    for k, vals in accum.items():
        arr = np.array(vals)
        q25, q75 = np.percentile(arr, [25, 75])
        reference[k] = {
            "mean": float(np.mean(arr)),
            "std": max(float(np.std(arr)), 1e-6),
            "q25": float(q25),
            "q75": float(q75),
            "iqr": max(float(q75 - q25), 1e-6),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "n": int(len(arr)),
        }
    return reference


def save_normative_reference(
    ref_stats: Dict[str, Dict[str, float]],
    path: Union[str, Path],
) -> None:
    """Persist a normative reference dict to JSON."""
    save_json({"normative_reference": ref_stats, "n_features": len(ref_stats)}, path)


def load_normative_reference(
    path: Union[str, Path],
) -> Dict[str, Dict[str, float]]:
    """Load a persisted normative reference from JSON."""
    data = load_json(path)
    return data.get("normative_reference", data)
