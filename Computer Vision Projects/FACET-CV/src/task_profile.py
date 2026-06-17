"""
Task profile module for facial motor and speech behavior analysis pipeline.

Accumulates baseline-session data into per-task reference profiles that
subsequent test sessions can use for anomaly detection and deviation scoring.
Each profile stores raw metric values, aggregated statistics, and
time-normalized activation patterns.

Each task group (A, B, C) has canonical task IDs (1-9 for A, 1-4 for B,
1-8 for C). Disorder-simulation tasks use higher IDs (e.g. A_10-A_17) and
are mapped to their canonical reference task via _DISORDER_TASK_CANONICAL_MAP.
This allows anomaly detection to always compare against a valid reference
profile even when a session uses a combined or simulated disorder task.

Fuzzy task-name matching uses Python's difflib.SequenceMatcher to handle
novel disorder task names not listed in _DISORDER_TASK_CANONICAL_MAP.
A similarity ratio threshold of 0.35 is used; this is intentionally
permissive to ensure a fallback reference is always found.
"""

import difflib
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from .utils import save_json, load_json, compute_statistics

logger = logging.getLogger("pipeline")


def _task_key(task_group: str, task_id: int) -> str:
    """Build a canonical ``<group>_<id>`` lookup key."""
    tg = str(task_group) if task_group and str(task_group) not in ("0", "nan", "None") else "0"
    tid = int(task_id) if task_id else 0
    return f"{tg}_{tid}"


_CANONICAL_TASK_ID_RANGES: Dict[str, range] = {
    "A": range(1, 10),
    "B": range(1, 5),
    "C": range(1, 9),
}

_BUCCOFACIAL_A_SUBSTITUTIONS: Dict[int, Tuple[str, str]] = {
    1: ("smile_closed",  "purse"),
    2: ("purse",         "smile_closed"),
    3: ("puff",          "smile_gentle"),
    4: ("tongue_out",    "tongue_out"),
    5: ("tongue_up",     "tongue_lr"),
    6: ("tongue_lr",     "tongue_up"),
    7: ("surprised",     "frown"),
    8: ("smile_gentle",  "puff"),
    9: ("frown",         "surprised"),
}

_BUCCOFACIAL_EXPECTED_REF: Dict[int, int] = {
    1: 2,
    2: 1,
    3: 8,
    4: 4,
    5: 6,
    6: 5,
    7: 9,
    8: 3,
    9: 7,
}

_DISORDER_TASK_CANONICAL_MAP: Dict[Tuple[str, int], Tuple[str, int]] = {
    ("A", 10): ("A", 1),
    ("A", 11): ("A", 3),
    ("A", 12): ("A", 3),
    ("A", 13): ("A", 5),
    ("A", 14): ("A", 5),
    ("A", 15): ("A", 7),
    ("A", 16): ("A", 8),
    ("A", 17): ("A", 9),
    ("B", 5):  ("B", 1),
    ("B", 6):  ("B", 2),
    ("B", 7):  ("B", 3),
    ("B", 8):  ("B", 4),
    ("C", 9):  ("C", 1),
    ("C", 10): ("C", 2),
    ("C", 11): ("C", 3),
    ("C", 12): ("C", 4),
    ("C", 13): ("C", 5),
    ("C", 14): ("C", 6),
    ("C", 15): ("C", 7),
    ("C", 16): ("C", 8),
}


class TaskProfile:
    """Per-subject reference profile built from one or more baseline sessions."""

    def __init__(self, subject_id: str):
        """Initialise an empty profile for the given subject."""
        self.subject_id = subject_id
        self.sessions_included: List[str] = []
        self.updated_at: Optional[str] = None
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.global_stats: Dict[str, Dict[str, float]] = {}

    def update_from_session(
        self,
        session_id: str,
        repetition_metrics_df: pd.DataFrame,
        features_df: Optional[pd.DataFrame] = None,
        task_metrics_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """Incorporate a baseline session into the profile."""
        if session_id in self.sessions_included:
            return
        if repetition_metrics_df is None or len(repetition_metrics_df) == 0:
            return

        has_task = (
            "task_group" in repetition_metrics_df.columns
            and repetition_metrics_df["task_group"].notna().any()
            and (repetition_metrics_df["task_group"].astype(str) != "0").any()
        )

        if has_task:
            groups = repetition_metrics_df.groupby(
                ["task_group", "task_id"], dropna=False
            )
        else:
            groups = [(("0", 0), repetition_metrics_df)]

        for group_key, group_df in groups:
            if isinstance(group_key, tuple):
                tg, tid = group_key
            else:
                tg, tid = "0", 0
            tg = str(tg) if pd.notna(tg) else "0"
            tid = int(tid) if pd.notna(tid) else 0
            tk = _task_key(tg, tid)

            task_name = ""
            if "task_name" in group_df.columns:
                names = group_df["task_name"].dropna().unique()
                names = [n for n in names if n != "(no task selected)"]
                if names:
                    task_name = str(names[0])

            if tk not in self.tasks:
                self.tasks[tk] = {
                    "task_group": tg,
                    "task_id": tid,
                    "task_name": task_name,
                    "n_sessions": 0,
                    "n_repetitions_total": 0,
                    "per_feature_stats": {},
                    "activation_pattern": {},
                    "_raw_values": {},
                }

            task_data = self.tasks[tk]
            task_data["n_sessions"] += 1
            task_data["n_repetitions_total"] += len(group_df)
            if task_name and not task_data["task_name"]:
                task_data["task_name"] = task_name

            numeric_cols = group_df.select_dtypes(include=[np.number]).columns
            exclude = {"repetition", "n_frames", "frame_index", "task_id"}
            metric_cols = [
                c for c in numeric_cols if c not in exclude and not c.startswith("_")
            ]

            for col in metric_cols:
                values = group_df[col].dropna().values
                if len(values) == 0:
                    continue
                if col not in task_data["_raw_values"]:
                    task_data["_raw_values"][col] = []
                task_data["_raw_values"][col].extend(values.tolist())

            if features_df is not None and len(features_df) > 0:
                self._update_activation_patterns(tk, tg, tid, features_df)

        self._recompute_all_stats()
        self._update_global_stats(repetition_metrics_df)

        self.sessions_included.append(session_id)
        self.updated_at = datetime.now().isoformat()

    def _update_activation_patterns(
        self,
        task_key: str,
        task_group: str,
        task_id: int,
        features_df: pd.DataFrame,
    ) -> None:
        """Bin and store time-normalized activation curves for the task."""
        if task_group != "0":
            tg_col = (
                features_df["task_group"].astype(str)
                if "task_group" in features_df.columns
                else pd.Series("", index=features_df.index)
            )
            tid_col = (
                features_df["task_id"].fillna(0).astype(int)
                if "task_id" in features_df.columns
                else pd.Series(0, index=features_df.index)
            )
            mask = (tg_col == str(task_group)) & (tid_col == int(task_id))
            task_features = features_df[mask]
        else:
            task_features = features_df

        if len(task_features) == 0:
            return

        exclude_pattern = {
            "frame_index", "timestamp_abs", "segment", "repetition",
            "detection_success", "detection_confidence", "time_rel_sec",
            "task_group", "task_id", "task_name", "occluded",
        }
        pattern_features = [
            c for c in task_features.columns
            if c not in exclude_pattern
            and not c.startswith("_")
            and task_features[c].dtype in [np.float64, np.float32, np.int64, np.int32]
        ]
        available = [f for f in pattern_features if f in task_features.columns]
        if not available:
            return

        if "repetition" not in task_features.columns or "timestamp_abs" not in task_features.columns:
            return

        n_bins = 50
        repetitions = sorted(
            r for r in task_features["repetition"].unique() if r != 0
        )
        task_data = self.tasks[task_key]

        for feature in available:
            if feature not in task_data["activation_pattern"]:
                task_data["activation_pattern"][feature] = {"curves": []}

            for rep in repetitions:
                rep_df = task_features[task_features["repetition"] == rep]
                if len(rep_df) < 5 or feature not in rep_df.columns:
                    continue
                start_t = rep_df["timestamp_abs"].min()
                time_rel = (rep_df["timestamp_abs"] - start_t).values
                vals = rep_df[feature].values
                duration = time_rel.max()
                if duration <= 0:
                    continue
                time_norm = time_rel / duration
                bins = np.linspace(0, 1, n_bins)
                binned = np.interp(bins, time_norm, vals)
                task_data["activation_pattern"][feature]["curves"].append(
                    binned.tolist()
                )

    @staticmethod
    def _mad(arr: np.ndarray) -> float:
        """Median absolute deviation."""
        return float(np.median(np.abs(arr - np.median(arr))))

    def _recompute_all_stats(self) -> None:
        """Re-derive per-feature statistics and activation-pattern summaries.

        Uses ddof=1 (unbiased) std throughout so that prediction intervals
        built from these statistics are not artificially tight at small n.
        coverage_std is derived from the observed range and provides a more
        conservative width floor than the biased std, which matters most at
        n=3 where the sample may not represent the tails of the distribution.
        """
        for task_data in self.tasks.values():
            raw = task_data.get("_raw_values", {})
            for col, values_list in raw.items():
                arr = np.array(values_list)
                n = len(arr)
                mean = float(np.mean(arr))
                std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
                obs_half_range = float(np.max(np.abs(arr - mean))) if n > 1 else 0.0
                coverage_std = obs_half_range / 2.0 if n > 1 else std
                task_data["per_feature_stats"][col] = {
                    "mean": mean,
                    "std": std,
                    "coverage_std": max(std, coverage_std),
                    "median": float(np.median(arr)),
                    "mad": self._mad(arr),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "q25": float(np.percentile(arr, 25)),
                    "q75": float(np.percentile(arr, 75)),
                    "n": n,
                }

            for pattern_data in task_data.get("activation_pattern", {}).values():
                curves = pattern_data.get("curves", [])
                if curves:
                    curves_arr = np.array(curves)
                    pattern_data["mean_pattern"] = np.mean(curves_arr, axis=0).tolist()
                    n_c = len(curves)
                    pattern_data["std_pattern"] = (
                        np.std(curves_arr, axis=0, ddof=1).tolist() if n_c > 1
                        else np.zeros(curves_arr.shape[1]).tolist()
                    )
                    mad_pattern = np.median(
                        np.abs(curves_arr - np.median(curves_arr, axis=0)), axis=0
                    )
                    pattern_data["mad_pattern"] = mad_pattern.tolist()
                    pattern_data["n_curves"] = n_c

    def _update_global_stats(self, repetition_metrics_df: pd.DataFrame) -> None:
        """Maintain running aggregate statistics across all tasks."""
        numeric_cols = repetition_metrics_df.select_dtypes(include=[np.number]).columns
        exclude = {"repetition", "n_frames", "frame_index", "task_id"}
        metric_cols = [
            c for c in numeric_cols if c not in exclude and not c.startswith("_")
        ]

        for col in metric_cols:
            values = repetition_metrics_df[col].dropna().values
            if len(values) == 0:
                continue
            if col not in self.global_stats:
                self.global_stats[col] = {"_all_values": []}
            self.global_stats[col]["_all_values"].extend(values.tolist())
            arr = np.array(self.global_stats[col]["_all_values"])
            self.global_stats[col].update(
                {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "median": float(np.median(arr)),
                    "mad": self._mad(arr),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "q25": float(np.percentile(arr, 25)),
                    "q75": float(np.percentile(arr, 75)),
                    "n": len(arr),
                }
            )


    def resolve_reference_task_key(
        self,
        task_group: str,
        task_id: int,
        task_name: Optional[str] = None,
    ) -> Tuple[str, int, str]:
        """Return the (group, id, key) of the best matching reference task.

        Resolution order:
        1. Direct match by task key (normal case).
        2. Explicit disorder-task canonical mapping (A_10-A_17 → A_1-A_9, etc.).
        3. Fuzzy string similarity against task names stored in the profile
           for the same group (handles novel disorder tasks automatically).
        4. Fallback: lowest-numbered task in the same group.

        The method never raises; it always returns a valid triple, though the
        returned key may still not be in ``self.tasks`` if the profile is empty.
        """
        tk = _task_key(task_group, task_id)

        if tk in self.tasks:
            return task_group, task_id, tk

        canonical = _DISORDER_TASK_CANONICAL_MAP.get((task_group, int(task_id)))
        if canonical is not None:
            ref_group, ref_id = canonical
            ref_tk = _task_key(ref_group, ref_id)
            if ref_tk in self.tasks:
                return ref_group, ref_id, ref_tk

        _canonical_range = _CANONICAL_TASK_ID_RANGES.get(task_group, range(0))
        _is_canonical_id = int(task_id) in _canonical_range
        if task_name and not _is_canonical_id:
            best_key, best_ratio = self._fuzzy_name_match(task_group, task_name)
            if best_key is not None and best_ratio >= 0.35:
                logger.debug(
                    "resolve_reference: fuzzy match %s_%s ('%s') → %s (ratio %.2f)",
                    task_group, task_id, task_name, best_key, best_ratio,
                )
                td = self.tasks[best_key]
                return (
                    str(td["task_group"]),
                    int(td["task_id"]),
                    best_key,
                )

        group_keys = sorted(
            tk2
            for tk2, td in self.tasks.items()
            if td.get("task_group") == task_group and tk2 != "0_0"
        )
        if group_keys:
            best_tk = group_keys[0]
            td = self.tasks[best_tk]
            return str(td["task_group"]), int(td["task_id"]), best_tk

        return task_group, int(task_id), tk

    def _fuzzy_name_match(
        self, task_group: str, query_name: str
    ) -> Tuple[Optional[str], float]:
        """Find the profile task in *task_group* whose name best matches *query_name*.

        Returns (task_key, ratio) where ratio is in [0, 1].
        """
        query_norm = query_name.lower().strip()
        best_key: Optional[str] = None
        best_ratio: float = 0.0
        for tk, td in self.tasks.items():
            if td.get("task_group") != task_group or tk == "0_0":
                continue
            ref_name = str(td.get("task_name", "")).lower().strip()
            if not ref_name:
                continue
            ratio = difflib.SequenceMatcher(None, query_norm, ref_name).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_key = tk
        return best_key, best_ratio

    def get_task_reference(
        self, task_group: str, task_id: int, task_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Return the accumulated profile data for a given task.

        Resolves disorder-simulation task IDs to their canonical baseline task
        via :meth:`resolve_reference_task_key` so anomaly detection always has
        a valid reference even for combined-session disorder tasks (A_10-A_17,
        etc.).  Attaches ``_ref_task_group`` and ``_ref_task_id`` metadata to
        the returned dict so callers can report which reference was used.
        """
        ref_group, ref_id, ref_tk = self.resolve_reference_task_key(
            task_group, task_id, task_name
        )
        if ref_tk in self.tasks:
            entry = self.tasks[ref_tk]
            entry = dict(entry)
            entry["_ref_task_group"] = ref_group
            entry["_ref_task_id"] = ref_id
            entry["_original_task_group"] = task_group
            entry["_original_task_id"] = int(task_id)
            entry["_is_mapped_reference"] = (
                ref_group != task_group or ref_id != int(task_id)
            )
            return entry
        if "0_0" in self.tasks:
            return self.tasks["0_0"]
        return None

    def get_task_feature_stats(
        self, task_group: str, task_id: int, task_name: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """Return per-feature descriptive statistics for a task."""
        ref = self.get_task_reference(task_group, task_id, task_name)
        if ref is None:
            return {}
        return ref.get("per_feature_stats", {})

    def get_task_activation_pattern(
        self,
        task_group: str,
        task_id: int,
        feature: str = "mean_activation",
        task_name: Optional[str] = None,
    ) -> Optional[Dict]:
        """Return the time-normalized activation curve data for a task + feature."""
        ref = self.get_task_reference(task_group, task_id, task_name)
        if ref is None:
            return None
        return ref.get("activation_pattern", {}).get(feature, None)

    def get_reference_metrics_df(
        self,
        task_group: Optional[str] = None,
        task_id: Optional[int] = None,
        task_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """Reconstruct a repetition-metrics-like DataFrame from stored raw values."""
        rows: List[Dict[str, Any]] = []

        if task_group is not None:
            ref_group, ref_id, ref_tk = self.resolve_reference_task_key(
                task_group, task_id or 0, task_name
            )
            targets = {ref_tk: self.tasks.get(ref_tk)} if ref_tk in self.tasks else {}
        else:
            targets = self.tasks

        for tk, task_data in targets.items():
            if task_data is None:
                continue
            raw = task_data.get("_raw_values", {})
            if not raw:
                continue
            max_len = max(len(v) for v in raw.values())
            for i in range(max_len):
                row: Dict[str, Any] = {
                    "task_group": task_data["task_group"],
                    "task_id": task_data["task_id"],
                    "repetition": i + 1,
                }
                for col, vals in raw.items():
                    if i < len(vals):
                        row[col] = vals[i]
                rows.append(row)

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    def compute_cross_task_matching(
        self,
        repetition_metrics_df: pd.DataFrame,
        expected_group: str,
        expected_task_id: int,
        task_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compare test repetitions against all reference task profiles in the same group.

        For each test repetition, compute z-score-based similarity to every
        reference task profile.  When a repetition matches a different task's
        profile better than its expected task, flag as a potential task
        substitution (key buccofacial apraxia indicator).

        Disorder-simulation tasks (A_10-A_17, etc.) are resolved to their
        canonical reference task via :meth:`resolve_reference_task_key` so that
        cross-task matching always operates against a valid expected profile.
        """
        ref_group, ref_id, expected_key = self.resolve_reference_task_key(
            expected_group, expected_task_id, task_name
        )
        group_tasks = {
            tk: td for tk, td in self.tasks.items()
            if td.get("task_group") == ref_group and tk != "0_0"
        }

        if len(group_tasks) < 2 or expected_key not in group_tasks:
            return {}

        feature_cols = self._select_matching_features(
            repetition_metrics_df, group_tasks, expected_key
        )
        if not feature_cols:
            return {}

        measurement_reps = (
            repetition_metrics_df[repetition_metrics_df["repetition"] != 0]
            if "repetition" in repetition_metrics_df.columns
            else repetition_metrics_df
        )

        per_rep_results: List[Dict[str, Any]] = []
        for _, rep_row in measurement_reps.iterrows():
            rep_id = int(rep_row.get("repetition", 0))
            similarities: Dict[str, float] = {}
            for tk, task_data in group_tasks.items():
                sim = self._compute_zscore_similarity(
                    rep_row, task_data.get("per_feature_stats", {}), feature_cols
                )
                if sim is not None:
                    similarities[tk] = sim

            if not similarities:
                continue

            expected_sim = similarities.get(expected_key, 0.0)
            alternate_sims = {k: v for k, v in similarities.items() if k != expected_key}
            best_alt_key = (
                max(alternate_sims, key=alternate_sims.get) if alternate_sims else None
            )
            best_alt_sim = alternate_sims.get(best_alt_key, 0.0) if best_alt_key else 0.0
            _margin = max(0.04, expected_sim * 0.08)
            is_substitution = best_alt_sim > expected_sim + _margin

            per_rep_results.append({
                "repetition": rep_id,
                "expected_task_similarity": expected_sim,
                "all_similarities": similarities,
                "best_alternate_task": best_alt_key,
                "best_alternate_similarity": best_alt_sim,
                "substitution_score": best_alt_sim - expected_sim,
                "is_substitution": is_substitution,
            })

        if not per_rep_results:
            return {}

        n_subs = sum(1 for r in per_rep_results if r["is_substitution"])

        return {
            "per_repetition": per_rep_results,
            "task_profile_similarity": float(
                np.mean([r["expected_task_similarity"] for r in per_rep_results])
            ),
            "mean_substitution_score": float(
                np.mean([r["substitution_score"] for r in per_rep_results])
            ),
            "n_substitutions": n_subs,
            "substitution_rate": float(n_subs / len(per_rep_results)),
            "n_repetitions_evaluated": len(per_rep_results),
            "expected_task": expected_key,
            "n_reference_tasks": len(group_tasks),
        }

    def _select_matching_features(
        self,
        repetition_metrics_df: pd.DataFrame,
        group_tasks: Dict[str, Dict],
        expected_key: str,
    ) -> List[str]:
        """Select numeric features present in both test data and all reference profiles."""
        expected_stats = group_tasks[expected_key].get("per_feature_stats", {})
        candidates: List[str] = []
        for col in repetition_metrics_df.select_dtypes(include=[np.number]).columns:
            if (
                col.endswith("_mean")
                and "asymmetry" not in col
                and "across" not in col
                and col in expected_stats
            ):
                candidates.append(col)
        return candidates

    def _compute_zscore_similarity(
        self,
        rep_row: pd.Series,
        ref_stats: Dict[str, Dict[str, float]],
        feature_cols: List[str],
    ) -> Optional[float]:
        """Compute exponential similarity from mean absolute z-score to a reference profile."""
        z_scores: List[float] = []
        for feat in feature_cols:
            if feat not in ref_stats:
                continue
            test_val = rep_row.get(feat, None)
            if test_val is None or (isinstance(test_val, float) and np.isnan(test_val)):
                continue
            ref_mean = ref_stats[feat].get("mean", 0.0)
            ref_std = max(ref_stats[feat].get("std", 0.1), 0.001)
            z_scores.append(abs(float(test_val) - ref_mean) / ref_std)

        if not z_scores:
            return None
        return float(np.exp(-np.mean(z_scores) / 2))

    def compute_activation_pattern_similarity(
        self,
        features_df: pd.DataFrame,
        task_group: str,
        task_id: int,
        feature: str = "mean_activation",
        task_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compare test activation curves against the reference mean pattern.

        Returns per-repetition pattern correlation and amplitude ratios.
        High correlation with lower amplitude suggests dysarthria (pattern
        preserved but degraded); low correlation suggests apraxia or
        substitution.
        """
        ref = self.get_task_reference(task_group, task_id, task_name)
        if ref is None:
            return {}

        pattern_data = ref.get("activation_pattern", {}).get(feature, {})
        ref_mean = pattern_data.get("mean_pattern")
        if ref_mean is None:
            return {}
        ref_mean_arr = np.array(ref_mean)

        if task_group != "0":
            tg_col = (
                features_df["task_group"].astype(str)
                if "task_group" in features_df.columns
                else pd.Series("", index=features_df.index)
            )
            tid_col = (
                features_df["task_id"].fillna(0).astype(int)
                if "task_id" in features_df.columns
                else pd.Series(0, index=features_df.index)
            )
            mask = (tg_col == str(task_group)) & (tid_col == int(task_id))
            task_features = features_df[mask]
        else:
            task_features = features_df

        if len(task_features) == 0 or feature not in task_features.columns:
            return {}
        if "repetition" not in task_features.columns or "timestamp_abs" not in task_features.columns:
            return {}

        n_bins = len(ref_mean_arr)
        repetitions = sorted(r for r in task_features["repetition"].unique() if r != 0)

        per_rep: List[Dict[str, Any]] = []
        for rep in repetitions:
            rep_df = task_features[task_features["repetition"] == rep]
            if len(rep_df) < 5:
                continue
            start_t = rep_df["timestamp_abs"].min()
            time_rel = (rep_df["timestamp_abs"] - start_t).values
            vals = rep_df[feature].values
            duration = time_rel.max()
            if duration <= 0:
                continue
            time_norm = time_rel / duration
            bins = np.linspace(0, 1, n_bins)
            test_binned = np.interp(bins, time_norm, vals)

            corr = (
                float(np.corrcoef(test_binned, ref_mean_arr)[0, 1])
                if np.std(test_binned) > 0
                else 0.0
            )
            amp_ratio = float(
                np.mean(np.abs(test_binned))
                / (np.mean(np.abs(ref_mean_arr)) + 0.001)
            )
            per_rep.append({
                "repetition": int(rep),
                "pattern_correlation": corr,
                "amplitude_ratio": amp_ratio,
            })

        if not per_rep:
            return {}

        return {
            "per_repetition": per_rep,
            "mean_pattern_correlation": float(
                np.mean([r["pattern_correlation"] for r in per_rep])
            ),
            "mean_amplitude_ratio": float(
                np.mean([r["amplitude_ratio"] for r in per_rep])
            ),
            "pattern_correlation_std": float(
                np.std([r["pattern_correlation"] for r in per_rep])
            ),
        }

    def save(self, path: Path) -> None:
        """Persist the task profile to a JSON file."""
        data: Dict[str, Any] = {
            "subject_id": self.subject_id,
            "sessions_included": self.sessions_included,
            "updated_at": self.updated_at,
            "tasks": {},
            "global_stats": {},
        }
        for tk, task_data in self.tasks.items():
            serialized = {k: v for k, v in task_data.items() if k != "_raw_values"}
            serialized["_raw_values"] = task_data.get("_raw_values", {})
            data["tasks"][tk] = serialized

        for col, stats in self.global_stats.items():
            data["global_stats"][col] = dict(stats)

        save_json(data, path)

    def load(self, path: Path) -> None:
        """Restore the task profile from a JSON file."""
        data = load_json(path)
        self.subject_id = data.get("subject_id", self.subject_id)
        self.sessions_included = data.get("sessions_included", [])
        self.updated_at = data.get("updated_at", None)
        self.tasks = data.get("tasks", {})
        self.global_stats = data.get("global_stats", {})

        for task_data in self.tasks.values():
            task_data.setdefault("_raw_values", {})
            task_data.setdefault("per_feature_stats", {})
            task_data.setdefault("activation_pattern", {})

    def is_loaded(self) -> bool:
        """Return whether the profile contains any task data."""
        return len(self.tasks) > 0 or len(self.sessions_included) > 0


def create_task_profile(subject_id: str) -> TaskProfile:
    """Factory: create a new empty TaskProfile."""
    return TaskProfile(subject_id)


def load_task_profile(path: Path, subject_id: str) -> Optional[TaskProfile]:
    """Load a TaskProfile from disk, returning None if the file does not exist."""
    if not path.exists():
        return None
    profile = TaskProfile(subject_id)
    profile.load(path)
    return profile
