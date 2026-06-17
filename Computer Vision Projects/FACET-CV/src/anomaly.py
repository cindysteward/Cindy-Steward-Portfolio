"""
Anomaly detection for facial motor and speech behavior analysis.

Task-relevant feature selection with t-distribution prediction intervals
for small-sample deviation testing (n=3-5 reference measurements).

Scoring layers:
  1. Feature deviation (primary): per-feature t-test against reference
     prediction intervals using task-relevant blendshapes from config.
  2. Mahalanobis distance in task-feature space with shrinkage covariance.
  3. Nearest-centroid Euclidean distance normalized by reference spread.
  4. Within-session relative outlier scoring (LOF + reference distance).

The ML model (OC-SVM / IsolationForest) is included only when
n_reference >= 10 (was not the case during this Master project, included for future possibilities). Below that threshold it contributes zero weight because
it is statistically unreliable with very few reference samples.

Composite score = weighted blend with feature deviation receiving highest
weight (~50-60 %). The decision rule requires both a minimum composite score
AND a minimum fraction of task-relevant features showing genuine deviation.

Supporting utilities: DTW pattern similarity, CUSUM drift monitoring,
bootstrap confidence intervals, and Hampel modified z-scores.

Key methodological references
==============================

DTW temporal pattern comparison
  Sakoe H and Chiba S (1978) IEEE TASSP 26, 43-49. The Sakoe-Chiba band
  constraint implemented in _dtw_distance() limits warping to a diagonal
  band, preventing degenerate alignments and reducing computation.
  https://doi.org/10.1109/TASSP.1978.1163055

  Allison et al. (2022) AJSLP 31, 1682 validated DTW-based kinematic
  variability (CV > 0.30) for detecting motor speech involvement.
  https://doi.org/10.1044/2022_AJSLP-21-00241

Snippet-based function preservation
  Kanno and Mikuni (2015) Neurol Med Chir 55, 287 describe the clinical
  principle of awake craniotomy monitoring: preserved function is concluded
  if ANY response is observed, not the average. This underpins the
  _snippet_function_preserved() logic.
  https://doi.org/10.2176/nmc.ra.2014-0395

Face mesh anomaly detection precedents
  Baig et al. (2023) achieved 98.93 % accuracy for binary paralysis
  classification using MobileNetV2 on MediaPipe 468-landmark meshes.
  t-SNE showed unhealthy subclusters within the healthy distribution,
  validating the need for person-specific reference baselines rather than
  population norms.
  hdl:10210/504453
"""

import logging
import warnings
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist, linregress as _linregress
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from .utils import save_json, load_json, get_numeric_feature_columns, _FRAME_META_COLUMNS

logger = logging.getLogger("pipeline")


_FEATURE_GROUP_KEYWORDS = {
    "facial_asymmetry":      ["asymmetry", "asymm", "signed_asym", "left_right"],
    "side_amplitude":        ["Left_", "Right_", "_left_", "_right_", "LeftSmile", "RightSmile"],
    "amplitude_reduction":   ["amplitude", "range", "activation", "mean_activation",
                               "mouthPucker", "mouthClose", "mouthFunnel", "jawOpen", "cheekPuff"],
    "temporal_distortion":   ["timing", "duration", "onset", "offset", "velocity", "jerk",
                               "rate_hz", "inter_syllable", "_vel_", "_acc_", "_vel", "_acc"],
    "articulation":          ["articulation", "phoneme", "syllable", "ddk", "fluency",
                               "consistency", "groping"],
    "kinematic_profile":     ["kin_", "kinematic", "trajectory"],
    "task_substitution":     ["substitution", "cross_task", "task_sim", "pattern_corr"],
}

_FEATURE_KEYWORD_ORDER = [
    "facial_asymmetry",
    "side_amplitude",
    "temporal_distortion",
    "articulation",
    "kinematic_profile",
    "task_substitution",
    "amplitude_reduction",
]

_SCALE_FEATURE_KEYWORDS = ("face_size", "face_scale", "face_width", "face_height")

_ASYMMETRY_PAIRS: Dict[str, str] = {
    "mouthSmileLeft":       "mouthSmileRight",
    "mouthSmileRight":      "mouthSmileLeft",
    "mouthFrownLeft":       "mouthFrownRight",
    "mouthFrownRight":      "mouthFrownLeft",
    "eyeWideLeft":          "eyeWideRight",
    "eyeWideRight":         "eyeWideLeft",
    "eyeSquintLeft":        "eyeSquintRight",
    "eyeSquintRight":       "eyeSquintLeft",
    "cheekSquintLeft":      "cheekSquintRight",
    "cheekSquintRight":     "cheekSquintLeft",
    "browDownLeft":         "browDownRight",
    "browDownRight":        "browDownLeft",
    "mouthDimpleLeft":      "mouthDimpleRight",
    "mouthDimpleRight":     "mouthDimpleLeft",
    "mouthLowerDownLeft":   "mouthLowerDownRight",
    "mouthLowerDownRight":  "mouthLowerDownLeft",
    "mouthUpperUpLeft":     "mouthUpperUpRight",
    "mouthUpperUpRight":    "mouthUpperUpLeft",
    "mouthPressLeft":       "mouthPressRight",
    "mouthPressRight":      "mouthPressLeft",
    "mouthStretchLeft":     "mouthStretchRight",
    "mouthStretchRight":    "mouthStretchLeft",
    "jawLeft":              "jawRight",
    "jawRight":             "jawLeft",
}

_STAT_SUFFIXES = ("mean", "std", "max", "min", "range", "time_to_peak")


def _compute_mad(values: np.ndarray) -> float:
    """Median absolute deviation."""
    med = np.median(values)
    return float(np.median(np.abs(values - med)))


def _dtw_distance(s1: np.ndarray, s2: np.ndarray, band: Optional[int] = None) -> float:
    """Dynamic time warping distance with optional Sakoe-Chiba band constraint."""
    n, m = len(s1), len(s2)
    if band is None:
        band = max(n, m)
    band = max(band, abs(n - m))

    dtw_mat = np.full((n + 1, m + 1), np.inf)
    dtw_mat[0, 0] = 0.0
    for i in range(1, n + 1):
        j_lo = max(1, i - band)
        j_hi = min(m, i + band)
        for j in range(j_lo, j_hi + 1):
            cost = abs(s1[i - 1] - s2[j - 1])
            dtw_mat[i, j] = cost + min(dtw_mat[i - 1, j],
                                        dtw_mat[i, j - 1],
                                        dtw_mat[i - 1, j - 1])
    return float(dtw_mat[n, m])


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid mapping to [0, 1]."""
    if x >= 0:
        z = np.exp(-x)
        return 1.0 / (1.0 + z)
    z = np.exp(x)
    return z / (1.0 + z)


def _snippet_function_preserved(
    activation_values: np.ndarray,
    threshold: float,
    n_snippets: int = 3,
) -> bool:
    """Return True if ANY snippet of the signal shows activation above threshold.

    Divides the signal into n_snippets equal windows and returns True if at
    least one snippet has a maximum activation at or above the threshold.
    Returns False only when all snippets fall below threshold, indicating
    consistent impairment across the entire task window.

    The key insight is that for disorders of execution consistency (such as
    dysarthria or buccofacial apraxia), a patient may produce one correct
    attempt among several deviant ones. Mean-based detection would flag the
    whole task as impaired even though preserved function was demonstrated.
    Requiring all snippets to be below threshold before concluding impairment
    mirrors how clinicians reason during awake craniotomy monitoring:
    preserved function is concluded when ANY response is observed in the
    monitoring window, not based on an average.

    Kanno and Mikuni (2015) Neurol Med Chir 55, 287 describe this clinical
    reasoning framework for intraoperative function monitoring.
    https://doi.org/10.2176/nmc.ra.2014-0395

    Parameters
    ----------
    activation_values : 1-D array of mean blendshape or kinematic values
    threshold         : value above which activation is considered preserved
    n_snippets        : number of equal-length windows to divide the signal into

    Returns
    -------
    bool
        True if at least one snippet shows activation >= threshold.
        False if all snippets are below threshold.
    """
    if len(activation_values) < n_snippets:
        return bool(np.any(activation_values >= threshold))

    snippets = np.array_split(activation_values, n_snippets)
    for snippet in snippets:
        if len(snippet) > 0 and float(np.max(snippet)) >= threshold:
            return True
    return False


def _classify_anomaly_type(feature_name: str) -> str:
    """Map a feature name to a granular anomaly type using priority-ordered keywords.

    Types (in priority order):
      facial_asymmetry    : left-right asymmetry features (paresis)
      side_amplitude      : per-side amplitude mismatch (peripheral palsy / paresis)
      temporal_distortion : rate/timing/velocity features (dysarthria, apraxia)
      articulation        : phoneme / articulation quality (speech apraxia, dysarthria)
      kinematic_profile   : shape trajectory deviation (apraxia of speech)
      task_substitution   : cross-task / pattern mismatch (buccofacial apraxia)
      amplitude_reduction : overall amplitude change (dysarthria / hypokinesia)
      unknown
    """
    orig = feature_name
    lower = feature_name.lower()
    for atype in _FEATURE_KEYWORD_ORDER:
        keywords = _FEATURE_GROUP_KEYWORDS[atype]
        if any(kw.lower() in lower for kw in keywords):
            return atype
    return "unknown"


def _derive_pair_base(name: str) -> str:
    """Strip Left/Right suffix from a blendshape name to get the pair base."""
    for suffix in ("Left", "Right"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


class CUSUMMonitor:
    """Cumulative sum control chart for sequential drift detection."""

    def __init__(self, k: float = 0.5, h: float = 5.0):
        """Initialise CUSUM with allowance k and decision threshold h."""
        self._k = k
        self._h = h
        self._references: Dict[str, Tuple[float, float]] = {}
        self._s_pos: Dict[str, float] = {}
        self._s_neg: Dict[str, float] = {}
        self._alarms: Dict[str, bool] = {}

    def set_reference(self, mean: float, std: float, feature: str) -> None:
        """Set reference mean and std for a single feature channel."""
        self._references[feature] = (mean, max(std, 1e-9))
        self._s_pos[feature] = 0.0
        self._s_neg[feature] = 0.0
        self._alarms[feature] = False

    def update(self, value: float, feature: str) -> bool:
        """Feed a single observation; return True if alarm triggered."""
        if feature not in self._references:
            return False
        mu, sigma = self._references[feature]
        z = (value - mu) / sigma
        self._s_pos[feature] = max(0.0, self._s_pos[feature] + z - self._k)
        self._s_neg[feature] = max(0.0, self._s_neg[feature] - z - self._k)
        alarm = self._s_pos[feature] > self._h or self._s_neg[feature] > self._h
        self._alarms[feature] = alarm
        return alarm

    def reset(self, feature: Optional[str] = None) -> None:
        """Reset accumulators for one or all features."""
        targets = [feature] if feature else list(self._references)
        for f in targets:
            self._s_pos[f] = 0.0
            self._s_neg[f] = 0.0
            self._alarms[f] = False

    def update_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process every row for all registered features; return alarm-annotated copy."""
        out = df.copy()
        for feat in self._references:
            if feat not in df.columns:
                continue
            alarm_col = []
            for val in df[feat]:
                if pd.notna(val):
                    alarm_col.append(self.update(float(val), feat))
                else:
                    alarm_col.append(False)
            out[f"{feat}_alarm"] = alarm_col
        alarm_cols = [c for c in out.columns if c.endswith("_alarm")]
        if alarm_cols:
            out["any_alarm"] = out[alarm_cols].any(axis=1)
        return out

    def get_state(self) -> Dict[str, Any]:
        """Snapshot of current accumulator and alarm states."""
        return {
            feat: {
                "s_pos": self._s_pos.get(feat, 0.0),
                "s_neg": self._s_neg.get(feat, 0.0),
                "alarm": self._alarms.get(feat, False),
            }
            for feat in self._references
        }


class _MahalanobisScorer:
    """Mahalanobis distance scorer using Ledoit-Wolf shrinkage covariance."""

    def __init__(self):
        """Initialise scorer with empty state before calling fit()."""
        self.centroid: Optional[np.ndarray] = None
        self.inv_cov: Optional[np.ndarray] = None
        self.ref_mean_dist: float = 1.0
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray) -> None:
        """Fit centroid and inverse covariance from reference vectors.

        At any sample size we use Ledoit-Wolf shrinkage, which adapts its
        regularisation parameter to the ratio n/d.  For very small n (< 2d)
        the analytical shrinkage coefficient will be large (close to 1),
        effectively pulling the estimate toward a scaled identity matrix and
        avoiding the singular-covariance problem that previously forced a
        diagonal-only fallback.  This means we always get a true Mahalanobis
        metric that accounts for feature correlations, even at n=3.
        """
        self.centroid = np.mean(X, axis=0)
        n, d = X.shape
        if n >= 2:
            try:
                lw = LedoitWolf()
                lw.fit(X)
                self.inv_cov = np.linalg.pinv(lw.covariance_)
            except Exception:
                self.inv_cov = np.diag(1.0 / (np.var(X, axis=0, ddof=0) + 1e-8))
        else:
            self.inv_cov = np.eye(d)
        ref_dists = [self._dist(x) for x in X]
        self.ref_mean_dist = max(float(np.mean(ref_dists)), 1e-6)
        self.is_fitted = True

    def _dist(self, x: np.ndarray) -> float:
        """Raw Mahalanobis distance from centroid."""
        diff = x - self.centroid
        return float(np.sqrt(max(diff @ self.inv_cov @ diff, 0.0)))

    def score(self, x: np.ndarray) -> float:
        """Return normalized distance: >1.0 means outside reference spread."""
        if not self.is_fitted:
            return 0.0
        return self._dist(x) / self.ref_mean_dist


class _NearestCentroidScorer:
    """Nearest-centroid and nearest-neighbor distance scorer."""

    def __init__(self):
        """Initialise scorer with empty state before calling fit()."""
        self.centroid: Optional[np.ndarray] = None
        self.ref_spread: float = 1.0
        self.X_ref: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray) -> None:
        """Fit centroid and reference spread from reference vectors."""
        self.centroid = np.mean(X, axis=0)
        ref_dists = np.linalg.norm(X - self.centroid, axis=1)
        self.ref_spread = max(float(np.mean(ref_dists)), 1e-6)
        self.X_ref = X.copy()
        self.is_fitted = True

    def score(self, x: np.ndarray) -> float:
        """Return normalized distance: >1.0 means outside reference spread."""
        if not self.is_fitted:
            return 0.0
        centroid_dist = np.linalg.norm(x - self.centroid) / self.ref_spread
        nn_dist = float(np.linalg.norm(x - self.X_ref, axis=1).min()) / self.ref_spread
        return float(max(centroid_dist, nn_dist))


class _WithinSessionOutlierScorer:
    """Combined LOF and reference-distance outlier scorer.

    When a reference centroid has been supplied, each point's distance to
    that centroid (normalised by reference spread) is combined with the LOF
    to produce the final score.
    """

    def __init__(self):
        """Initialise scorer with empty state before calling fit_reference()."""
        self._ref_centroid: Optional[np.ndarray] = None
        self._ref_spread: float = 1.0

    def fit_reference(self, X_ref: np.ndarray) -> None:
        """Store the reference centroid and mean distance for scaling."""
        self._ref_centroid = np.mean(X_ref, axis=0)
        dists = np.linalg.norm(X_ref - self._ref_centroid, axis=1)
        self._ref_spread = max(float(np.mean(dists)), 1e-6)

    def score_batch(self, X: np.ndarray, k: int = 2) -> List[float]:
        """Return per-sample outlier score combining LOF and reference distance."""
        from scipy.spatial.distance import pdist, squareform

        n = len(X)
        if n < 2:
            if self._ref_centroid is not None and n == 1:
                d = float(np.linalg.norm(X[0] - self._ref_centroid)) / self._ref_spread
                return [d]
            return [0.0] * n

        k = min(k, n - 1)
        dist_matrix = squareform(pdist(X, metric='euclidean'))

        sorted_dists = np.sort(dist_matrix, axis=1)
        k_dists = sorted_dists[:, k]

        lrd = np.zeros(n)
        for i in range(n):
            nn_indices = np.argsort(dist_matrix[i])[1:k + 1]
            reach_dists = np.maximum(dist_matrix[i, nn_indices], k_dists[nn_indices])
            avg_reach = np.mean(reach_dists)
            lrd[i] = 1.0 / max(avg_reach, 1e-10)

        lof_scores = np.zeros(n)
        for i in range(n):
            nn_indices = np.argsort(dist_matrix[i])[1:k + 1]
            lof_scores[i] = np.mean(lrd[nn_indices]) / max(lrd[i], 1e-10)

        if self._ref_centroid is not None:
            ref_dists = np.array([
                float(np.linalg.norm(X[i] - self._ref_centroid)) / self._ref_spread
                for i in range(n)
            ])
            return [float(max(lof, rd)) for lof, rd in zip(lof_scores, ref_dists)]

        return [float(s) for s in lof_scores]


class AnomalyDetector:
    """Task-relevant anomaly detector with t-distribution prediction intervals.

    Selects features based on the task's primary_blendshapes and symmetry
    pairs from the task configuration, then applies proper small-sample
    statistical testing via prediction intervals.  Geometric scorers
    (Mahalanobis, centroid, within-session) operate in the task-relevant
    feature space.  ML model scoring is only activated when n_reference >= 10.
    """

    _EXTRA_EXCLUDES = frozenset({"n_frames", "duration_sec", "detection_rate"})

    _DDK_EXCLUDE_STAT_SUFFIX = "time_to_peak"

    def __init__(self, decision_rules_config: Dict[str, Any], tasks_config: Optional[Dict[str, Any]] = None):
        """Initialise detector from loaded decision_rules YAML and optional tasks YAML."""
        self.config = decision_rules_config
        self.anomaly_config = decision_rules_config.get("anomaly_detection", {})
        self.tasks_config = tasks_config or {}

        if_config = self.anomaly_config.get("isolation_forest", {})
        self._if_params = {
            "n_estimators": if_config.get("n_estimators", 100),
            "contamination": if_config.get("contamination", 0.1),
            "max_samples": if_config.get("max_samples", "auto"),
            "random_state": if_config.get("random_state", 42),
        }

        self.deviation_threshold = self.anomaly_config.get("deviation_threshold_std", 2.0)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.reference_stats: Dict[str, Dict[str, float]] = {}
        self.feature_names: List[str] = []
        self.task_profile_stats: Optional[Dict[str, Dict[str, float]]] = None
        self._feature_weights: Dict[str, float] = {}
        self._use_ocsvm: bool = False
        self._n_reference: int = 0
        self.n_reference: int = 0
        self._model: Any = None

        self.pca: Optional[PCA] = None
        self.pca_feature_names: List[str] = []
        self.n_pca_components: int = 0
        self.pca_explained_variance: List[float] = []
        self.selected_features: List[str] = []
        self.learned_importance: Dict[str, float] = {}

        self.task_group: Optional[str] = None
        self.task_id: Optional[int] = None
        self._task_config: Dict[str, Any] = {}
        self._task_relevant_features: List[str] = []

        self._mahal_scorer = _MahalanobisScorer()
        self._centroid_scorer = _NearestCentroidScorer()
        self._within_scorer = _WithinSessionOutlierScorer()

        self._ref_score_dist: Dict[str, Dict[str, float]] = {}

    def _filter_ddk_count_sensitive(self, features: List[str]) -> List[str]:
        """Drop features whose value depends on syllable count for B (DDK) tasks.

        Only active when ``self.task_group == "B"``.  All other task groups
        are passed through unchanged.
        """
        if self.task_group != "B":
            return features
        return [f for f in features if not f.endswith(f"_{self._DDK_EXCLUDE_STAT_SUFFIX}")]

    def _resolve_task_config(self, task_group: Optional[str], task_id: Optional[int]) -> Dict[str, Any]:
        """Look up the task definition from tasks_config for the given group and id.

        Task-level settings take precedence; group-level settings are used as
        fallback for keys not present in the specific task entry.  This allows
        group-wide defaults (e.g. primary_blendshapes for group C) to be
        inherited by every task in that group.
        """
        if not task_group or task_id is None:
            return {}
        groups = self.tasks_config.get("task_groups", {})
        group = groups.get(task_group, {})
        group_defaults = {k: v for k, v in group.items() if k != "tasks"}
        tasks = group.get("tasks", {})
        task_entry = tasks.get(task_id, tasks.get(str(task_id), {}))
        merged = {**group_defaults, **task_entry}
        return merged

    def _get_task_relevant_features(
        self,
        task_config: Dict[str, Any],
        available_features: List[str],
    ) -> List[str]:
        """Select features relevant to the task based on primary_blendshapes and symmetry_pairs.

        For each primary blendshape, includes all stat columns (mean, std,
        max, min, range, time_to_peak).  For each symmetry pair, includes
        asymmetry ratio columns and signed asymmetry.  Members of symmetry
        pairs that are not already primary blendshapes are also included.
        """
        primary_bs = task_config.get("primary_blendshapes", [])
        symmetry_pairs = task_config.get("symmetry_pairs", [])
        avail_set = set(available_features)
        selected = set()

        active_stat_suffixes = tuple(
            s for s in _STAT_SUFFIXES
            if not (self.task_group == "B" and s == self._DDK_EXCLUDE_STAT_SUFFIX)
        )

        for bs in primary_bs:
            for stat in active_stat_suffixes:
                col = f"{bs}_{stat}"
                if col in avail_set:
                    selected.add(col)

        seen_bases = set()
        for pair in symmetry_pairs:
            if len(pair) != 2:
                continue
            base = _derive_pair_base(pair[0])
            seen_bases.add(base)
            for stat in active_stat_suffixes:
                col = f"asymmetry_ratio_{base}_{stat}"
                if col in avail_set:
                    selected.add(col)
            sa_col = f"signed_asymmetry_{base}"
            if sa_col in avail_set:
                selected.add(sa_col)
            for member in pair:
                if member not in primary_bs:
                    for stat in active_stat_suffixes:
                        col = f"{member}_{stat}"
                        if col in avail_set:
                            selected.add(col)

        for bs in primary_bs:
            base = _derive_pair_base(bs)
            if base not in seen_bases:
                for stat in active_stat_suffixes:
                    col = f"asymmetry_ratio_{base}_{stat}"
                    if col in avail_set:
                        selected.add(col)
                sa_col = f"signed_asymmetry_{base}"
                if sa_col in avail_set:
                    selected.add(sa_col)

        return sorted(selected)

    def _compute_prediction_interval(self, n_ref: int, alpha: float = 0.05) -> float:
        """Return the t-distribution critical value for a two-sided prediction interval."""
        if n_ref < 2:
            return 4.0
        df = n_ref - 1
        return float(t_dist.ppf(1.0 - alpha / 2.0, df))

    @staticmethod
    def _adaptive_std_floor(ref_mean: float, ref_std: float, ref_range: float, n_ref: int) -> float:
        """Compute an adaptive std floor that is principled for small n.

        The floor serves two purposes:
        1. Prevent division-by-zero when a feature shows no variance in the
           reference (e.g. the person always blinks exactly the same amount).
        2. At very small n the observed std underestimates the true population
           std, so we add a component based on the observed range — which
           shrinks as n grows and we become more confident in the std estimate.

        Floor = max(
            range_component,   # range / (2 * sqrt(n)), shrinks with n
            mean_fraction,     # 1 % of |mean|, prevents over-sensitivity
            hard_min,          # absolute minimum
        )
        """
        range_component = ref_range / (2.0 * max(np.sqrt(n_ref), 1.0))
        mean_fraction = abs(ref_mean) * 0.01
        hard_min = 1e-4
        return max(range_component, mean_fraction, hard_min)

    def _compute_feature_deviation(
        self,
        value: float,
        feature_name: str,
    ) -> Tuple[float, bool, str]:
        """Compute t-statistic-based deviation for a single feature value.

        Returns (deviation_magnitude, is_deviant, direction) where
        deviation_magnitude is abs(t_stat) / t_critical so that values
        above 1.0 indicate the test point falls outside the prediction
        interval.
        """
        stats = self.reference_stats.get(feature_name, {})
        if not stats:
            return 0.0, False, "within"

        n_ref = stats.get("n", self._n_reference)
        ref_mean = stats.get("mean", 0.0)
        ref_std = stats.get("std", 0.0)
        ref_range = stats.get("ref_range", abs(ref_mean) * 0.1)

        std_floor = self._adaptive_std_floor(ref_mean, ref_std, ref_range, n_ref)
        effective_std = max(ref_std, std_floor)

        se_pred = effective_std * np.sqrt(1.0 + 1.0 / max(n_ref, 2))
        t_stat = (value - ref_mean) / se_pred
        t_crit = self._compute_prediction_interval(n_ref)

        deviation_magnitude = abs(t_stat) / max(t_crit, 0.01)

        if value > ref_mean + 1e-9:
            direction = "above"
        elif value < ref_mean - 1e-9:
            direction = "below"
        else:
            direction = "within"

        is_deviant = deviation_magnitude > 1.0

        return float(deviation_magnitude), is_deviant, direction

    def _compute_range_deviation(self, value: float, feature_name: str) -> float:
        """Compute range-based deviation for backward compatibility with visualizations.

        Uses the reference range expanded by the prediction interval factor
        to define the tolerance band.
        """
        stats = self.reference_stats.get(feature_name, {})
        if not stats:
            return 0.0

        ref_mean = stats.get("mean", 0.0)
        ref_std = stats.get("std", 0.0)
        ref_range = stats.get("ref_range", abs(ref_mean) * 0.1)
        n_ref = stats.get("n", self._n_reference)

        std_floor = self._adaptive_std_floor(ref_mean, ref_std, ref_range, n_ref)
        effective_std = max(ref_std, std_floor)

        t_crit = self._compute_prediction_interval(n_ref)
        tolerance = effective_std * np.sqrt(1.0 + 1.0 / max(n_ref, 2)) * t_crit
        tolerance = max(tolerance, abs(ref_mean) * 0.05 + 1e-4)

        deviation = abs(value - ref_mean) / tolerance
        return float(deviation)

    @staticmethod
    def _compute_modified_z(value: float, median: float, mad: float) -> float:
        """Hampel modified z-score: 0.6745 * (x - median) / MAD."""
        if mad < 1e-12:
            return 0.0
        return 0.6745 * (value - median) / mad

    @staticmethod
    def _soft_sigmoid(s: float, center: float = 1.5, steep: float = 2.0) -> float:
        """Soft sigmoid mapping for score normalization."""
        return float(1.0 / (1.0 + np.exp(-steep * (s - center))))

    def set_task_feature_weights(self, task_config: Dict[str, Any]) -> None:
        """Weight primary_blendshapes features 2×, others 1×."""
        primary = set(task_config.get("primary_blendshapes", []))
        self._feature_weights = {}
        for feat in self.feature_names:
            self._feature_weights[feat] = 2.0 if any(p in feat for p in primary) else 1.0

    def _get_task_group_weights(self) -> Dict[str, float]:
        """Return composite weights adapted to the current task group and n_ref.

        Feature deviation (t-prediction intervals) receives the highest weight
        because it is the most reliable signal at small n.  Geometric distances
        (Mahalanobis, centroid) become more useful as n grows — they need
        enough samples to estimate covariance and spread robustly.

        n_ref scaling:
          - At n=3 (minimum pilot reference): geometric weights are halved;
            feature deviation is maximised.
          - At n≥10: geometric weights rise to their nominal values; ML model
            contributes.  Linear ramp between n=3 and n=10.
        """
        n_ref = self._n_reference
        use_ml = n_ref >= 10

        n_scale = min(1.0, max(0.0, (n_ref - 3) / 7.0))

        base = {
            "w_model":    (1.0 if use_ml else 0.0),
            "w_mahal":    0.30 + 0.40 * n_scale,
            "w_centroid": 0.20 + 0.25 * n_scale,
            "w_feature":  3.50 - 0.50 * n_scale,
            "w_within":   0.40 + 0.30 * n_scale,
        }

        if self.task_group == "A":
            base["w_feature"] += 0.2
            base["w_mahal"]   += 0.05 * n_scale
        elif self.task_group == "B":
            base["w_centroid"] += 0.10
        elif self.task_group == "C":
            base["w_within"]  += 0.10
            base["w_feature"] += 0.10

        return base

    def _calibrate_score(self, raw: float, method: str) -> float:
        """Normalize a raw score using the reference distribution for that method.

        For small reference sets (n < 10) calibration is skipped because the
        reference score distribution is too tight and dividing by its std
        inflates scores dramatically.
        """
        if self._n_reference < 10:
            return raw
        dist = self._ref_score_dist.get(method)
        if not dist:
            return raw
        ref_mean = dist.get("mean", 0.0)
        ref_std = dist.get("std", 1.0)
        if ref_std < 1e-8:
            return max(0.0, raw - ref_mean)
        return max(0.0, (raw - ref_mean) / ref_std)

    @staticmethod
    def _bootstrap_ci(
        scores: List[float],
        composite: float,
        n_bootstrap: int = 200,
        alpha: float = 0.05,
    ) -> Tuple[float, float]:
        """Percentile-bootstrap 95 % CI for a composite anomaly score."""
        if len(scores) < 4:
            half = composite * 0.10 + 0.01
            return max(0.0, composite - half), min(1.0, composite + half)
        arr = np.array(scores)
        rng = np.random.default_rng(42)
        boot_means: List[float] = []
        for _ in range(n_bootstrap):
            sample = rng.choice(arr, size=len(arr), replace=True)
            boot_means.append(float(np.mean(sample)))
        lo = float(np.percentile(boot_means, 100 * alpha / 2))
        hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
        return max(0.0, lo), min(1.0, hi)

    def _build_model(self, n_samples: int) -> None:
        """Instantiate the ML model; only meaningful for n >= 10."""
        self._n_reference = n_samples
        if n_samples < 15:
            self._use_ocsvm = True
            self._model = OneClassSVM(kernel="rbf", nu=0.1)
        else:
            self._use_ocsvm = False
            self._model = IsolationForest(**self._if_params)

    def _select_task_features(
        self,
        X_df: pd.DataFrame,
        available_features: List[str],
    ) -> List[str]:
        """Select task-relevant features or fall back to variance-based selection."""
        if self._task_config:
            task_feats = self._get_task_relevant_features(self._task_config, available_features)
            if len(task_feats) >= 3:
                return task_feats

        if self._feature_weights:
            weighted = [f for f in available_features if self._feature_weights.get(f, 1.0) > 1.0]
            if len(weighted) >= 3:
                return weighted

        if len(X_df) >= 2 and len(available_features) >= 3:
            stds = X_df[available_features].std()
            means = X_df[available_features].mean().abs() + 1e-8
            cvs = stds / means
            top = cvs.nlargest(min(30, len(available_features))).index.tolist()
            if len(top) >= 3:
                return top

        return available_features[:30] if len(available_features) > 30 else available_features

    def _prepare_feature_matrix(
        self,
        X_df: pd.DataFrame,
        features: List[str],
        is_fitting: bool = False,
    ) -> np.ndarray:
        """Scale features and optionally fit PCA for visualization.

        PCA is fitted for visualization support only; the geometric scorers
        operate on the scaled (non-PCA) feature space.
        """
        X = X_df[features].fillna(0).values

        col_vars = np.var(X, axis=0)
        nonzero_mask = col_vars > 1e-10
        if not nonzero_mask.any():
            nonzero_mask[:] = True
        X = X[:, nonzero_mask]
        active_features = [features[i] for i in range(len(features)) if nonzero_mask[i]]

        if is_fitting:
            self.selected_features = active_features
            self.scaler.fit(X)

            n_components = min(len(X) - 1, X.shape[1], 5)
            n_components = max(1, n_components)
            self.pca = PCA(n_components=n_components)
            self.pca.fit(X)
            self.pca_feature_names = active_features
            self.n_pca_components = n_components
            self.pca_explained_variance = self.pca.explained_variance_ratio_.tolist()

        X_scaled = self.scaler.transform(X)
        return X_scaled

    def fit(
        self,
        reference_metrics_df: pd.DataFrame,
        task_group: Optional[str] = None,
        task_id: Optional[int] = None,
    ) -> None:
        """Fit anomaly detection models on reference data."""
        if len(reference_metrics_df) < 2:
            raise ValueError(f"Need at least 2 reference samples, got {len(reference_metrics_df)}")

        logger.debug("Fitting anomaly detector with %d reference samples", len(reference_metrics_df))

        self.n_reference = len(reference_metrics_df)
        self._n_reference = len(reference_metrics_df)
        self.task_group = task_group
        self.task_id = task_id
        self._task_config = self._resolve_task_config(task_group, task_id)

        all_features = get_numeric_feature_columns(
            reference_metrics_df, self._EXTRA_EXCLUDES
        )
        if not all_features:
            raise ValueError("No valid feature columns found")

        all_features = self._filter_ddk_count_sensitive(all_features)

        self.feature_names = all_features
        if not self._feature_weights:
            self._feature_weights = {f: 1.0 for f in all_features}

        if self._task_config and not any(v > 1.0 for v in self._feature_weights.values()):
            self.set_task_feature_weights(self._task_config)

        n = len(reference_metrics_df)
        for feat in all_features:
            values = reference_metrics_df[feat].dropna().values
            if len(values) > 0:
                self.reference_stats[feat] = {
                    "mean":          float(np.mean(values)),
                    "std":           float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "median":        float(np.median(values)),
                    "mad":           _compute_mad(values),
                    "ref_min":       float(np.min(values)),
                    "ref_max":       float(np.max(values)),
                    "ref_range":     float(np.max(values) - np.min(values)),
                    "ref_median_abs": float(abs(np.median(values))),
                    "q25":           float(np.percentile(values, 25)),
                    "q75":           float(np.percentile(values, 75)),
                    "n":             int(len(values)),
                }

        self._task_relevant_features = self._select_task_features(
            reference_metrics_df, all_features
        )
        logger.debug(
            "Task %s_%s: %d task-relevant features selected from %d total",
            task_group, task_id, len(self._task_relevant_features), len(all_features),
        )

        X_scaled = self._prepare_feature_matrix(
            reference_metrics_df, self._task_relevant_features, is_fitting=True
        )

        self._build_model(n)
        if n >= 10:
            X_pca_scaled = self.scaler.transform(
                reference_metrics_df[self.selected_features].fillna(0).values
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._model.fit(X_pca_scaled)

        self._mahal_scorer.fit(X_scaled)
        self._centroid_scorer.fit(X_scaled)
        self._within_scorer.fit_reference(X_scaled)

        self._compute_reference_score_dist(X_scaled)

        self.is_fitted = True

    def _compute_reference_score_dist(self, X_scaled: np.ndarray) -> None:
        """Score reference samples to build per-method calibration distributions."""
        mahal_scores = np.array([self._mahal_scorer.score(x) for x in X_scaled])
        centroid_scores = np.array([self._centroid_scorer.score(x) for x in X_scaled])

        for name, arr in [
            ("mahal", mahal_scores),
            ("centroid", centroid_scores),
        ]:
            self._ref_score_dist[name] = {
                "mean": float(np.mean(arr)),
                "std": max(float(np.std(arr)), 1e-8),
                "p75": float(np.percentile(arr, 75)),
                "p90": float(np.percentile(arr, 90)),
                "max": float(np.max(arr)),
            }

        if self._n_reference >= 10 and self._model is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if self._use_ocsvm:
                    model_scores = -self._model.decision_function(X_scaled)
                else:
                    model_scores = -self._model.score_samples(X_scaled)
            self._ref_score_dist["model"] = {
                "mean": float(np.mean(model_scores)),
                "std": max(float(np.std(model_scores)), 1e-8),
                "p75": float(np.percentile(model_scores, 75)),
                "p90": float(np.percentile(model_scores, 90)),
                "max": float(np.max(model_scores)),
            }

    def fit_from_task_profile(
        self,
        task_feature_stats: Dict[str, Dict[str, float]],
        reference_metrics_df: Optional[pd.DataFrame] = None,
        task_group: Optional[str] = None,
        task_id: Optional[int] = None,
    ) -> None:
        """Fit using pre-computed per-feature stats from a task profile."""
        if task_group is not None:
            self.task_group = task_group
        if task_id is not None:
            self.task_id = task_id
        if not self._task_config and (self.task_group or self.task_id is not None):
            self._task_config = self._resolve_task_config(self.task_group, self.task_id)
        self.task_profile_stats = task_feature_stats

        for feat, stats in task_feature_stats.items():
            raw_std = stats.get("std", 1.0)
            ref_std = stats.get("coverage_std", raw_std)
            if stats.get("n", 0) > 1 and ref_std == 0.0:
                ref_std = max(
                    (stats.get("max", 0.0) - stats.get("min", 0.0)) / 3.46, 1e-6
                )
            self.reference_stats[feat] = {
                "mean":          stats.get("mean", 0.0),
                "std":           ref_std,
                "median":        stats.get("median", 0.0),
                "mad":           stats.get("mad", 0.0),
                "ref_min":       stats.get("min", 0.0),
                "ref_max":       stats.get("max", 0.0),
                "ref_range":     stats.get("max", 0.0) - stats.get("min", 0.0),
                "ref_median_abs": abs(stats.get("median", 0.0)),
                "q25":           stats.get("q25", 0.0),
                "q75":           stats.get("q75", 0.0),
                "n":             stats.get("n", 0),
            }
        self.feature_names = list(task_feature_stats.keys())
        self.feature_names = self._filter_ddk_count_sensitive(self.feature_names)

        if not self._feature_weights:
            self._feature_weights = {f: 1.0 for f in self.feature_names}

        if self._task_config and not any(v > 1.0 for v in self._feature_weights.values()):
            self.set_task_feature_weights(self._task_config)

        if reference_metrics_df is not None and len(reference_metrics_df) > 0:
            available = [
                f for f in self.feature_names
                if f in reference_metrics_df.columns
            ]
            if len(available) >= 2:
                n = len(reference_metrics_df)
                self._n_reference = n
                self.n_reference = n

                self._task_relevant_features = self._select_task_features(
                    reference_metrics_df, available
                )

                X_scaled = self._prepare_feature_matrix(
                    reference_metrics_df, self._task_relevant_features, is_fitting=True
                )

                self._build_model(n)
                if n >= 10:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        self._model.fit(X_scaled)

                self._mahal_scorer.fit(X_scaled)
                self._centroid_scorer.fit(X_scaled)
                self._within_scorer.fit_reference(X_scaled)
                self._compute_reference_score_dist(X_scaled)
                self.feature_names = available
                self.is_fitted = True
        else:
            n_total = sum(
                1 for s in task_feature_stats.values()
                if s.get("n", 0) > 0
            )
            self._n_reference = max(n_total, 1)
            self.n_reference = max(n_total, 1)

    def compute_dtw_pattern_deviation(
        self, test_series: np.ndarray, reference_curves: List[np.ndarray]
    ) -> Dict[str, Any]:
        """Compute DTW distance between a test curve and all reference curves.

        Classifies the temporal anomaly type:
        - is_temporal_slowdown: test signal is >= 30 % longer than mean reference
          length AND the DTW distance is elevated.  Corresponds to preserved shape
          but stretched duration — the canonical dysarthria temporal signature.
        - is_temporal_delay: peak activation of the test curve arrives significantly
          later than the reference peak (mean peak-time ratio > 1.25), but the
          overall signal length is comparable.  Corresponds to increased reaction /
          initiation latency.
        - is_shape_anomaly: DTW distance exceeds mean inter-reference variability by
          > 2 SD.  Indicates a genuine shape change irrespective of timing.

        The DTW distance uses a Sakoe-Chiba band constraint (Sakoe and Chiba
        1978, doi:10.1109/TASSP.1978.1163055) to limit warping to a local
        diagonal band proportional to the signal length, preventing degenerate
        alignments that would otherwise inflate distance scores.
        """
        if len(reference_curves) == 0:
            return {
                "mean_dtw": 0.0, "min_dtw": 0.0,
                "is_shape_anomaly": False,
                "is_temporal_slowdown": False,
                "is_temporal_delay": False,
                "temporal_type": "none",
                "stretch_ratio": 1.0,
                "peak_time_ratio": 1.0,
            }

        ref_range = max(
            float(np.ptp(c)) for c in reference_curves if len(c) > 0
        )
        ref_range = max(ref_range, 1e-9)

        band = max(10, len(test_series) // 5)

        distances = []
        for rc in reference_curves:
            d = _dtw_distance(test_series, rc, band=band)
            distances.append(d / (ref_range * len(test_series)))

        mean_d = float(np.mean(distances))
        min_d = float(np.min(distances))

        inter_ref = []
        for i in range(len(reference_curves)):
            for j in range(i + 1, len(reference_curves)):
                d = _dtw_distance(reference_curves[i], reference_curves[j], band=band)
                inter_ref.append(d / (ref_range * len(reference_curves[i])))
        if inter_ref:
            ref_mean = float(np.mean(inter_ref))
            ref_std = float(np.std(inter_ref)) if len(inter_ref) > 1 else ref_mean * 0.5
            threshold = ref_mean + 2.0 * max(ref_std, 1e-9)
        else:
            threshold = mean_d * 2.0

        is_shape_anomaly = mean_d > threshold

        mean_ref_len = float(np.mean([len(rc) for rc in reference_curves]))
        test_len = float(len(test_series))
        stretch_ratio = test_len / max(mean_ref_len, 1.0)
        is_temporal_slowdown = (stretch_ratio >= 1.30) and is_shape_anomaly

        def _peak_time(arr: np.ndarray) -> float:
            """Return normalised time of peak absolute value (0.0 = start, 1.0 = end)."""
            if len(arr) == 0:
                return 0.5
            idx = int(np.argmax(np.abs(arr)))
            return float(idx) / max(len(arr) - 1, 1)

        ref_peak_times = [_peak_time(rc) for rc in reference_curves]
        mean_ref_peak = float(np.mean(ref_peak_times)) if ref_peak_times else 0.5
        test_peak = _peak_time(test_series)
        peak_time_ratio = test_peak / max(mean_ref_peak, 0.05)
        is_temporal_delay = (
            peak_time_ratio > 1.25
            and not is_temporal_slowdown
            and stretch_ratio < 1.30
        )

        if is_temporal_slowdown:
            temporal_type = "slowdown"
        elif is_temporal_delay:
            temporal_type = "delay"
        elif is_shape_anomaly:
            temporal_type = "shape_change"
        else:
            temporal_type = "none"

        return {
            "mean_dtw":              mean_d,
            "min_dtw":               min_d,
            "is_shape_anomaly":      is_shape_anomaly,
            "is_temporal_slowdown":  is_temporal_slowdown,
            "is_temporal_delay":     is_temporal_delay,
            "temporal_type":         temporal_type,
            "stretch_ratio":         round(stretch_ratio, 3),
            "peak_time_ratio":       round(peak_time_ratio, 3),
        }

    def detect_anomalies(
        self,
        metrics_df: pd.DataFrame,
        kin_summary: Optional[Dict[str, Any]] = None,
        dtw_results: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Score each repetition using task-relevant feature deviations and geometric distances.
        
        Optional kin_summary dict containing kinematic profile deviations for B/C groups.
        For B/C groups only, kinematic deviation is blended at 10% weight into composite.

        Optional dtw_results dict (keyed by "{task_group}_{task_id}") containing
        per-repetition DTW pattern deviation results.  When provided, the DTW
        shape-anomaly signal is blended at 10% weight for B/C tasks, and the
        temporal_type field from DTW is reflected in the per-rep anomaly_type output.
        """
        n_ref = self._n_reference
        results: Dict[str, Any] = {
            "anomaly_scores":        [],
            "is_anomaly":            [],
            "deviations":            [],
            "feature_deviations":    {},
            "deviation_score":       [],
            "score_confidence":      [],
            "anomaly_type":          [],
            "contributing_features": [],
            "mahalanobis_score":     [],
            "centroid_score":        [],
            "within_session_score":  [],
            "method_votes":          [],
            "weighted_votes":        [],
            "method_sigmoid_scores": [],
            "method_weighted_components": [],
            "mahalanobis_ci_lower":  [],
            "mahalanobis_ci_upper":  [],
            "deviation_ci_lower":    [],
            "deviation_ci_upper":    [],
            "model_type":   "OC-SVM" if self._use_ocsvm else "IsolationForest",
            "n_reference":  n_ref,
            "n_pca_components": self.n_pca_components,
            "pca_explained_variance": self.pca_explained_variance,
            "effective_threshold": self._compute_prediction_interval(n_ref),
            "ml_metadata": {
                "n_pca_components":       self.n_pca_components,
                "pca_explained_variance": self.pca_explained_variance,
                "n_features_selected":    len(self.selected_features),
                "n_task_relevant":        len(self._task_relevant_features),
                "scoring_method":         "task_relevant_t_prediction_interval",
            },
            "summary": {},
        }

        if len(metrics_df) == 0:
            return results

        available_features = [
            f for f in self.feature_names if f in metrics_df.columns
        ]
        if not available_features:
            n = len(metrics_df)
            results["anomaly_scores"]        = [0.0] * n
            results["is_anomaly"]            = [False] * n
            results["deviations"]            = [{}] * n
            results["deviation_score"]       = [0.0] * n
            results["score_confidence"]      = [0.0] * n
            results["anomaly_type"]          = ["unknown"] * n
            results["contributing_features"] = [[]] * n
            results["mahalanobis_score"]     = [0.0] * n
            results["centroid_score"]        = [0.0] * n
            results["within_session_score"]  = [0.0] * n
            results["method_votes"]          = [[]] * n
            return results

        task_feats = [f for f in self._task_relevant_features if f in metrics_df.columns]
        if len(task_feats) < 3:
            task_feats = self._select_task_features(metrics_df, available_features)
        score_features = task_feats if task_feats else available_features

        X_scaled = None
        if self.is_fitted and self.selected_features:
            missing_sel = [f for f in self.selected_features if f not in metrics_df.columns]
            X_raw = np.zeros((len(metrics_df), len(self.selected_features)))
            for _fi, _feat in enumerate(self.selected_features):
                if _feat in metrics_df.columns:
                    X_raw[:, _fi] = metrics_df[_feat].fillna(0).values
            X_scaled = self.scaler.transform(X_raw)

        use_ml = self._n_reference >= 10 and self._model is not None
        if use_ml and X_scaled is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if self._use_ocsvm:
                    raw_model_scores = -self._model.decision_function(X_scaled)
                else:
                    raw_model_scores = -self._model.score_samples(X_scaled)
            results["anomaly_scores"] = raw_model_scores.tolist()
        else:
            results["anomaly_scores"] = [0.0] * len(metrics_df)

        if X_scaled is not None and self._mahal_scorer.is_fitted:
            mahal_scores = [self._mahal_scorer.score(X_scaled[i])
                            for i in range(len(metrics_df))]
            centroid_scores = [self._centroid_scorer.score(X_scaled[i])
                               for i in range(len(metrics_df))]
        else:
            mahal_scores = [0.0] * len(metrics_df)
            centroid_scores = [0.0] * len(metrics_df)
        results["mahalanobis_score"] = mahal_scores
        results["centroid_score"] = centroid_scores

        if X_scaled is not None:
            within_scores = self._within_scorer.score_batch(X_scaled)
        else:
            within_scores = [0.0] * len(metrics_df)
        results["within_session_score"] = within_scores

        if X_scaled is not None and self.pca is not None:
            pca_projected = self.pca.transform(X_scaled)
            results["pca_projected"] = pca_projected.tolist()
            results["pca_loadings"] = self.pca.components_.tolist()
            results["pca_feature_names"] = list(self.selected_features)

        score_set = set(score_features)

        _dtw_by_rep: Dict[int, Dict[str, Any]] = {}
        if dtw_results is not None and self.task_group in ("B", "C"):
            _task_key = f"{self.task_group}_{self.task_id}"
            _per_rep_dtw = dtw_results.get(_task_key, {}).get("repetitions", [])
            for _dr in _per_rep_dtw:
                if isinstance(_dr, dict) and "repetition" in _dr:
                    _dtw_by_rep[int(_dr["repetition"])] = _dr

        for i in range(len(metrics_df)):
            row_deviations: Dict[str, Dict[str, Any]] = {}
            contributing: List[str] = []
            type_votes: Dict[str, float] = {}

            for feat in available_features:
                value = metrics_df[feat].iloc[i]
                if pd.isna(value) or feat not in self.reference_stats:
                    continue

                ref = self.reference_stats[feat]
                median = ref.get("median", ref["mean"])
                mad = ref.get("mad", 0.0)
                mod_z = self._compute_modified_z(float(value), median, mad)
                dev_mag, is_dev, direction = self._compute_feature_deviation(float(value), feat)
                rng_dev = self._compute_range_deviation(float(value), feat)
                weight = self._feature_weights.get(feat, 1.0)

                row_deviations[feat] = {
                    "value":        float(value),
                    "modified_z":   float(mod_z),
                    "range_dev":    float(rng_dev),
                    "t_deviation":  float(dev_mag),
                    "weighted_dev": float(dev_mag * weight),
                    "weight":       weight,
                    "is_deviant":   is_dev,
                    "z_score":      float(mod_z),
                    "direction":    direction,
                }

                if is_dev and feat in score_set:
                    contributing.append(feat)
                    if not any(kw in feat.lower() for kw in _SCALE_FEATURE_KEYWORDS):
                        atype = _classify_anomaly_type(feat)
                        type_votes[atype] = (
                            type_votes.get(atype, 0.0) + dev_mag * weight
                        )

            results["deviations"].append(row_deviations)

            task_devs = []
            for f in score_features:
                if f in row_deviations and not any(kw in f.lower() for kw in _SCALE_FEATURE_KEYWORDS):
                    task_devs.append(row_deviations[f]["t_deviation"])
            if not task_devs:
                task_devs = [
                    row_deviations[f]["t_deviation"]
                    for f in row_deviations
                    if not any(kw in f.lower() for kw in _SCALE_FEATURE_KEYWORDS)
                ]

            n_task_feats = max(len(task_devs), 1)
            n_deviant_task = sum(1 for d in task_devs if d > 1.0)
            deviant_ratio = n_deviant_task / n_task_feats

            sorted_devs = sorted(task_devs, reverse=True)
            top_k = max(1, len(sorted_devs) // 4)
            severity = float(np.mean(sorted_devs[:top_k])) if sorted_devs else 0.0

            n_near_deviant = sum(1 for d in task_devs if 0.70 < d <= 1.0)
            soft_deviant_count = n_deviant_task + 0.5 * n_near_deviant
            breadth = min(1.0, (soft_deviant_count / n_task_feats) * 2.5)

            depth = min(1.0, severity / 3.0)

            _deviant_dirs = [
                row_deviations[f]["direction"]
                for f in score_features
                if f in row_deviations and row_deviations[f]["is_deviant"]
            ]
            if len(_deviant_dirs) >= 2:
                from collections import Counter as _Counter
                _dir_counts = _Counter(_deviant_dirs)
                _max_dir = _dir_counts.most_common(1)[0][1]
                direction_consistency = float(_max_dir) / len(_deviant_dirs)
            elif len(_deviant_dirs) == 1:
                direction_consistency = 1.0
            else:
                direction_consistency = 0.5

            feature_dev_score = (
                0.50 * breadth
                + 0.35 * depth
                + 0.15 * direction_consistency
            )

            cal_mahal = self._calibrate_score(mahal_scores[i], "mahal")
            cal_centroid = self._calibrate_score(centroid_scores[i], "centroid")

            mahal_s = self._soft_sigmoid(cal_mahal, 2.0, 2.0)
            centroid_s = self._soft_sigmoid(cal_centroid, 2.0, 2.0)
            within_s = self._soft_sigmoid(within_scores[i], 2.0, 2.0)

            if use_ml:
                cal_model = self._calibrate_score(results["anomaly_scores"][i], "model")
                model_s = self._soft_sigmoid(cal_model, 0.5, 3.0)
            else:
                model_s = 0.0

            tgw = self._get_task_group_weights()
            w_model = tgw["w_model"]
            w_mahal = tgw["w_mahal"]
            w_centroid = tgw["w_centroid"]
            w_feature = tgw["w_feature"]
            w_within = tgw["w_within"]
            total_w = w_model + w_mahal + w_centroid + w_feature + w_within

            kin_dev_score = 0.0
            if (
                kin_summary is not None
                and isinstance(kin_summary, dict)
                and self.task_group in ["B", "C"]
            ):
                if "overall_deviation" in kin_summary:
                    kin_dev_score = min(1.0, kin_summary["overall_deviation"] / 3.0)
                elif "mean_deviation" in kin_summary:
                    kin_dev_score = min(1.0, kin_summary["mean_deviation"] / 3.0)

            _rep_num = int(metrics_df["repetition"].iloc[i]) if "repetition" in metrics_df.columns else i + 1
            _dtw_rep = _dtw_by_rep.get(_rep_num, {})
            dtw_score = 0.0
            dtw_temporal_type = "none"
            if _dtw_rep:
                _raw_dtw = float(_dtw_rep.get("mean_dtw", 0.0))
                dtw_score = float(1.0 / (1.0 + np.exp(-8.0 * (_raw_dtw - 0.10))))
                dtw_temporal_type = _dtw_rep.get("temporal_type", "none")

            base_weighted = (
                (w_model * model_s
                + w_mahal * mahal_s
                + w_centroid * centroid_s
                + w_feature * feature_dev_score
                + w_within * within_s) / total_w
            )

            if _dtw_by_rep:
                composite = (
                    0.80 * base_weighted
                    + 0.10 * kin_dev_score
                    + 0.10 * dtw_score
                )
            else:
                composite = (
                    0.85 * base_weighted
                    + 0.15 * kin_dev_score
                )

            results["deviation_score"].append(round(composite, 4))

            sample_conf = min(1.0, n_ref / 10.0)
            n_scoreable = len([
                f for f in score_features
                if self.reference_stats.get(f, {}).get("std", 0) > 1e-6
                or self.reference_stats.get(f, {}).get("ref_range", 0) > 1e-4
            ])
            spread_conf = min(1.0, n_scoreable / max(len(score_features) * 0.3, 1))
            conf = round(sample_conf * 0.7 + spread_conf * 0.3, 3)
            results["score_confidence"].append(conf)

            if type_votes:
                total_vote = sum(type_votes.values())
                prominent = [
                    atype for atype, w in sorted(type_votes.items(), key=lambda x: -x[1])
                    if w / max(total_vote, 1e-9) >= 0.15
                ]
                top_type = "+".join(prominent[:3]) if prominent else "unknown"
            else:
                top_type = "unknown"

            if dtw_temporal_type != "none":
                if dtw_temporal_type == "slowdown":
                    top_type = (
                        "temporal_slowdown+" + top_type
                        if top_type != "unknown" else "temporal_slowdown"
                    )
                elif dtw_temporal_type == "delay":
                    if "temporal" not in top_type:
                        top_type = top_type + "+temporal_delay" if top_type != "unknown" else "temporal_delay"
                elif dtw_temporal_type == "shape_change":
                    if "kinematic" not in top_type:
                        top_type = top_type + "+kinematic_profile" if top_type != "unknown" else "kinematic_profile"
            results["anomaly_type"].append(top_type)
            results["contributing_features"].append(contributing[:15])

            vote_model = model_s > 0.50 if use_ml else False
            vote_mahal = mahal_s > 0.50
            vote_centroid = centroid_s > 0.50
            vote_within = within_s > 0.50
            votes_weighted = (
                (1.0 if vote_model else 0.0)
                + (1.0 if vote_mahal else 0.0)
                + (1.0 if vote_centroid else 0.0)
                + (1.0 if vote_within else 0.0)
            )

            method_votes_list = [
                vote_model,
                vote_mahal,
                vote_centroid,
                vote_within,
            ]
            results["method_votes"].append(method_votes_list)
            results["weighted_votes"].append(float(votes_weighted))
            results["method_sigmoid_scores"].append([
                float(model_s), float(mahal_s),
                float(centroid_s), float(within_s),
            ])
            results["method_weighted_components"].append({
                "ML Model":        round(w_model * model_s / total_w, 4),
                "Mahalanobis":     round(w_mahal * mahal_s / total_w, 4),
                "Nearest Centroid": round(w_centroid * centroid_s / total_w, 4),
                "Feature Dev":     round(w_feature * feature_dev_score / total_w, 4),
                "Within-Session":  round(w_within * within_s / total_w, 4),
            })

            _deviant_ratio_thresh = max(0.10, 0.20 - 0.02 * max(0, n_ref - 3))
            _single_rep_test = len(metrics_df) == 1
            min_deviant_abs = 1 if _single_rep_test else max(1, int(n_task_feats * 0.10))
            has_sufficient_deviance = (
                deviant_ratio >= _deviant_ratio_thresh
                and n_deviant_task >= min_deviant_abs
            )
            high_directionality = direction_consistency >= 0.85 and n_deviant_task >= 2
            is_anom = (
                (composite > 0.40 and has_sufficient_deviance)
                or (feature_dev_score > 0.55 and n_deviant_task >= min_deviant_abs + 1)
                or (high_directionality and composite > 0.30)
            )
            results["is_anomaly"].append(is_anom)

            component_scores = [
                w_model * model_s / total_w,
                w_mahal * mahal_s / total_w,
                w_centroid * centroid_s / total_w,
                w_feature * feature_dev_score / total_w,
                w_within * within_s / total_w,
            ]
            if self.n_reference >= 5:
                ci_lo, ci_hi = self._bootstrap_ci(
                    component_scores, composite, n_bootstrap=200
                )
            else:
                ci_half = composite * 0.30 + 0.03
                ci_lo = max(0.0, composite - ci_half)
                ci_hi = min(1.0, composite + ci_half)
            results["deviation_ci_lower"].append(ci_lo)
            results["deviation_ci_upper"].append(ci_hi)

            mahal_ci = mahal_scores[i] * (0.25 if self.n_reference < 5 else 0.12)
            results["mahalanobis_ci_lower"].append(max(0.0, mahal_scores[i] - mahal_ci))
            results["mahalanobis_ci_upper"].append(mahal_scores[i] + mahal_ci)

        for feat in available_features:
            feat_devs = [
                d[feat] for d in results["deviations"] if feat in d
            ]
            if feat_devs:
                rng_vals = [d["range_dev"] for d in feat_devs]
                mz_vals = [d["modified_z"] for d in feat_devs]
                t_vals = [d["t_deviation"] for d in feat_devs]
                w = self._feature_weights.get(feat, 1.0)

                dirs = [d.get("direction", "within") for d in feat_devs
                        if d.get("is_deviant", False)]
                if dirs:
                    from collections import Counter
                    dcounts = Counter(dirs)
                    dominant_dir = dcounts.most_common(1)[0][0]
                else:
                    dominant_dir = "within"

                results["feature_deviations"][feat] = {
                    "mean_range_dev":     float(np.mean(rng_vals)),
                    "max_range_dev":      float(np.max(rng_vals)),
                    "mean_t_deviation":   float(np.mean(t_vals)),
                    "max_t_deviation":    float(np.max(t_vals)),
                    "mean_modified_z":    float(np.mean(mz_vals)),
                    "max_abs_modified_z": float(np.max(np.abs(mz_vals))),
                    "n_deviant": sum(1 for d in feat_devs if d["is_deviant"]),
                    "weight": w,
                    "mean_z_score":    float(np.mean(mz_vals)),
                    "max_abs_z_score": float(np.max(np.abs(mz_vals))),
                    "dominant_direction": dominant_dir,
                }

        n_total = len(metrics_df)
        n_anom = sum(results["is_anomaly"])

        _score_arr = np.array(results["anomaly_scores"], dtype=float)
        _composite_threshold = 0.40
        results["snippet_function_preserved"] = bool(
            _snippet_function_preserved(
                1.0 - _score_arr,
                threshold=1.0 - _composite_threshold,
                n_snippets=min(3, max(1, n_total)),
            )
        )

        _type_totals: Dict[str, float] = {}
        for _atype_list in results["anomaly_type"]:
            if isinstance(_atype_list, str):
                _type_totals[_atype_list] = _type_totals.get(_atype_list, 0.0) + 1.0
        _type_feature_weight: Dict[str, float] = {}
        for feat, fd in results["feature_deviations"].items():
            if feat not in score_set:
                continue
            if fd.get("n_deviant", 0) > 0:
                if any(kw in feat.lower() for kw in _SCALE_FEATURE_KEYWORDS):
                    continue
                atype = _classify_anomaly_type(feat)
                _type_feature_weight[atype] = (
                    _type_feature_weight.get(atype, 0.0)
                    + fd["mean_t_deviation"] * fd.get("weight", 1.0)
                )
        dominant_type = (
            max(_type_feature_weight, key=_type_feature_weight.get)
            if _type_feature_weight else "unknown"
        )

        results["summary"] = {
            "n_samples":              n_total,
            "n_anomalies":            n_anom,
            "anomaly_rate":           n_anom / n_total if n_total > 0 else 0.0,
            "mean_deviation_score":   float(np.mean(results["deviation_score"])),
            "mean_score_confidence":  float(np.mean(results["score_confidence"])),
            "mean_anomaly_score":     float(np.mean(results["anomaly_scores"])),
            "model_type":             results["model_type"],
            "n_reference":            n_ref,
            "n_pca_components":       self.n_pca_components,
            "pca_explained_variance": self.pca_explained_variance,
            "n_task_relevant_features": len(self._task_relevant_features),
            "effective_threshold":    self._compute_prediction_interval(n_ref),
            "n_features_with_deviations": sum(
                1 for d in results["feature_deviations"].values()
                if d["n_deviant"] > 0
            ),
            "dominant_anomaly_type":  dominant_type,
            "anomaly_type_breakdown": _type_feature_weight,
        }

        if "repetition" in metrics_df.columns:
            results["repetitions"] = (
                metrics_df["repetition"].astype(int).tolist()
            )
        else:
            results["repetitions"] = list(range(n_total))

        if "task_group" in metrics_df.columns:
            results["task_groups"] = (
                metrics_df["task_group"].fillna("0").astype(str).tolist()
            )
        else:
            results["task_groups"] = ["0"] * n_total

        if "task_id" in metrics_df.columns:
            results["task_ids"] = (
                metrics_df["task_id"].fillna(0).astype(int).tolist()
            )
        else:
            results["task_ids"] = [0] * n_total

        return results

    def compute_deviation_from_baseline(
        self, current_value: float, feature_name: str
    ) -> Dict[str, float]:
        """Compute t-deviation and range deviation for a single value against reference."""
        if feature_name not in self.reference_stats:
            return {
                "modified_z": 0.0,
                "z_score": 0.0,
                "range_dev": 0.0,
                "t_deviation": 0.0,
                "percentile_deviation": 0.0,
                "is_deviant": False,
            }

        ref = self.reference_stats[feature_name]
        median = ref.get("median", ref["mean"])
        mad = ref.get("mad", 0.0)
        mod_z = self._compute_modified_z(current_value, median, mad)
        rng_dev = self._compute_range_deviation(current_value, feature_name)
        t_dev, is_dev, _ = self._compute_feature_deviation(current_value, feature_name)

        return {
            "modified_z": float(mod_z),
            "z_score": float(mod_z),
            "range_dev": float(rng_dev),
            "t_deviation": float(t_dev),
            "percentile_deviation": float(rng_dev),
            "is_deviant": is_dev,
        }

    def save_model(self, path: str) -> None:
        """Persist model state to JSON."""
        model_data = {
            "reference_stats":        self.reference_stats,
            "feature_names":          self.feature_names,
            "selected_features":      self.selected_features,
            "task_relevant_features": self._task_relevant_features,
            "learned_importance":     self.learned_importance,
            "pca_feature_names":      self.pca_feature_names,
            "n_pca_components":       self.n_pca_components,
            "pca_explained_variance": self.pca_explained_variance,
            "pca_mean":   self.pca.mean_.tolist() if self.pca else [],
            "pca_components": (
                self.pca.components_.tolist() if self.pca else []
            ),
            "deviation_threshold":    self.deviation_threshold,
            "scaler_mean":  self.scaler.mean_.tolist() if self.is_fitted else [],
            "scaler_scale": self.scaler.scale_.tolist() if self.is_fitted else [],
            "mahal_centroid": (
                self._mahal_scorer.centroid.tolist()
                if self._mahal_scorer.is_fitted else []
            ),
            "mahal_ref_mean_dist":    self._mahal_scorer.ref_mean_dist,
            "centroid_centroid": (
                self._centroid_scorer.centroid.tolist()
                if self._centroid_scorer.is_fitted else []
            ),
            "centroid_ref_spread":    self._centroid_scorer.ref_spread,
            "is_fitted":              self.is_fitted,
            "model_type": "OC-SVM" if self._use_ocsvm else "IsolationForest",
            "n_reference":            self._n_reference,
            "feature_weights":        self._feature_weights,
            "ref_score_dist":         self._ref_score_dist,
        }
        save_json(model_data, path)

    def load_model(self, path: str) -> None:
        """Restore model state from a previously saved JSON file."""
        model_data = load_json(path)
        self.reference_stats = model_data.get("reference_stats", {})
        self.feature_names = model_data.get("feature_names", [])
        self.selected_features = model_data.get("selected_features", [])
        self._task_relevant_features = model_data.get("task_relevant_features", [])
        self.learned_importance = model_data.get("learned_importance", {})
        self.pca_feature_names = model_data.get("pca_feature_names", [])
        self.n_pca_components = model_data.get("n_pca_components", 0)
        self.pca_explained_variance = model_data.get("pca_explained_variance", [])
        self.deviation_threshold = model_data.get("deviation_threshold", 2.0)
        self.is_fitted = model_data.get("is_fitted", False)
        self._n_reference = model_data.get("n_reference", 0)
        self._feature_weights = model_data.get("feature_weights", {})
        self._ref_score_dist = model_data.get("ref_score_dist", {})

        if self.is_fitted:
            pca_components = model_data.get("pca_components", [])
            pca_mean = model_data.get("pca_mean", [])
            if pca_components and pca_mean:
                self.pca = PCA(n_components=self.n_pca_components)
                self.pca.components_ = np.array(pca_components)
                self.pca.mean_ = np.array(pca_mean)

            if model_data.get("scaler_mean"):
                self.scaler.mean_ = np.array(model_data["scaler_mean"])
                self.scaler.scale_ = np.array(model_data["scaler_scale"])

            mc = model_data.get("mahal_centroid", [])
            if mc:
                self._mahal_scorer.centroid = np.array(mc)
                self._mahal_scorer.ref_mean_dist = model_data.get(
                    "mahal_ref_mean_dist", 1.0
                )
                self._mahal_scorer.is_fitted = True

            cc = model_data.get("centroid_centroid", [])
            if cc:
                self._centroid_scorer.centroid = np.array(cc)
                self._centroid_scorer.ref_spread = model_data.get(
                    "centroid_ref_spread", 1.0
                )
                self._centroid_scorer.is_fitted = True

    def get_reference_stats(self) -> Dict[str, Dict[str, float]]:
        """Return the stored per-feature reference statistics."""
        return self.reference_stats


def create_anomaly_detector(
    decision_rules_config: Dict[str, Any],
    tasks_config: Optional[Dict[str, Any]] = None,
) -> AnomalyDetector:
    """Factory: build an AnomalyDetector from decision-rules configuration."""
    return AnomalyDetector(decision_rules_config, tasks_config)


def create_cusum_monitor(k: float = 0.5, h: float = 5.0) -> CUSUMMonitor:
    """Factory: build a CUSUMMonitor with the given sensitivity parameters."""
    return CUSUMMonitor(k=k, h=h)


class ContinuousBaselineEstimator:
    """
    Establishes a behavioural baseline for a continuous recording session.

    Two reference levels:
      - Session-internal: stats computed from the explicit neutral/baseline
        segment (or the first `baseline_duration_s` seconds if absent).
      - Cross-session normative: loaded from a persisted JSON produced by
        `build_normative_reference()`.

    All computations are feature-wise (per column); no cross-feature covariance
    is assumed here (that belongs to AnomalyDetector).
    """

    def __init__(
        self,
        baseline_duration_s: float = 30.0,
        min_baseline_frames: int = 30,
    ):
        """Initialise estimator; call fit() to compute per-feature statistics from data."""
        self.baseline_duration_s = baseline_duration_s
        self.min_baseline_frames = min_baseline_frames
        self.session_stats: Dict[str, Dict[str, float]] = {}
        self._fitted = False
        self.baseline_quality: str = "ok"
        self.baseline_n_frames: int = 0

    def fit(self, features_df: pd.DataFrame) -> "ContinuousBaselineEstimator":
        """
        Fit baseline from the neutral segment of features_df.

        Priority:
          1. Rows where ``segment == 'neutral'``.
          2. First ``baseline_duration_s`` seconds if (1) is insufficient.
          3. All rows where ``detection_success == True`` (or all rows) as a
             last resort — sets ``baseline_quality = 'contaminated_full_session'``
             and emits a warning so callers can react appropriately.

        Sets ``self.session_stats``, ``self.baseline_quality``, and
        ``self._fitted = True``.  Returns self for chaining.
        """
        neutral = features_df[features_df.get("segment", pd.Series()) == "neutral"] \
            if "segment" in features_df.columns else pd.DataFrame()

        if len(neutral) >= self.min_baseline_frames:
            self.baseline_quality = "ok"
        else:
            t0 = features_df["timestamp_abs"].min()
            neutral = features_df[
                features_df["timestamp_abs"] <= t0 + self.baseline_duration_s
            ]
            if len(neutral) >= self.min_baseline_frames:
                self.baseline_quality = "first_N_seconds"
            else:
                if "detection_success" in features_df.columns:
                    neutral = features_df[features_df["detection_success"] == True]
                else:
                    neutral = features_df
                self.baseline_quality = "contaminated_full_session"
                logger.warning(
                    "ContinuousBaselineEstimator: insufficient neutral frames "
                    "(%d); using full recording as baseline — anomaly scores "
                    "will reflect within-session variation only, not absolute "
                    "deviation from a true neutral state.  Interpret with caution.",
                    len(neutral),
                )

        feat_cols = get_numeric_feature_columns(neutral)
        feat_cols = [c for c in feat_cols if c not in _FRAME_META_COLUMNS and not str(c).startswith("_")]
        for col in feat_cols:
            vals = neutral[col].dropna().to_numpy()
            if len(vals) == 0:
                continue
            q25, q75 = np.percentile(vals, [25, 75])
            self.session_stats[col] = {
                "mean": float(np.mean(vals)),
                "std": max(float(np.std(vals)), 1e-6),
                "median": float(np.median(vals)),
                "q25": float(q25),
                "q75": float(q75),
                "iqr": max(float(q75 - q25), 1e-6),
                "n": int(len(vals)),
            }
        self.baseline_n_frames = len(neutral)
        self._fitted = True
        return self

    def z_score_frame(self, row_values: Dict[str, float]) -> Dict[str, float]:
        """Return per-feature z-scores relative to session baseline."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        return {
            col: (val - self.session_stats[col]["mean"]) / self.session_stats[col]["std"]
            for col, val in row_values.items()
            if col in self.session_stats
        }

    def z_score_window(self, window_df: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame of z-scores for every row in window_df."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        feat_cols = [c for c in get_numeric_feature_columns(window_df) if c in self.session_stats]
        result = window_df[feat_cols].copy()
        for col in feat_cols:
            s = self.session_stats[col]
            result[col] = (window_df[col] - s["mean"]) / s["std"]
        return result


class ContinuousAnomalyDetector:
    """
    Detects anomalous periods in a continuous facial motor recording.

    Detection layers:
      1. Rolling z-score: |z| > z_threshold sustained for > min_duration_s
      2. IQR fence:       value outside [Q1 - k*IQR, Q3 + k*IQR] baseline fences
      3. CUSUM:           cumulative sum over z-scores for each feature; flags
                          when CUSUM exceeds `cusum_threshold` (already exists
                          in CUSUMMonitor — this class wraps it per feature).
      4. Sustained multi-feature elevation: rolling window mean |z| > composite_threshold.
      5. Change-point:    simple PELT-lite (cumulative-sum breakpoint scan)
                          to find sudden shifts in feature distributions.

    Each detected period is assigned a composite anomaly score (0–1) and a
    label: 'transient_spike', 'sustained_elevation', 'sustained_depression',
    'change_point', or 'multi_feature'.

    Cross-session normative comparison is optional: if `normative_stats` is
    provided, each window's feature means are also compared against the normative
    distribution, giving a second anomaly signal independent of the within-session
    baseline.
    """

    def __init__(
        self,
        baseline_estimator: ContinuousBaselineEstimator,
        normative_stats: Optional[Dict[str, Dict[str, float]]] = None,
        window_size_s: float = 2.0,
        step_size_s: float = 0.5,
        z_threshold: float = 2.0,
        iqr_k: float = 1.5,
        composite_threshold: float = 2.0,
        min_duration_s: float = 1.5,
        cusum_k: float = 0.5,
        cusum_h: float = 5.0,
        feature_cols: Optional[List[str]] = None,
    ):
        """Initialise monitor from a fitted baseline estimator and optional normative stats."""
        self.baseline = baseline_estimator
        self.normative = normative_stats
        self.window_size_s = window_size_s
        self.step_size_s = step_size_s
        self.z_threshold = z_threshold
        self.iqr_k = iqr_k
        self.composite_threshold = composite_threshold
        self.min_duration_s = min_duration_s
        self.cusum_k = cusum_k
        self.cusum_h = cusum_h
        self._feature_cols = feature_cols

    def _get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        """Return the list of feature columns to monitor, intersected with df columns."""
        if self._feature_cols:
            return [c for c in self._feature_cols if c in df.columns]
        numeric_cols = get_numeric_feature_columns(df)
        numeric_cols = [c for c in numeric_cols if c not in _FRAME_META_COLUMNS and not str(c).startswith("_")]
        return [c for c in numeric_cols if c in self.baseline.session_stats]

    def detect(
        self,
        features_df: pd.DataFrame,
        kin_df: Optional[pd.DataFrame] = None,
        kinematic_reference_profiles: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Run all anomaly detection layers on a continuous features DataFrame.

        features_df must have: 'timestamp_abs', numeric feature columns.
        Optional kin_df for kinematic profile deviation scoring.
        Optional kinematic_reference_profiles for normative kinematic comparison.
        Returns a report dict with anomalous periods, change points, etc.
        """
        feat_cols = self._get_feature_cols(features_df)
        if not feat_cols:
            return {"error": "No feature columns available after filtering."}

        ts = features_df["timestamp_abs"].to_numpy()
        t_min, t_max = ts.min(), ts.max()

        windows = []
        t = t_min
        while t + self.window_size_s <= t_max + self.step_size_s:
            mask = (ts >= t) & (ts < t + self.window_size_s)
            if mask.sum() < 3:
                t += self.step_size_s
                continue
            win_df = features_df.loc[mask, feat_cols]
            win_means = win_df.mean()

            z_scores = {
                col: (win_means[col] - self.baseline.session_stats[col]["median"])
                      / (self.baseline.session_stats[col]["iqr"] / 1.35)
                for col in feat_cols
                if col in self.baseline.session_stats
            }

            iqr_flags = {
                col: (
                    win_means[col] < self.baseline.session_stats[col]["q25"]
                                    - self.iqr_k * self.baseline.session_stats[col]["iqr"]
                    or
                    win_means[col] > self.baseline.session_stats[col]["q75"]
                                    + self.iqr_k * self.baseline.session_stats[col]["iqr"]
                )
                for col in z_scores
            }

            z_arr = np.array(list(z_scores.values()))
            frac_deviant = float(np.mean(np.abs(z_arr) > self.z_threshold)) if len(z_arr) else 0.0
            mean_abs_z = float(np.mean(np.abs(z_arr))) if len(z_arr) else 0.0
            iqr_frac = float(sum(iqr_flags.values()) / max(len(iqr_flags), 1))

            kin_dev_score = 0.0
            if (
                kin_df is not None
                and not kin_df.empty
                and kinematic_reference_profiles is not None
            ):
                kin_mask = (kin_df["timestamp_abs"] >= t) & (kin_df["timestamp_abs"] < t + self.window_size_s)
                if kin_mask.sum() > 0:
                    kin_scores = []
                    for col in kin_df.columns:
                        if col.startswith("kin_") and col in kinematic_reference_profiles:
                            kin_vals = kin_df.loc[kin_mask, col].dropna()
                            if len(kin_vals) > 0:
                                kin_mean = kin_vals.mean()
                                ref = kinematic_reference_profiles[col]
                                if isinstance(ref, dict) and "mean" in ref and "std" in ref:
                                    ref_mean = ref["mean"]
                                    ref_std = ref["std"]
                                    if ref_std > 0:
                                        kin_z = (kin_mean - ref_mean) / ref_std
                                        kin_scores.append(min(abs(kin_z) / 3.0, 1.0))
                    if kin_scores:
                        kin_dev_score = float(np.mean(kin_scores))

            composite = float(np.clip(
                0.20 * min(mean_abs_z / max(self.z_threshold, 1), 1.0)
                + 0.15 * frac_deviant
                + 0.40 * iqr_frac
                + 0.25 * kin_dev_score,
                0.0, 1.0
            ))

            norm_z = None
            if self.normative:
                norm_z = {}
                for col in z_scores:
                    if col in self.normative:
                        ns = self.normative[col]
                        norm_z[col] = (
                            (win_means[col] - ns["mean"]) / max(ns["std"], 1e-6)
                        )

            n_above_threshold = int(np.sum(np.abs(z_arr) > self.z_threshold)) if len(z_arr) else 0
            is_asymmetry_flag = any(
                col in _ASYMMETRY_PAIRS
                and abs(z_scores.get(col, 0)) > self.z_threshold
                and abs(z_scores.get(_ASYMMETRY_PAIRS[col], 0)) <= self.z_threshold
                for col in z_scores
            )
            is_multi_feature = n_above_threshold >= 2 or is_asymmetry_flag

            task_context: Dict[str, Any] = {}
            for ctx_col in ("task_group", "task_id", "segment", "task_name"):
                if ctx_col in features_df.columns:
                    ctx_vals = features_df.loc[mask, ctx_col].dropna()
                    if len(ctx_vals) > 0:
                        mode_val = ctx_vals.mode()
                        task_context[ctx_col] = mode_val.iloc[0] if len(mode_val) > 0 else None

            windows.append({
                "start_s": float(t),
                "end_s": float(t + self.window_size_s),
                "composite": composite,
                "z_scores": z_scores,
                "norm_z": norm_z,
                "top_features": sorted(z_scores, key=lambda k: abs(z_scores[k]), reverse=True)[:5],
                "frac_deviant": frac_deviant,
                "mean_abs_z": mean_abs_z,
                "n_above_threshold": n_above_threshold,
                "is_multi_feature": is_multi_feature,
                "is_asymmetry_flag": is_asymmetry_flag,
                "task_context": task_context,
            })
            t += self.step_size_s

        anomalous_periods = self._merge_anomalous_windows(windows)

        change_points, per_feature_cusum_flags = self._detect_change_points(
            features_df, feat_cols, ts
        )

        n_anom = len(anomalous_periods)
        total_anom_s = sum(p["duration_s"] for p in anomalous_periods)
        session_dur = float(t_max - t_min)

        return {
            "n_windows": len(windows),
            "n_anomalous_windows": sum(1 for w in windows if w["composite"] > 0.5),
            "anomalous_periods": anomalous_periods,
            "change_points": change_points,
            "per_feature_cusum_flags": per_feature_cusum_flags,
            "baseline_quality": getattr(self.baseline, "baseline_quality", "ok"),
            "baseline_n_frames": getattr(self.baseline, "baseline_n_frames", 0),
            "summary": {
                "n_anomalous_periods": n_anom,
                "total_anomalous_duration_s": round(total_anom_s, 2),
                "session_duration_s": round(session_dur, 2),
                "anomaly_fraction": round(total_anom_s / max(session_dur, 1), 4),
                "n_change_points": len(change_points),
            },
        }

    def _merge_anomalous_windows(
        self,
        windows: List[Dict],
        bridge_gap_s: float = 1.0,
    ) -> List[Dict]:
        """Merge anomalous windows into contiguous periods.

        A window is anomalous when composite > 0.5 AND at least 2 features
        simultaneously exceed z_threshold (multi-feature gate).  Short normal
        gaps <= bridge_gap_s between two anomalous stretches are bridged so
        that near-contiguous disruptions appear as one period.

        Bridging is measured by counting consecutive non-anomalous windows
        (each representing step_size_s of time).  This avoids the pitfall
        where consecutive overlapping windows always have end_s differences
        equal to step_size_s, which would make every gap appear bridgeable.
        """
        periods = []
        current: Optional[Dict] = None
        pending: List[Dict] = []
        max_bridge_windows = max(1, round(bridge_gap_s / self.step_size_s))

        for w in windows:
            is_anom = w["composite"] > 0.50 and w.get("is_multi_feature", True)

            if is_anom:
                if current is None:
                    pending.clear()
                    current = {
                        "start_s": w["start_s"],
                        "end_s": w["end_s"],
                        "_scores": [w["composite"]],
                        "_z_scores": [w["z_scores"]],
                        "_norm_z": [w["norm_z"]] if w["norm_z"] else [],
                        "_top_features": w["top_features"],
                        "_frac_deviant": [w.get("frac_deviant", 0.0)],
                        "_mean_abs_z": [w.get("mean_abs_z", 0.0)],
                        "_task_contexts": [w.get("task_context", {})],
                    }
                else:
                    for pw in pending:
                        current["end_s"] = pw["end_s"]
                        current["_scores"].append(pw["composite"])
                        current["_z_scores"].append(pw["z_scores"])
                        current["_frac_deviant"].append(pw.get("frac_deviant", 0.0))
                        current["_mean_abs_z"].append(pw.get("mean_abs_z", 0.0))
                        if pw.get("norm_z"):
                            current["_norm_z"].append(pw["norm_z"])
                        current["_task_contexts"].append(pw.get("task_context", {}))
                    pending.clear()
                    current["end_s"] = w["end_s"]
                    current["_scores"].append(w["composite"])
                    current["_z_scores"].append(w["z_scores"])
                    current["_frac_deviant"].append(w.get("frac_deviant", 0.0))
                    current["_mean_abs_z"].append(w.get("mean_abs_z", 0.0))
                    if w["norm_z"]:
                        current["_norm_z"].append(w["norm_z"])
                    current["_task_contexts"].append(w.get("task_context", {}))
            else:
                if current is not None:
                    pending.append(w)
                    if len(pending) > max_bridge_windows:
                        period = self._finalise_period(current)
                        if period["duration_s"] >= self.min_duration_s:
                            periods.append(period)
                        current = None
                        pending.clear()

        if current is not None:
            period = self._finalise_period(current)
            if period["duration_s"] >= self.min_duration_s:
                periods.append(period)

        return periods

    def _finalise_period(self, current: Dict) -> Dict:
        """Convert the in-progress period dict to a completed period record with duration and top features."""
        dur = current["end_s"] - current["start_s"]
        scores = current["_scores"]
        all_z = current["_z_scores"]
        all_feats = set().union(*[set(d.keys()) for d in all_z])
        mean_z = {
            f: float(np.mean([d[f] for d in all_z if f in d]))
            for f in all_feats
        }
        top5 = sorted(mean_z, key=lambda k: abs(mean_z[k]), reverse=True)[:5]
        pos_count = sum(1 for k in top5 if mean_z.get(k, 0) > 0)
        neg_count = len(top5) - pos_count

        mean_abs_z_list = current.get("_mean_abs_z", [])
        frac_dev_list = current.get("_frac_deviant", [])

        if len(mean_abs_z_list) >= 4:
            z_arr = np.array(mean_abs_z_list)
            slope = np.polyfit(np.arange(len(z_arr)), z_arr, 1)[0]
            is_drift = slope > 0.02 and z_arr[-1] > z_arr[0]
        else:
            is_drift = False

        kin_top = sum(1 for f in top5 if "kin_" in f)
        is_kinematic = kin_top >= 2

        if len(scores) == 1:
            atype = "transient_spike"
        elif is_drift:
            atype = "drift"
        elif is_kinematic:
            atype = "kinematic_deviation"
        elif pos_count > neg_count:
            atype = "sustained_elevation"
        elif neg_count > pos_count:
            atype = "sustained_depression"
        else:
            atype = "pattern_shift"

        norm_z = None
        if current["_norm_z"]:
            all_nf = set().union(*[set(d.keys()) for d in current["_norm_z"]])
            norm_z = {
                f: float(np.mean([d[f] for d in current["_norm_z"] if f in d]))
                for f in all_nf
            }

        task_contexts = current.get("_task_contexts", [])
        task_context_out: Dict[str, Any] = {}
        for ctx_col in ("task_group", "task_id", "segment", "task_name"):
            vals = [tc[ctx_col] for tc in task_contexts if tc.get(ctx_col) is not None]
            if vals:
                try:
                    from collections import Counter
                    task_context_out[ctx_col] = Counter(vals).most_common(1)[0][0]
                except Exception:
                    task_context_out[ctx_col] = vals[0]

        return {
            "start_s": current["start_s"],
            "end_s": current["end_s"],
            "duration_s": round(dur, 3),
            "composite_score": round(float(np.mean(scores)), 4),
            "anomaly_type": atype,
            "task_context": task_context_out,
            "top_features": top5,
            "mean_z_scores": {k: round(v, 4) for k, v in mean_z.items() if k in top5},
            "normative_z": norm_z,
        }

    def _detect_change_points(
        self, df: pd.DataFrame, feat_cols: List[str], ts: np.ndarray
    ) -> Tuple[List[float], Dict[str, List[float]]]:
        """
        CUSUM change-point scan per feature.

        Uses a robust z-score (median + IQR) consistent with the window-level
        detection.  Returns:
          - change_points: list of timestamps where >= max(2, n_features // 8)
            features simultaneously triggered a CUSUM alarm.  The 12.5 %
            threshold (vs the old 33 %) makes distributed changes detectable
            without requiring every feature to fire at once.
          - per_feature_cusum_flags: {feature: [alarm_timestamps]}
        """
        per_feature: Dict[str, List[float]] = {}
        alarm_at: Dict[float, int] = {}

        for col in feat_cols:
            if col not in self.baseline.session_stats:
                continue
            s = self.baseline.session_stats[col]
            series = df[col].fillna(s["median"]).to_numpy()
            cusum_pos = 0.0
            cusum_neg = 0.0
            flags: List[float] = []
            robust_scale = s["iqr"] / 1.35

            for i, val in enumerate(series):
                z = (val - s["median"]) / robust_scale if robust_scale > 0 else 0.0
                cusum_pos = max(0.0, cusum_pos + z - self.cusum_k)
                cusum_neg = max(0.0, cusum_neg - z - self.cusum_k)
                if cusum_pos > self.cusum_h or cusum_neg > self.cusum_h:
                    t = float(ts[i])
                    flags.append(t)
                    alarm_at[t] = alarm_at.get(t, 0) + 1
                    cusum_pos = 0.0
                    cusum_neg = 0.0

            if flags:
                per_feature[col] = flags

        threshold = max(2, len(feat_cols) // 8)
        change_points = sorted(t for t, cnt in alarm_at.items() if cnt >= threshold)
        return change_points, per_feature


class FatigueDriftMonitor:
    """Continuous fatigue and motor drift monitor using long sliding windows.

    Implements the continuous analysis methodology synthesised from three
    empirical studies on facial/ocular fatigue biomarkers:

    Baseline (first ``baseline_duration_s`` seconds, default 120 s / 2 min):
        Mean blendshape velocities, L–R symmetry index, and per-feature
        dynamic range (MAX − MIN) are computed per anatomical region (eye,
        brow, mouth) to establish the participant's individual resting motor
        profile.  The dynamic range is stored per feature and used to
        normalise all subsequent window measurements.

    Continuous analysis (sliding ``window_size_s`` windows, default 60 s,
    ``step_size_s`` step, default 10 s):
        Per-region velocity, L–R asymmetry index, and ROM activation range
        within each window are compared against baseline.

    Flags (any combination triggers a fatigue / drift alert):
        - Velocity decay:    |z| > ``velocity_z_thresh`` (default 2.0)
                             AND percent change < -``velocity_pct_thresh``
                             (default 25 %).  Rationale: velocity is a freely
                             moving, voluntary-control-free correlate of motor
                             effort; its decline under fatigue is well-
                             established (Di Stasi et al. 2014).
        - Asymmetry creep:  absolute change in asymmetry index from baseline
                             > ``asymmetry_pct_thresh`` / 100 (default 0.10).
                             The asymmetry index is |L − R| / (|L| + |R| + ε),
                             averaged over symmetric blendshape pairs within
                             the region.  Using absolute values in the
                             denominator ensures the index is valid for the
                             pose-corrected signed feature values used in this
                             pipeline.  A change > 0.10 on the [0, 1] scale
                             corresponds to the Kong et al. (2021) threshold
                             (r=0.72–0.89 with PVT performance).
        - ROM tightening:   |z| > ``rom_z_thresh`` (default 2.0) on intra-
                             window ROM standard deviation decrease.  Indicates
                             a shrinking range of facial motion, consistent
                             with progressive motor fatigue or inhibition.

    Fatigue risk index — activation range %
    ----------------------------------------
    An eye–brow–jaw composite is reported each window using the blendshapes
    with the highest correlation to psychomotor vigilance task (PVT)
    performance found by Kong et al. (2021): eyeSquintLeft/Right (Lid Tighten
    proxy, r=0.89), browInnerUp (Inner Brow Raise, r=0.89), eyeBlinkLeft/Right
    (Eye Closure, r=0.88), jawOpen (Jaw Drop, r=0.86), browOuterUpLeft/Right
    (Brow Raise, r=0.82).

    For each fatigue-risk blendshape and for each window, the activation range
    is expressed as a percentage of the baseline dynamic range:

        activation_range_pct = (window_MAX − window_MIN) / baseline_range × 100 %

    where ``baseline_range = baseline_MAX − baseline_MIN`` (stored in
    ``fit()``).  100 % means the same ROM as during quiet rest; values above
    100 % indicate task engagement exceeding the resting range; a progressive
    decline toward 100 % (or below) over the session is the fatigue signal.
    The metric is capped at 200 % to suppress outlier windows.

    Design rationale: the pose-corrected blendshape features used in this
    pipeline are not bounded to [0, 1] — they are scaled ~100× with negative
    values (e.g. jawOpen −18 to +742).  The original Brach & VanSwearingen
    (1995) formula (MAX − MIN) / MAX × 100 assumes non-negative signals and
    breaks for negative minima.  The normalised dynamic-range approach above
    is sign-agnostic and produces meaningful values regardless of the feature
    scale.

    Session-level trend significance
    ---------------------------------
    ``_summarize()`` fits a linear regression (scipy.stats.linregress) over
    the per-window ``mean_activation_range_pct`` values and reports:
        slope_pct_per_min, r_squared, p_value, significant_decline (p<0.05
        AND slope < 0).

    References
    ----------
    Kong Y, Posada-Quintero HF, Daley MS, Chon KH, Bolkhovsky J (2021).
      "Facial features and head movements obtained with a webcam correlate
      with performance deterioration during prolonged wakefulness."
      Atten Percept Psychophys 83:525–540.
      https://doi.org/10.3758/s13414-020-02199-5

    Brach JS, VanSwearingen J (1995).
      "Measuring Fatigue Related to Facial Muscle Function."
      Arch Phys Med Rehabil 76:905–908. PMID: 7668964
      (Original percent-fatigue formula; adapted here for signed features
      via dynamic-range normalisation.)

    Di Stasi LL, McCamy MB, Macknik SL, Mankin JA, Hooft N, Catena A,
      Martinez-Conde S (2014).
      "Saccadic Eye Movement Metrics Reflect Surgical Residents' Fatigue."
      Ann Surg 259:824–829.
      https://doi.org/10.1097/SLA.0000000000000260
    """

    _EYE_BLENDSHAPES: List[str] = [
        "eyeBlinkLeft", "eyeBlinkRight",
        "eyeLookDownLeft", "eyeLookDownRight",
        "eyeLookUpLeft", "eyeLookUpRight",
        "eyeSquintLeft", "eyeSquintRight",
        "eyeWideLeft", "eyeWideRight",
    ]
    _BROW_BLENDSHAPES: List[str] = [
        "browDownLeft", "browDownRight",
        "browInnerUp",
        "browOuterUpLeft", "browOuterUpRight",
    ]
    _MOUTH_BLENDSHAPES: List[str] = [
        "jawOpen", "mouthClose", "mouthFunnel", "mouthPucker",
        "mouthSmileLeft", "mouthSmileRight",
        "mouthFrownLeft", "mouthFrownRight",
        "mouthLowerDownLeft", "mouthLowerDownRight",
        "mouthUpperUpLeft", "mouthUpperUpRight",
    ]

    _FATIGUE_RISK_BLENDSHAPES: List[str] = [
        "eyeSquintLeft", "eyeSquintRight",
        "browInnerUp",
        "eyeBlinkLeft", "eyeBlinkRight",
        "jawOpen",
        "browOuterUpLeft", "browOuterUpRight",
    ]

    _REGION_PAIRS: Dict[str, List[Tuple[str, str]]] = {
        "eye": [
            ("eyeBlinkLeft",   "eyeBlinkRight"),
            ("eyeSquintLeft",  "eyeSquintRight"),
            ("eyeWideLeft",    "eyeWideRight"),
        ],
        "brow": [
            ("browDownLeft",     "browDownRight"),
            ("browOuterUpLeft",  "browOuterUpRight"),
        ],
        "mouth": [
            ("mouthSmileLeft",       "mouthSmileRight"),
            ("mouthFrownLeft",       "mouthFrownRight"),
            ("mouthLowerDownLeft",   "mouthLowerDownRight"),
        ],
    }

    def __init__(
        self,
        baseline_duration_s: float = 120.0,
        window_size_s: float = 60.0,
        step_size_s: float = 10.0,
        min_window_frames: int = 10,
        velocity_z_thresh: float = 2.0,
        velocity_pct_thresh: float = 25.0,
        asymmetry_pct_thresh: float = 10.0,
        rom_z_thresh: float = 2.0,
    ):
        """Initialise fatigue drift monitor with window parameters and per-flag thresholds."""
        self.baseline_duration_s = baseline_duration_s
        self.window_size_s = window_size_s
        self.step_size_s = step_size_s
        self.min_window_frames = min_window_frames
        self.velocity_z_thresh = velocity_z_thresh
        self.velocity_pct_thresh = velocity_pct_thresh
        self.asymmetry_pct_thresh = asymmetry_pct_thresh
        self.rom_z_thresh = rom_z_thresh

        self._baseline: Dict[str, float] = {}
        self._fitted: bool = False


    @staticmethod
    def _region_cols(df: pd.DataFrame, candidates: List[str]) -> List[str]:
        """Return the subset of candidate column names that are present in df."""
        return [c for c in candidates if c in df.columns]

    @staticmethod
    def _velocity_array(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
        """Frame-by-frame mean absolute blendshape velocity across *cols*.

        velocity_i = mean_features(|ΔX_i|) / Δt_i  for each frame i.

        Returns a 1-D array of length len(df) (zero-padded at first frame).
        """
        if not cols or "timestamp_abs" not in df.columns or len(df) < 2:
            return np.zeros(len(df))
        ts = df["timestamp_abs"].values
        dt = np.concatenate([[1.0 / 30.0], np.diff(ts)])
        dt[dt == 0.0] = 1.0 / 30.0
        vals = df[cols].fillna(0.0).values
        delta = np.abs(np.diff(vals, axis=0, prepend=vals[:1]))
        return delta.mean(axis=1) / dt

    @staticmethod
    def _region_asymmetry(df: pd.DataFrame, pairs: List[Tuple[str, str]]) -> float:
        """Mean L–R asymmetry index over all valid symmetric pairs.

        asymmetry_index = |L − R| / (|L| + |R| + ε),  averaged over frames
        and then over pairs.

        The denominator uses absolute values to ensure the index is bounded
        [0, 1] for pose-corrected signed features (which can be negative).
        Using (L + R + ε) as denominator is incorrect when L and R are both
        negative: the sum approaches zero, yielding asymmetry ratios >> 1.
        """
        eps = 1e-3
        ratios: List[float] = []
        for left, right in pairs:
            if left not in df.columns or right not in df.columns:
                continue
            l_v = df[left].fillna(0.0).values
            r_v = df[right].fillna(0.0).values
            ratios.append(float(np.mean(np.abs(l_v - r_v) / (np.abs(l_v) + np.abs(r_v) + eps))))
        return float(np.mean(ratios)) if ratios else 0.0

    @staticmethod
    def _region_rom_std(df: pd.DataFrame, cols: List[str]) -> float:
        """Intra-window ROM proxy: mean per-feature standard deviation.

        Larger std → wider range of motion over the window.
        Declining std relative to baseline → ROM tightening.
        """
        if not cols or len(df) < 2:
            return 0.0
        return float(np.mean(df[cols].fillna(0.0).std(axis=0)))


    def fit(self, features_df: pd.DataFrame) -> "FatigueDriftMonitor":
        """Establish per-region baseline from the first ``baseline_duration_s`` s.

        If a ``'neutral'`` segment is present it is preferred over the
        first-N-seconds heuristic, ensuring the baseline reflects the same
        resting face used for blendshape z-scoring throughout the session.

        Methodology note: the 2-minute default baseline window is derived from
        the continuous analysis framework proposed from Kong et al. (2021) and
        Brach & VanSwearingen (1995), where a task-free resting period precedes
        sustained contraction testing.
        """
        if "timestamp_abs" not in features_df.columns:
            logger.warning(
                "FatigueDriftMonitor.fit: timestamp_abs missing; cannot fit."
            )
            return self

        if "segment" in features_df.columns:
            neutral = features_df[features_df["segment"] == "neutral"]
        else:
            neutral = pd.DataFrame()

        if len(neutral) < 5:
            t0 = float(features_df["timestamp_abs"].min())
            neutral = features_df[
                features_df["timestamp_abs"] <= t0 + self.baseline_duration_s
            ]

        if len(neutral) == 0:
            neutral = features_df
            logger.warning(
                "FatigueDriftMonitor: baseline window empty; using full recording."
            )

        for region, candidates, pairs in [
            ("eye",   self._EYE_BLENDSHAPES,   self._REGION_PAIRS["eye"]),
            ("brow",  self._BROW_BLENDSHAPES,  self._REGION_PAIRS["brow"]),
            ("mouth", self._MOUTH_BLENDSHAPES, self._REGION_PAIRS["mouth"]),
        ]:
            cols = self._region_cols(neutral, candidates)

            vel_arr  = self._velocity_array(neutral, cols)
            asym     = self._region_asymmetry(neutral, pairs)
            rom_std  = self._region_rom_std(neutral, cols)

            self._baseline[f"{region}_vel_mean"] = float(np.mean(vel_arr))
            self._baseline[f"{region}_vel_std"]  = max(float(np.std(vel_arr)), 1e-6)
            self._baseline[f"{region}_asym_mean"] = asym
            self._baseline[f"{region}_rom_std_mean"] = rom_std
            self._baseline[f"{region}_rom_std_std"] = max(rom_std * 0.15, 1e-6)

        for col in self._FATIGUE_RISK_BLENDSHAPES:
            if col in neutral.columns:
                vals = neutral[col].dropna().values
                if len(vals):
                    self._baseline[f"fr_{col}_mean"] = float(np.mean(vals))
                    drange = float(np.max(vals) - np.min(vals)) if len(vals) > 1 else 1.0
                    self._baseline[f"fr_{col}_range"] = max(drange, 1e-6)
                else:
                    self._baseline[f"fr_{col}_mean"] = 0.0
                    self._baseline[f"fr_{col}_range"] = 1.0

        self._fitted = True
        return self

    @staticmethod
    def percent_fatigue(window_df: pd.DataFrame, feature: str) -> float:
        """Legacy per-feature percent-fatigue (kept for API compatibility).

        .. deprecated::
            Use ``_feature_activation_pct()`` instead.  This method used the
            formula ``(MAX − MIN) / MAX × 100``, which assumes non-negative
            signals and breaks when the pose-corrected features in this
            pipeline have negative values (e.g. jawOpen MIN = −18).  The
            replacement method normalises against the *baseline dynamic range*
            which is sign-agnostic.

        Reference
        ---------
        Brach JS, VanSwearingen J (1995). Measuring Fatigue Related to Facial
        Muscle Function. Arch Phys Med Rehabil 76:905–908. PMID: 7668964
        """
        if feature not in window_df.columns:
            return 0.0
        vals = window_df[feature].dropna().values
        if len(vals) == 0:
            return 0.0
        max_v = float(np.max(vals))
        if max_v < 1e-6:
            return 0.0
        return (max_v - float(np.min(vals))) / max_v * 100.0

    def _feature_activation_pct(self, window_df: pd.DataFrame, feature: str) -> float:
        """Within-window dynamic range as % of baseline dynamic range.

        activation_pct = (window_MAX − window_MIN) / baseline_range × 100 %

        Adapted from Brach & VanSwearingen (1995) for corrected (signed) blendshape
        values.  The original EMG formula (MAX−MIN)/MAX assumes positive-only signals;
        this version normalises by the baseline dynamic range instead, which:
          - Is valid for corrected features that can be negative
          - Preserves the clinical meaning (100% = same ROM as baseline)
          - Values < 100% indicate ROM tightening / fatigue
          - Values > 100% indicate more active than baseline
        Capped at 200 % to prevent outlier windows from dominating the axis.
        """
        if feature not in window_df.columns:
            return 0.0
        vals = window_df[feature].dropna().values
        if len(vals) == 0:
            return 0.0
        window_range = float(np.max(vals) - np.min(vals))
        b_range = self._baseline.get(f"fr_{feature}_range", 1.0)
        return min(200.0, window_range / b_range * 100.0)

    def analyze(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Run fatigue drift analysis over *features_df* and return a report.

        Slides ``window_size_s`` windows (step ``step_size_s``) over the
        full recording.  For each window:
          - Per-region (eye, brow, mouth) velocity, L–R asymmetry, and ROM std
            are computed and compared against baseline.
          - Three flag types are evaluated:
              * velocity_decay     : z < -``velocity_z_thresh`` AND
                                     pct_change < -``velocity_pct_thresh``
              * asymmetry_creep   : absolute index change > ``asymmetry_pct_thresh`` / 100
              * rom_tightening    : z < -``rom_z_thresh`` on ROM std
          - A fatigue risk composite score is computed.
          - Per-feature activation range % vs baseline is reported.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before analyze().")
        if "timestamp_abs" not in features_df.columns or len(features_df) == 0:
            return {"error": "timestamp_abs missing or empty DataFrame"}

        ts = features_df["timestamp_abs"].values
        t_min = float(ts.min())
        t_max = float(ts.max())

        windows: List[Dict[str, Any]] = []
        t = t_min

        while t + self.window_size_s <= t_max + self.step_size_s:
            mask = (ts >= t) & (ts < t + self.window_size_s)
            if int(mask.sum()) < self.min_window_frames:
                t += self.step_size_s
                continue

            win = features_df.loc[mask]
            w: Dict[str, Any] = {
                "start_s":  round(float(t), 2),
                "end_s":    round(float(t + self.window_size_s), 2),
                "n_frames": int(mask.sum()),
                "regions":  {},
                "fatigue_risk": {},
                "flags":    [],
            }

            for region, candidates, pairs in [
                ("eye",   self._EYE_BLENDSHAPES,   self._REGION_PAIRS["eye"]),
                ("brow",  self._BROW_BLENDSHAPES,  self._REGION_PAIRS["brow"]),
                ("mouth", self._MOUTH_BLENDSHAPES, self._REGION_PAIRS["mouth"]),
            ]:
                cols = self._region_cols(win, candidates)
                if not cols:
                    continue

                vel_arr  = self._velocity_array(win, cols)
                vel      = float(np.mean(vel_arr))
                asym     = self._region_asymmetry(win, pairs)
                rom_std  = self._region_rom_std(win, cols)

                b_vel_mean  = self._baseline.get(f"{region}_vel_mean", vel)
                b_vel_std   = self._baseline.get(f"{region}_vel_std", 1e-6)
                b_asym      = self._baseline.get(f"{region}_asym_mean", asym)
                b_rom_mean  = self._baseline.get(f"{region}_rom_std_mean", rom_std)
                b_rom_std_s = self._baseline.get(f"{region}_rom_std_std", 1e-6)

                vel_z        = (vel - b_vel_mean) / b_vel_std
                vel_pct      = (vel - b_vel_mean) / max(b_vel_mean, 1e-8) * 100.0
                asym_change  = asym - b_asym
                rom_z        = (rom_std - b_rom_mean) / max(b_rom_std_s, 1e-6)

                w["regions"][region] = {
                    "velocity_mean":        round(float(vel), 6),
                    "velocity_z":           round(float(vel_z), 3),
                    "velocity_pct_change":  round(float(vel_pct), 2),
                    "asymmetry_index":      round(float(asym), 4),
                    "asymmetry_change":     round(float(asym_change), 4),
                    "rom_std":              round(float(rom_std), 4),
                    "rom_z":                round(float(rom_z), 3),
                }

                if vel_z < -self.velocity_z_thresh and vel_pct < -self.velocity_pct_thresh:
                    w["flags"].append({
                        "type":        "velocity_decay",
                        "region":      region,
                        "z":           round(float(vel_z), 3),
                        "pct_change":  round(float(vel_pct), 2),
                    })

                if asym_change > self.asymmetry_pct_thresh / 100.0:
                    w["flags"].append({
                        "type":         "asymmetry_creep",
                        "region":       region,
                        "asym_change":  round(float(asym_change), 4),
                    })

                if rom_z < -self.rom_z_thresh:
                    w["flags"].append({
                        "type":   "rom_tightening",
                        "region": region,
                        "z":      round(float(rom_z), 3),
                    })

            fr_cols = self._region_cols(win, self._FATIGUE_RISK_BLENDSHAPES)
            if fr_cols:
                pf = {c: round(self._feature_activation_pct(win, c), 2) for c in fr_cols}
                mean_act = float(win[fr_cols].mean().mean())
                w["fatigue_risk"] = {
                    "composite_mean_activation": round(mean_act, 4),
                    "activation_range_pct_by_feature": pf,
                    "mean_activation_range_pct": round(
                        float(np.mean(list(pf.values()))) if pf else 0.0, 2
                    ),
                }

            w["flagged"] = len(w["flags"]) > 0
            windows.append(w)
            t += self.step_size_s

        return {
            "windows": windows,
            "summary": self._summarize(windows),
            "baseline_duration_s": self.baseline_duration_s,
            "window_size_s":        self.window_size_s,
            "step_size_s":          self.step_size_s,
            "methodology": (
                f"Sliding-window fatigue monitor. Baseline: first "
                f"{self.baseline_duration_s:.0f} s (or neutral segment). "
                f"Windows: {self.window_size_s:.0f} s, step {self.step_size_s:.0f} s. "
                "Velocity decay flag: z < -2 AND pct_change < -25 % "
                "(Di Stasi et al. 2014 Ann Surg 259:824, "
                "doi:10.1097/SLA.0000000000000260). "
                "Asymmetry creep flag: abs_change > 0.10 (ratio units on [0,1] scale) "
                "(Kong et al. 2021 Atten Percept Psychophys 83:525, "
                "doi:10.3758/s13414-020-02199-5). "
                "ROM tightening: z < -2 on intra-window ROM std. "
                "Activation range = (window_range/baseline_range) x100 % "
                "(adapted from Brach & VanSwearingen 1995 Arch Phys Med Rehabil 76:905, "
                "PMID:7668964; normalised for signed corrected features)."
            ),
        }

    def _summarize(self, windows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Produce a session-level summary from per-window results."""
        if not windows:
            return {
                "n_windows": 0, "n_flagged": 0,
                "flag_fraction": 0.0, "flag_counts_by_type": {},
                "region_velocity_trends": {}, "mean_activation_range_pct": 0.0,
                "mean_percent_fatigue": 0.0,
            }

        n_flagged = sum(1 for w in windows if w["flagged"])

        flag_counts: Dict[str, int] = {}
        for w in windows:
            for f in w["flags"]:
                flag_counts[f["type"]] = flag_counts.get(f["type"], 0) + 1

        region_trends: Dict[str, Any] = {}
        for region in ("eye", "brow", "mouth"):
            vels = [
                w["regions"][region]["velocity_mean"]
                for w in windows if region in w.get("regions", {})
            ]
            if len(vels) >= 4:
                half = len(vels) // 2
                first_v = float(np.mean(vels[:half]))
                last_v  = float(np.mean(vels[half:]))
                change_pct = (last_v - first_v) / max(first_v, 1e-8) * 100.0
                region_trends[region] = {
                    "first_half_mean_vel": round(first_v, 6),
                    "last_half_mean_vel":  round(last_v, 6),
                    "change_pct":          round(change_pct, 2),
                    "direction": (
                        "increasing" if change_pct > 5.0
                        else "decreasing" if change_pct < -5.0
                        else "stable"
                    ),
                }

        mean_pf_vals = [
            w["fatigue_risk"].get("mean_activation_range_pct",
                w["fatigue_risk"].get("mean_percent_fatigue", 0.0))
            for w in windows
            if w.get("fatigue_risk")
        ]

        rom_trend: Optional[Dict[str, Any]] = None
        if len(mean_pf_vals) >= 6:
            try:
                times_s = np.array([
                    (w["start_s"] + w["end_s"]) / 2.0
                    for w in windows if w.get("fatigue_risk")
                ], dtype=float)
                vals_arr = np.array(mean_pf_vals, dtype=float)
                lr = _linregress(times_s, vals_arr)
                p = float(lr.pvalue)
                rom_trend = {
                    "slope_pct_per_min": round(float(lr.slope) * 60.0, 3),
                    "r_squared":         round(float(lr.rvalue) ** 2, 4),
                    "p_value":           round(p, 4),
                    "significant_decline": bool(p < 0.05 and lr.slope < 0),
                }
            except Exception:
                pass

        return {
            "n_windows":             len(windows),
            "n_flagged":             n_flagged,
            "flag_fraction":         round(n_flagged / max(len(windows), 1), 4),
            "flag_counts_by_type":   flag_counts,
            "region_velocity_trends": region_trends,
            "mean_activation_range_pct": round(
                float(np.mean(mean_pf_vals)) if mean_pf_vals else 0.0, 2
            ),
            "mean_percent_fatigue": round(
                float(np.mean(mean_pf_vals)) if mean_pf_vals else 0.0, 2
            ),
            "rom_trend": rom_trend,
        }
