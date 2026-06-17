"""
Longitudinal trend tracking for facial motor and speech behavior analysis.

Implements the Mann-Kendall trend test, the Theil-Sen slope estimator, and
composite progression scoring across sessions for each feature channel.

The Mann-Kendall test is used rather than OLS regression because it makes no
distributional assumptions about the feature values, is robust to outlier
sessions, and performs well on short time series (3-10 sessions). The Theil-Sen
slope (median of all pairwise slopes) is similarly robust to single outlier
sessions that might otherwise dominate a least-squares fit.

A feature trend is flagged as significant when p < alpha (default 0.05). The
composite progression_score is the fraction of analyzed features showing a
significant trend. The overall_direction is determined by the sign of the
sum of Mann-Kendall tau values across significant features.

References
----------
Kendall M (1975) Rank Correlation Methods, 4th ed. Hodder Arnold, London.
  Original formulation of the non-parametric S-statistic and Kendall's tau
  used in _mann_kendall().

Mann HB (1945) Nonparametric tests against trend. Econometrica 13(3):245-259.
  Early statement of the trend-detection framework.

Sen PK (1968) Estimates of the regression coefficient based on Kendall's tau.
  J Am Stat Assoc 63(324):1379-1389.
  Median-slope estimator implemented in _sens_slope().

Theil H (1950) A rank-invariant method of linear and polynomial regression
  analysis. Proc Koninkl Ned Akad Wetensch A 53:386-392.
  Independent derivation of Sen's estimator. The combined method is commonly
  cited as the Theil-Sen estimator.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats as sp_stats


def _mann_kendall(x: np.ndarray) -> Tuple[float, float]:
    """Non-parametric Mann-Kendall trend test.

    Returns (tau, p_value).  tau > 0 indicates an increasing trend.
    """
    n = len(x)
    if n < 3:
        return 0.0, 1.0

    s = 0
    for k in range(n - 1):
        for j in range(k + 1, n):
            diff = x[j] - x[k]
            if diff > 0:
                s += 1
            elif diff < 0:
                s -= 1

    var_s = n * (n - 1) * (2 * n + 5) / 18.0
    if var_s == 0:
        return 0.0, 1.0

    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0

    p_value = 2.0 * (1.0 - sp_stats.norm.cdf(abs(z)))
    tau = s / (n * (n - 1) / 2.0)
    return float(tau), float(p_value)


def _sens_slope(x: np.ndarray) -> float:
    """Theil-Sen slope estimator: median of all pairwise slopes."""
    n = len(x)
    if n < 2:
        return 0.0

    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            slopes.append((x[j] - x[i]) / (j - i))

    return float(np.median(slopes)) if slopes else 0.0


class TrendAnalyzer:
    """Analyse longitudinal trends across multiple sessions."""

    def __init__(self, significance_level: float = 0.05):
        """Initialise analyser with the alpha threshold for significance testing."""
        self.alpha = significance_level

    def analyze_trends(
        self,
        session_summaries: List[Dict[str, Any]],
        feature_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run trend analysis over a list of session summary dicts.

        Each entry in *session_summaries* should contain at minimum:
            - 'session_id': identifier
            - 'timestamp': ISO string or ordinal
            - feature values as top-level keys (or nested under 'metrics')

        Returns a dict with per-feature trend results and a composite
        progression score.
        """
        if len(session_summaries) < 3:
            return {
                "n_sessions": len(session_summaries),
                "trends": {},
                "progression_score": 0.0,
                "overall_direction": "insufficient_data",
            }

        if feature_keys is None:
            feature_keys = self._discover_features(session_summaries)

        trends: Dict[str, Dict[str, Any]] = {}
        significant_trends = 0
        direction_weights = 0.0

        for feat in feature_keys:
            values = self._extract_feature_series(session_summaries, feat)
            if len(values) < 3:
                continue

            tau, p_val = _mann_kendall(values)
            slope = _sens_slope(values)
            is_significant = p_val < self.alpha

            trend_dir = "stable"
            if is_significant:
                trend_dir = "increasing" if tau > 0 else "decreasing"
                significant_trends += 1
                direction_weights += tau

            trends[feat] = {
                "mann_kendall_tau": round(tau, 4),
                "p_value": round(p_val, 4),
                "sens_slope": round(slope, 6),
                "direction": trend_dir,
                "is_significant": is_significant,
                "n_observations": len(values),
                "first_value": float(values[0]),
                "last_value": float(values[-1]),
                "change_pct": round(
                    (values[-1] - values[0]) / (abs(values[0]) + 1e-8) * 100, 2
                ),
            }

        n_features = max(len(trends), 1)
        progression_score = significant_trends / n_features

        if direction_weights > 0.1:
            overall = "worsening"
        elif direction_weights < -0.1:
            overall = "improving"
        else:
            overall = "stable"

        return {
            "n_sessions": len(session_summaries),
            "n_features_analyzed": len(trends),
            "n_significant_trends": significant_trends,
            "trends": trends,
            "progression_score": round(progression_score, 4),
            "overall_direction": overall,
        }

    def _discover_features(self, summaries: List[Dict[str, Any]]) -> List[str]:
        """Identify numeric keys present in most session summaries."""
        key_counts: Dict[str, int] = {}
        for s in summaries:
            metrics = s.get("metrics", s)
            for key, val in metrics.items():
                if isinstance(val, (int, float)):
                    key_counts[key] = key_counts.get(key, 0) + 1

        threshold = len(summaries) * 0.6
        return [k for k, c in key_counts.items() if c >= threshold]

    def _extract_feature_series(
        self, summaries: List[Dict[str, Any]], feature_key: str
    ) -> np.ndarray:
        """Pull a single feature's values across sessions in chronological order."""
        values = []
        for s in summaries:
            metrics = s.get("metrics", s)
            val = metrics.get(feature_key)
            if val is not None and isinstance(val, (int, float)) and np.isfinite(val):
                values.append(float(val))
        return np.array(values)


def create_trend_analyzer(significance_level: float = 0.05) -> TrendAnalyzer:
    """Factory: build a TrendAnalyzer."""
    return TrendAnalyzer(significance_level)
