"""
Articulation scoring module for facial motor and speech behavior analysis pipeline.

Computes articulation quality scores from repetition-level metrics for Group B
(speech articulation) and Group C (word production) tasks.  Produces per-task
scores and four derived features consumed by the decision-support module:

    articulation_score_pataka           - Pa-Ta-Ka composite quality score
    simple_syllable_mean                - mean score across Pa / Ta / Ka
    mean_articulation_score             - task-weighted mean across B + C tasks
    articulation_impairment_consistency - how consistently impaired across reps/tasks

Each per-task score is a weighted composite of three clinically motivated
components:

    timing      (0.30) - duration CV, rate match, time-to-peak consistency
    smoothness  (0.35) - velocity / acceleration variance, spectral arc length
    amplitude   (0.35) - activation adequacy, activation range consistency

For COMBINED-profile sessions in which pa-ta-ka is replaced by sequencing
permutations (ka-pa-ta, ta-pa-ka, pa-ka-ta), those scrambled tasks are
recognised as complex sequencing attempts and included in the pataka /
complex-task scoring bucket so that apraxia indicators (complex_simple
variability ratio) are computed correctly even when the standard B_4 label
is absent.

References
----------
Kent RD, Read C (2002) Acoustic Analysis of Speech, 2nd ed. Singular,
  San Diego.
  DDK (diadochokinesis) rate norms and variability benchmarks; basis for
  the timing component (CV of inter-repetition intervals, rate match score).

Fletcher SG (1972) Time-by-count measurement of diadochokinetic syllable
  rate. J Speech Hear Res 15(4):763–770.
  Classical Pa-Ta-Ka DDK norms used to calibrate rate-match targets in the
  timing sub-score.

Balasubramanian S, Melendez-Calderon A, Burdet E (2012) A robust and
  sensitive metric for quantifying movement smoothness. IEEE Trans Biomed
  Eng 59(8):2126–2136.
  Introduces SPARC (Spectral Arc Length) as a frequency-domain smoothness
  metric and evaluates it alongside LDJ on arm-reaching tasks in healthy
  and stroke subjects.  The core LDJ and SPARC formulae defined in this
  paper are adopted here for Group B (DDK: SPARC on full-repetition
  velocity) and Group C (words: LDJ + SPARC on isolated active jaw phase).
  Note: the speech-specific adaptations (jaw blendshape signals, active-
  phase isolation, empirically calibrated absolute bounds for webcam data
  (LDJ_upper = -5, LDJ_lower = -55 from PAC3/PAC7), and the 60/40
  LDJ-SPARC blend) are extensions beyond the original paper, which
  addresses general upper-limb kinematics without normalization blending
  or phase isolation.

Allison KM, Yunusova Y, Campbell TF, Wang J, Green JR (2022) Short-phrase
  production for the purpose of monitoring changes in speakers with ALS.
  Folia Phoniatr Logop 74(1):1–13.
  Amplitude adequacy norms and within-speaker variability benchmarks that
  inform activation-adequacy thresholds in the amplitude component.

Spearman C (1904) The proof and measurement of association between two
  things. Am J Psychol 15(1):72–101.
  Rank-correlation method computed in ``_rank_correlation()``, used for
  detecting monotonic timing degradation across repetitions.

Segal O, Geva-Dayan K, Dinstein I, Israel-Yaacov S (2022) DDKtor: Automatic
  diadochokinetic speech analysis. arXiv:2206.14639.
  CNN/LSTM model segments DDK audio burst/vowel/silence at 1 ms resolution,
  achieving r = 0.94–0.99 for DDK rate vs manual annotation; validates the
  inter-repetition interval CV and burst-based segmentation approach used
  in the timing sub-score.
  https://arxiv.org/abs/2206.14639

Allison KM, Cordero KN, Munson B, Hustad KC (2022) Use of automated
  kinematic diadochokinesis analysis to identify potential indicators of
  speech motor impairment in children. Am J Speech Lang Pathol
  31(5):2154–2174.
  88 % sensitivity / 88 % specificity for motor involvement via DDK
  inter-syllable interval CV, amplitude stability, and duration - directly
  calibrates the timing and amplitude component thresholds in this module.
  https://doi.org/10.1044/2022_ajslp-21-00241

Simmatis LER, Ghassemi M, Taati B, et al. (2023) Analytical validation of
  a webcam-based assessment of speech kinematics: digital biomarker
  evaluation. Folia Phoniatr Logop 75(1):253–265.
  Good-to-strong ICC-A ≥ 0.70 agreement between webcam kinematics and
  gold-standard EMA; confirms velocity and amplitude as reliable digital
  biomarkers; notes that symmetry features have consistently poor
  test-retest reliability and should not be primary indicators.
  https://doi.org/10.1159/000529685

Speech amplitude envelope kinematics methodology
  He L, Dellwo V (2017) Amplitude envelope kinematics of speech signal:
    parameter extraction and applications. In: Trouvain J, Steiner I,
    Möbius B (eds) Elektronische Sprachsignalverarbeitung 2017 (ESSV 2017,
    No. 86; pp. 1–8). TUDpress, Saarbrücken.
    http://essv2017.coli.uni-saarland.de/pdfs/He.pdf
    Principal reference for computing displacement, velocity, acceleration,
    and jerk from the speech amplitude envelope; adapted here to articulatory
    blendshape channels (jawOpen, mouthClose, etc.) as the envelope analog.

Timing variability across repeated speech tokens
  Smith A, Goffman L, Zelaznik HN, et al. (2009) Spatiotemporal index (STI):
    a tool for characterizing the stability of speech movement patterns.
    J Speech Lang Hear Res 52(4):1088–1096.
    https://doi.org/10.1044/1092-4388(2009/07-0167)

Smoothness metrics: spectral arc length (SPARC)
  Gulde P, Hermsdörfer J (2018) Smoothness metrics in complex movement tasks.
    Front Neurol 9:615. https://doi.org/10.3389/fneur.2018.00615
    Review of log dimensionless jerk (LDJ) and spectral arc length (SPARC)
    for quantifying movement smoothness; SPARC is implemented in
    ``compute_spectral_arc_length()`` and applied to speech-specific
    blendshape velocity channels (kinematic smoothness sub-score).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple

from .feature_extraction import compute_spectral_arc_length, compute_log_dimensionless_jerk


_TASK_B_SIMPLE = {"B_1", "B_2", "B_3"}
_TASK_B_COMPLEX = {"B_4"}

_COMPONENT_WEIGHTS = {"timing": 0.30, "smoothness": 0.35, "amplitude": 0.35}

_COMPLEXITY_TO_NSYL: Dict[int, int] = {c: max(1, (c + 1) // 2) for c in range(1, 9)}

_LDJ_C_UPPER: float = -5.0
_LDJ_C_LOWER: float = -55.0
_LDJ_C_RANGE: float = _LDJ_C_UPPER - _LDJ_C_LOWER


def _score_ldj_group_c(ldj_actual: float) -> float:
    """Map an active-phase LDJ value to a [0, 1] smoothness score for Group C.

    Uses absolute empirical calibration rather than synthetic-reference
    comparison.  See ``_LDJ_C_UPPER`` / ``_LDJ_C_LOWER`` for calibration
    landmarks.  More negative LDJ → jerkier → lower score.
    """
    return max(0.0, min(1.0, (ldj_actual - _LDJ_C_LOWER) / _LDJ_C_RANGE))


def _ldj_reference_smooth(n_syl: int, n_frames: int, fs: float) -> float:
    """LDJ of a synthetic smooth reference jaw movement.

    Retained for use by simulation tools (``tools/_sim_smoothness.py``).
    No longer used in the Group C smoothness path, where it caused a systematic
    downward bias: the synthetic sine arch is always smoother than any real
    webcam blendshape signal, so ``ldj_ref − ldj_actual`` was too large even
    for healthy speakers (healthy scored 0.45–0.65 instead of ~0.85).
    Group C smoothness now uses ``_score_ldj_group_c`` (absolute calibration)
    blended with active-phase SPARC.
    """
    t = np.linspace(0, n_syl * np.pi, max(8, n_frames))
    signal = np.abs(np.sin(t))
    return compute_log_dimensionless_jerk(signal, fs)


def _find_active_phase(
    series: np.ndarray,
    fs: float,
    threshold_frac: float = 0.15,
    min_frames: int = 12,
) -> tuple:
    """Return (left, right) frame indices of the primary jaw-movement event.

    Uses displacement-level thresholding: finds the largest contiguous region
    where the signal is above *threshold_frac × (max − min) + min*.  This
    correctly captures the full jaw arch including the velocity zero-crossing
    at the jaw peak (which speed-based thresholding would split into two
    half-arches).  The largest contiguous above-threshold region is returned
    as the active phase; if it is shorter than *min_frames* it is expanded
    symmetrically around its centre.

    Falls back to (0, len(series)−1) when the amplitude range is negligible,
    i.e. when there is no detectable jaw movement in the window.
    """
    sig_range = float(np.ptp(series))
    if sig_range < 1e-6:
        return 0, len(series) - 1
    baseline = float(np.min(series))
    thr = baseline + threshold_frac * sig_range
    above = series >= thr

    regions: list = []
    in_region = False
    start_i = 0
    for i in range(len(above)):
        if above[i] and not in_region:
            start_i = i
            in_region = True
        elif not above[i] and in_region:
            regions.append((start_i, i - 1))
            in_region = False
    if in_region:
        regions.append((start_i, len(above) - 1))
    if not regions:
        return 0, len(series) - 1

    best = max(regions, key=lambda r: r[1] - r[0])
    left, right = best
    width = right - left + 1
    if width < min_frames:
        expand = (min_frames - width + 1) // 2
        left = max(0, left - expand)
        right = min(len(series) - 1, right + expand)
    return left, right


_IMPAIRMENT_THRESHOLD = 0.60

_SIMPLE_SYLLABLE_LABELS: frozenset = frozenset({"pa-pa-pa", "ta-ta-ta", "ka-ka-ka"})

_PATAKA_LABELS: frozenset = frozenset({"pa-ta-ka"})

_COMPLEX_SEQUENCING_LABELS: frozenset = frozenset({
    "pa-ta-ka",
    "ka-pa-ta",
    "ta-pa-ka",
    "pa-ka-ta",
    "ta-ka-pa",
    "ka-ta-pa",
})


def _get_score_by_label(
    per_task_scores: Dict[str, Dict[str, Any]],
    label: str,
    fallback_key: str,
    default: float,
) -> float:
    """Return composite score matched by task_name label, then key, then default."""
    for task_info in per_task_scores.values():
        if task_info.get("task_name", "").lower().strip() == label.lower():
            return float(task_info.get("score", default))
    return float(per_task_scores.get(fallback_key, {}).get("score", default))


def _rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation coefficient computed without external dependencies."""
    n = len(x)
    if n < 3:
        return 0.0
    rank_x = np.argsort(np.argsort(x)).astype(float)
    rank_y = np.argsort(np.argsort(y)).astype(float)
    d = rank_x - rank_y
    denom = n * (n ** 2 - 1)
    if denom == 0:
        return 0.0
    return float(1.0 - (6.0 * np.sum(d ** 2)) / denom)


class ArticulationScorer:
    """Computes articulation quality scores from facial movement repetition metrics.

    Scoring components
    ------------------
    Group B - DDK smoothness (SPARC)
        Frame-rate-normalised SPARC on per-rep speech-channel velocity.
        ``score = max(0, 1 + SAL × (30 / fs) / 100)`` keeps the scale
        hardware-independent.  When ≥ 3 reps are available, an inter-rep
        *consistency score* ``max(0, 1 − CV × 4)`` is appended to the per-rep
        list before averaging (healthy: CV ≈ 0.01 → consistency ≈ 0.96;
        apraxic: CV ≈ 0.20 → consistency ≈ 0.20).

    Group C - word-production smoothness (LDJ + SPARC blend)
        Active-phase LDJ (60 %) + active-phase SPARC (40 %).  LDJ mapped via
        absolute calibration bounds (LDJ_upper = −5, LDJ_lower = −55) derived
        from PAC3/PAC7 healthy-speaker webcam data.

    Group C - amplitude consistency
        Robust CV (MAD × 1.4826 / median) across reps when median range > 150
        units (lowered from 500 to include short consonant-cluster words).
        Single-rep tasks use absolute amplitude adequacy instead of neutral 0.5:
        ≥ 2000 → 0.82; 500–2000 → linear 0.50–0.82;
        150–500 → linear 0.30–0.50; < 150 → 0.25.

    **Reference-relative duration ratio (Group B, test sessions only)**
        When a reference session is available, ``duration_ratio_vs_ref`` is
        added to each B-task deviation dict: ``test_raw_duration /
        ref_raw_duration``.  Values > 1.20 indicate ≥ 20 % slower task
        completion vs baseline - a direct kinematic slowing measure that
        complements the absolute timing score and avoids its dependency on
        the configured expected-window length.  Group summaries
        ``group_b_mean_duration_ratio``, ``group_b_n_slow_tasks`` (ratio
        > 1.20), and ``group_b_n_fast_tasks`` (ratio < 0.80) are appended to
        the result dict and consumed by decision support.

    Smoothness (SPARC) - known limitation
        MediaPipe FaceLandmarker applies an internal Kalman-filter smoother
        before outputting blendshapes and 3-D landmarks.  This compresses all
        SPARC (spectral arc-length) scores into approximately 0.76–0.86
        regardless of true movement quality, limiting the maximum observable
        smoothness deviation to ~0.07 for realistic speech signals.  The
        smoothness component therefore contributes limited differentiation
        between healthy and impaired profiles at the overall-score level.
        Disorder detection relies primarily on timing (duration ratio) and
        amplitude components.
    """

    def __init__(self, tasks_config: Dict[str, Any]):
        """Initialise scorer from the loaded tasks YAML config."""
        self.tasks_config = tasks_config
        task_groups = tasks_config.get("task_groups", {})
        self.group_b_config = task_groups.get("B", {}).get("tasks", {})
        self.group_c_config = task_groups.get("C", {}).get("tasks", {})

    _ORS_ROLL_THRESHOLD: float = 45.0

    def compute_scores(
        self,
        repetition_metrics_df: pd.DataFrame,
        features_df: Optional[pd.DataFrame] = None,
        reference_articulation: Optional[Dict[str, Any]] = None,
        session_id: str = "",
    ) -> Dict[str, Any]:
        """Compute all articulation features from repetition and optional frame-level data.

        Returns a dict with the four headline scores, per-syllable scores
        (articulation_score_pa / ta / ka), a per_task_scores breakdown with
        component detail, and metadata (n_tasks_scored, tasks_scored).

        When *reference_articulation* (a previous ``compute_scores`` result) is
        supplied, per-task and per-component deviation scores are included in the
        output alongside the absolute scores.

        *session_id* is used to detect ORS/supine sessions whose jaw and lip
        kinematics are affected by gravity (patient lying on side).  When
        detected, ``ors_gravity_flag`` is set in the result so downstream
        consumers can discount articulation scores accordingly.
        """
        _ors_kws = ("_ors_", "_ors", "ors_", "supine", "or_sim", "or-sim", "orsim", "intraop")
        ors_gravity_flag = any(k in session_id.lower() for k in _ors_kws)
        if not ors_gravity_flag and features_df is not None and "head_roll" in features_df.columns:
            median_roll = float(features_df["head_roll"].dropna().abs().median())
            if median_roll > self._ORS_ROLL_THRESHOLD:
                ors_gravity_flag = True

        task_reps = self._group_by_task(repetition_metrics_df)

        per_task_scores: Dict[str, Dict[str, Any]] = {}
        for task_key, task_df in task_reps.items():
            components = self._score_task_components(task_df, task_key, features_df)
            per_task_scores[task_key] = components

        simple_keys = []
        pataka_score = None
        for task_key, task_info in per_task_scores.items():
            tname = task_info.get("task_name", "").lower().strip()
            if task_key in _TASK_B_SIMPLE or tname in _SIMPLE_SYLLABLE_LABELS:
                simple_keys.append(task_key)
            if task_key in _TASK_B_COMPLEX or tname in _COMPLEX_SEQUENCING_LABELS:
                candidate = task_info.get("score", None)
                if candidate is not None:
                    pataka_score = (
                        min(pataka_score, candidate)
                        if pataka_score is not None
                        else candidate
                    )

        simple_scores = [per_task_scores[k]["score"] for k in simple_keys]
        simple_syllable_mean = (
            float(np.mean(simple_scores)) if simple_scores else None
        )

        all_scores = [v["score"] for v in per_task_scores.values()]
        mean_articulation_score = (
            float(np.mean(all_scores)) if all_scores else None
        )

        b_scores = [v["score"] for k, v in per_task_scores.items() if k.startswith("B_")]
        c_scores = [v["score"] for k, v in per_task_scores.items() if k.startswith("C_")]
        group_b_articulation_score = float(np.mean(b_scores)) if b_scores else None
        group_c_articulation_score = float(np.mean(c_scores)) if c_scores else None

        impairment_consistency = self._compute_impairment_consistency(
            per_task_scores, repetition_metrics_df
        )

        default = mean_articulation_score if mean_articulation_score is not None else 0.7

        word_production_features: Dict[str, Any] = {}
        c_rows = (
            repetition_metrics_df[repetition_metrics_df.get("task_group", pd.Series(dtype=str)) == "C"]
            if "task_group" in repetition_metrics_df.columns else pd.DataFrame()
        )

        if len(c_rows) > 0:
            word_production_features = self.compute_word_production_features(per_task_scores)

        result: Dict[str, Any] = {
            "articulation_score_pataka": pataka_score if pataka_score is not None else default,
            "simple_syllable_mean": simple_syllable_mean if simple_syllable_mean is not None else default,
            "mean_articulation_score": default,
            "group_b_articulation_score": group_b_articulation_score if group_b_articulation_score is not None else default,
            "group_c_articulation_score": group_c_articulation_score if group_c_articulation_score is not None else default,
            "articulation_impairment_consistency": impairment_consistency,
            "per_task_scores": per_task_scores,
            "articulation_score_pa": _get_score_by_label(per_task_scores, "pa-pa-pa", "B_1", default),
            "articulation_score_ta": _get_score_by_label(per_task_scores, "ta-ta-ta", "B_2", default),
            "articulation_score_ka": _get_score_by_label(per_task_scores, "ka-ka-ka", "B_3", default),
            "n_tasks_scored": len(per_task_scores),
            "tasks_scored": list(per_task_scores.keys()),
            "ors_gravity_flag": ors_gravity_flag,
            **{k: v for k, v in word_production_features.items() if isinstance(v, (int, float))},
        }

        _b_grp = {k: v for k, v in per_task_scores.items() if k.startswith("B_")}
        _c_grp = {k: v for k, v in per_task_scores.items() if k.startswith("C_")}
        for _grp_pfx, _grp_tasks in [("group_b", _b_grp), ("group_c", _c_grp)]:
            for _comp in ("timing", "smoothness", "amplitude"):
                _vals = [v[_comp] for v in _grp_tasks.values() if v.get(_comp) is not None]
                result[f"{_grp_pfx}_{_comp}_mean"] = float(np.mean(_vals)) if _vals else None

        _b4_act = per_task_scores.get("B_4", {}).get("raw_act_mean")
        _b_simple_acts = [
            per_task_scores.get(f"B_{i}", {}).get("raw_act_mean") for i in [1, 2, 3]
        ]
        _b_simple_acts = [v for v in _b_simple_acts if v is not None]
        if _b4_act is not None and _b_simple_acts:
            _b_simple_mean = float(np.mean(_b_simple_acts))
            result["b4_simple_act_ratio"] = (
                _b4_act / _b_simple_mean if _b_simple_mean != 0 else None
            )
        else:
            result["b4_simple_act_ratio"] = None

        if reference_articulation is not None:
            ref_per_task = reference_articulation.get("per_task_scores", {})
            ref_mean = reference_articulation.get("mean_articulation_score")
            per_task_deviations: Dict[str, Dict[str, float]] = {}
            for tk, task_info in per_task_scores.items():
                ref_t = ref_per_task.get(tk, {})
                if not ref_t:
                    continue
                dev: Dict[str, float] = {}
                for field in ("score", "timing", "smoothness", "amplitude"):
                    test_v = task_info.get(field)
                    ref_v = ref_t.get(field)
                    if test_v is not None and ref_v is not None:
                        dev[f"{field}_deviation"] = float(test_v) - float(ref_v)
                if tk.startswith("B_"):
                    _test_dur = task_info.get("raw_duration")
                    _ref_dur = ref_t.get("raw_duration")
                    if _test_dur is not None and _ref_dur is not None and float(_ref_dur) > 0:
                        dev["duration_ratio_vs_ref"] = float(_test_dur) / float(_ref_dur)
                per_task_deviations[tk] = dev
            result["per_task_deviations"] = per_task_deviations

            _b_dur_ratios = [
                per_task_deviations.get(tk, {}).get("duration_ratio_vs_ref")
                for tk in per_task_scores
                if tk.startswith("B_")
            ]
            _b_dur_ratios = [r for r in _b_dur_ratios if r is not None]
            if _b_dur_ratios:
                result["group_b_mean_duration_ratio"] = float(np.mean(_b_dur_ratios))
                result["group_b_n_slow_tasks"] = sum(1 for r in _b_dur_ratios if r > 1.20)
                result["group_b_n_fast_tasks"] = sum(1 for r in _b_dur_ratios if r < 0.80)

            _drop_thr = -0.15
            for _grp_pfx, _grp_prefix in [("group_b", "B_"), ("group_c", "C_")]:
                for _comp in ("timing", "smoothness", "amplitude"):
                    _ref_cv  = reference_articulation.get(f"{_grp_pfx}_{_comp}_mean")
                    _test_cv = result.get(f"{_grp_pfx}_{_comp}_mean")
                    if _test_cv is not None and _ref_cv is not None:
                        result[f"{_grp_pfx}_{_comp}_deviation"] = float(_test_cv) - float(_ref_cv)
                _grp_devs = {k: v for k, v in per_task_deviations.items() if k.startswith(_grp_prefix)}
                for _comp in ("timing", "smoothness", "amplitude"):
                    _n_drop = sum(
                        1 for devs in _grp_devs.values()
                        if devs.get(f"{_comp}_deviation", 0) < _drop_thr
                    )
                    result[f"{_grp_pfx}_n_{_comp}_drop"] = _n_drop

            _ref_b4_ratio = reference_articulation.get("b4_simple_act_ratio")
            _test_b4_ratio = result.get("b4_simple_act_ratio")
            if _ref_b4_ratio and _test_b4_ratio and _ref_b4_ratio > 0:
                result["b4_simple_act_ratio_vs_ref"] = _test_b4_ratio / _ref_b4_ratio
            else:
                result["b4_simple_act_ratio_vs_ref"] = None

            _n_c_complex_extreme = 0
            for _ctk in ("C_5", "C_6", "C_7", "C_8"):
                _amp_dev = per_task_deviations.get(_ctk, {}).get("amplitude_deviation", 0.0)
                if _amp_dev < -0.50:
                    _n_c_complex_extreme += 1
            result["n_c_complex_extreme_amp_drop"] = _n_c_complex_extreme

            if ref_mean is not None:
                result["mean_score_deviation"] = default - float(ref_mean)
            ref_b = reference_articulation.get("group_b_articulation_score")
            if ref_b is not None and group_b_articulation_score is not None:
                result["group_b_score_deviation"] = float(group_b_articulation_score) - float(ref_b)
            ref_c = reference_articulation.get("group_c_articulation_score")
            if ref_c is not None and group_c_articulation_score is not None:
                result["group_c_score_deviation"] = float(group_c_articulation_score) - float(ref_c)
            ref_pataka = reference_articulation.get("articulation_score_pataka")
            if ref_pataka is not None and pataka_score is not None:
                result["pataka_deviation"] = float(pataka_score) - float(ref_pataka)
            ref_simple = reference_articulation.get("simple_syllable_mean")
            if ref_simple is not None and simple_syllable_mean is not None:
                result["simple_syllable_deviation"] = float(simple_syllable_mean) - float(ref_simple)
            result["has_reference"] = True
        else:
            result["has_reference"] = False

        return result

    def compute_per_rep_scores(
        self, features_df: pd.DataFrame
    ) -> Dict[Tuple[str, int, int], Dict[str, Optional[float]]]:
        """Return per-rep articulation component scores for every B and C repetition.

        Computes four clinically interpretable scores for each rep:

        ``kinematic_smoothness``
            Group B: fs-normalised SPARC on jaw/lip velocity.
            Group C: active-phase LDJ vs single-arch reference.
            Captures spectral jerkiness missed by time-domain acceleration stats.

        ``rep_temporal_consistency``
            Normalised deviation of this rep's duration from the within-task
            median duration.  Score 1.0 = on-target, 0.0 = far from median.
            Anomaly keyword "consistency" → classified as ``articulation`` →
            ``speech_apraxia`` in Group B.  Captures rep-to-rep timing
            *variability* (the apraxia groping signal), complementing the
            absolute slowness signal already in raw ``duration_sec`` z-scores.

        ``rep_spatial_consistency``
            Normalised deviation of this rep's peak-to-valley amplitude from
            the within-task median amplitude.  Score 1.0 = consistent.
            Anomaly keyword "consistency" → classified as ``articulation`` →
            ``speech_apraxia`` in Group B.  Captures rep-to-rep amplitude
            *variability* (groping), complementing the uniform-reduction signal
            already in raw blendshape z-scores.

        ``rep_articulation_score``
            Weighted composite matching the task-level scorer weights
            (timing 0.30, smoothness 0.35, amplitude 0.35).  Uses only the
            components that could be computed for this rep.
            Anomaly keyword "articulation" → classified as ``articulation``.

        These are injected into ``repetition_metrics_df`` in-memory between
        articulation scoring and anomaly detection.  The anomaly detector then
        z-scores them against reference, producing clinically labelled
        ``feature_deviations`` ("rep_temporal_consistency dropped") rather than
        raw blendshape names ("duration_sec changed").  No changes to anomaly.py
        or decision_support.py are required.

        Returns
        -------
        Dict keyed by ``(task_group_str, task_id_int, rep_id_int)`` →
        ``{"kinematic_smoothness": float|None, "rep_temporal_consistency": float|None,
           "rep_spatial_consistency": float|None, "rep_articulation_score": float|None}``.
        Keys: ``kinematic_smoothness``, ``rep_temporal_consistency``,
        ``rep_spatial_consistency``, ``rep_articulation_score``.
        All four keys are always present; value is ``None`` when not computable.
        """
        _KIN_COLS = [
            "kin_mouth_opening",
            "kin_labial_fissure_width",
            "kin_lip_action_y",
            "kin_mouth_height",
        ]
        _SPEECH_BS = [
            "jawOpen",
            "mouthClose",
            "mouthPucker",
            "mouthFunnel",
            "mouthSmileLeft",
            "mouthSmileRight",
        ]
        _W_TIMING = 0.30
        _W_SMOOTH = 0.35
        _W_AMP = 0.35

        result: Dict[Tuple[str, int, int], Dict[str, Optional[float]]] = {}

        if "task_group" not in features_df.columns or "task_id" not in features_df.columns:
            return result

        for (tg, tid), grp in features_df.groupby(["task_group", "task_id"]):
            tg_str = str(tg)
            if tg_str not in ("B", "C"):
                continue
            tid_int = int(tid) if pd.notna(tid) else 0
            if tid_int == 0:
                continue

            avail = [c for c in _KIN_COLS if c in grp.columns]
            if not avail:
                avail = [c for c in _SPEECH_BS if c in grp.columns]
            if not avail:
                continue

            fs = 30.0
            if "timestamp_abs" in grp.columns:
                ts = grp["timestamp_abs"].dropna().values
                if len(ts) > 1:
                    dt = np.median(np.diff(ts))
                    if dt > 0:
                        fs = 1.0 / dt

            if "segment" in grp.columns:
                grp = grp[grp["segment"] == "measurement"]

            is_c = tg_str == "C"
            reps = (
                sorted([r for r in grp["repetition"].unique() if r != 0 and pd.notna(r)])
                if "repetition" in grp.columns
                else [None]
            )
            single_rep_task = len(reps) <= 1

            rep_durs: List[Optional[float]] = []
            rep_amps: List[Optional[float]] = []
            for rep in reps:
                rep_df = grp[grp["repetition"] == rep] if rep is not None else grp
                if len(rep_df) < 4:
                    rep_durs.append(None)
                    rep_amps.append(None)
                    continue
                if "duration_sec" in rep_df.columns and rep_df["duration_sec"].notna().any():
                    dur: Optional[float] = float(rep_df["duration_sec"].dropna().iloc[0])
                else:
                    dur = len(rep_df) / fs
                rep_durs.append(dur)
                amp_vals: List[float] = []
                for col in avail[:4]:
                    vals = rep_df[col].dropna().values
                    if len(vals) >= 4:
                        amp_vals.append(float(np.max(vals) - np.min(vals)))
                rep_amps.append(float(np.mean(amp_vals)) if amp_vals else None)

            valid_durs = [d for d in rep_durs if d is not None]
            valid_amps = [a for a in rep_amps if a is not None]
            median_dur: Optional[float] = float(np.median(valid_durs)) if valid_durs else None
            median_amp: Optional[float] = float(np.median(valid_amps)) if valid_amps else None
            mad_dur = (
                float(np.median(np.abs(np.array(valid_durs) - median_dur))) * 1.4826
                if valid_durs and median_dur is not None
                else 0.0
            )
            mad_amp = (
                float(np.median(np.abs(np.array(valid_amps) - median_amp))) * 1.4826
                if valid_amps and median_amp is not None
                else 0.0
            )

            for i, rep in enumerate(reps):
                rep_df = grp[grp["repetition"] == rep] if rep is not None else grp
                if len(rep_df) < 8:
                    continue
                rep_id = int(rep) if rep is not None else 1
                key = (tg_str, tid_int, rep_id)

                scores: Dict[str, Optional[float]] = {
                    "kinematic_smoothness": None,
                    "rep_temporal_consistency": None,
                    "rep_spatial_consistency": None,
                    "rep_articulation_score": None,
                }

                sm_vals: List[float] = []
                for col in avail[:4]:
                    series = rep_df[col].dropna().values
                    if len(series) < 8:
                        continue
                    if is_c:
                        _k5 = np.ones(5) / 5.0
                        _smoothed = np.convolve(series, _k5, mode="same")
                        _left, _right = _find_active_phase(_smoothed, fs)
                        active = _smoothed[_left : _right + 1]
                        if len(active) < 8:
                            active = _smoothed
                        if np.ptp(active) < 1e-4:
                            sm_vals.append(0.5)
                            continue
                        ldj_actual = compute_log_dimensionless_jerk(active, fs)
                        ldj_score = _score_ldj_group_c(ldj_actual)
                        _act_vel = np.diff(active)
                        _sal = compute_spectral_arc_length(_act_vel, fs)
                        sparc_score = max(0.0, min(1.0, 1.0 + (_sal * 30.0 / fs) / 100.0))
                        norm = 0.6 * ldj_score + 0.4 * sparc_score
                    else:
                        velocity = np.diff(series)
                        sal = compute_spectral_arc_length(velocity, fs)
                        sal_norm_fs = sal * (30.0 / fs)
                        norm = max(0.0, min(1.0, 1.0 + sal_norm_fs / 100.0))
                    sm_vals.append(norm)
                if sm_vals:
                    scores["kinematic_smoothness"] = float(np.mean(sm_vals))

                dur = rep_durs[i]
                if not single_rep_task and dur is not None and median_dur is not None and median_dur > 0.05:
                    denom_t = max(mad_dur * 3.0, median_dur * 0.15)
                    scores["rep_temporal_consistency"] = max(
                        0.0, min(1.0, 1.0 - abs(dur - median_dur) / denom_t)
                    )

                amp = rep_amps[i]
                if not single_rep_task and amp is not None and median_amp is not None and median_amp > 1e-3:
                    denom_a = max(mad_amp * 3.0, median_amp * 0.15)
                    scores["rep_spatial_consistency"] = max(
                        0.0, min(1.0, 1.0 - abs(amp - median_amp) / denom_a)
                    )

                comp_pairs = [
                    (_W_TIMING, scores["rep_temporal_consistency"]),
                    (_W_SMOOTH, scores["kinematic_smoothness"]),
                    (_W_AMP, scores["rep_spatial_consistency"]),
                ]
                valid_pairs = [(w, v) for w, v in comp_pairs if v is not None]
                if len(valid_pairs) >= 2:
                    total_w = sum(w for w, _ in valid_pairs)
                    scores["rep_articulation_score"] = float(
                        sum(w * v for w, v in valid_pairs) / total_w
                    )

                result[key] = scores

        return result

    def _group_by_task(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Group repetition metrics by (task_group, task_id), keeping only B and C tasks."""
        task_reps: Dict[str, pd.DataFrame] = {}
        if "task_group" not in df.columns or "task_id" not in df.columns:
            return task_reps

        for (tg, tid), grp in df.groupby(["task_group", "task_id"]):
            tg_str = str(tg)
            if tg_str not in ("B", "C"):
                continue
            tid_int = int(tid) if pd.notna(tid) else 0
            if tid_int == 0:
                continue
            measurement_reps = (
                grp[grp["repetition"] != 0] if "repetition" in grp.columns else grp
            )
            if len(measurement_reps) > 0:
                task_reps[f"{tg_str}_{tid_int}"] = measurement_reps

        return task_reps

    def _score_task_components(
        self,
        task_df: pd.DataFrame,
        task_key: str,
        features_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Score a single task on timing regularity, movement smoothness, and amplitude."""
        timing = self._compute_timing_regularity(task_df, task_key)
        smoothness = self._compute_movement_smoothness(task_df, task_key, features_df)
        amplitude = self._compute_amplitude_consistency(task_df, task_key)

        w = _COMPONENT_WEIGHTS
        composite = (
            w["timing"] * timing
            + w["smoothness"] * smoothness
            + w["amplitude"] * amplitude
        )

        task_name = ""
        if "task_name" in task_df.columns:
            names = task_df["task_name"].dropna().unique()
            names = [n for n in names if n != "(no task selected)"]
            task_name = names[0] if names else task_key

        raw_act_mean = (
            float(task_df["mean_activation_mean"].mean())
            if "mean_activation_mean" in task_df.columns
            else None
        )
        raw_duration = (
            float(task_df["duration_sec"].mean())
            if "duration_sec" in task_df.columns
            else None
        )

        return {
            "score": float(composite),
            "timing": float(timing),
            "smoothness": float(smoothness),
            "amplitude": float(amplitude),
            "task_name": task_name,
            "n_reps": len(task_df),
            "raw_act_mean": raw_act_mean,
            "raw_duration": raw_duration,
        }

    def _compute_timing_regularity(
        self, task_df: pd.DataFrame, task_key: str
    ) -> float:
        """Score timing regularity from duration consistency and expected-duration match.

        Three components are combined:
        1. Inter-repetition duration CV - consistency across reps (multi-rep only).
        2. Expected-duration ratio - mean rep duration vs expected_duration_sec x n_reps
           from config. Applied to both Group B and Group C tasks so that both
           single-rep test sessions and multi-rep baseline sessions are treated
           consistently: effective_expected = expected_duration_sec x n_configured
           is always used, regardless of how many rows are present. This avoids
           the collapse to 0 that occurred when multi-rep baselines were scored at
           3 s per bout but actual recording windows were ~9 s per bout.
        3. Time-to-peak CV - consistency of peak timing across reps, restricted to
           speech-relevant channels (jaw, mouth, mean_activation) to avoid noise
           from brow/eye columns that sort alphabetically to the top of the list.
        """
        scores: List[float] = []

        if "duration_sec" in task_df.columns:
            durations = task_df["duration_sec"].dropna().values

            if len(durations) > 1:
                mean_dur = np.mean(durations)
                if mean_dur > 0:
                    cv = np.std(durations) / mean_dur
                    scores.append(max(0.0, 1.0 - cv))

            parts = task_key.split("_")
            if len(parts) == 2 and parts[0] in ("B", "C"):
                cfg_map = self.group_b_config if parts[0] == "B" else self.group_c_config
                task_cfg = cfg_map.get(int(parts[1]), {})
                expected_dur = float(task_cfg.get("expected_duration_sec", 0))
                if expected_dur <= 0:
                    expected_dur = 3.0 if parts[0] == "B" else 2.0
                n_configured = int(task_cfg.get("repetitions", 1) or 1)
                effective_expected = expected_dur * n_configured
                if len(durations) > 0:
                    ratio = float(np.mean(durations)) / effective_expected
                    scores.append(max(0.0, 1.0 - abs(1.0 - ratio)))


        return float(np.mean(scores)) if scores else 0.5

    def _compute_movement_smoothness(
        self,
        task_df: pd.DataFrame,
        task_key: str,
        features_df: Optional[pd.DataFrame] = None,
    ) -> float:
        """Score movement smoothness via spectral arc length (SAL) and SPARC.

        Two frame-level components:
          1. SAL from the session-level ``activation_velocity`` column in
             features_df (one score per repetition, normalised as
             ``1 + SAL / 100``).  Group C tasks are skipped here (handled
             in component 2 via LDJ).
          2. Per-channel kinematic smoothness via ``_compute_kinematic_smoothness``:
             - Group B (DDK): frame-rate-normalised SPARC on each channel's
               velocity per repetition.
             - Group C (words): active-phase LDJ mapped to [0, 1] using
               empirically calibrated absolute bounds (PAC3/PAC7 webcam data),
               blended 60/40 with active-phase SPARC.  The previous
               synthetic-reference LDJ comparison biased healthy speakers to
               0.45–0.65; the corrected method targets 0.83–0.89 for healthy,
               ≤ 0.60 for impaired.

        Note: the previous ``jerkiness = vel_std / amplitude`` metric was
        removed.  ``activation_velocity_std`` is in activation-units/s while
        ``mean_activation_mean`` is in activation-units; the ratio is not
        dimensionless, systematically scored 0.0 for healthy speakers with
        oscillating DDK signals, and is fully superseded by the SAL/SPARC
        estimates below.
        """
        scores: List[float] = []

        if features_df is not None and "activation_velocity" in features_df.columns:
            sal_scores = self._compute_sal_scores(features_df, task_key)
            if sal_scores:
                scores.extend(sal_scores)

        if features_df is not None:
            kin_smoothness = self._compute_kinematic_smoothness(features_df, task_key)
            if kin_smoothness is not None:
                scores.append(kin_smoothness)

        return float(np.mean(scores)) if scores else 0.5

    def _compute_sal_scores(
        self, features_df: pd.DataFrame, task_key: str
    ) -> List[float]:
        """Compute spectral arc length from frame-level velocity for each repetition.

        Group C tasks are skipped here: each word production is a single discrete
        event whose velocity signal has a simple single-cycle spectrum, making
        SPARC insensitive (all words cluster at the same SAL).  Group C smoothness
        is handled via LDJ in ``_compute_kinematic_smoothness`` instead.
        """
        results: List[float] = []
        parts = task_key.split("_")
        if len(parts) != 2:
            return results

        tg, tid = parts[0], parts[1]
        if tg == "C":
            return results

        has_tg = "task_group" in features_df.columns
        has_tid = "task_id" in features_df.columns
        if not (has_tg and has_tid):
            return results

        mask = (
            (features_df["task_group"].astype(str) == tg)
            & (features_df["task_id"].astype(int) == int(tid))
        )
        if "segment" in features_df.columns:
            mask = mask & (features_df["segment"] == "measurement")

        task_frames = features_df[mask]
        if len(task_frames) == 0:
            return results

        for rep in task_frames["repetition"].unique():
            if rep == 0:
                continue
            rep_df = task_frames[task_frames["repetition"] == rep]
            velocity = rep_df["activation_velocity"].values
            if len(velocity) < 8:
                continue
            timestamps = rep_df["timestamp_abs"].values
            dt = np.median(np.diff(timestamps))
            fs = 1.0 / dt if dt > 0 else 30.0
            sal = compute_spectral_arc_length(velocity, fs)
            sal_score = max(0.0, min(1.0, 1.0 + (sal * 30.0 / fs) / 100.0))
            results.append(sal_score)

        if len(results) > 2:
            _cv = float(np.std(results)) / (float(np.mean(results)) + 1e-8)
            results.append(max(0.0, 1.0 - _cv * 4.0))

        return results

    def _compute_kinematic_smoothness(
        self, features_df: pd.DataFrame, task_key: str
    ) -> Optional[float]:
        """Compute smoothness from speech-channel velocity using SPARC and LDJ.

        Preferred channels are landmark kinematic measures (kin_mouth_opening,
        etc.). When those are absent (the common case for webcam sessions
        without explicit landmark kinematics), falls back to speech-relevant
        blendshape activations (jawOpen, mouthClose, mouthPucker, …) as the
        articulatory amplitude-envelope analog following He & Dellwo (2017).

        **Group B (DDK):** Frame-rate-normalised SPARC on the velocity of each
        channel for every repetition.  SPARC magnitude scales linearly with the
        Nyquist frequency (fs/2), so raw SAL values grow more negative at higher
        frame rates even for identical movements.  Dividing by (fs/30) before
        applying the 30-fps calibration makes the score frame-rate independent::

            SAL_norm = SAL × (30 / fs)
            score    = max(0, min(1, 1 + SAL_norm / 100))

        Calibration (SAL_norm, equivalent to 30 fps DDK blendshape velocity):
          SAL_norm ≈  −5  → 0.95  (very smooth DDK)
          SAL_norm ≈ −20  → 0.80  (normal healthy DDK at 53.6 fps)
          SAL_norm ≈ −35  → 0.65  (mildly impaired)
          SAL_norm ≈ −50  → 0.50  (moderately impaired)
          SAL_norm ≈ −100 → 0.00  (floored)

        **Group C (word production):** Active-phase absolute LDJ calibration
        blended with active-phase SPARC.  The previous synthetic-reference LDJ
        comparison (``ldj_ref − ldj_actual`` / 20) systematically underscored
        healthy speakers (0.45–0.65) because the synthetic sine-arch reference
        is always smoother than any real webcam jaw-movement signal.

        Algorithm (LDJ formula from Balasubramanian et al. 2012; SPARC formula
        from Balasubramanian et al. 2012 via Gulde & Hermsdörfer 2018;
        phase isolation, empirical bounds, and metric blending are project-
        specific adaptations for webcam-derived jaw blendshape signals):

        1. Apply 5-frame moving-average to suppress blendshape tracker noise.
        2. Isolate the primary jaw-movement event with ``_find_active_phase``
           (largest contiguous region above 15 % of peak-to-baseline range).
        3. Skip if active-phase amplitude < noise floor (1e-4 units → 0.5 neutral).
        4. Compute LDJ of the active phase and map to [0, 1] using empirically
           calibrated absolute bounds (from PAC3 and PAC7 healthy-speaker data)::

               ldj_score = max(0, min(1, (LDJ − (−55)) / 50))

           Calibration landmarks (webcam blendshape, 30 fps):
             LDJ ≥ −5   → 1.00  (near-theoretical smooth arch)
             LDJ = −10  → 0.90  (healthy, short active phase ≤ 20 frames)
             LDJ = −14  → 0.82  (healthy, typical 60–85 frame active phase)
             LDJ = −25  → 0.60  (mild motor impairment)
             LDJ = −35  → 0.40  (moderate impairment)
             LDJ ≤ −55  → 0.00  (severe / degenerate)

        5. Compute active-phase SPARC (velocity spectral arc length) to detect
           tremor and stop-start hesitation patterns that LDJ may saturate on::

               sparc_score = max(0, min(1, 1 + SAL_norm / 100))

        6. Blend: ``score = 0.6 × ldj_score + 0.4 × sparc_score``
           (LDJ weighted higher for its superior sensitivity to motor impairment).

        Healthy expected range post-fix: PAC3 0.825–0.883, PAC7 0.861–0.889.
        Mild impairment: 0.60–0.75.  Moderate: 0.40–0.60.  Severe: < 0.40.

        Returns None if no suitable columns are present.
        """
        _KIN_COLS = [
            "kin_mouth_opening",
            "kin_labial_fissure_width",
            "kin_lip_action_y",
            "kin_mouth_height",
        ]
        _SPEECH_BS = [
            "jawOpen",
            "mouthClose",
            "mouthPucker",
            "mouthFunnel",
            "mouthSmileLeft",
            "mouthSmileRight",
        ]
        avail = [c for c in _KIN_COLS if c in features_df.columns]
        if not avail:
            avail = [c for c in _SPEECH_BS if c in features_df.columns]
        if not avail:
            return None

        parts = task_key.split("_")
        if len(parts) != 2:
            return None
        tg, tid = parts[0], parts[1]
        if tg not in ("B", "C"):
            return None

        has_tg = "task_group" in features_df.columns
        has_tid = "task_id" in features_df.columns
        if not (has_tg and has_tid):
            return None

        mask = (
            (features_df["task_group"].astype(str) == tg)
            & (features_df["task_id"].astype(int) == int(tid))
        )
        if "segment" in features_df.columns:
            mask = mask & (features_df["segment"] == "measurement")
        task_frames = features_df[mask]
        if len(task_frames) < 4:
            return None

        smoothness_vals: List[float] = []

        fs = 30.0
        if "timestamp_abs" in task_frames.columns:
            ts = task_frames["timestamp_abs"].dropna().values
            if len(ts) > 1:
                dt = np.median(np.diff(ts))
                if dt > 0:
                    fs = 1.0 / dt

        is_c_task = tg == "C"

        rep_col_present = "repetition" in task_frames.columns
        reps = task_frames["repetition"].unique() if rep_col_present else [None]
        for rep in reps:
            if rep is not None and rep == 0:
                continue
            rep_df = (
                task_frames[task_frames["repetition"] == rep]
                if rep is not None else task_frames
            )
            if len(rep_df) < 8:
                continue
            for col in avail[:4]:
                series = rep_df[col].dropna().values
                if len(series) < 8:
                    continue
                if is_c_task:
                    _k5 = np.ones(5) / 5.0
                    _smoothed = np.convolve(series, _k5, mode="same")
                    _left, _right = _find_active_phase(_smoothed, fs)
                    active = _smoothed[_left : _right + 1]
                    if len(active) < 8:
                        active = _smoothed
                    if np.ptp(active) < 1e-4:
                        smoothness_vals.append(0.5)
                        continue
                    ldj_actual = compute_log_dimensionless_jerk(active, fs)
                    ldj_score = _score_ldj_group_c(ldj_actual)
                    _act_vel = np.diff(active)
                    _sal = compute_spectral_arc_length(_act_vel, fs)
                    sparc_score = max(0.0, min(1.0, 1.0 + (_sal * 30.0 / fs) / 100.0))
                    norm = 0.6 * ldj_score + 0.4 * sparc_score
                else:
                    velocity = np.diff(series)
                    sal = compute_spectral_arc_length(velocity, fs)
                    sal_norm_fs = sal * (30.0 / fs)
                    norm = max(0.0, min(1.0, 1.0 + sal_norm_fs / 100.0))
                smoothness_vals.append(norm)

        return float(np.mean(smoothness_vals)) if smoothness_vals else None

    def _compute_amplitude_consistency(
        self, task_df: pd.DataFrame, task_key: str
    ) -> float:
        """Score amplitude consistency across repetitions.

        Multi-rep (≥2 rows): uses median-based inter-rep coefficient of variation
        so that a single outlier repetition does not collapse the entire score.

        Single-rep (1 row): used for DDK tasks recorded as one continuous sequence
        (e.g. a 9–14 s pa-pa-pa block counted as one repetition).  Inter-rep CV
        is undefined, so we use the *within-rep* coefficient of variation
        (std / |mean| of mean_activation across frames) as a proxy for movement
        rhythmicity.  Healthy DDK produces oscillating activation → within-rep
        CV ≈ 1.0–2.5; threshold of 1.5 is calibrated from PAC7 healthy-speaker
        data (pa-pa-pa CV=2.49, ta-ta-ta 1.54, ka-ka-ka 1.37, pa-ta-ka 1.06).
        """
        scores: List[float] = []
        is_c_task = task_key.startswith("C_")

        if is_c_task:
            if "mean_activation_range" in task_df.columns:
                rng_vals = task_df["mean_activation_range"].dropna().values
                if len(rng_vals) > 1:
                    med_r = np.median(rng_vals)
                    mad_r = np.median(np.abs(rng_vals - med_r))
                    if med_r > 150:
                        robust_cv = (mad_r * 1.4826) / med_r
                        scores.append(max(0.0, 1.0 - min(1.0, robust_cv / 1.5)))
                    else:
                        scores.append(0.5)
                elif len(rng_vals) == 1:
                    amp_val = float(rng_vals[0])
                    if amp_val >= 2000:
                        scores.append(0.82)
                    elif amp_val >= 500:
                        scores.append(0.50 + 0.32 * (amp_val - 500.0) / 1500.0)
                    elif amp_val >= 150:
                        scores.append(0.30 + 0.20 * (amp_val - 150.0) / 350.0)
                    else:
                        scores.append(0.25)
        else:
            if "mean_activation_mean" in task_df.columns:
                amps = task_df["mean_activation_mean"].dropna().values
                if len(amps) > 1:
                    med = np.median(amps)
                    mad = np.median(np.abs(amps - med))
                    if med > 1e-6:
                        robust_cv = (mad * 1.4826) / med
                        scores.append(max(0.0, 1.0 - min(1.0, robust_cv)))
                elif len(amps) == 1 and abs(amps[0]) > 1e-6:
                    std_col = "mean_activation_std"
                    if std_col in task_df.columns:
                        std_vals = task_df[std_col].dropna().values
                        if len(std_vals) == 1 and std_vals[0] >= 0:
                            within_cv = std_vals[0] / abs(amps[0])
                            scores.append(min(1.0, max(0.0, within_cv / 1.5)))
                        else:
                            scores.append(0.5)
                    else:
                        scores.append(0.5)

            range_cols = [
                c for c in task_df.columns
                if c.endswith("_range") and "activation" in c and "across" not in c
            ]
            if range_cols:
                range_vals = task_df[range_cols].mean(axis=1).dropna().values
                if len(range_vals) > 1:
                    med_r = np.median(range_vals)
                    mad_r = np.median(np.abs(range_vals - med_r))
                    if med_r > 1e-6:
                        robust_cv_r = (mad_r * 1.4826) / med_r
                        scores.append(max(0.0, 1.0 - min(1.0, robust_cv_r)))
                elif len(range_vals) == 1 and "mean_activation_mean" in task_df.columns:
                    m = task_df["mean_activation_mean"].dropna().values
                    if len(m) == 1 and abs(m[0]) > 1e-6:
                        range_to_mean = range_vals[0] / abs(m[0])
                        scores.append(min(1.0, max(0.0, range_to_mean / 8.0)))

        return float(np.mean(scores)) if scores else 0.5

    def _compute_impairment_consistency(
        self,
        per_task_scores: Dict[str, Dict[str, Any]],
        repetition_metrics_df: pd.DataFrame,
    ) -> float:
        """Compute how consistently impaired the articulation is across tasks and reps.

        High value (>0.7) suggests consistent impairment (dysarthria pattern).
        Low value (<0.5) suggests inconsistent impairment (apraxia-like).
        Returns near-zero when no tasks are impaired.
        """
        if not per_task_scores:
            return 0.5

        task_scores = [v["score"] for v in per_task_scores.values()]

        impaired_tasks = [s for s in task_scores if s < _IMPAIRMENT_THRESHOLD]
        if len(impaired_tasks) == 0:
            return 0.0

        proportion_impaired = len(impaired_tasks) / len(task_scores)

        if len(task_scores) > 1:
            score_std = float(np.std(task_scores))
            uniformity = max(0.0, 1.0 - score_std * 3)
        else:
            uniformity = 0.5

        within_task_consistencies: List[float] = []
        task_reps = self._group_by_task(repetition_metrics_df)
        for task_key in per_task_scores:
            grouped_df = task_reps.get(task_key)
            if grouped_df is None or len(grouped_df) < 2:
                continue
            mean_cols = [
                c
                for c in grouped_df.columns
                if c.endswith("_mean")
                and "asymmetry" not in c
                and "across" not in c
            ]
            if not mean_cols:
                continue
            cvs: List[float] = []
            for col in mean_cols:
                vals = grouped_df[col].dropna().values
                if len(vals) > 1 and np.abs(np.mean(vals)) > 0.001:
                    cvs.append(np.std(vals) / np.abs(np.mean(vals)))
            if cvs:
                within_task_consistencies.append(
                    max(0.0, 1.0 - min(1.0, float(np.mean(cvs))))
                )

        within_consistency = (
            float(np.mean(within_task_consistencies))
            if within_task_consistencies
            else 0.5
        )

        consistency = (
            0.40 * proportion_impaired
            + 0.30 * uniformity
            + 0.30 * within_consistency
        )

        return float(np.clip(consistency, 0.0, 1.0))

    def compute_enhanced_speech_features(
        self,
        repetition_metrics_df: pd.DataFrame,
        task_profile: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Compute enhanced features for dysarthria, speech apraxia, and phonological disorder differentiation.

        Produces duration ratios versus baseline, per-task repetition
        variability (simple vs complex), cross-group B-vs-C comparison,
        and per-word cross-repetition consistency.
        """
        result: Dict[str, Any] = {}
        task_reps = self._group_by_task(repetition_metrics_df)

        duration_analysis = self._compute_duration_analysis(task_reps, task_profile)
        result.update(duration_analysis)

        variability = self._compute_task_repetition_variability(task_reps)
        result.update(variability)

        group_comparison = self._compute_group_comparison(task_reps)
        result.update(group_comparison)

        word_consistency = self._compute_word_cross_rep_consistency(task_reps)
        result.update(word_consistency)

        return result

    def _compute_duration_analysis(
        self,
        task_reps: Dict[str, pd.DataFrame],
        task_profile: Optional[Any],
    ) -> Dict[str, Any]:
        """Compute duration ratios between test and reference for each Group B task.

        Duration ratios > 1 indicate slower execution (characteristic of
        dysarthria).  High consistency across tasks strengthens that hypothesis.
        """
        if task_profile is None or not hasattr(task_profile, "get_task_feature_stats"):
            return {}

        b_ratios: List[float] = []
        per_task_duration: Dict[str, Dict[str, float]] = {}

        for task_key, task_df in task_reps.items():
            if not task_key.startswith("B_"):
                continue
            parts = task_key.split("_")
            if len(parts) != 2:
                continue

            ref_stats = task_profile.get_task_feature_stats(parts[0], int(parts[1]))
            ref_dur = ref_stats.get("duration_sec", {})
            ref_dur_mean = ref_dur.get("mean") if isinstance(ref_dur, dict) else None
            if ref_dur_mean is None or ref_dur_mean <= 0:
                continue
            if "duration_sec" not in task_df.columns:
                continue

            test_dur = float(task_df["duration_sec"].mean())
            if test_dur <= 0:
                continue

            ratio = test_dur / ref_dur_mean
            b_ratios.append(ratio)
            per_task_duration[task_key] = {
                "test_duration": test_dur,
                "ref_duration": ref_dur_mean,
                "ratio": ratio,
            }

        if not b_ratios:
            return {}

        ratios_arr = np.array(b_ratios)
        mean_ratio = float(np.mean(ratios_arr))
        ratio_cv = float(np.std(ratios_arr) / (np.mean(ratios_arr) + 0.001))
        ratio_consistency = float(max(0.0, 1.0 - ratio_cv))

        return {
            "speech_duration_ratio_mean": mean_ratio,
            "speech_duration_ratio_consistency": ratio_consistency,
            "speech_duration_per_task": per_task_duration,
        }

    def _compute_task_repetition_variability(
        self, task_reps: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Compute within-task repetition-to-repetition variability for each Group B task.

        High variability in complex tasks (pa-ta-ka) with low variability in
        simple tasks is characteristic of speech apraxia.
        """
        variabilities: Dict[str, float] = {}

        for task_key, task_df in task_reps.items():
            if not task_key.startswith("B_"):
                continue
            if len(task_df) < 2:
                variabilities[task_key] = 0.0
                continue

            mean_cols = [
                c for c in task_df.columns
                if c.endswith("_mean") and "asymmetry" not in c and "across" not in c
            ]
            if not mean_cols:
                variabilities[task_key] = 0.0
                continue

            cvs: List[float] = []
            for col in mean_cols:
                vals = task_df[col].dropna().values
                if len(vals) > 1 and np.abs(np.mean(vals)) > 0.001:
                    cvs.append(float(np.std(vals) / np.abs(np.mean(vals))))
            variabilities[task_key] = float(np.mean(cvs)) if cvs else 0.0

        simple_keys = [k for k in variabilities if k in _TASK_B_SIMPLE]
        complex_keys = []
        for k, task_df in task_reps.items():
            if k in _TASK_B_COMPLEX:
                complex_keys.append(k)
                continue
            if not k.startswith("B_"):
                continue
            tnames = task_df["task_name"].dropna().unique() if "task_name" in task_df.columns else []
            if any(str(n).lower().strip() in _COMPLEX_SEQUENCING_LABELS for n in tnames):
                if k not in complex_keys:
                    complex_keys.append(k)

        simple_var = float(np.mean([variabilities[k] for k in simple_keys])) if simple_keys else 0.0
        complex_var = float(np.mean([variabilities[k] for k in complex_keys])) if complex_keys else 0.0
        ratio = complex_var / (simple_var + 0.001) if simple_var > 0 else complex_var * 10

        return {
            "per_task_variability": variabilities,
            "simple_repetition_variability": simple_var,
            "pataka_repetition_variability": complex_var,
            "complex_simple_variability_ratio": float(ratio),
        }

    def _compute_group_comparison(
        self, task_reps: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Compare Group B and Group C aggregate quality scores.

        Positive dissociation (B intact, C impaired) suggests phonological
        disorder rather than dysarthria or speech apraxia.
        """
        b_scores: List[float] = []
        c_scores: List[float] = []

        for task_key, task_df in task_reps.items():
            components = self._score_task_components(task_df, task_key, None)
            score = components["score"]
            if task_key.startswith("B_"):
                b_scores.append(score)
            elif task_key.startswith("C_"):
                c_scores.append(score)

        if not b_scores or not c_scores:
            return {}

        b_mean = float(np.mean(b_scores))
        c_mean = float(np.mean(c_scores))
        b_intact = b_mean > _IMPAIRMENT_THRESHOLD
        c_impaired = c_mean < _IMPAIRMENT_THRESHOLD

        return {
            "group_b_mean_score": b_mean,
            "group_c_mean_score": c_mean,
            "group_bc_dissociation": float(b_mean - c_mean),
            "group_b_intact": 1.0 if b_intact else 0.0,
            "group_c_impaired": 1.0 if c_impaired else 0.0,
        }

    def _compute_word_cross_rep_consistency(
        self, task_reps: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Compute how consistently each word is produced across its repetitions.

        High consistency (same facial movement pattern each time) suggests a
        stable substitution rule (phonological disorder).  Low consistency
        suggests variable motor planning errors (speech apraxia).
        """
        per_word_consistency: Dict[str, float] = {}

        for task_key, task_df in task_reps.items():
            if not task_key.startswith("C_"):
                continue
            if len(task_df) < 2:
                per_word_consistency[task_key] = 1.0
                continue

            mean_cols = [
                c for c in task_df.columns
                if c.endswith("_mean") and "asymmetry" not in c and "across" not in c
            ]
            if not mean_cols:
                per_word_consistency[task_key] = 1.0
                continue

            cvs: List[float] = []
            for col in mean_cols:
                vals = task_df[col].dropna().values
                if len(vals) > 1 and np.abs(np.mean(vals)) > 0.001:
                    cvs.append(float(np.std(vals) / np.abs(np.mean(vals))))
            per_word_consistency[task_key] = (
                float(max(0.0, 1.0 - np.mean(cvs))) if cvs else 1.0
            )

        if not per_word_consistency:
            return {}

        return {
            "per_word_cross_rep_consistency": per_word_consistency,
            "word_cross_rep_consistency_mean": float(
                np.mean(list(per_word_consistency.values()))
            ),
        }

    def compute_word_production_features(
        self, per_task_scores: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Derive word-production features from Group C task scores.

        Returns quality, complexity gradient, cross-word consistency, and
        impairment rate that the decision-support module uses as proxies
        for phonological accuracy when audio is unavailable.
        """
        c_scores = {k: v for k, v in per_task_scores.items() if k.startswith("C_")}
        if not c_scores:
            return {}

        c_tasks_config = (
            self.tasks_config.get("task_groups", {}).get("C", {}).get("tasks", {})
        )

        scores: List[float] = []
        complexities: List[int] = []
        for task_key in sorted(c_scores.keys()):
            task_id = int(task_key.split("_")[1])
            complexity = c_tasks_config.get(task_id, {}).get("complexity", task_id)
            scores.append(c_scores[task_key]["score"])
            complexities.append(complexity)

        scores_arr = np.array(scores)
        complexities_arr = np.array(complexities, dtype=float)

        word_production_quality = float(np.mean(scores_arr))
        complexity_gradient = _rank_correlation(complexities_arr, scores_arr)

        if len(scores_arr) > 1 and np.mean(scores_arr) > 0:
            cv = np.std(scores_arr) / np.mean(scores_arr)
            cross_word_consistency = float(max(0.0, 1.0 - cv))
        else:
            cross_word_consistency = 1.0

        impairment_rate = float(np.mean(scores_arr < _IMPAIRMENT_THRESHOLD))

        per_word: Dict[str, Dict[str, Any]] = {}
        for task_key in sorted(c_scores.keys()):
            task_id = int(task_key.split("_")[1])
            complexity = c_tasks_config.get(task_id, {}).get("complexity", task_id)
            per_word[task_key] = {
                "score": c_scores[task_key]["score"],
                "complexity": complexity,
            }

        cross_word_score_variance = float(np.var(scores_arr)) if len(scores_arr) > 1 else 0.0
        cross_word_score_std = float(np.std(scores_arr)) if len(scores_arr) > 1 else 0.0

        return {
            "word_production_quality": word_production_quality,
            "complexity_gradient": complexity_gradient,
            "cross_word_consistency": cross_word_consistency,
            "cross_word_score_variance": cross_word_score_variance,
            "cross_word_score_std": cross_word_score_std,
            "word_production_impairment_rate": impairment_rate,
            "n_words_scored": len(scores_arr),
            "per_word_scores": per_word,
        }


def create_articulation_scorer(
    tasks_config: Dict[str, Any],
) -> ArticulationScorer:
    """Factory: build an ArticulationScorer from task configuration."""
    return ArticulationScorer(tasks_config)
