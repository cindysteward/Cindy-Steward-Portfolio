"""
Decision support module for facial motor and speech behavior analysis pipeline.

Loads and applies decision rules from configuration to generate screening
indications. Evaluates task groups A (non-speech facial), B (speech
articulation), and C (word production) independently, then computes a
composite confidence score.

Key clinical references underpinning the disorder screening logic
=================================================================

Intraoperative speech and language error taxonomy
  Collee et al. (2022) Cancers 14, 5466 showed that intraoperative
  production errors (dysarthria, stuttering) independently predict acute
  postoperative language deficits (OR=2.06), and anomia predicts them too
  (OR=2.09). Production errors map to Group B evaluation; anomia and
  word-finding errors map to Group C evaluation.
  https://doi.org/10.3390/cancers14215466

  Collee et al. (2023) Neurosurg Rev maps specific intraoperative error
  types to cortical and subcortical anatomy: precentral gyrus for
  dysarthria/speech arrest, IFOF for semantic errors, AF for phonemic
  errors. This anatomical framework underpins the disorder-profile taxonomy
  used in this module.
  https://doi.org/10.1007/s10143-022-01943-9

Facial paresis detection
  Oliveira et al. (2024) CMPB 258, 108195 achieved 82 % accuracy
  distinguishing post-stroke patients from healthy controls using AU7, AU20,
  AU23 (mouth-area AUs), confirming that perioral AUs are most specific for
  central paresis.
  https://doi.org/10.1016/j.cmpb.2024.108195

  Ruiter et al. (2023) Ann Clin Transl Neurol achieved AUC 0.88 for severity
  classification of facial weakness in myasthenia gravis; AU6 (cheek raiser)
  correlated with disease severity and a 3D-CNN outperformed neurologists on
  the same isolated video task.
  https://doi.org/10.1002/acn3.51823

  Baig et al. (2023) achieved 98.93 % accuracy for binary paralysis
  classification using MobileNetV2 on MediaPipe 468-landmark meshes; t-SNE
  showed unhealthy subclusters within the healthy distribution, supporting
  the need for patient-specific baseline comparison rather than population
  norms.
  hdl:10210/504453

  Ozmen et al. (2025) confirmed multi-feature superiority for facial
  paralysis detection: smile index ratio, commissure displacement, and teeth
  area combined achieved 86 % accuracy.
  https://doi.org/10.1097/01.GOX.0001112148.28567.85

Apraxia of speech detection rationale
  Allison et al. (2020) J Speech Lang Hear Res validated kinematic features
  for differential diagnosis of apraxia of speech from other motor speech
  disorders, supporting the use of selective complex-task decline as the
  primary apraxia detection criterion.
  https://doi.org/10.1044/2020_JSLHR-20-00061

  Allison et al. (2022) AJSLP 31, 1682 demonstrated that CV across DDK
  repetitions (the D_dtw metric) achieves 88 % sensitivity / specificity
  for detecting motor involvement. CV > 0.30 separates pathological from
  typical motor variability, underpinning the b4_rep_dtw_cv gate.
  https://doi.org/10.1044/2022_AJSLP-21-00241

  Duffy JR (2013) Motor Speech Disorders, 3rd ed., Elsevier/Mosby.
  Standard clinical reference for dysarthria and apraxia classification,
  informing the severity tier thresholds and the dysarthria/apraxia
  dissociation logic throughout this module.

Threshold notes for Group B
  apraxia_selective = 0.10 (config/decision_rules.yaml):
    Represents a 10 % selective pa-ta-ka decline while simple syllables
    remain intact. The selective criterion is appropriate because the AOS
    signal is the DIFFERENTIAL between complex (B4) and simple (B1-3)
    tasks, not the absolute performance level.

  b4_dtw_vs_ref (test B4 mean DTW / baseline B4 mean DTW), gate > 2.0:
    Reference-relative detection: tests whether the participant's own
    pa-ta-ka kinematics have changed dramatically from their healthy
    baseline. Immune to between-participant variation in baseline DTW level.

  b4_rep_dtw_cv (CV across B4 repetition DTW values), gate > 0.30:
    High trial-to-trial variability in B4 DTW indicates inconsistent
    articulatory search behaviour. Complements the ratio-based detection.
    Based on the D_dtw metric validated in Allison et al. (2022).
"""

import logging
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger("pipeline")


@dataclass
class ScreeningIndication:
    """A single screening finding produced by the decision tree."""

    indication_type: str
    severity: str
    confidence: float
    supporting_features: Dict[str, float]
    source_node: str
    description: str
    task_group: str = "0"
    task_id: int = 0


@dataclass
class ConfidenceComponents:
    """Breakdown of the composite confidence score."""

    data_quality: float
    consistency: float
    model_rule_agreement: float
    overall: float


class DecisionSupport:
    """Applies clinical decision rules to generate screening indications."""

    def __init__(self, decision_rules_config: Dict[str, Any]):
        """Initialise from the loaded decision_rules YAML config."""
        self.config = decision_rules_config
        self.decision_tree = decision_rules_config.get("decision_tree", {})
        self.thresholds = decision_rules_config.get("thresholds", {})
        self.confidence_weights = decision_rules_config.get(
            "confidence_weights",
            {"data_quality": 0.35, "consistency": 0.35, "model_rule_agreement": 0.30},
        )
        self.screening_indications_config = decision_rules_config.get(
            "screening_indications", {}
        )

        self.is_baseline_session = False
        self.has_reference_baseline = False
        self.reference_baseline_stats: Optional[Dict] = None
        self.reference_articulation: Optional[Dict[str, Any]] = None
        self.reference_asymmetry_stats: Optional[Dict[str, float]] = None
        self.is_ors_session: bool = False

        self.current_task_group: str = "0"
        self.current_task_id: int = 0
        self._ors_b_apraxia_found: bool = False
        self._b_dysarthria_found: bool = False
        self._c_dysarthria_found: bool = False

    def set_session_context(
        self,
        is_baseline: bool = False,
        has_reference: bool = False,
        reference_stats: Optional[Dict] = None,
        task_group: str = "0",
        task_id: int = 0,
        reference_articulation: Optional[Dict[str, Any]] = None,
        reference_asymmetry_stats: Optional[Dict[str, float]] = None,
        is_ors_session: bool = False,
        reference_head_yaw: Optional[float] = None,
    ) -> None:
        """Set context about the current session for proper evaluation.

        For baseline sessions without a reference, pathology is only flagged
        when asymmetry is extremely severe.  For test sessions, deviations
        from the baseline are evaluated normally.

        *reference_articulation* supplies the baseline articulation scores so
        that Group B evaluation uses deviation-based thresholds rather than
        absolute cut-offs.

        *reference_asymmetry_stats* supplies participant-specific Group-A
        asymmetry statistics (``mean`` and ``std``) computed from the reference
        session's repetition metrics.  When provided, the paresis threshold is
        set to ``max(config_mild, ref_mean + 3 * ref_std)`` so that a
        participant who naturally shows higher baseline face asymmetry is not
        incorrectly flagged.

        *reference_head_yaw* is the mean head yaw angle (degrees) in the
        reference (baseline) session.  When provided, the between-session yaw
        change is used to correct the apparent asymmetry ratio before threshold
        comparison.  A head yaw rotation of Δθ degrees changes the projected
        face geometry by |sin(Δθ)|, creating spurious left-right asymmetry in
        landmark-derived features.  The correction removes this geometric bias
        so that a head position change between sessions does not produce false
        paresis detections.
        """
        self.is_baseline_session = is_baseline
        self.has_reference_baseline = has_reference
        self.reference_baseline_stats = reference_stats
        self.reference_articulation = reference_articulation
        self.reference_asymmetry_stats = reference_asymmetry_stats
        self.is_ors_session = is_ors_session
        self.reference_head_yaw = reference_head_yaw
        self.current_task_group = task_group if task_group else "0"
        self.current_task_id = task_id if task_id else 0

    def evaluate(
        self,
        session_metrics: Dict[str, Any],
        task_metrics_df: Any,
        repetition_metrics_df: Any,
        anomaly_results: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run the full decision tree and return screening results."""
        features = self._prepare_features(
            session_metrics, task_metrics_df, repetition_metrics_df
        )

        if self.is_baseline_session:
            return {
                "indications": [],
                "confidence": {
                    "data_quality": 1.0,
                    "consistency": 1.0,
                    "model_rule_agreement": 1.0,
                    "overall": 1.0,
                },
                "features_used": features,
                "n_indications": 0,
                "indication_types": [],
            }

        indications: List[ScreeningIndication] = []

        _b4_hint_dtw = anomaly_results.get("b4_dtw_summary", {}) if anomaly_results else {}
        _b4_hint_vs_simple = float(_b4_hint_dtw.get("b4_vs_simple_ratio", 1.0))
        _b4_hint_n_anom = int(_b4_hint_dtw.get("b4_n_shape_anom", 0))
        _b4_dtw_vs_ref = float(_b4_hint_dtw.get("b4_dtw_vs_ref") or 1.0)
        self._b4_dtw_vs_ref_hint = _b4_dtw_vs_ref
        _b4_rep_dtw_cv = float(_b4_hint_dtw.get("b4_rep_dtw_cv") or 0.0)
        self._b4_selective_hint = (
            (_b4_hint_vs_simple > 1.5 and _b4_hint_n_anom >= 2)
            or (_b4_dtw_vs_ref > 1.6 and _b4_hint_n_anom >= 2)
            or (_b4_rep_dtw_cv > 0.25 and _b4_hint_n_anom >= 1)
            or (self.is_ors_session and _b4_hint_vs_simple > 1.70 and _b4_hint_n_anom >= 2)
        )
        self._b4_ratio_raw = _b4_hint_vs_simple
        self._wpq_c = float(features.get("word_production_quality", 1.0))
        self._ors_b_apraxia_found = False
        self._b_dysarthria_found = False
        self._c_dysarthria_found = False

        a_indications = self._evaluate_group_a(features)
        indications.extend(a_indications)
        b_indications = self._evaluate_group_b(features)
        indications.extend(b_indications)
        _b_speech_disorder = any(
            i.indication_type in ("dysarthria", "speech_apraxia") for i in b_indications
        )
        if _b_speech_disorder:
            features["group_b_intact"] = 0.0
        self._b_dysarthria_found = any(
            i.indication_type == "dysarthria" for i in b_indications
        )
        _ar_c_pre = (anomaly_results or {}).get("c_dtw_summary", {})
        self._max_c_task_dtw_precomputed = float(
            _ar_c_pre.get("max_c_task_dtw", 0.0) or 0.0
        )
        indications.extend(self._evaluate_group_c(features))
        indications.extend(self._evaluate_anomaly_results(anomaly_results))

        if any(i.indication_type == "speech_apraxia" for i in indications):
            indications = [
                i for i in indications if i.indication_type != "phonological_disorder"
            ]

        if any(i.indication_type == "facial_paresis" for i in indications):
            indications = [i for i in indications if i.indication_type != "buccofacial_apraxia"]

        by_type: Dict[str, ScreeningIndication] = {}
        for ind in indications:
            prev = by_type.get(ind.indication_type)
            if prev is None or ind.confidence > prev.confidence:
                by_type[ind.indication_type] = ind
        indications = [by_type[k] for k in by_type]

        confidence = self._compute_confidence(features, indications, anomaly_results)

        for ind in indications:
            ind.confidence = min(ind.confidence, confidence.overall)

        c_anomaly_note = None
        if anomaly_results:
            _per_task = anomaly_results.get("per_task_results", [anomaly_results])
            _c_anom_count = 0
            for _tr in _per_task:
                _names = _tr.get("task_names", [])
                _first = _names[0] if _names else ""
                if _first.startswith("C:") or (not _first and self.current_task_group == "C"):
                    _c_anom_count += sum(1 for v in _tr.get("is_anomaly", []) if v)
            _c_indicated = any(
                ind.task_group == "C" or ind.indication_type == "phonological_disorder"
                for ind in indications
            )
            if _c_anom_count > 0 and not _c_indicated:
                c_anomaly_note = (
                    f"{_c_anom_count} C-section repetition(s) showed statistical deviation "
                    f"from the reference profile but did not meet the clinical threshold for "
                    f"a phonological indication (within expected variation)."
                )

        return {
            "indications": [self._indication_to_dict(ind) for ind in indications],
            "confidence": {
                "data_quality": confidence.data_quality,
                "consistency": confidence.consistency,
                "model_rule_agreement": confidence.model_rule_agreement,
                "overall": confidence.overall,
            },
            "features_used": features,
            "n_indications": len(indications),
            "indication_types": list(set(ind.indication_type for ind in indications)),
            **({"c_section_note": c_anomaly_note} if c_anomaly_note else {}),
        }

    def _prepare_features(
        self,
        session_metrics: Dict[str, Any],
        task_metrics_df: Any,
        repetition_metrics_df: Any,
    ) -> Dict[str, float]:
        """Flatten session, task, and repetition data into a single feature dict."""
        features: Dict[str, float] = {}

        for key, value in session_metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                features[key] = float(value)

        if hasattr(task_metrics_df, "columns") and len(task_metrics_df) > 0:
            for col in task_metrics_df.select_dtypes(include=[np.number]).columns:
                values = task_metrics_df[col].dropna().values
                if len(values) > 0:
                    features[f"task_{col}"] = float(np.mean(values))

        if hasattr(repetition_metrics_df, "columns") and len(repetition_metrics_df) > 0:
            for col in repetition_metrics_df.select_dtypes(include=[np.number]).columns:
                values = repetition_metrics_df[col].dropna().values
                if len(values) > 0:
                    features[f"rep_{col}"] = float(np.mean(values))

        features.setdefault(
            "mean_asymmetry_ratio", features.get("overall_mean_asymmetry", 0.0)
        )
        features.setdefault(
            "asymmetry_across_tasks",
            features.get("overall_max_asymmetry", 0.0) > 0.15,
        )
        features.setdefault("asymmetry_consistency", 0.50)
        features.setdefault("dominant_side", "symmetric")
        features.setdefault("mean_signed_asymmetry", 0.0)
        features.setdefault("execution_correctness_score", 0.8)
        features.setdefault("error_consistency", 0.7)
        features.setdefault("articulation_impairment_consistency", 0.5)
        features.setdefault("mean_articulation_score", 0.7)

        if hasattr(repetition_metrics_df, "columns") and "task_group" in repetition_metrics_df.columns and len(repetition_metrics_df) > 0:
            _a_only = repetition_metrics_df[repetition_metrics_df["task_group"].astype(str) == "A"]
            _asym_col = next(
                (c for c in ("mean_asymmetry_ratio", "asymmetry_ratio_mean",
                             "mean_asymmetry", "overall_mean_asymmetry")
                 if c in repetition_metrics_df.columns),
                None,
            )
            if _asym_col and len(_a_only) > 0:
                _a_asym_vals = _a_only[_asym_col].dropna().values
                if len(_a_asym_vals) > 0:
                    features["group_a_mean_asymmetry"] = float(np.mean(_a_asym_vals))
                    _half = features["group_a_mean_asymmetry"] * 0.5
                    features["group_a_asym_consistency"] = float(np.mean(_a_asym_vals >= _half)) if _half > 0 else 0.5
        features.setdefault("group_a_mean_asymmetry",   features.get("mean_asymmetry_ratio", 0.0))
        features.setdefault("group_a_asym_consistency", features.get("asymmetry_consistency", 0.50))

        if hasattr(repetition_metrics_df, "columns") and len(repetition_metrics_df) > 0:
            _rep_df = repetition_metrics_df
            if "task_group" in _rep_df.columns and "task_id" in _rep_df.columns and "duration_sec" in _rep_df.columns:
                _b4_mask = (
                    _rep_df["task_group"].astype(str) == "B"
                ) & (
                    _rep_df["task_id"].astype(str).isin(["4"])
                ) & (
                    _rep_df.get("repetition", pd.Series(dtype=float)).reindex(_rep_df.index, fill_value=1).astype(str) != "0"
                )
                _simple_mask = (
                    _rep_df["task_group"].astype(str) == "B"
                ) & (
                    _rep_df["task_id"].astype(str).isin(["1", "2", "3"])
                ) & (
                    _rep_df.get("repetition", pd.Series(dtype=float)).reindex(_rep_df.index, fill_value=1).astype(str) != "0"
                )
                _b4_durs = _rep_df.loc[_b4_mask, "duration_sec"].dropna().values
                _sim_durs = _rep_df.loc[_simple_mask, "duration_sec"].dropna().values
                def _robust_cv(vals: np.ndarray) -> float:
                    """Coefficient of variation using median and MAD instead of mean/std."""
                    if len(vals) < 2:
                        return 0.0
                    med = np.median(vals)
                    if med < 1e-8:
                        return 0.0
                    mad = np.median(np.abs(vals - med))
                    return float((mad * 1.4826) / med)
                pataka_var = _robust_cv(_b4_durs)
                simple_var = _robust_cv(_sim_durs)
                features["pataka_repetition_variability"] = pataka_var
                features["simple_repetition_variability"] = simple_var
                _denom = max(simple_var, 0.005)
                features["complex_simple_variability_ratio"] = float(pataka_var / _denom)

                for _score_col in ("mean_activation_mean", "mean_activation",
                                   "activation_score", "mean_score"):
                    if _score_col in _rep_df.columns:
                        _b4_sc = _rep_df.loc[_b4_mask, _score_col].dropna().values
                        _sim_sc = _rep_df.loc[_simple_mask, _score_col].dropna().values
                        _pvar_sc = _robust_cv(_b4_sc)
                        _svar_sc = _robust_cv(_sim_sc)
                        features["pataka_score_variability"] = _pvar_sc
                        features["simple_score_variability"] = _svar_sc
                        features["complex_simple_score_var_ratio"] = float(
                            _pvar_sc / max(_svar_sc, 0.005)
                        )
                        break

        mean_artic_default = features["mean_articulation_score"]
        features.setdefault("articulation_score_pataka", mean_artic_default)
        features.setdefault("simple_syllable_mean", mean_artic_default)
        features.setdefault("articulation_score_pa", mean_artic_default)
        features.setdefault("articulation_score_ta", mean_artic_default)
        features.setdefault("articulation_score_ka", mean_artic_default)

        features.setdefault("word_production_quality", 0.7)
        features.setdefault("complexity_gradient", 0.0)
        features.setdefault("cross_word_consistency", 0.8)
        features.setdefault("cross_word_score_variance", 0.0)
        features.setdefault("cross_word_score_std", 0.0)
        features.setdefault("n_words_scored", 0)

        features.setdefault("task_profile_similarity", 1.0)
        features.setdefault("substitution_rate", 0.0)
        features.setdefault("mean_substitution_score", 0.0)

        features.setdefault("speech_duration_ratio_mean", 1.0)
        features.setdefault("speech_duration_ratio_consistency", 0.5)
        features.setdefault("mean_pattern_correlation", 0.8)
        features.setdefault("pataka_repetition_variability", 0.0)
        features.setdefault("simple_repetition_variability", 0.0)
        features.setdefault("complex_simple_variability_ratio", 1.0)
        features.setdefault("pataka_score_variability", 0.0)
        features.setdefault("simple_score_variability", 0.0)
        features.setdefault("complex_simple_score_var_ratio", 1.0)

        _b_score = features.get(
            "group_b_articulation_score", features.get("mean_articulation_score", 0.8)
        )
        _c_score = features.get(
            "group_c_articulation_score", features.get("word_production_quality", 0.8)
        )
        features["group_b_mean_score"] = _b_score
        features["group_c_mean_score"] = _c_score
        _impairment_threshold = 0.60
        features["group_bc_dissociation"] = float(_b_score - _c_score)
        features["group_b_intact"] = 1.0 if _b_score > _impairment_threshold else 0.0
        features["group_c_impaired"] = 1.0 if _c_score < _impairment_threshold else 0.0
        features.setdefault("word_cross_rep_consistency_mean", 0.8)

        return features

    def _evaluate_group_a(self, features: Dict[str, float]) -> List[ScreeningIndication]:
        """Evaluate non-speech facial tasks (Group A) for facial paresis and buccofacial apraxia.

        Facial paresis: persistent left-right asymmetry across tasks.

        Buccofacial apraxia: incorrect execution detected via cross-task
        profile matching (task substitution), low expected-task similarity,
        poor execution correctness, and/or inconsistent errors, all in the
        absence of meaningful facial asymmetry.

        Both conditions can be flagged simultaneously for mixed profiles.

        Buccofacial asymmetry gate: buccofacial apraxia is a motor-planning
        disorder with no structural asymmetry component. Facial paresis
        distorts movement geometry and causes the cross-task matching
        algorithm to report elevated substitution rates as an artefact.
        To avoid false positives, buccofacial detection is gated on
        mean_asym < 0.29. This threshold was validated against PAC7 pilot
        data where non-paresis profiles showed mean_asym around 0.27 and
        mild-to-moderate paresis profiles showed mean_asym >= 0.31.

        Neuroanatomical rationale: lower-face muscles (lip corners,
        commissures, chin) receive predominantly contralateral cortical input
        via the corticobulbar tract. Cortical or subcortical lesions produce
        contralateral lower-face weakness with asymmetric smile and commissure
        droop while sparing brow movement. Tasks targeting the lower face
        (Showing Teeth, Smiling Broadly, Puffing Cheeks) therefore carry
        stronger diagnostic weight for cortical paresis than brow-lift or
        eye-closure tasks.

        Oliveira et al. (2024) CMPB 258, 108195 achieved 82 % accuracy
        separating post-stroke patients from healthy controls using mouth-area
        AUs (AU7, AU20, AU23) with best sensitivity (91 %) on KISS and SPREAD
        tasks. Perioral AUs, the targets of Group A tasks 1 (lip purse), 2/3
        (smiling), and 8 (cheek puff), are the most diagnostically specific
        features for central paresis.
        https://doi.org/10.1016/j.cmpb.2024.108195

        Ruiter et al. (2023) achieved AUC 0.82 for diagnosis and 0.88 for
        severity in myasthenia gravis using facial video analysis; AU6 (cheek
        raiser) correlated with disease severity.
        https://doi.org/10.1002/acn3.51823

        Baig et al. (2023) achieved 98.93 % accuracy for binary paralysis
        classification using MobileNetV2 on MediaPipe 468-landmark meshes
        (hdl:10210/504453). Ozmen et al. (2025) confirmed multi-feature
        superiority (smile index ratio, commissure, teeth area, 86 % accuracy)
        over any single asymmetry metric.
        https://doi.org/10.1097/01.GOX.0001112148.28567.85
        """
        indications: List[ScreeningIndication] = []

        asym_threshold = self.thresholds.get("asymmetry", {})
        mean_asym = features.get("group_a_mean_asymmetry", features.get("mean_asymmetry_ratio", 0.0))
        asym_consistency = features.get("group_a_asym_consistency", features.get("asymmetry_consistency", 0.0))

        _ref_n = float(
            self.reference_asymmetry_stats.get("n", 0)
            if self.reference_asymmetry_stats else 0
        )

        if self.is_baseline_session and not self.has_reference_baseline:
            mild_t = asym_threshold.get("severe", 0.55)
            moderate_t = 0.65
            severe_t = 0.75
            consistency_t = 0.92
        else:
            _config_mild = asym_threshold.get("mild", 0.42)
            moderate_t = asym_threshold.get("moderate", 0.55)
            severe_t = asym_threshold.get("severe", 0.65)
            consistency_t = 0.90
            if self.reference_asymmetry_stats and _ref_n >= 5:
                _ref_a_mean = float(self.reference_asymmetry_stats.get("mean", 0.0))
                _ref_a_std  = float(self.reference_asymmetry_stats.get("std", 0.05))
                _sigma = 4.0 if self.is_ors_session else 3.0
                mild_t = max(_config_mild, _ref_a_mean + _sigma * _ref_a_std)
                _scale = mild_t / max(_config_mild, 0.01)
                moderate_t = min(0.75, asym_threshold.get("moderate", 0.55) * _scale)
                severe_t   = min(0.90, asym_threshold.get("severe",   0.65) * _scale)
            else:
                mild_t = _config_mild

        _test_yaw = features.get("rep_head_yaw_mean", 0.0)
        _ref_yaw  = getattr(self, "reference_head_yaw", None) or 0.0
        _yaw_offset_deg = abs(_test_yaw - _ref_yaw)
        _yaw_bias = abs(math.sin(math.radians(_yaw_offset_deg))) * 0.95
        mean_asym_for_threshold = max(0.0, mean_asym - _yaw_bias)

        self._group_a_mean_asym_corrected = mean_asym_for_threshold
        self._group_a_mild_t = mild_t

        asym_significant = True
        if self.reference_asymmetry_stats and _ref_n >= 5:
            ref_mean = float(self.reference_asymmetry_stats.get("mean", 0.0))
            ref_std  = float(self.reference_asymmetry_stats.get("std", 0.05))
            deviation = abs(mean_asym_for_threshold - ref_mean)
            asym_significant = (
                (deviation / max(ref_std, 0.001)) > 2.5
                if ref_std > 0
                else deviation > mild_t
            )

        if asym_significant and mean_asym_for_threshold > mild_t and asym_consistency > consistency_t:
            severity = (
                "severe" if mean_asym > severe_t
                else "moderate" if mean_asym > moderate_t
                else "mild"
            )
            conf = min(1.0, asym_consistency) * (
                0.7 if self.is_baseline_session else 1.0
            )
            dom_side = features.get("dominant_side", "symmetric")
            signed_val = features.get("mean_signed_asymmetry", 0.0)
            if dom_side == "right":
                side_note = "right side dominant (left may be weaker)"
            elif dom_side == "left":
                side_note = "left side dominant (right may be weaker)"
            else:
                side_note = "no consistent lateralization"
            indications.append(
                ScreeningIndication(
                    indication_type="facial_paresis",
                    severity=severity,
                    confidence=conf,
                    supporting_features={
                        "mean_asymmetry_ratio": mean_asym,
                        "asymmetry_consistency": asym_consistency,
                        "dominant_side": dom_side,
                        "mean_signed_asymmetry": signed_val,
                        "is_baseline_session": 1.0 if self.is_baseline_session else 0.0,
                    },
                    source_node="check_asymmetry_persistent",
                    description=(
                        f"Persistent left-right asymmetry detected "
                        f"(ratio: {mean_asym:.3f}, {side_note})"
                    ),
                    task_group=self.current_task_group,
                    task_id=self.current_task_id,
                )
            )

        substitution_rate = features.get("substitution_rate", 0.0)
        task_sim = features.get("task_profile_similarity", 1.0)

        _mean_asym_for_bucc = mean_asym_for_threshold if _yaw_bias > 0.01 else mean_asym
        if self.is_ors_session or not self.reference_asymmetry_stats or _ref_n < 5:
            has_low_asym = _mean_asym_for_bucc < 0.29
        else:
            _r_mean = float(self.reference_asymmetry_stats.get("mean", 0.0))
            _r_std  = float(self.reference_asymmetry_stats.get("std", 0.05))
            has_low_asym = _mean_asym_for_bucc < (_r_mean + 2.0 * _r_std)
        _paresis_already = any(
            i.indication_type == "facial_paresis" for i in indications
        )
        if (not has_low_asym and not _paresis_already
                and substitution_rate >= 0.50 and not self.is_ors_session):
            has_low_asym = True
        sub_score = features.get("mean_substitution_score", 0.0)
        exec_score = features.get("execution_correctness_score", 1.0)
        error_cons = features.get("error_consistency", 1.0)

        exec_t = self.thresholds.get("execution_correctness", {})
        sim_t = self.thresholds.get("task_profile_similarity", {}).get("poor", 0.40)

        evidence: List[Tuple[str, float]] = []
        support: Dict[str, float] = {"mean_asymmetry_ratio": mean_asym}

        _b_timing_compound = int(features.get("group_b_n_timing_drop", 0)) >= 3
        _sub_evidence_gate = 0.20 if _b_timing_compound else 0.30
        _n_a_eval = int(features.get("n_a_reps_evaluated", 9))
        if substitution_rate > _sub_evidence_gate and _n_a_eval >= 2:
            evidence.append(("substitution", substitution_rate))
            support["substitution_rate"] = substitution_rate
            support["mean_substitution_score"] = sub_score

        if task_sim < sim_t and task_sim < 0.35:
            evidence.append(("poor_profile_match", 1.0 - task_sim))
            support["task_profile_similarity"] = task_sim

        if exec_score < exec_t.get("acceptable", 0.60):
            evidence.append(("poor_execution", 1.0 - exec_score))
            support["execution_correctness_score"] = exec_score

        if error_cons < 0.5:
            evidence.append(("inconsistent_errors", 1.0 - error_cons))
            support["error_consistency"] = error_cons

        _has_substitution_ev = any(name == "substitution" for name, _ in evidence)
        _bucc_gate = 1 if _has_substitution_ev else 2
        if len(evidence) >= _bucc_gate and has_low_asym:
            max_strength = max(s for _, s in evidence)
            severity = (
                "severe" if max_strength > 0.5
                else "moderate" if max_strength > 0.3
                else "mild"
            )

            conf = min(1.0, 0.5 + len(evidence) * 0.15)
            if _has_substitution_ev:
                conf = min(1.0, conf + 0.15)
            if not has_low_asym:
                conf *= 0.7

            desc_parts: List[str] = []
            if any(n == "substitution" for n, _ in evidence):
                desc_parts.append(
                    f"task substitution ({substitution_rate:.0%})"
                )
            if any(n == "poor_profile_match" for n, _ in evidence):
                desc_parts.append(f"low profile match ({task_sim:.2f})")
            if any(n == "poor_execution" for n, _ in evidence):
                desc_parts.append(f"poor execution ({exec_score:.2f})")
            if any(n == "inconsistent_errors" for n, _ in evidence):
                desc_parts.append("inconsistent errors")

            indications.append(
                ScreeningIndication(
                    indication_type="buccofacial_apraxia",
                    severity=severity,
                    confidence=conf,
                    supporting_features=support,
                    source_node="check_execution_correctness_a",
                    description=(
                        f"Motor planning disruption: {'; '.join(desc_parts)}"
                    ),
                    task_group=self.current_task_group,
                    task_id=self.current_task_id,
                )
            )

        elif (
            self.is_ors_session
            and not any(i.indication_type == "facial_paresis" for i in indications)
            and substitution_rate > 0.50
            and task_sim < 0.40
            and sub_score < 0.35
        ):
            indications.append(ScreeningIndication(
                indication_type="buccofacial_apraxia",
                severity="mild",
                confidence=min(0.72, 0.45 + substitution_rate * 0.35),
                supporting_features={
                    "mean_asymmetry_ratio": mean_asym,
                    "substitution_rate": substitution_rate,
                    "task_profile_similarity": task_sim,
                },
                source_node="check_execution_correctness_a_ors",
                description=(
                    f"Motor planning disruption (ORS): high substitution "
                    f"({substitution_rate:.0%}) + low profile similarity ({task_sim:.2f})"
                ),
                task_group=self.current_task_group,
                task_id=self.current_task_id,
            ))

        return indications

    def _evaluate_group_b(self, features: Dict[str, float]) -> List[ScreeningIndication]:
        """Evaluate speech articulation tasks (Group B) for dysarthria and speech apraxia.

        Dysarthria: uniform articulation degradation and/or uniform slowness
        across all syllable tasks, with pattern preservation (same movement
        shape, but slower/weaker).

        Speech apraxia: selective failure on complex sequencing (pa-ta-ka)
        while simple syllable tasks remain intact, with high repetition-to-
        repetition variability on the complex task.

        Both conditions can be flagged simultaneously for mixed profiles.

        Clinical validation for Group B detection logic
        ------------------------------------------------
        Collee et al. (2022) Cancers 14, 5466 showed that intraoperative
        production errors (dysarthria, stuttering) are independent predictors
        of acute postoperative language deficits (OR=2.06), establishing
        production error detection as the primary intraoperative Group B goal.
        https://doi.org/10.3390/cancers14215466

        Allison et al. (2022) AJSLP 31, 1682 confirmed that DDK rate,
        temporal variability (Tsd), and spatiotemporal index (STI) each
        achieve 88 % sensitivity and specificity for motor speech involvement,
        supporting the kinematic thresholds applied in this group.
        https://doi.org/10.1044/2022_AJSLP-21-00241

        Duffy (2013) Motor Speech Disorders provides the clinical framework
        for distinguishing dysarthria (uniform degradation with preserved
        movement pattern) from apraxia of speech (selective complex-task
        failure with high repetition variability).
        """
        indications: List[ScreeningIndication] = []

        artic_threshold = self.thresholds.get("articulation", {})
        artic_impairment = features.get("articulation_impairment_consistency", 0.0)
        mean_artic = features.get("group_b_articulation_score", features.get("mean_articulation_score", 1.0))
        pataka_score = features.get("articulation_score_pataka", mean_artic)
        simple_mean = features.get("simple_syllable_mean", mean_artic)

        duration_ratio = features.get("speech_duration_ratio_mean", 1.0)
        duration_consistency = features.get("speech_duration_ratio_consistency", 0.5)
        pattern_corr = features.get("mean_pattern_correlation", 0.8)
        pataka_var = features.get("pataka_repetition_variability", 0.0)
        simple_var = features.get("simple_repetition_variability", 0.0)
        var_ratio = features.get("complex_simple_variability_ratio", 1.0)
        pataka_score_var = features.get("pataka_score_variability", 0.0)
        simple_score_var = features.get("simple_score_variability", 0.0)
        score_var_ratio   = features.get("complex_simple_score_var_ratio", 1.0)
        _combined_var_elevated = (
            (var_ratio > 1.3 and pataka_var > 0.08)
            or (score_var_ratio > 1.4 and pataka_score_var > 0.05)
        )

        dur_t = self.thresholds.get("duration_ratio", {})
        slow_t = dur_t.get("slow", 1.15)

        if self.is_baseline_session and not self.has_reference_baseline:
            if mean_artic < artic_threshold.get("poor", 0.40) and artic_impairment > 0.7:
                severity = "severe" if mean_artic < 0.20 else "moderate"
                indications.append(
                    ScreeningIndication(
                        indication_type="dysarthria",
                        severity=severity,
                        confidence=artic_impairment * 0.7,
                        supporting_features={
                            "articulation_impairment_consistency": artic_impairment,
                            "mean_articulation_score": mean_artic,
                            "is_baseline_session": 1.0,
                        },
                        source_node="check_articulation_all_syllables",
                        description=(
                            f"Severe articulation impairment at baseline "
                            f"(score: {mean_artic:.3f})"
                        ),
                        task_group=self.current_task_group,
                        task_id=self.current_task_id,
                    )
                )
            return indications

        ref = self.reference_articulation
        has_ref = ref is not None and "mean_articulation_score" in ref

        if has_ref:
            ref_mean = ref["mean_articulation_score"]
            ref_pataka = ref.get("articulation_score_pataka", ref_mean)
            ref_simple = ref.get("simple_syllable_mean", ref_mean)

            delta_mean = mean_artic - ref_mean
            delta_pataka = pataka_score - ref_pataka
            delta_simple = simple_mean - ref_simple

            features["delta_mean_articulation"] = delta_mean
            features["delta_articulation_pataka"] = delta_pataka
            features["delta_articulation_simple"] = delta_simple

            dev_t = self.thresholds.get("articulation_deviation", {})
            mild_drop = dev_t.get("mild", 0.38)
            moderate_drop = dev_t.get("moderate", 0.48)
            severe_drop = dev_t.get("severe", 0.58)
            apraxia_selective_drop = dev_t.get("apraxia_selective", 0.10)

            _uniform_drop_noise = (
                delta_simple < -0.15
                and delta_pataka < -0.15
                and abs(delta_pataka) < abs(delta_simple) * 1.4
            )

            dysarthria_evidence: List[Tuple[str, float]] = []
            dysarthria_support: Dict[str, float] = {
                "mean_articulation_score": mean_artic,
                "reference_mean_articulation": ref_mean,
                "delta_mean_articulation": delta_mean,
            }

            if delta_mean < -mild_drop and artic_impairment > 0.5:
                dysarthria_evidence.append(("articulation_decline", abs(delta_mean)))

            if duration_ratio > slow_t and duration_consistency > 0.5:
                dysarthria_evidence.append(("uniform_slowness", duration_ratio - 1.0))
                dysarthria_support["speech_duration_ratio_mean"] = duration_ratio
                dysarthria_support["speech_duration_ratio_consistency"] = duration_consistency

            if pattern_corr > 0.5 and delta_mean < -mild_drop:
                dysarthria_evidence.append(("pattern_preserved_degraded", pattern_corr))
                dysarthria_support["mean_pattern_correlation"] = pattern_corr

            simple_also_dropped = delta_simple < -mild_drop * 0.5
            if delta_mean < -mild_drop and simple_also_dropped:
                dysarthria_evidence.append(("uniform_across_tasks", artic_impairment))
                dysarthria_support["articulation_impairment_consistency"] = artic_impairment

            _b_t_dev = features.get("group_b_timing_deviation", 0.0)
            _b_s_dev = features.get("group_b_smoothness_deviation", 0.0)
            _b_a_dev = features.get("group_b_amplitude_deviation", 0.0)
            _b_t_n   = int(features.get("group_b_n_timing_drop", 0))
            _b_s_n   = int(features.get("group_b_n_smoothness_drop", 0))
            _b_a_n   = int(features.get("group_b_n_amplitude_drop", 0))
            if _b_t_dev < -0.18 and (_b_t_n >= 3 or (_b_t_n >= 2 and len(dysarthria_evidence) > 0)):
                dysarthria_evidence.append(("timing_drop", abs(_b_t_dev)))
                dysarthria_support["group_b_timing_deviation"] = _b_t_dev
                dysarthria_support["group_b_n_timing_drop"] = float(_b_t_n)
            if _b_s_dev < -0.06 and _b_s_n >= 2:
                dysarthria_evidence.append(("smoothness_drop", abs(_b_s_dev)))
                dysarthria_support["group_b_smoothness_deviation"] = _b_s_dev
            if _b_a_dev < -0.18 and _b_a_n >= 2:
                dysarthria_evidence.append(("amplitude_drop", abs(_b_a_dev)))
                dysarthria_support["group_b_amplitude_deviation"] = _b_a_dev
            _b_dur_ratio = float(features.get("group_b_mean_duration_ratio") or 1.0)
            _b_n_slow = int(features.get("group_b_n_slow_tasks") or 0)
            if _b_dur_ratio > 1.20 and _b_n_slow >= 2 and delta_mean < 0.015:
                dysarthria_evidence.append(("speech_slowing_vs_baseline", _b_dur_ratio - 1.0))
                dysarthria_support["group_b_mean_duration_ratio"] = _b_dur_ratio
                dysarthria_support["group_b_n_slow_tasks"] = float(_b_n_slow)

            if dysarthria_evidence:
                max_delta = max(abs(delta_mean), abs(delta_simple))
                _dur_excess = max(0.0, _b_dur_ratio - 1.0) if _b_n_slow >= 2 else 0.0
                _comp_max = max(abs(_b_t_dev), abs(_b_s_dev), abs(_b_a_dev), _dur_excess)
                _comp_sev = (
                    "severe"   if _comp_max > 0.50
                    else "moderate" if _comp_max > 0.30
                    else "mild"    if _comp_max > 0.12
                    else None
                )
                severity = (
                    "severe"   if max_delta > severe_drop
                    else "moderate" if max_delta > moderate_drop
                    else "mild"
                )
                _sev_rank = {"mild": 1, "moderate": 2, "severe": 3}
                if _comp_sev and _sev_rank.get(_comp_sev, 0) > _sev_rank.get(severity, 0):
                    severity = _comp_sev
                conf = min(1.0, artic_impairment * 0.9 + 0.1)
                if len(dysarthria_evidence) >= 3:
                    conf = min(1.0, conf + 0.1)
                _n_comp_ev = sum(1 for ev, _ in dysarthria_evidence if ev.endswith("_drop"))
                if _n_comp_ev >= 1:
                    conf = min(1.0, conf + 0.07 * _n_comp_ev)
                _comp_only = all(
                    ev.endswith("_drop") or ev == "speech_slowing_vs_baseline"
                    for ev, _ in dysarthria_evidence
                )
                if _comp_only:
                    _c_floor = 0.40 + min(0.35, _comp_max * 0.75) + 0.05 * min(_b_t_n, 4)
                    conf = max(conf, min(1.0, _c_floor))

                desc_parts: List[str] = []
                if any(n == "articulation_decline" for n, _ in dysarthria_evidence):
                    desc_parts.append(
                        f"articulation declined "
                        f"({ref_mean:.2f}\u2192{mean_artic:.2f})"
                    )
                if any(n == "uniform_slowness" for n, _ in dysarthria_evidence):
                    desc_parts.append(
                        f"uniformly slower ({duration_ratio:.1f}x)"
                    )
                if any(n == "pattern_preserved_degraded" for n, _ in dysarthria_evidence):
                    desc_parts.append(
                        f"pattern preserved (r={pattern_corr:.2f})"
                    )
                if any(n == "timing_drop" for n, _ in dysarthria_evidence):
                    desc_parts.append(
                        f"timing declined ({_b_t_dev:+.2f}, {_b_t_n} tasks)"
                    )
                if any(n == "smoothness_drop" for n, _ in dysarthria_evidence):
                    desc_parts.append(
                        f"smoothness declined ({_b_s_dev:+.2f})"
                    )
                if any(n == "amplitude_drop" for n, _ in dysarthria_evidence):
                    desc_parts.append(
                        f"amplitude declined ({_b_a_dev:+.2f}, {_b_a_n} tasks)"
                    )
                if any(n == "speech_slowing_vs_baseline" for n, _ in dysarthria_evidence):
                    desc_parts.append(
                        f"speech slowing vs baseline ({_b_dur_ratio:.2f}x, {_b_n_slow} tasks)"
                    )

                indications.append(
                    ScreeningIndication(
                        indication_type="dysarthria",
                        severity=severity,
                        confidence=conf,
                        supporting_features=dysarthria_support,
                        source_node="check_articulation_all_syllables",
                        description=(
                            f"Consistent articulation impairment: "
                            f"{'; '.join(desc_parts)}"
                        ),
                        task_group=self.current_task_group,
                        task_id=self.current_task_id,
                    )
                )

            apraxia_evidence: List[Tuple[str, float]] = []
            apraxia_support: Dict[str, float] = {
                "articulation_score_pataka": pataka_score,
                "simple_syllable_mean": simple_mean,
            }

            ref_dissociation = ref_simple - ref_pataka
            test_dissociation = simple_mean - pataka_score
            dissociation_change = test_dissociation - ref_dissociation

            _dysarthria_already_in_b = any(
                ind.indication_type == "dysarthria" for ind in indications
            )
            if (not _uniform_drop_noise and not _dysarthria_already_in_b
                    and delta_pataka < -apraxia_selective_drop and dissociation_change > 0.02):
                apraxia_evidence.append(
                    ("complex_simple_dissociation", dissociation_change)
                )
                apraxia_support["dissociation_change"] = dissociation_change
                apraxia_support["reference_pataka"] = ref_pataka
                apraxia_support["delta_pataka"] = delta_pataka
                apraxia_support["delta_simple"] = delta_simple

            if (not _uniform_drop_noise and not _dysarthria_already_in_b
                    and delta_pataka < 0
                    and delta_pataka < delta_simple - mild_drop * 0.5):
                _disproportionate = delta_simple - delta_pataka
                apraxia_evidence.append(
                    ("disproportionate_complex_decline", _disproportionate)
                )
                apraxia_support["delta_pataka"] = delta_pataka
                apraxia_support["delta_simple"] = delta_simple

            if _combined_var_elevated:
                _var_strength = max(var_ratio, score_var_ratio)
                apraxia_evidence.append(
                    ("complex_task_inconsistency", _var_strength)
                )
                apraxia_support["pataka_repetition_variability"] = pataka_var
                apraxia_support["simple_repetition_variability"] = simple_var
                apraxia_support["complex_simple_variability_ratio"] = var_ratio
                apraxia_support["complex_simple_score_var_ratio"] = score_var_ratio

            _b4_ratio_vs_ref = 1.0
            if has_ref:
                _b4_ratio_vs_ref = float(features.get("b4_simple_act_ratio_vs_ref") or 1.0)
            _b4_ratio_has_quality = (
                pataka_score < simple_mean - 0.04
                or pataka_var > 0.08
            )
            _b4_act_ratio_within = float(features.get("b4_simple_act_ratio", 0.0))
            if (has_ref and _b4_ratio_vs_ref > 1.7 and not self.is_ors_session
                    and _b4_ratio_has_quality
                    and _b4_act_ratio_within > 1.0):
                apraxia_evidence.append(
                    ("b4_excess_effort_ratio", _b4_ratio_vs_ref - 1.0)
                )
                apraxia_support["b4_simple_act_ratio_vs_ref"] = _b4_ratio_vs_ref

            if self.is_ors_session and has_ref:
                _ors_dissoc = pataka_score - simple_mean
                _uniform_noise_ors = (
                    delta_simple < -0.15
                    and delta_pataka < -0.15
                    and abs(delta_pataka) < abs(delta_simple) * 1.4
                )
                if (not _uniform_noise_ors
                        and _ors_dissoc < -0.05
                        and delta_simple > -0.05
                        and not _dysarthria_already_in_b):
                    apraxia_evidence.append(
                        ("ors_score_dissociation", abs(_ors_dissoc))
                    )
                    apraxia_support["ors_pataka_dissociation"] = _ors_dissoc
                    apraxia_support["ors_delta_simple"] = delta_simple

            self._ors_b_apraxia_found = any(
                name == "ors_score_dissociation" for name, _ in apraxia_evidence
            )

            if apraxia_evidence:
                max_strength = max(s for _, s in apraxia_evidence)
                _only_effort = (
                    len(apraxia_evidence) == 1
                    and apraxia_evidence[0][0] == "b4_excess_effort_ratio"
                )
                if _only_effort:
                    severity = (
                        "severe"   if max_strength > 1.50
                        else "moderate" if max_strength > 0.70
                        else "mild"
                    )
                else:
                    severity = (
                        "severe"   if max_strength > 0.30
                        else "moderate" if max_strength > 0.15
                        else "mild"
                    )
                conf = 0.65
                if len(apraxia_evidence) >= 2:
                    conf = 0.80

                desc_parts = []
                if any(n == "complex_simple_dissociation" for n, _ in apraxia_evidence):
                    desc_parts.append(
                        f"complex task declined "
                        f"({ref_pataka:.2f}\u2192{pataka_score:.2f}) "
                        f"while simple preserved"
                    )
                if any(n == "complex_task_inconsistency" for n, _ in apraxia_evidence):
                    desc_parts.append(
                        f"complex variability {var_ratio:.1f}x higher "
                        f"than simple"
                    )
                if any(n == "b4_excess_effort_ratio" for n, _ in apraxia_evidence):
                    desc_parts.append(
                        f"B4 excess effort ({_b4_ratio_vs_ref:.2f}x reference ratio)"
                    )

                indications.append(
                    ScreeningIndication(
                        indication_type="speech_apraxia",
                        severity=severity,
                        confidence=conf,
                        supporting_features=apraxia_support,
                        source_node="check_articulation_complex_only",
                        description=(
                            f"Sequencing/planning disruption: "
                            f"{'; '.join(desc_parts)}"
                        ),
                        task_group=self.current_task_group,
                        task_id=self.current_task_id,
                    )
                )

            return indications

        conf_penalty = 0.8

        if artic_impairment > 0.7 and mean_artic < artic_threshold.get("acceptable", 0.60):
            severity = "mild"
            if mean_artic < artic_threshold.get("poor", 0.40):
                severity = "severe" if mean_artic < 0.20 else "moderate"

            dysarthria_conf = artic_impairment * conf_penalty
            if duration_ratio > slow_t:
                dysarthria_conf = min(1.0, dysarthria_conf + 0.1)

            indications.append(
                ScreeningIndication(
                    indication_type="dysarthria",
                    severity=severity,
                    confidence=dysarthria_conf,
                    supporting_features={
                        "articulation_impairment_consistency": artic_impairment,
                        "mean_articulation_score": mean_artic,
                        "speech_duration_ratio_mean": duration_ratio,
                        "has_reference": 0.0,
                    },
                    source_node="check_articulation_all_syllables",
                    description=(
                        f"Consistent articulation impairment "
                        f"(score: {mean_artic:.3f}, no baseline reference)"
                    ),
                    task_group=self.current_task_group,
                    task_id=self.current_task_id,
                )
            )

        diff_no_ref = simple_mean - pataka_score
        if diff_no_ref > 0.08 and _combined_var_elevated:
            _strength = diff_no_ref
            severity = (
                "severe" if _strength > 0.30
                else "moderate" if _strength > 0.15
                else "mild"
            )
            apraxia_conf = 0.60 * conf_penalty
            if _combined_var_elevated:
                apraxia_conf = min(1.0, apraxia_conf + 0.12)
            indications.append(
                ScreeningIndication(
                    indication_type="speech_apraxia",
                    severity=severity,
                    confidence=apraxia_conf,
                    supporting_features={
                        "articulation_score_pataka": pataka_score,
                        "simple_syllable_mean": simple_mean,
                        "pataka_simple_difference": diff_no_ref,
                        "complex_simple_variability_ratio": var_ratio,
                        "complex_simple_score_var_ratio": score_var_ratio,
                        "has_reference": 0.0,
                    },
                    source_node="check_articulation_complex_only",
                    description=(
                        f"Complex-task impairment with inconsistency "
                        f"(Pa-Ta-Ka: {pataka_score:.3f}, gap: {diff_no_ref:.3f}, "
                        f"var ratio: {max(var_ratio, score_var_ratio):.2f})"
                    ),
                    task_group=self.current_task_group,
                    task_id=self.current_task_id,
                )
            )
        elif diff_no_ref > 0.12:
            severity = (
                "severe" if diff_no_ref > 0.35
                else "moderate" if diff_no_ref > 0.20
                else "mild"
            )
            apraxia_conf = 0.55 * conf_penalty
            indications.append(
                ScreeningIndication(
                    indication_type="speech_apraxia",
                    severity=severity,
                    confidence=apraxia_conf,
                    supporting_features={
                        "articulation_score_pataka": pataka_score,
                        "simple_syllable_mean": simple_mean,
                        "pataka_simple_difference": diff_no_ref,
                        "has_reference": 0.0,
                    },
                    source_node="check_articulation_complex_only",
                    description=(
                        f"Complex-task impairment "
                        f"(Pa-Ta-Ka: {pataka_score:.3f}, no baseline reference)"
                    ),
                    task_group=self.current_task_group,
                    task_id=self.current_task_id,
                )
            )
        elif _combined_var_elevated and pataka_var > 0.15:
            _var_strength = max(var_ratio, score_var_ratio)
            apraxia_conf = min(0.70, 0.45 * conf_penalty + _var_strength * 0.06)
            indications.append(
                ScreeningIndication(
                    indication_type="speech_apraxia",
                    severity="mild",
                    confidence=apraxia_conf,
                    supporting_features={
                        "pataka_repetition_variability": pataka_var,
                        "simple_repetition_variability": simple_var,
                        "complex_simple_variability_ratio": var_ratio,
                        "complex_simple_score_var_ratio": score_var_ratio,
                        "has_reference": 0.0,
                    },
                    source_node="check_variability_complex_only",
                    description=(
                        f"High complex-task inconsistency suggests articulatory "
                        f"planning difficulty (duration var ratio: {var_ratio:.2f}, "
                        f"score var ratio: {score_var_ratio:.2f})"
                    ),
                    task_group=self.current_task_group,
                    task_id=self.current_task_id,
                )
            )

        return indications

    def _evaluate_group_c(self, features: Dict[str, float]) -> List[ScreeningIndication]:
        """Evaluate word production tasks (Group C) for phonological disorder and speech apraxia.

        Phonological disorder: consistent production-level errors in Group C
        while Group B remains intact, with stable substitution patterns across
        repetitions of the same word and no complexity-driven gradient shift.

        Speech apraxia (from C): inconsistent word production errors that
        vary across repetitions, possibly with disproportionate difficulty
        at higher complexity levels.

        Both conditions can be flagged for mixed profiles.

        Clinical validation
        -------------------
        Collee et al. (2022) Cancers 14, 5466 showed that spontaneous speech
        deficits arose after ALL intraoperative error categories — making
        continuous Group C monitoring (word production quality under natural
        speaking load) the most globally sensitive post-resection marker.
        https://doi.org/10.3390/cancers14215466
        """
        indications: List[ScreeningIndication] = []

        wpq = features.get("word_production_quality", 0.0)
        gradient = features.get("complexity_gradient", 0.0)
        consistency = features.get("cross_word_consistency", 0.0)
        n_words = features.get("n_words_scored", 0)

        word_cross_rep = features.get("word_cross_rep_consistency_mean", 0.8)
        group_b_score = features.get("group_b_mean_score", 0.8)
        group_b_intact_flag = features.get("group_b_intact", 1.0)
        bc_dissociation = features.get("group_bc_dissociation", 0.0)
        cross_word_score_std = features.get("cross_word_score_std", 0.0)

        if n_words < 2:
            return indications

        dev_t = self.thresholds.get("word_production_deviation", {})
        mild_drop = dev_t.get("mild", 0.10)
        moderate_drop = dev_t.get("moderate", 0.20)
        severe_drop = dev_t.get("severe", 0.30)

        abs_t = self.thresholds.get("word_production", {})
        poor = abs_t.get("poor", 0.40)
        acceptable = abs_t.get("acceptable", 0.60)

        if self.is_baseline_session and not self.has_reference_baseline:
            if wpq < poor and consistency > 0.7:
                severity = (
                    "severe" if wpq < 0.25
                    else "moderate" if wpq < 0.35
                    else "mild"
                )
                indications.append(
                    ScreeningIndication(
                        indication_type="phonological_disorder",
                        severity=severity,
                        confidence=consistency * 0.7,
                        supporting_features={
                            "word_production_quality": wpq,
                            "cross_word_consistency": consistency,
                            "complexity_gradient": gradient,
                        },
                        source_node="check_word_production_baseline",
                        description=(
                            f"Consistent word production impairment at baseline "
                            f"(quality: {wpq:.3f})"
                        ),
                        task_group=self.current_task_group,
                        task_id=self.current_task_id,
                    )
                )
            return indications

        if self.has_reference_baseline and self.reference_articulation:
            ref_wpq = self.reference_articulation.get("word_production_quality", None)
            if ref_wpq is not None:
                ref_gradient = self.reference_articulation.get(
                    "complexity_gradient", 0.0
                )
                ref_consistency = self.reference_articulation.get(
                    "cross_word_consistency", 0.8
                )

                delta_wpq = wpq - ref_wpq
                delta_gradient = gradient - ref_gradient
                delta_consistency = consistency - ref_consistency

                features["delta_word_production_quality"] = delta_wpq
                features["delta_complexity_gradient"] = delta_gradient
                features["delta_cross_word_consistency"] = delta_consistency

                if delta_wpq < -mild_drop:
                    phono_evidence: List[Tuple[str, float]] = []
                    phono_support: Dict[str, float] = {
                        "word_production_quality": wpq,
                        "reference_word_production_quality": ref_wpq,
                        "delta_word_production_quality": delta_wpq,
                    }

                    if group_b_intact_flag > 0.5 and bc_dissociation > 0.1:
                        phono_evidence.append(
                            ("b_intact_c_impaired", bc_dissociation)
                        )
                        phono_support["group_b_mean_score"] = group_b_score
                        phono_support["group_bc_dissociation"] = bc_dissociation

                    if word_cross_rep > 0.6 and consistency > 0.5:
                        phono_evidence.append(
                            ("consistent_error_pattern", word_cross_rep)
                        )
                        phono_support["word_cross_rep_consistency_mean"] = (
                            word_cross_rep
                        )
                        phono_support["cross_word_consistency"] = consistency

                    if delta_gradient > -0.20:
                        phono_evidence.append(
                            ("no_complexity_gradient_shift", 1.0)
                        )
                        phono_support["complexity_gradient"] = gradient

                    if phono_evidence and group_b_intact_flag > 0.5 and (
                        consistency > 0.5
                        or word_cross_rep > 0.6
                    ):
                        severity = (
                            "severe" if delta_wpq < -severe_drop
                            else "moderate" if delta_wpq < -moderate_drop
                            else "mild"
                        )
                        conf = min(1.0, 0.5 + len(phono_evidence) * 0.15)
                        if group_b_intact_flag > 0.5:
                            conf = min(1.0, conf + 0.1)

                        desc_parts: List[str] = [
                            f"word production declined "
                            f"({ref_wpq:.2f}\u2192{wpq:.2f})"
                        ]
                        if any(
                            n == "b_intact_c_impaired"
                            for n, _ in phono_evidence
                        ):
                            desc_parts.append(
                                f"Group B intact ({group_b_score:.2f})"
                            )
                        if any(
                            n == "consistent_error_pattern"
                            for n, _ in phono_evidence
                        ):
                            desc_parts.append(
                                f"consistent pattern ({word_cross_rep:.2f})"
                            )

                        indications.append(
                            ScreeningIndication(
                                indication_type="phonological_disorder",
                                severity=severity,
                                confidence=conf,
                                supporting_features=phono_support,
                                source_node=(
                                    "check_word_production_decline_consistent"
                                ),
                                description=(
                                    f"Consistent phonological impairment: "
                                    f"{'; '.join(desc_parts)}"
                                ),
                                task_group=self.current_task_group,
                                task_id=self.current_task_id,
                            )
                        )

                    apraxia_evidence: List[Tuple[str, float]] = []
                    apraxia_support: Dict[str, float] = {
                        "word_production_quality": wpq,
                        "reference_word_production_quality": ref_wpq,
                        "delta_word_production_quality": delta_wpq,
                    }

                    if word_cross_rep < 0.5:
                        apraxia_evidence.append(
                            ("inconsistent_words", 1.0 - word_cross_rep)
                        )
                        apraxia_support["word_cross_rep_consistency_mean"] = (
                            word_cross_rep
                        )

                    if delta_gradient < -0.20:
                        apraxia_evidence.append(
                            ("complexity_shift", abs(delta_gradient))
                        )
                        apraxia_support["delta_complexity_gradient"] = (
                            delta_gradient
                        )

                    if delta_consistency < -0.15:
                        apraxia_evidence.append(
                            ("consistency_dropped", abs(delta_consistency))
                        )
                        apraxia_support["delta_cross_word_consistency"] = (
                            delta_consistency
                        )

                    if apraxia_evidence:
                        severity = (
                            "severe" if delta_wpq < -severe_drop
                            else "moderate" if delta_wpq < -moderate_drop
                            else "mild"
                        )
                        conf = 0.55
                        if len(apraxia_evidence) >= 2:
                            conf = 0.70

                        desc_parts = [
                            f"word production declined "
                            f"({ref_wpq:.2f}\u2192{wpq:.2f})"
                        ]
                        if any(
                            n == "inconsistent_words"
                            for n, _ in apraxia_evidence
                        ):
                            desc_parts.append(
                                f"inconsistent across repetitions "
                                f"({word_cross_rep:.2f})"
                            )
                        if any(
                            n == "complexity_shift"
                            for n, _ in apraxia_evidence
                        ):
                            desc_parts.append(
                                f"complexity-dependent "
                                f"(\u0394gradient: {delta_gradient:+.2f})"
                            )

                        indications.append(
                            ScreeningIndication(
                                indication_type="speech_apraxia",
                                severity=severity,
                                confidence=conf,
                                supporting_features=apraxia_support,
                                source_node=(
                                    "check_word_production_decline_inconsistent"
                                ),
                                description=(
                                    f"Inconsistent word production errors: "
                                    f"{'; '.join(desc_parts)}"
                                ),
                                task_group=self.current_task_group,
                                task_id=self.current_task_id,
                            )
                        )

                ref_cross_word_std = self.reference_articulation.get(
                    "cross_word_score_std", None
                )
                _elevated_variance = cross_word_score_std > 0.065
                if ref_cross_word_std is not None:
                    _elevated_variance = cross_word_score_std > ref_cross_word_std * 1.5
                _phono_already = any(
                    ind.indication_type == "phonological_disorder"
                    for ind in indications
                )
                if (
                    _elevated_variance
                    and not _phono_already
                    and group_b_intact_flag > 0.5
                    and delta_wpq < -mild_drop
                ):
                    _sel_severity = (
                        "moderate" if cross_word_score_std > 0.10
                        else "mild"
                    )
                    indications.append(
                        ScreeningIndication(
                            indication_type="phonological_disorder",
                            severity=_sel_severity,
                            confidence=min(0.75, 0.50 + cross_word_score_std * 2.0),
                            supporting_features={
                                "cross_word_score_std": cross_word_score_std,
                                "cross_word_consistency": consistency,
                                "group_b_intact": group_b_intact_flag,
                                "word_production_quality": wpq,
                            },
                            source_node="check_selective_word_errors",
                            description=(
                                f"Selective word-level errors: high cross-word score "
                                f"variability (std={cross_word_score_std:.3f}) with "
                                f"Group B intact — consistent with phonological disorder"
                            ),
                            task_group=self.current_task_group,
                            task_id=self.current_task_id,
                        )
                    )

                _c_t_dev = features.get("group_c_timing_deviation", 0.0)
                _c_t_n   = int(features.get("group_c_n_timing_drop", 0))
                _c_s_dev = features.get("group_c_smoothness_deviation", 0.0)
                _c_s_n   = int(features.get("group_c_n_smoothness_drop", 0))
                _c_phono_already = any(
                    ind.indication_type == "phonological_disorder" for ind in indications
                )
                _b_n_timing_drop = int(features.get("group_b_n_timing_drop", 0))
                _b_effectively_intact = group_b_intact_flag > 0.5 and _b_n_timing_drop <= 1
                _c_quality_declined = float(features.get("group_c_score_deviation", 0.0)) < -0.06
                if _c_t_dev < -0.15 and _c_t_n >= 3 and not _c_phono_already:
                    _c_sev = "moderate" if _c_t_dev < -0.25 else "mild"
                    _c_ind_type = (
                        "phonological_disorder" if (_b_effectively_intact and _c_quality_declined)
                        else "dysarthria"
                    )
                    if _c_ind_type == "dysarthria":
                        self._c_dysarthria_found = True
                    indications.append(
                        ScreeningIndication(
                            indication_type=_c_ind_type,
                            severity=_c_sev,
                            confidence=0.58,
                            supporting_features={
                                "group_c_timing_deviation": _c_t_dev,
                                "group_c_n_timing_drop": float(_c_t_n),
                                "word_production_quality": wpq,
                            },
                            source_node="check_component_drop_c_timing",
                            description=(
                                f"Word production timing declined "
                                f"({_c_t_dev:+.2f} vs reference; {_c_t_n} tasks affected)"
                            ),
                            task_group=self.current_task_group,
                            task_id=self.current_task_id,
                        )
                    )
                if _c_s_dev < -0.06 and _c_s_n >= 3 and not _c_phono_already:
                    indications.append(
                        ScreeningIndication(
                            indication_type="phonological_disorder",
                            severity="mild",
                            confidence=0.52,
                            supporting_features={
                                "group_c_smoothness_deviation": _c_s_dev,
                                "group_c_n_smoothness_drop": float(_c_s_n),
                            },
                            source_node="check_component_drop_c_smoothness",
                            description=(
                                f"Word production smoothness declined "
                                f"({_c_s_dev:+.2f} vs reference; {_c_s_n} tasks affected)"
                            ),
                            task_group=self.current_task_group,
                            task_id=self.current_task_id,
                        )
                    )

                _n_c_cpx_extreme = int(features.get("n_c_complex_extreme_amp_drop", 0))
                _c_phono_already2 = any(
                    ind.indication_type == "phonological_disorder" for ind in indications
                )
                _b4_selective_hint = getattr(self, "_b4_selective_hint", False)
                _b4_ratio_raw = getattr(self, "_b4_ratio_raw", 1.0)
                _cpx_extreme_gate = 2 if self.is_ors_session else 1
                _delta_wpq_c = features.get("delta_word_production_quality", 0.0)
                _cpx_ors_single_evidence = (
                    self.is_ors_session
                    and _n_c_cpx_extreme == 1
                    and _delta_wpq_c < -0.10
                    and _b4_ratio_raw > 1.50
                )
                _cpx_ambiguous_b4_ceiling = 1.55 if self.is_ors_session else 1.15
                _cpx_ambiguous = wpq < 0.74 and _b4_ratio_raw < _cpx_ambiguous_b4_ceiling
                _cpx_phono_evidence_strong = (
                    _cpx_ambiguous
                    and _delta_wpq_c < -0.15
                    and _b4_ratio_raw > 1.40
                )
                _c_max_dtw_artifact = getattr(self, "_max_c_task_dtw_precomputed", 0.0) > 0.50
                if ((_n_c_cpx_extreme >= _cpx_extreme_gate or _cpx_ors_single_evidence)
                        and _b_effectively_intact
                        and not _c_phono_already2 and not _b4_selective_hint
                        and not _c_max_dtw_artifact
                        and (not _cpx_ambiguous or _cpx_phono_evidence_strong)):
                    _cpx_sev = "moderate" if _n_c_cpx_extreme >= 2 else "mild"
                    indications.append(
                        ScreeningIndication(
                            indication_type="phonological_disorder",
                            severity=_cpx_sev,
                            confidence=min(0.75, 0.55 + _n_c_cpx_extreme * 0.10),
                            supporting_features={
                                "n_c_complex_extreme_amp_drop": float(_n_c_cpx_extreme),
                                "group_b_intact": group_b_intact_flag,
                                "word_production_quality": wpq,
                            },
                            source_node="check_c_complex_kinematic_shift",
                            description=(
                                f"Complex word kinematic pattern shift: "
                                f"{_n_c_cpx_extreme} complex C task(s) with extreme "
                                f"amplitude deviation vs reference — consistent with "
                                f"phonological substitution"
                            ),
                            task_group=self.current_task_group,
                            task_id=self.current_task_id,
                        )
                    )

                _apraxia_already2 = any(
                    ind.indication_type == "speech_apraxia" for ind in indications
                )
                _ors_cpx_apraxia = self.is_ors_session and _n_c_cpx_extreme >= 3
                if (_n_c_cpx_extreme >= 1 and _b_effectively_intact
                        and not _c_phono_already2 and not _b4_selective_hint
                        and _cpx_ambiguous
                        and (not self.is_ors_session or _ors_cpx_apraxia)
                        and not _apraxia_already2):
                    indications.append(
                        ScreeningIndication(
                            indication_type="speech_apraxia",
                            severity="mild",
                            confidence=min(0.60, 0.40 + _n_c_cpx_extreme * 0.10),
                            supporting_features={
                                "n_c_complex_extreme_amp_drop": float(_n_c_cpx_extreme),
                                "word_production_quality": wpq,
                                "b4_vs_simple_ratio": _b4_ratio_raw,
                            },
                            source_node="check_c_complex_kinematic_apraxia",
                            description=(
                                f"Complex word amplitude deviation ({_n_c_cpx_extreme} task(s)); "
                                f"reduced WPQ ({wpq:.2f}) with non-elevated B4 ({_b4_ratio_raw:.2f}) "
                                f"— motor planning load pattern consistent with speech apraxia"
                            ),
                            task_group=self.current_task_group,
                            task_id=self.current_task_id,
                        )
                    )


                return indications

        conf_penalty = 0.8

        if wpq < acceptable:
            if consistency > 0.5 and word_cross_rep > 0.6:
                severity = (
                    "severe" if wpq < 0.3
                    else "moderate" if wpq < poor
                    else "mild"
                )
                conf = consistency * conf_penalty
                if group_b_intact_flag > 0.5:
                    conf = min(1.0, conf + 0.1)

                indications.append(
                    ScreeningIndication(
                        indication_type="phonological_disorder",
                        severity=severity,
                        confidence=conf,
                        supporting_features={
                            "word_production_quality": wpq,
                            "cross_word_consistency": consistency,
                            "word_cross_rep_consistency_mean": word_cross_rep,
                            "group_b_intact": group_b_intact_flag,
                            "has_reference": 0.0,
                        },
                        source_node="check_word_production_absolute",
                        description=(
                            f"Consistent word production impairment "
                            f"(quality: {wpq:.3f}, no baseline reference)"
                        ),
                        task_group=self.current_task_group,
                        task_id=self.current_task_id,
                    )
                )

            if consistency < 0.4 or word_cross_rep < 0.5:
                severity = (
                    "severe" if wpq < 0.3
                    else "moderate" if wpq < poor
                    else "mild"
                )
                indications.append(
                    ScreeningIndication(
                        indication_type="speech_apraxia",
                        severity=severity,
                        confidence=0.6 * conf_penalty,
                        supporting_features={
                            "word_production_quality": wpq,
                            "cross_word_consistency": consistency,
                            "word_cross_rep_consistency_mean": word_cross_rep,
                            "complexity_gradient": gradient,
                            "has_reference": 0.0,
                        },
                        source_node=(
                            "check_word_production_inconsistent_absolute"
                        ),
                        description=(
                            f"Inconsistent word production impairment "
                            f"(quality: {wpq:.3f}, no baseline reference)"
                        ),
                        task_group=self.current_task_group,
                        task_id=self.current_task_id,
                    )
                )

        return indications

    def _evaluate_anomaly_results(
        self,
        anomaly_results: Optional[Dict[str, Any]],
    ) -> List[ScreeningIndication]:
        """Generate screening indications from anomaly detection output.

        Aggregates at the task-GROUP level (A / B / C), not per individual task.
        This prevents condition-effect noise (systematic differences between
        baseline recording conditions and test conditions) from generating
        spurious per-task indications.

        Quality gates applied to each task before it can contribute:
          - total_reps >= 2  (single-rep tasks are statistically unreliable)
          - anom_rate >= 0.60  (majority of repetitions must be anomalous)
          - mean_dev >= 1.0   (anomaly must be meaningfully above threshold)

        B- and C-group tasks with facial_asymmetry as dominant type are ignored:
        asymmetry in speech tasks does not indicate neurological paresis.

        Disorder-simulation A tasks (named "A: A10", "A: A11" etc.) follow
        a different rule than canonical A tasks (named e.g. "A: Showing Teeth"):
        even a single disorder task with 100% anomaly rate is meaningful.

        Buccofacial apraxia is only detected from disorder A tasks (A_10+),
        never from canonical A tasks alone (canonical A anomalies reflect
        condition differences, not expression substitutions).

        **B4 speech_apraxia via OC-SVM gate**:
        When B4 (pa-ta-ka) is OC-SVM anomalous but B1-B3 are not, the default
        ``_b4_ocsvm_only`` path fires speech_apraxia.  However, pa-ta-ka
        kinematics are structurally more complex than simple syllables, so the
        OC-SVM routinely flags B4 as anomalous across all profiles (including
        healthy ones) due to cross-session measurement differences.  To prevent
        false positives, the OC-SVM path additionally requires that the B4 vs
        simple DTW ratio exceeds 1.35 — the threshold validated against the
        PAC7 pilot dataset where healthy profiles show ratios of 1.18–1.33
        and the lowest genuine apraxia signal is 1.37.
        """
        if not anomaly_results:
            return []

        summary = anomaly_results.get("summary", {})
        n_anomalies = int(summary.get("n_anomalies", 0))
        if n_anomalies == 0:
            return []

        _TG_TO_TYPE: Dict[str, str] = {
            "A": "facial_paresis",
            "B": "dysarthria",
            "C": "phonological_disorder",
        }
        _ANOM_TYPE_TO_DISORDER_A: Dict[str, str] = {
            "facial_asymmetry":    "facial_paresis",
            "side_amplitude":      "facial_paresis",
            "kinematic_profile":   "buccofacial_apraxia",
            "task_substitution":   "buccofacial_apraxia",
        }
        _ANOM_TYPE_TO_DISORDER_B: Dict[str, str] = {
            "temporal_distortion": "dysarthria",
            "amplitude_reduction": "dysarthria",
            "side_amplitude":      "dysarthria",
            "articulation":        "speech_apraxia",
            "kinematic_profile":   "speech_apraxia",
        }
        _ANOM_TYPE_TO_DISORDER_C: Dict[str, str] = {
            "kinematic_profile":   "phonological_disorder",
            "articulation":        "phonological_disorder",
            "temporal_distortion": "speech_apraxia",
        }
        _ANOM_BY_GROUP = {"A": _ANOM_TYPE_TO_DISORDER_A,
                          "B": _ANOM_TYPE_TO_DISORDER_B,
                          "C": _ANOM_TYPE_TO_DISORDER_C}

        per_task: List[Dict[str, Any]] = anomaly_results.get("per_task_results", [])
        if not per_task:
            per_task = [anomaly_results]

        def _is_disorder_a_task(task_name: str) -> bool:
            """True if this is a numbered disorder-simulation A task (A: A10 etc.)."""
            suffix = task_name[3:].strip() if task_name.startswith("A:") else ""
            return (
                len(suffix) >= 2
                and suffix[0] == "A"
                and suffix[1:].isdigit()
            )

        _MIN_ANOM_RATE_A_CANON   = 0.60
        _MIN_MEAN_DEV_A_CANON    = 0.55
        _MIN_ANOM_RATE_A_DIS     = 0.50
        _MIN_MEAN_DEV_A_DIS      = 0.40
        _MIN_ANOM_RATE_B         = 0.60
        _MIN_MEAN_DEV_B          = 0.78
        _MIN_ANOM_RATE_B4        = 0.60
        _MIN_MEAN_DEV_B4         = 0.40
        _MIN_ANOM_RATE_C         = 0.60
        _MIN_MEAN_DEV_C          = 0.60

        group_qualifying: Dict[str, List[Dict]] = {"A": [], "B": [], "C": []}
        disorder_a_qualifying: List[Dict] = []
        canonical_a_qualifying: List[Dict] = []
        b_simple_qualifying: List[Dict] = []
        b4_qualifying: List[Dict] = []

        for tr in per_task:
            task_is_anomaly: List[bool] = tr.get("is_anomaly", [])
            task_n = sum(1 for v in task_is_anomaly if v)
            total_reps = len(task_is_anomaly)
            task_names: List[str] = tr.get("task_names", [])
            first_name = task_names[0] if task_names else ""

            if first_name.startswith("A:"):
                tg = "A"
            elif first_name.startswith("B:"):
                tg = "B"
            elif first_name.startswith("C:"):
                tg = "C"
            else:
                _tr_tg = str(tr.get("task_group", self.current_task_group or "0"))
                tg = _tr_tg if _tr_tg in ("A", "B", "C") else (self.current_task_group or "0")

            if task_n == 0:
                continue

            dev_scores = [
                float(d) for d in tr.get("deviation_score", [])
                if d is not None
            ]
            mean_dev = float(np.mean(dev_scores)) if dev_scores else 0.0
            anom_rate = task_n / max(total_reps, 1)
            dom_type = tr.get("summary", {}).get("dominant_anomaly_type", "")

            if tg == "B" and dom_type in ("facial_asymmetry", "unknown"):
                continue
            _MIN_MEAN_DEV_B_SA = 0.80
            if tg == "B" and dom_type == "side_amplitude" and mean_dev < _MIN_MEAN_DEV_B_SA:
                continue
            _name_lower_early = first_name.lower()
            _is_b4_early = tg == "B" and any(
                seq in _name_lower_early
                for seq in ("pa-ta-ka", "ta-pa-ka", "ka-pa-ta", "pa-ka-ta",
                            "ka-ta-pa", "ta-ka-pa", "pa ta ka", "pataka")
            )
            if tg == "B" and not _is_b4_early and total_reps < 2:
                continue

            task_rec = {
                "tg": tg,
                "name": first_name,
                "n_anom": task_n,
                "n_total": total_reps,
                "anom_rate": anom_rate,
                "mean_dev": mean_dev,
                "dom_type": dom_type,
                "tr": tr,
            }

            _tr_task_id = tr.get("task_id")
            _task_id_disorder = (
                _tr_task_id is not None and int(_tr_task_id) >= 10
            )
            is_disorder_a = tg == "A" and (_is_disorder_a_task(first_name) or _task_id_disorder)
            _name_lower = first_name.lower()
            is_b4 = tg == "B" and any(
                seq in _name_lower
                for seq in ("pa-ta-ka", "ta-pa-ka", "ka-pa-ta", "pa-ka-ta",
                            "ka-ta-pa", "ta-ka-pa", "pa ta ka", "pataka")
            )

            if tg == "A":
                if is_disorder_a:
                    passes = anom_rate >= _MIN_ANOM_RATE_A_DIS and mean_dev >= _MIN_MEAN_DEV_A_DIS
                else:
                    passes = anom_rate >= _MIN_ANOM_RATE_A_CANON and mean_dev >= _MIN_MEAN_DEV_A_CANON
            elif tg == "B":
                if is_b4:
                    passes = anom_rate >= _MIN_ANOM_RATE_B4 and mean_dev >= _MIN_MEAN_DEV_B4
                else:
                    passes = anom_rate >= _MIN_ANOM_RATE_B and mean_dev >= _MIN_MEAN_DEV_B
            elif tg == "C":
                passes = anom_rate >= _MIN_ANOM_RATE_C and mean_dev >= _MIN_MEAN_DEV_C
            else:
                passes = False

            if not passes:
                continue

            if tg == "A":
                if is_disorder_a:
                    disorder_a_qualifying.append(task_rec)
                else:
                    canonical_a_qualifying.append(task_rec)
            elif tg == "B":
                if is_b4:
                    b4_qualifying.append(task_rec)
                else:
                    b_simple_qualifying.append(task_rec)
                    group_qualifying["B"].append(task_rec)
            elif tg == "C":
                group_qualifying["C"].append(task_rec)

        indications: List[ScreeningIndication] = []

        if canonical_a_qualifying and not self.is_ors_session:
            asym_tasks = [
                t for t in canonical_a_qualifying
                if t["dom_type"] in ("facial_asymmetry", "side_amplitude")
            ]
            kin_tasks_canon = [
                t for t in canonical_a_qualifying
                if t["dom_type"] in ("kinematic_profile", "task_substitution")
            ]
            if len(asym_tasks) >= 3:
                _rates = [t["anom_rate"] for t in asym_tasks]
                _devs  = [t["mean_dev"]  for t in asym_tasks]
                mean_anom_rate = float(np.mean(_rates))
                mean_dev_a     = float(np.mean(_devs))
                if mean_dev_a <= 0.80:
                    asym_tasks = []
            _ocsvm_asym_ok = (
                getattr(self, "_group_a_mean_asym_corrected", 0.0)
                >= getattr(self, "_group_a_mild_t", 0.43)
            )
            if len(asym_tasks) >= 3 and not _ocsvm_asym_ok:
                asym_tasks = []
            if len(asym_tasks) >= 3:
                _rates = [t["anom_rate"] for t in asym_tasks]
                _devs  = [t["mean_dev"]  for t in asym_tasks]
                mean_anom_rate = float(np.mean(_rates))
                mean_dev_a     = float(np.mean(_devs))
                severity = ("severe"   if mean_dev_a > 1.20
                            else "moderate" if mean_dev_a > 0.90
                            else "mild")
                conf = min(0.75, 0.40 + mean_anom_rate * 0.35)
                feat_counts: Dict[str, int] = {}
                for t in asym_tasks:
                    for cf in t["tr"].get("contributing_features", []):
                        if isinstance(cf, list):
                            for f in cf:
                                feat_counts[f] = feat_counts.get(f, 0) + 1
                top_feats = sorted(feat_counts, key=feat_counts.get, reverse=True)[:5]
                indications.append(ScreeningIndication(
                    indication_type="facial_paresis",
                    severity=severity,
                    confidence=conf,
                    supporting_features={
                        "n_qualifying_a_tasks": float(len(asym_tasks)),
                        "mean_anomaly_rate": mean_anom_rate,
                        "mean_deviation_score": mean_dev_a,
                        "top_features": ", ".join(top_feats),
                    },
                    source_node="anomaly_detection_group_a",
                    description=(
                        f"Facial asymmetry across {len(asym_tasks)} Group A tasks "
                        f"(mean rate: {mean_anom_rate:.0%}, dev: {mean_dev_a:.2f})"
                    ),
                    task_group="A",
                    task_id=self.current_task_id,
                ))
            elif len(kin_tasks_canon) >= 2:
                _rates_k = [t["anom_rate"] for t in kin_tasks_canon]
                _devs_k  = [t["mean_dev"]  for t in kin_tasks_canon]
                mean_r_k = float(np.mean(_rates_k))
                mean_d_k = float(np.mean(_devs_k))
                severity = ("severe"   if mean_d_k > 0.80
                            else "moderate" if mean_d_k > 0.65
                            else "mild")
                feat_counts_k: Dict[str, int] = {}
                for t in kin_tasks_canon:
                    for cf in t["tr"].get("contributing_features", []):
                        if isinstance(cf, list):
                            for f in cf:
                                feat_counts_k[f] = feat_counts_k.get(f, 0) + 1
                top_feats_k = sorted(feat_counts_k, key=feat_counts_k.get, reverse=True)[:5]
                indications.append(ScreeningIndication(
                    indication_type="buccofacial_apraxia",
                    severity=severity,
                    confidence=min(0.65, 0.38 + mean_r_k * 0.25),
                    supporting_features={
                        "n_kinematic_a_tasks": float(len(kin_tasks_canon)),
                        "mean_anomaly_rate": mean_r_k,
                        "mean_deviation_score": mean_d_k,
                        "top_features": ", ".join(top_feats_k),
                    },
                    source_node="anomaly_detection_group_a_kin",
                    description=(
                        f"Kinematic profile mismatch across {len(kin_tasks_canon)} "
                        f"Group A tasks (mean rate: {mean_r_k:.0%}, dev: {mean_d_k:.2f})"
                    ),
                    task_group="A",
                    task_id=self.current_task_id,
                ))

        if disorder_a_qualifying:
            _d_rates = [t["anom_rate"] for t in disorder_a_qualifying]
            _d_devs  = [t["mean_dev"]  for t in disorder_a_qualifying]
            mean_anom_rate_d = float(np.mean(_d_rates))
            mean_dev_d       = float(np.mean(_d_devs))
            severity = ("severe"   if mean_dev_d > 0.80
                        else "moderate" if mean_dev_d > 0.65
                        else "mild")
            conf = min(0.82, 0.50 + len(disorder_a_qualifying) * 0.05 + mean_anom_rate_d * 0.15)
            _d_asym_count  = sum(1 for t in disorder_a_qualifying
                                 if t["dom_type"] in ("facial_asymmetry", "side_amplitude"))
            _d_kin_count   = sum(1 for t in disorder_a_qualifying
                                 if t["dom_type"] == "kinematic_profile")
            _d_subst_count = sum(1 for t in disorder_a_qualifying
                                 if t["dom_type"] == "task_substitution")
            if _d_subst_count > _d_asym_count + _d_kin_count:
                indication_type_d = "buccofacial_apraxia"
            else:
                indication_type_d = "facial_paresis"
            feat_counts_d: Dict[str, int] = {}
            for t in disorder_a_qualifying:
                for cf in t["tr"].get("contributing_features", []):
                    if isinstance(cf, list):
                        for f in cf:
                            feat_counts_d[f] = feat_counts_d.get(f, 0) + 1
            top_feats_d = sorted(feat_counts_d, key=feat_counts_d.get, reverse=True)[:5]
            indications.append(ScreeningIndication(
                indication_type=indication_type_d,
                severity=severity,
                confidence=conf,
                supporting_features={
                    "n_disorder_a_tasks_anomalous": float(len(disorder_a_qualifying)),
                    "mean_anomaly_rate": mean_anom_rate_d,
                    "mean_deviation_score": mean_dev_d,
                    "n_kinematic_tasks": float(_d_kin_count),
                    "n_asymmetry_tasks": float(_d_asym_count),
                    "n_substitution_tasks": float(_d_subst_count),
                    "top_features": ", ".join(top_feats_d),
                },
                source_node="anomaly_detection_disorder_a",
                description=(
                    f"{len(disorder_a_qualifying)} disorder A task(s) deviate "
                    f"from reference (mean rate: {mean_anom_rate_d:.0%}, "
                    f"dom: {indication_type_d.replace('_',' ')})"
                ),
                task_group="A",
                task_id=self.current_task_id,
            ))

        b_tasks = group_qualifying.get("B", [])
        if len(b_tasks) >= 2:
            _b_dysarthria = sum(
                1 for t in b_tasks
                if t["dom_type"] in ("amplitude_reduction", "temporal_distortion")
                or _ANOM_TYPE_TO_DISORDER_B.get(t["dom_type"], "") == "dysarthria"
            )
            _b_apraxia = sum(
                1 for t in b_tasks
                if t["dom_type"] in ("kinematic_profile", "articulation")
                or _ANOM_TYPE_TO_DISORDER_B.get(t["dom_type"], "") == "speech_apraxia"
            )
            b_type = (
                "dysarthria" if _b_dysarthria >= _b_apraxia
                else "speech_apraxia"
            )
            _b_rates = [t["anom_rate"] for t in b_tasks]
            _b_devs  = [t["mean_dev"]  for t in b_tasks]
            mean_r_b = float(np.mean(_b_rates))
            mean_d_b = float(np.mean(_b_devs))
            severity = ("severe"   if mean_d_b > 0.80
                        else "moderate" if mean_d_b > 0.65
                        else "mild")
            conf_b = min(0.80, 0.45 + len(b_tasks) * 0.10 + mean_r_b * 0.20)
            feat_counts_b: Dict[str, int] = {}
            for t in b_tasks:
                for cf in t["tr"].get("contributing_features", []):
                    if isinstance(cf, list):
                        for f in cf:
                            feat_counts_b[f] = feat_counts_b.get(f, 0) + 1
            top_b = sorted(feat_counts_b, key=feat_counts_b.get, reverse=True)[:5]
            indications.append(ScreeningIndication(
                indication_type=b_type,
                severity=severity,
                confidence=conf_b,
                supporting_features={
                    "n_qualifying_b_tasks": float(len(b_tasks)),
                    "mean_anomaly_rate": mean_r_b,
                    "mean_deviation_score": mean_d_b,
                    "n_dysarthria_type": float(_b_dysarthria),
                    "n_apraxia_type": float(_b_apraxia),
                    "top_features": ", ".join(top_b),
                },
                source_node="anomaly_detection_group_b",
                description=(
                    f"{len(b_tasks)} Group B task(s) anomalous — "
                    f"dominant: {b_type.replace('_',' ')} "
                    f"(mean rate: {mean_r_b:.0%}, dev: {mean_d_b:.2f})"
                ),
                task_group="B",
                task_id=self.current_task_id,
            ))

        _b4_dtw         = anomaly_results.get("b4_dtw_summary", {}) if anomaly_results else {}
        _b4_vs_simple   = float(_b4_dtw.get("b4_vs_simple_ratio", 1.0))
        _b4_mean_dtw    = float(_b4_dtw.get("b4_mean_dtw", 0.0))
        _b4_n_anom      = int(_b4_dtw.get("b4_n_shape_anom", 0))
        _b4_dtw_ref     = float(_b4_dtw.get("b4_dtw_vs_ref") or 1.0)
        _b4_rep_cv      = float(_b4_dtw.get("b4_rep_dtw_cv") or 0.0)
        _no_dysarthria = (not getattr(self, "_b_dysarthria_found", False)
                         and not getattr(self, "_c_dysarthria_found", False))
        _b4_dtw_selective = (
            (_b4_vs_simple > 1.3 and _b4_n_anom >= 2 and not self.is_ors_session and _no_dysarthria)
            or (_b4_dtw_ref > 1.6 and _b4_n_anom >= 2 and _no_dysarthria)
            or (_b4_rep_cv > 0.25 and _b4_n_anom >= 1 and _no_dysarthria)
            or (self.is_ors_session and _b4_vs_simple > 1.50 and _b4_n_anom >= 2 and _no_dysarthria)
        )
        _paresis_likely = (
            getattr(self, "_group_a_mean_asym_corrected", 0.0)
            >= getattr(self, "_group_a_mild_t", 0.43)
        )
        _b4_ocsvm_min_anom = 2 if _paresis_likely else 1
        _b4_ocsvm_only  = (
            bool(b4_qualifying)
            and not b_simple_qualifying
            and _b4_n_anom >= _b4_ocsvm_min_anom
            and _no_dysarthria
            and (
                (_b4_vs_simple > 1.37 and not self.is_ors_session)
                or (self.is_ors_session and _b4_vs_simple > 1.70)
            )
        )
        _b4_shape_anom_only = (
            _b4_n_anom >= 2
            and not b_simple_qualifying
            and _no_dysarthria
            and _b4_vs_simple > 0.90
        )
        _b4_cv_only = (
            _b4_rep_cv > 0.40
            and not b_simple_qualifying
            and _no_dysarthria
        )
        b4_only_impaired = _b4_ocsvm_only or _b4_dtw_selective or _b4_shape_anom_only or _b4_cv_only
        _c_nrel_early = int(
            (anomaly_results or {}).get("c_dtw_summary", {}).get("c_n_high_relative", 0)
        )
        _b4_ocsvm_suppressed = (
            _b4_ocsvm_only
            and not (_b4_dtw_selective or _b4_shape_anom_only or _b4_cv_only)
            and _c_nrel_early >= 2
        )

        if b4_only_impaired and not _b4_ocsvm_suppressed:
            _b4_rates = [t["anom_rate"] for t in b4_qualifying] if b4_qualifying else [1.0]
            _b4_devs  = [t["mean_dev"]  for t in b4_qualifying] if b4_qualifying else [_b4_mean_dtw]
            mean_r_b4 = float(np.mean(_b4_rates))
            mean_d_b4 = float(np.mean(_b4_devs))
            feat_counts_b4: Dict[str, int] = {}
            for t in b4_qualifying:
                for cf in t["tr"].get("contributing_features", []):
                    if isinstance(cf, list):
                        for f in cf:
                            feat_counts_b4[f] = feat_counts_b4.get(f, 0) + 1
            top_b4 = sorted(feat_counts_b4, key=feat_counts_b4.get, reverse=True)[:5]
            _simple_b_intact = not b_simple_qualifying
            if _b4_dtw_selective and not _simple_b_intact:
                b4_desc = (
                    f"Complex sequencing (pa-ta-ka) selectively elevated vs simple syllables "
                    f"(DTW ratio: {_b4_vs_simple:.2f}, {_b4_n_anom} shape anomalies; "
                    f"rate: {mean_r_b4:.0%}, dev: {mean_d_b4:.2f})"
                )
            else:
                b4_desc = (
                    f"Complex sequencing (pa-ta-ka) impaired while simple "
                    f"syllables intact (rate: {mean_r_b4:.0%}, dev: {mean_d_b4:.2f})"
                )
            indications.append(ScreeningIndication(
                indication_type="speech_apraxia",
                severity="mild",
                confidence=min(0.72, 0.50 + mean_r_b4 * 0.20),
                supporting_features={
                    "b4_anomaly_rate": mean_r_b4,
                    "b4_mean_deviation": mean_d_b4,
                    "simple_b_intact": float(_simple_b_intact),
                    "b4_vs_simple_dtw_ratio": _b4_vs_simple,
                    "b4_n_shape_anom": float(_b4_n_anom),
                    "top_features": ", ".join(top_b4),
                },
                source_node="anomaly_detection_b4_selective",
                description=b4_desc,
                task_group="B",
                task_id=self.current_task_id,
            ))

        c_tasks = group_qualifying.get("C", [])
        _b_severely_impaired = len(b_tasks) >= 2

        _c_dtw     = anomaly_results.get("c_dtw_summary", {}) if anomaly_results else {}
        _c_dtw_mean    = float(_c_dtw.get("c_mean_dtw", 0.0))
        _c_dtw_n_high  = int(_c_dtw.get("c_n_high_dtw", 0))
        _c_n_high_relative = _c_nrel_early
        _max_c_task_dtw = float(_c_dtw.get("max_c_task_dtw", 0.0))
        _ocsvm_paresis = any(i.indication_type == "facial_paresis" for i in indications)
        _paresis_any = _paresis_likely or _ocsvm_paresis
        _c_dtw_gate    = (
            _c_dtw_n_high >= 7
            and _c_dtw_mean > 0.14
            and not (_paresis_any and _c_n_high_relative < 3)
        )
        _c_abs_many = (
            _c_dtw_n_high >= 6
            and _c_dtw_mean > 0.11
            and _c_n_high_relative == 0
            and _max_c_task_dtw > 0.10
            and _max_c_task_dtw < 0.50
            and not getattr(self, "_b4_selective_hint", False)
            and not _b_severely_impaired
        )
        _paresis_no_b4 = (
            _paresis_likely
            and not getattr(self, "_b4_selective_hint", False)
            and not b4_only_impaired
        )
        _c_relative_gate = (
            (_c_n_high_relative >= 3 and _c_dtw_mean > 0.08)
            or (_c_n_high_relative >= 2 and _c_dtw_mean > 0.08 and _max_c_task_dtw < 0.50
                and _c_dtw_n_high < 7
                and not _paresis_no_b4)
            or _c_abs_many
        )

        _c_kin_tasks = [t for t in c_tasks if t["dom_type"] == "kinematic_profile"]
        _any_dysarthria_found = (
            getattr(self, "_b_dysarthria_found", False)
            or getattr(self, "_c_dysarthria_found", False)
        )
        _c_fires = (
            (_c_dtw_gate or _c_relative_gate or (len(c_tasks) >= 7 and _c_dtw_n_high >= 7 and _c_dtw_mean > 0.14))
            and not _b_severely_impaired
            and not _any_dysarthria_found
        )

        if _c_fires:
            _c_rates = [t["anom_rate"] for t in c_tasks]
            _c_devs  = [t["mean_dev"]  for t in c_tasks]
            mean_r_c = float(np.mean(_c_rates)) if _c_rates else 0.0
            mean_d_c = float(np.mean(_c_devs))  if _c_devs  else 0.0
            _c_dom_types = [t["dom_type"] for t in c_tasks]
            _c_unique_types = len(set(_c_dom_types))
            _b4_hint_overrides = (
                getattr(self, "_b4_selective_hint", False)
                and _c_n_high_relative < 2
            )
            if ((b4_only_impaired and not _b4_ocsvm_suppressed)
                    or (_c_unique_types >= 3 and not _c_relative_gate)
                    or _b4_hint_overrides):
                c_indication = "speech_apraxia"
            elif getattr(self, "_b_dysarthria_found", False):
                c_indication = "dysarthria"
            else:
                c_indication = "phonological_disorder"
            severity = ("severe"   if mean_d_c > 0.80
                        else "moderate" if mean_d_c > 0.65
                        else "mild")
            conf_c = min(0.70, 0.35 + max(len(c_tasks), _c_dtw_n_high) * 0.04 + mean_r_c * 0.15)
            feat_counts_c: Dict[str, int] = {}
            for t in c_tasks:
                for cf in t["tr"].get("contributing_features", []):
                    if isinstance(cf, list):
                        for f in cf:
                            feat_counts_c[f] = feat_counts_c.get(f, 0) + 1
            top_c = sorted(feat_counts_c, key=feat_counts_c.get, reverse=True)[:5]
            gate_desc = (
                f"DTW gate: {_c_dtw_n_high} tasks elevated (mean DTW {_c_dtw_mean:.3f})"
                if _c_dtw_gate else
                f"OC-SVM fallback: {len(c_tasks)} tasks anomalous"
            )
            indications.append(ScreeningIndication(
                indication_type=c_indication,
                severity=severity,
                confidence=conf_c,
                supporting_features={
                    "n_qualifying_c_tasks":     float(len(c_tasks)),
                    "n_kinematic_profile_tasks": float(len(_c_kin_tasks)),
                    "b_simple_intact":          float(not _b_severely_impaired),
                    "mean_anomaly_rate":         mean_r_c,
                    "mean_deviation_score":      mean_d_c,
                    "c_dtw_gate_passed":         float(_c_dtw_gate),
                    "c_mean_dtw":               _c_dtw_mean,
                    "c_n_high_dtw_tasks":       float(_c_dtw_n_high),
                    "top_features":             ", ".join(top_c),
                },
                source_node="anomaly_detection_group_c",
                description=(
                    f"Group C word tasks — {c_indication.replace('_',' ')}. "
                    f"{gate_desc}. "
                    f"Mean OC-SVM rate: {mean_r_c:.0%}, dev: {mean_d_c:.2f}"
                ),
                task_group="C",
                task_id=self.current_task_id,
            ))

        _wpq_c = getattr(self, "_wpq_c", 1.0)
        _ors_phono_fallback = (
            self.is_ors_session
            and not _c_fires
            and _c_dtw_n_high >= 5
            and _c_dtw_mean > 0.12
            and _wpq_c < 0.75
            and not b4_only_impaired
            and not getattr(self, "_b4_selective_hint", False)
            and not getattr(self, "_ors_b_apraxia_found", False)
            and not _any_dysarthria_found
            and not _b_severely_impaired
        )
        if _ors_phono_fallback:
            _phono_exists = any(
                ind.indication_type == "phonological_disorder" for ind in indications
            )
            if not _phono_exists:
                indications.append(ScreeningIndication(
                    indication_type="phonological_disorder",
                    severity="mild",
                    confidence=0.52,
                    supporting_features={
                        "c_n_high_dtw": float(_c_dtw_n_high),
                        "c_mean_dtw": _c_dtw_mean,
                        "word_production_quality": _wpq_c,
                        "b4_vs_simple_ratio": float(getattr(self, "_b4_ratio_raw", 1.0)),
                        "ors_fallback_gate": 1.0,
                    },
                    source_node="anomaly_detection_group_c_ors_fallback",
                    description=(
                        f"Group C (ORS): {_c_dtw_n_high} tasks elevated "
                        f"(mean DTW {_c_dtw_mean:.3f}), WPQ {_wpq_c:.2f} — ORS gravity "
                        f"suppresses absolute gate; moderate elevation with degraded word "
                        f"quality consistent with phonological disorder"
                    ),
                    task_group="C",
                    task_id=self.current_task_id,
                ))


        return indications

    def _compute_confidence(
        self,
        features: Dict[str, float],
        indications: List[ScreeningIndication],
        anomaly_results: Optional[Dict[str, Any]],
    ) -> ConfidenceComponents:
        """Compute the composite confidence score from data quality, consistency, and agreement."""
        detection_rate = features.get(
            "overall_detection_rate", features.get("rep_detection_rate", 0.8)
        )
        sample_size = features.get(
            "total_repetitions", features.get("rep_repetition", 3)
        )
        data_quality = min(1.0, detection_rate * min(1.0, sample_size / 3))

        cv_features = [
            k for k in features if "_cv" in k or "consistency" in k.lower()
        ]
        if cv_features:
            consistency_scores = [
                features[k] for k in cv_features if 0 <= features[k] <= 1
            ]
            consistency = np.mean(consistency_scores) if consistency_scores else 0.7
        else:
            consistency = 0.7

        model_rule_agreement = 0.8
        if anomaly_results and "is_anomaly" in anomaly_results:
            anomaly_flags = anomaly_results["is_anomaly"]
            indication_present = len(indications) > 0
            any_anomaly = bool(anomaly_flags and any(anomaly_flags))
            anom_rate = float(sum(1 for a in anomaly_flags if a) / max(len(anomaly_flags), 1)) if anomaly_flags else 0.0

            summary_block = anomaly_results.get("summary", {})
            mean_dev = float(summary_block.get("mean_deviation", summary_block.get("mean_dev", 0.0)))

            logger.debug(
                "confidence | any_anomaly=%s indication_present=%s mean_dev=%.3f anom_rate=%.3f",
                any_anomaly, indication_present, mean_dev, anom_rate,
            )

            per_task: List[Dict[str, Any]] = anomaly_results.get("per_task_results", [])

            if per_task and indications:
                _ANOM_TYPE_TO_DISORDER_A: Dict[str, str] = {
                    "facial_asymmetry": "facial_paresis",
                    "side_amplitude": "facial_paresis",
                    "kinematic_profile": "buccofacial_apraxia",
                    "task_substitution": "buccofacial_apraxia",
                }
                _ANOM_TYPE_TO_DISORDER_B: Dict[str, str] = {
                    "temporal_distortion": "dysarthria",
                    "amplitude_reduction": "dysarthria",
                    "side_amplitude": "dysarthria",
                    "articulation": "speech_apraxia",
                    "kinematic_profile": "speech_apraxia",
                }
                _ANOM_TYPE_TO_DISORDER_C: Dict[str, str] = {
                    "kinematic_profile": "phonological_disorder",
                    "articulation": "phonological_disorder",
                    "temporal_distortion": "speech_apraxia",
                }
                _ANOM_BY_GROUP: Dict[str, Dict[str, str]] = {
                    "A": _ANOM_TYPE_TO_DISORDER_A,
                    "B": _ANOM_TYPE_TO_DISORDER_B,
                    "C": _ANOM_TYPE_TO_DISORDER_C,
                }

                indicated_disorders = {ind.indication_type for ind in indications}

                matched_pairs = 0
                total_pairs = 0
                strength_sum = 0.0
                for task in per_task:
                    t_summary = task.get("summary", {})
                    dom_type = t_summary.get("dominant_anomaly_type", "")
                    task_names = task.get("task_names", [])
                    first_name = task_names[0] if task_names else ""
                    if first_name.startswith("A:"):
                        tg = "A"
                    elif first_name.startswith("B:"):
                        tg = "B"
                    elif first_name.startswith("C:"):
                        tg = "C"
                    else:
                        tg = str(task.get("task_groups", ["A"])[0] if task.get("task_groups") else "A")
                    is_anom_list = task.get("is_anomaly", [])
                    task_rate = float(sum(1 for a in is_anom_list if a) / max(len(is_anom_list), 1)) if is_anom_list else float(t_summary.get("anomaly_rate", 0.0))
                    dev_scores = [float(d) for d in task.get("deviation_score", []) if d is not None]
                    task_dev = float(np.mean(dev_scores)) if dev_scores else float(t_summary.get("mean_deviation_score", 0.0))
                    if not dom_type or task_rate < 0.3:
                        continue
                    _fname_lower = first_name.lower()
                    _is_b4 = tg == "B" and any(
                        seq in _fname_lower for seq in (
                            "pa-ta-ka", "ta-pa-ka", "ka-pa-ta", "pa-ka-ta",
                            "ka-ta-pa", "ta-ka-pa", "pa ta ka", "pataka",
                        )
                    )
                    if _is_b4 and dom_type == "temporal_distortion":
                        tg_map = dict(_ANOM_TYPE_TO_DISORDER_B)
                        tg_map["temporal_distortion"] = "speech_apraxia"
                    else:
                        tg_map = _ANOM_BY_GROUP.get(tg, {})
                    mapped_disorder = tg_map.get(dom_type, "")
                    if not mapped_disorder:
                        continue
                    total_pairs += 1
                    strength = min(1.0, task_rate * min(task_dev / 1.0, 1.0))
                    if mapped_disorder in indicated_disorders:
                        matched_pairs += 1
                        strength_sum += strength
                    else:
                        strength_sum -= strength * 0.3

                if total_pairs > 0:
                    base_match = matched_pairs / total_pairs
                    strength_factor = min(1.0, max(0.0, strength_sum / total_pairs))
                    model_rule_agreement = 0.5 + 0.45 * base_match + 0.05 * strength_factor
                else:
                    model_rule_agreement = 0.85 if any_anomaly == indication_present else 0.65
            else:
                strength_bonus = min(0.05, mean_dev * 0.02 + anom_rate * 0.03)
                if any_anomaly == indication_present:
                    model_rule_agreement = 0.85 + strength_bonus
                else:
                    model_rule_agreement = max(0.5, 0.65 - strength_bonus)

            model_rule_agreement = float(np.clip(model_rule_agreement, 0.0, 1.0))

        weights = self.confidence_weights
        overall = (
            weights.get("data_quality", 0.35) * data_quality
            + weights.get("consistency", 0.35) * consistency
            + weights.get("model_rule_agreement", 0.30) * model_rule_agreement
        )

        return ConfidenceComponents(
            data_quality=float(data_quality),
            consistency=float(consistency),
            model_rule_agreement=float(model_rule_agreement),
            overall=float(overall),
        )

    @staticmethod
    def _indication_to_dict(indication: ScreeningIndication) -> Dict[str, Any]:
        """Serialise a ScreeningIndication to a plain dictionary."""
        return {
            "indication_type": indication.indication_type,
            "severity": indication.severity,
            "confidence": indication.confidence,
            "supporting_features": indication.supporting_features,
            "source_node": indication.source_node,
            "description": indication.description,
            "task_group": indication.task_group,
            "task_id": indication.task_id,
        }

    def get_indication_description(self, indication_type: str) -> str:
        """Return the human-readable description for a screening indication type."""
        if indication_type in self.screening_indications_config:
            return self.screening_indications_config[indication_type].get("description", "")
        return ""


def create_decision_support(decision_rules_config: Dict[str, Any]) -> DecisionSupport:
    """Factory: build a DecisionSupport engine from decision-rules configuration."""
    return DecisionSupport(decision_rules_config)
