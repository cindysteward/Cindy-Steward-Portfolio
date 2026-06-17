"""
Feature extraction for the FACET-CV pipeline.

Computes baseline-corrected features and derived metrics from MediaPipe
blendshape data including facial asymmetry, activation intensity, temporal
dynamics, and movement smoothness.

Key methodological references
==============================
Facial motor kinematics classification
  Palmer et al. (2024) A machine-learning approach to classifying facial
  movement kinematics. Sensors 24(22):7235.
  doi:10.3390/s24227235

Smoothness metrics for movement quality
  Balasubramanian et al. (2012) A robust and sensitive metric for quantifying
  movement smoothness. IEEE Trans Biomed Eng 59(8):2126-2136.
  doi:10.1109/TBME.2011.2179545

  Gulde & Hermsdorfer (2018) Smoothness metrics in complex movement tasks.
  Front Neurol 9:615.
  doi:10.3389/fneur.2018.00615

MediaPipe facial landmark framework
  Lugaresi et al. (2019) MediaPipe: A framework for building perception
  pipelines. CVPR Workshop on Computer Vision for AR/VR.

Facial fatigue indicators
  Kong et al. (2021) Looking fatigued: the effects of continuous visual task
  performance on facial features. Atten Percept Psychophys 83:730-748.
  doi:10.3758/s13414-020-02199-5

  Brach & VanSwearingen (1995) Physical therapy for facial paralysis: a
  tailored treatment approach. Phys Ther 75(12):1060-1070.
  doi:10.1016/S0003-9993(95)80064-6
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

from .utils import safe_divide, sanitize_events_df
from .head_pose import HeadPoseNormalizer


class FeatureExtractor:
    """Derives asymmetry, intensity, and temporal features from corrected blendshape data."""

    def __init__(self, features_config: Dict[str, Any], tasks_config: Dict[str, Any]):
        """Initialise the extractor with feature and task configuration dicts.

        Args:
            features_config: Dict loaded from features.yaml, containing symmetry
                pairs, blendshape lists, activation ranges, and smoothing settings.
            tasks_config: Dict loaded from tasks.yaml, used for task-specific
                feature queries.
        """
        self.features_config = features_config
        self.tasks_config = tasks_config
        self.symmetry_pairs = features_config.get("symmetry_pairs", {})
        self.derived_features = features_config.get("derived_features", {})
        self.epsilon = 0.001
        self.head_pose_normalizer = HeadPoseNormalizer()
        self.fallback_ranges = self._build_activation_ranges()
        self.configured_blendshapes = self._collect_blendshape_names()
        self._baseline_stats: Optional[Dict[str, Dict[str, float]]] = None
        self._observed_ranges: Optional[Dict[str, float]] = None

    def extract_features(self, corrected_df: pd.DataFrame, events_df: pd.DataFrame,
                         baseline_stats: Optional[Dict[str, Dict[str, float]]] = None,
                         observed_ranges: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """Run the full feature extraction pipeline on corrected blendshape data.

        Applies temporal smoothing, adds relative timestamps, then sequentially
        computes head pose, asymmetry, intensity, and temporal features.

        Args:
            corrected_df: Baseline-corrected blendshape DataFrame from the
                preprocessing stage.
            events_df: Pipeline events DataFrame used to assign segment-relative
                timestamps and propagate task identifiers.
            baseline_stats: Per-feature mean/std from the reference baseline
                session, used for range normalisation.
            observed_ranges: Per-feature 95th-percentile range from baseline
                measurement data.

        Returns:
            A copy of corrected_df enriched with all derived feature columns.
        """
        self._baseline_stats = baseline_stats
        self._observed_ranges = observed_ranges
        features_df = corrected_df.copy()

        smooth_win = int(self.features_config.get("smoothing_window", 5))
        if smooth_win > 1:
            exclude = {
                "frame_index", "timestamp_abs", "segment", "repetition",
                "detection_success", "time_rel_sec", "task_group", "task_id",
                "task_name", "brightness", "occluded",
            }
            blendshape_cols = [
                c for c in features_df.columns
                if c not in exclude and not c.startswith("asymmetry") and not c.startswith("_")
            ]
            if blendshape_cols:
                features_df[blendshape_cols] = (
                    features_df[blendshape_cols]
                    .rolling(window=smooth_win, min_periods=1, center=True)
                    .mean()
                )

        if "occluded" not in features_df.columns:
            if "detection_success" in features_df.columns:
                features_df["occluded"] = ~features_df["detection_success"].astype(bool)
            elif "detection_confidence" in features_df.columns:
                thresh = float(self.features_config.get("occlusion_confidence_thresh", 0.5))
                features_df["occluded"] = features_df["detection_confidence"].fillna(1.0) < thresh
            else:
                features_df["occluded"] = False

        features_df = self._add_relative_time(features_df, events_df)
        features_df = self._compute_head_pose_features(features_df)
        features_df = self._compute_asymmetry_features(features_df)
        features_df = self._compute_intensity_features(features_df)
        features_df = self._compute_temporal_features(features_df)
        return features_df

    def _compute_head_pose_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estimate head pose from landmark columns and add yaw/pitch/roll features.

        If the baseline (neutral) segment is present in df, the reference pose
        is set from its median values so that subsequent frames report deviation
        from that reference.  When landmark columns are absent, all pose columns
        are filled with 0.0.

        Fully vectorised over all frames using numpy column operations, mirroring
        HeadPoseNormalizer.estimate_pose() without a Python row loop.
        """
        has_landmarks = all(
            c in df.columns for c in ("noseTip_x", "leftEye_x", "rightEye_x")
        )
        if not has_landmarks:
            df["head_yaw"] = 0.0
            df["head_pitch"] = 0.0
            df["head_roll"] = 0.0
            df["head_pose_deviation"] = 0.0
            return df

        _nt_x = df["noseTip_x"].fillna(0.0).to_numpy(dtype=np.float64)
        _nt_y = df["noseTip_y"].fillna(0.0).to_numpy(dtype=np.float64)
        _le_x = df["leftEye_x"].fillna(0.0).to_numpy(dtype=np.float64)
        _le_y = df["leftEye_y"].fillna(0.0).to_numpy(dtype=np.float64)
        _re_x = df["rightEye_x"].fillna(0.0).to_numpy(dtype=np.float64)
        _re_y = df["rightEye_y"].fillna(0.0).to_numpy(dtype=np.float64)
        _ml_y = df["mouthLeft_y"].fillna(0.0).to_numpy(dtype=np.float64) if "mouthLeft_y" in df.columns else np.zeros(len(df))
        _mr_y = df["mouthRight_y"].fillna(0.0).to_numpy(dtype=np.float64) if "mouthRight_y" in df.columns else np.zeros(len(df))

        _eye_mid_x = (_le_x + _re_x) / 2.0
        _eye_mid_y = (_le_y + _re_y) / 2.0
        _mouth_mid_y = (_ml_y + _mr_y) / 2.0

        _eye_span = np.abs(_re_x - _le_x)
        _nose_off_x = _nt_x - _eye_mid_x
        _half_span = np.maximum(_eye_span / 2.0, 1e-6)
        _yaw = np.degrees(np.arctan2(_nose_off_x, _half_span))

        _vert_line = _mouth_mid_y - _eye_mid_y
        _nose_offset_y = _nt_y - _eye_mid_y
        _pitch = np.degrees(np.arctan2(
            _nose_offset_y - _vert_line / 2.0,
            np.maximum(np.abs(_vert_line), 1e-6),
        ))

        _eye_dy = _re_y - _le_y
        _roll = np.degrees(np.arctan2(_eye_dy, np.maximum(_eye_span, 1e-6)))

        df["head_yaw"] = _yaw
        df["head_pitch"] = _pitch
        df["head_roll"] = _roll

        neutral_mask = (df["segment"] == "neutral") if "segment" in df.columns else pd.Series(False, index=df.index)
        if neutral_mask.any():
            _nm = neutral_mask.to_numpy(dtype=bool)
            ref_pose = {
                "yaw":   float(np.median(_yaw[_nm])),
                "pitch": float(np.median(_pitch[_nm])),
                "roll":  float(np.median(_roll[_nm])),
            }
            self.head_pose_normalizer.set_reference_pose(ref_pose)
            _d_yaw   = _yaw   - ref_pose["yaw"]
            _d_pitch = _pitch - ref_pose["pitch"]
            _d_roll  = _roll  - ref_pose["roll"]
            df["head_pose_deviation"] = np.sqrt(_d_yaw ** 2 + _d_pitch ** 2 + _d_roll ** 2)
        else:
            df["head_pose_deviation"] = 0.0

        return df

    def _add_relative_time(self, df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
        """Add segment-relative timestamps and propagate task info from the events table.

        For each segment-start event in events_df, computes time_rel_sec as the
        elapsed time since that segment began.  Also copies task_group, task_id,
        and task_name from the event onto the matching frame rows when the
        DataFrame does not already carry that information.  Any frames that remain
        unassigned after the per-segment pass get relative time relative to the
        first frame timestamp.
        """
        df["time_rel_sec"] = float("nan")

        if "timestamp_abs" in df.columns:
            df["timestamp_abs"] = pd.to_numeric(df["timestamp_abs"], errors="coerce").fillna(0.0)
        if events_df is not None and len(events_df) > 0:
            try:
                events_df = sanitize_events_df(events_df)
            except Exception:
                events_df = events_df.copy()
                if "timestamp_abs" in events_df.columns:
                    events_df["timestamp_abs"] = pd.to_numeric(
                        events_df["timestamp_abs"], errors="coerce"
                    )
                    events_df = events_df.dropna(subset=["timestamp_abs"]).reset_index(drop=True)

        has_existing_task_info = (
            "task_group" in df.columns
            and df["task_group"].notna().any()
            and (df["task_group"] != "0").any()
            and (df["task_group"] != "None").any()
        )

        for col, default in (("task_group", "0"), ("task_id", 0), ("task_name", "(no task selected)")):
            if col not in df.columns:
                df[col] = default

        if events_df is None or len(events_df) == 0:
            if "timestamp_abs" in df.columns and len(df) > 0:
                df["time_rel_sec"] = df["timestamp_abs"] - df["timestamp_abs"].iloc[0]
            else:
                df["time_rel_sec"] = 0.0
            return df

        segment_starts = events_df[events_df["event_type"].isin(["neutral", "measurement"])]

        if len(segment_starts) == 0:
            if len(df) > 0 and "timestamp_abs" in df.columns:
                df["time_rel_sec"] = df["timestamp_abs"] - df["timestamp_abs"].iloc[0]
            else:
                df["time_rel_sec"] = 0.0
            return df

        for _, event in segment_starts.iterrows():
            try:
                start_time = float(event["timestamp_abs"])
                segment_type = event["event_type"]
                mask = (df["segment"] == segment_type) & (df["timestamp_abs"] >= start_time)

                if not mask.any():
                    continue

                next_ends = events_df[
                    (events_df["timestamp_abs"] > start_time) & (events_df["event_type"] == "segment_end")
                ]
                if len(next_ends) > 0:
                    mask = mask & (df["timestamp_abs"] <= next_ends.iloc[0]["timestamp_abs"])

                df.loc[mask, "time_rel_sec"] = df.loc[mask, "timestamp_abs"] - start_time

                if not has_existing_task_info:
                    tg = event.get("task_group", "0") if pd.notna(event.get("task_group")) else "0"
                    tid = event.get("task_id", 0) if pd.notna(event.get("task_id")) else 0
                    tn = event.get("task_name", "(no task selected)") if pd.notna(event.get("task_name")) else "(no task selected)"
                    df.loc[mask, "task_group"] = str(tg) if tg else "0"
                    df.loc[mask, "task_id"] = int(tid) if tid else 0
                    df.loc[mask, "task_name"] = tn if tn else "(no task selected)"
            except Exception:
                continue

        unassigned = df["time_rel_sec"].isna()
        if unassigned.any() and "timestamp_abs" in df.columns and len(df) > 0:
            df.loc[unassigned, "time_rel_sec"] = (
                df.loc[unassigned, "timestamp_abs"] - df["timestamp_abs"].iloc[0]
            )

        df["time_rel_sec"] = df["time_rel_sec"].fillna(0.0)

        return df

    _LM_NOSE_TIP: int = 1
    _LM_CHEEK_L: List[int] = [116, 123, 147]
    _LM_CHEEK_R: List[int] = [345, 352, 376]
    _LM_NOSE_ALA_L: List[int] = [129, 49]
    _LM_NOSE_ALA_R: List[int] = [358, 279]
    _LM_COMMISSURE_L: int = 61
    _LM_COMMISSURE_R: int = 291
    _LM_LIP_UPPER_MID: int = 13
    _LM_LIP_LOWER_MID: int = 14
    _LM_INNER_CANTHUS_L: int = 133
    _LM_INNER_CANTHUS_R: int = 362
    _BLENDSHAPE_NOISE_THRESHOLD: float = 1e-4

    def _compute_asymmetry_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute left-right asymmetry difference and ratio features.

        Blendshape-based asymmetry is computed for all configured pairs.
        For cheekSquint and noseSneer regions whose blendshapes sit near the
        noise floor, landmark-based structural asymmetry is added instead using
        malar (cheek) and alar (nose) x-distances from the nose-tip midline.
        """
        cheek_blendshape_present = False
        nose_blendshape_present = False

        for pairs in self.symmetry_pairs.values():
            for left_feat, right_feat in pairs:
                if left_feat not in df.columns or right_feat not in df.columns:
                    continue
                left_vals = df[left_feat].values
                right_vals = df[right_feat].values
                base = left_feat.replace("Left", "").replace("Right", "")
                diff = right_vals - left_vals
                abs_diff = np.abs(diff)
                df[f"asymmetry_{base}"] = diff
                total = np.abs(left_vals) + np.abs(right_vals) + self.epsilon
                df[f"asymmetry_ratio_{base}"] = safe_divide(abs_diff, total)

                if "cheekSquint" in base.lower():
                    if float(np.nanmean(abs_diff)) >= self._BLENDSHAPE_NOISE_THRESHOLD:
                        cheek_blendshape_present = True
                if "noseSneer" in base.lower():
                    if float(np.nanmean(abs_diff)) >= self._BLENDSHAPE_NOISE_THRESHOLD:
                        nose_blendshape_present = True

        if "_landmarks_3d" in df.columns and (not cheek_blendshape_present or not nose_blendshape_present):
            _have_precomputed = (
                "asymmetry_commissure" in df.columns and "asymmetry_eyelid_ar" in df.columns
                and (cheek_blendshape_present or "asymmetry_cheekSquint" in df.columns)
                and (nose_blendshape_present or "asymmetry_noseSneer" in df.columns)
            )
            if not _have_precomputed:
                df = self._add_landmark_asymmetry(
                    df,
                    add_cheek=not cheek_blendshape_present,
                    add_nose=not nose_blendshape_present,
                )

        return df

    _LM_EYE_UPPER_L: List[int] = [159, 145, 160, 144]
    _LM_EYE_LOWER_L: List[int] = [386, 374, 387, 373]
    _LM_EYE_UPPER_R: List[int] = [386, 374, 387, 373]
    _LM_EYE_LOWER_R: List[int] = [159, 145, 160, 144]

    def _add_landmark_asymmetry(
        self,
        df: pd.DataFrame,
        add_cheek: bool = True,
        add_nose: bool = True,
    ) -> pd.DataFrame:
        """Supplement near-zero blendshape asymmetry with landmark-based distance measures.

        Computes interocular-normalised asymmetry so that results are invariant
        to camera distance and head angle changes.  All distances are divided by
        the inter-canthus distance (IOD) before computing ratios.

        Eyelid asymmetry uses the Sum-of-Distances Asymmetry Ratio (AR):
          AR = |SD_left - SD_right| / (SD_left + SD_right + epsilon)
        where SD is the summed Euclidean distance between upper/lower eyelid
        landmark pairs for each eye.

        Measures added to the DataFrame:
          - asymmetry_cheekSquint / asymmetry_ratio_cheekSquint: malar x-offsets
          - asymmetry_noseSneer / asymmetry_ratio_noseSneer: alar x-offsets
          - asymmetry_commissure / asymmetry_ratio_commissure: commissure droop
          - asymmetry_eyelid_ar / asymmetry_ratio_eyelid: eyelid AR

        Each ratio formula: |d_L - d_R| / (d_L + d_R + epsilon), yielding [0, 1].

        Rows with an unparseable _landmarks_3d JSON string or fewer landmark
        coordinates than required are left as NaN.
        """
        import json as _json

        min_lm_idx = max(
            self._LM_NOSE_TIP,
            max(self._LM_CHEEK_L), max(self._LM_CHEEK_R),
            max(self._LM_NOSE_ALA_L), max(self._LM_NOSE_ALA_R),
            self._LM_COMMISSURE_L, self._LM_COMMISSURE_R,
            self._LM_LIP_UPPER_MID, self._LM_LIP_LOWER_MID,
            self._LM_INNER_CANTHUS_L, self._LM_INNER_CANTHUS_R,
        )
        needed_len = (min_lm_idx + 1) * 3

        n_rows = len(df)
        cheek_asym    = np.full(n_rows, np.nan)
        cheek_signed  = np.full(n_rows, np.nan)
        nose_asym     = np.full(n_rows, np.nan)
        nose_signed   = np.full(n_rows, np.nan)
        comm_asym     = np.full(n_rows, np.nan)
        comm_signed   = np.full(n_rows, np.nan)
        eyelid_ar     = np.full(n_rows, np.nan)
        eyelid_signed = np.full(n_rows, np.nan)

        valid_idx: List[int] = []
        flat_list = []
        for row_i, lm_str in enumerate(df["_landmarks_3d"].values):
            try:
                flat = _json.loads(str(lm_str))
                if len(flat) >= needed_len:
                    valid_idx.append(row_i)
                    flat_list.append(flat)
            except Exception:
                pass

        if not flat_list:
            return df

        lm_all = np.array(flat_list, dtype=np.float32).reshape(len(flat_list), -1, 3)
        vidx = np.array(valid_idx)

        iod = np.linalg.norm(
            lm_all[:, self._LM_INNER_CANTHUS_L] - lm_all[:, self._LM_INNER_CANTHUS_R],
            axis=1,
        )
        iod = np.maximum(iod, 1e-6)

        nose_x = lm_all[:, self._LM_NOSE_TIP, 0]

        if add_cheek:
            x_L = lm_all[:, self._LM_CHEEK_L, 0].mean(axis=1)
            x_R = lm_all[:, self._LM_CHEEK_R, 0].mean(axis=1)
            dL = np.abs(x_L - nose_x) / iod
            dR = np.abs(x_R - nose_x) / iod
            cheek_signed[vidx] = dR - dL
            cheek_asym[vidx]   = np.abs(dL - dR) / (dL + dR + 1e-6)

        if add_nose:
            x_L = lm_all[:, self._LM_NOSE_ALA_L, 0].mean(axis=1)
            x_R = lm_all[:, self._LM_NOSE_ALA_R, 0].mean(axis=1)
            dL = np.abs(x_L - nose_x) / iod
            dR = np.abs(x_R - nose_x) / iod
            nose_signed[vidx] = dR - dL
            nose_asym[vidx]   = np.abs(dL - dR) / (dL + dR + 1e-6)

        lip_mid = (lm_all[:, self._LM_LIP_UPPER_MID] + lm_all[:, self._LM_LIP_LOWER_MID]) / 2.0
        dL_c = np.linalg.norm(lm_all[:, self._LM_COMMISSURE_L] - lip_mid, axis=1) / iod
        dR_c = np.linalg.norm(lm_all[:, self._LM_COMMISSURE_R] - lip_mid, axis=1) / iod
        comm_signed[vidx] = dR_c - dL_c
        comm_asym[vidx]   = np.abs(dL_c - dR_c) / (dL_c + dR_c + 1e-6)

        n_eye_lm = max(
            max(self._LM_EYE_UPPER_L), max(self._LM_EYE_LOWER_L),
            max(self._LM_EYE_UPPER_R), max(self._LM_EYE_LOWER_R),
        )
        if lm_all.shape[1] > n_eye_lm:
            sd_l = np.zeros(len(flat_list), dtype=np.float32)
            for u, lo in zip(self._LM_EYE_UPPER_L, self._LM_EYE_LOWER_L):
                sd_l += np.linalg.norm(lm_all[:, u] - lm_all[:, lo], axis=1)
            sd_l /= iod
            sd_r = np.zeros(len(flat_list), dtype=np.float32)
            for u, lo in zip(self._LM_EYE_UPPER_R, self._LM_EYE_LOWER_R):
                sd_r += np.linalg.norm(lm_all[:, u] - lm_all[:, lo], axis=1)
            sd_r /= iod
            eyelid_signed[vidx] = sd_r - sd_l
            eyelid_ar[vidx]     = np.abs(sd_l - sd_r) / (sd_l + sd_r + 1e-6)

        if add_cheek and not np.all(np.isnan(cheek_asym)):
            df["asymmetry_cheekSquint"]       = cheek_signed
            df["asymmetry_ratio_cheekSquint"] = cheek_asym
        if add_nose and not np.all(np.isnan(nose_asym)):
            df["asymmetry_noseSneer"]       = nose_signed
            df["asymmetry_ratio_noseSneer"] = nose_asym
        if not np.all(np.isnan(comm_asym)):
            df["asymmetry_commissure"]       = comm_signed
            df["asymmetry_ratio_commissure"] = comm_asym
        if not np.all(np.isnan(eyelid_ar)):
            df["asymmetry_eyelid_ar"]    = eyelid_signed
            df["asymmetry_ratio_eyelid"] = eyelid_ar

        return df

    def _compute_intensity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute range-normalised aggregate activation intensity features across blendshapes.

        Each blendshape is scaled by its expected physiological range so that
        features with inherently larger movement (e.g. mouth) become comparable
        to features with smaller ranges (e.g. brows).
        """
        present_blendshapes = [c for c in self.configured_blendshapes if c in df.columns]

        if not present_blendshapes:
            return df

        normalized = self._range_normalize_activations(df, present_blendshapes)
        df["mean_activation"] = normalized.mean(axis=1)
        df["max_activation"] = normalized.max(axis=1)
        df["activation_range"] = normalized.max(axis=1) - normalized.min(axis=1)
        return df

    def _range_normalize_activations(self, df: pd.DataFrame,
                                     blendshape_cols: List[str]) -> pd.DataFrame:
        """Normalise z-scored activations by the per-subject observed range.

        Uses subject-specific observed ranges (95th percentile from baseline
        measurement data) so each feature is expressed as a proportion of that
        subject's own movement capacity.  Falls back to config-defined activation
        ranges when empirical data is unavailable, and further falls back to
        z-score normalisation using baseline mean and std when those are provided.
        """
        normalized = pd.DataFrame(index=df.index)

        for col in blendshape_cols:
            obs_range = (
                self._observed_ranges.get(col)
                if self._observed_ranges
                else None
            )

            if obs_range and obs_range > self.epsilon:
                normalized[col] = df[col] / obs_range
            else:
                fallback_max = self.fallback_ranges.get(col, 1.0)
                if self._baseline_stats and col in self._baseline_stats:
                    baseline_std = max(self._baseline_stats[col].get("std", 1.0), self.epsilon)
                    baseline_mean = self._baseline_stats[col].get("mean", 0.0)
                    z_max = max((fallback_max - baseline_mean) / baseline_std, self.epsilon)
                    normalized[col] = df[col] / z_max
                else:
                    normalized[col] = df[col] / max(fallback_max, self.epsilon)

        return normalized

    def _build_activation_ranges(self) -> Dict[str, float]:
        """Flatten per-region activation_ranges config into a feature-to-max fallback dict."""
        ranges_config = self.features_config.get("activation_ranges", {})
        flat: Dict[str, float] = {}
        for region_ranges in ranges_config.values():
            if isinstance(region_ranges, dict):
                flat.update(region_ranges)
        return flat

    def _collect_blendshape_names(self) -> List[str]:
        """Return all blendshape names defined in the blendshapes section of features config."""
        blendshapes_config = self.features_config.get("blendshapes", {})
        names: List[str] = []
        for region_list in blendshapes_config.values():
            if isinstance(region_list, list):
                names.extend(region_list)
        return names

    def _compute_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute frame-level velocity and acceleration of mean activation.

        Uses finite differences on mean_activation over the absolute timestamp
        axis.  Frames where the inter-frame interval is zero are assigned 1/30 s
        to avoid division by zero.  Returns the DataFrame unchanged (with columns
        set to 0.0) when mean_activation is absent or fewer than two frames exist.
        """
        if "mean_activation" not in df.columns or len(df) <= 1:
            df["activation_velocity"] = 0.0
            df["activation_acceleration"] = 0.0
            return df

        dt = np.diff(df["timestamp_abs"].values)
        dt = np.concatenate([[dt[0] if len(dt) > 0 else 1 / 30], dt])
        dt[dt == 0] = 1 / 30

        activation_diff = np.diff(df["mean_activation"].values, prepend=df["mean_activation"].values[0])
        df["activation_velocity"] = activation_diff / dt

        velocity_diff = np.diff(df["activation_velocity"].values, prepend=df["activation_velocity"].values[0])
        df["activation_acceleration"] = velocity_diff / dt
        return df

    def extract_repetition_features(self, features_df: pd.DataFrame, repetition: int) -> pd.DataFrame:
        """Return feature rows for a single repetition with time reset to local zero.

        Returns an empty DataFrame when the repetition number is not found.
        """
        rep_df = features_df[features_df["repetition"] == repetition].copy()
        if len(rep_df) == 0:
            return pd.DataFrame()
        rep_df["time_rel_sec"] = rep_df["timestamp_abs"] - rep_df["timestamp_abs"].min()
        return rep_df

    def get_task_features(self, features_df: pd.DataFrame, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """Compute task-specific summary features from a features DataFrame.

        Computes mean, max, and std for each blendshape listed under
        primary_blendshapes in task_config, and mean asymmetry for each pair
        listed under symmetry_pairs.  Also computes duration_sec from either
        timestamp_abs or time_rel_sec.

        Returns an empty dict when features_df is empty.
        """
        task_features: Dict[str, Any] = {}

        for bs in task_config.get("primary_blendshapes", []):
            if bs not in features_df.columns:
                continue
            values = features_df[bs].dropna().values
            if len(values) > 0:
                task_features[f"{bs}_mean"] = float(np.mean(values))
                task_features[f"{bs}_max"] = float(np.max(values))
                task_features[f"{bs}_std"] = float(np.std(values))

        for left, right in task_config.get("symmetry_pairs", []):
            if left in features_df.columns and right in features_df.columns:
                l_vals = features_df[left].values
                r_vals = features_df[right].values
                asym = np.abs(r_vals - l_vals) / (np.abs(l_vals) + np.abs(r_vals) + self.epsilon)
                task_features[f'asymmetry_{left.replace("Left", "")}'] = float(np.mean(asym))

        if "timestamp_abs" in features_df.columns and len(features_df) > 0:
            _t = features_df["timestamp_abs"].dropna()
            task_features["duration_sec"] = float(_t.max() - _t.min()) if len(_t) >= 2 else 0.0
        elif "time_rel_sec" in features_df.columns and len(features_df) > 0:
            _tr = features_df["time_rel_sec"].dropna()
            task_features["duration_sec"] = float(_tr.max() - _tr.min()) if len(_tr) >= 2 else 0.0

        return task_features


def compute_spectral_arc_length(velocity: np.ndarray, fs: float = 30.0) -> float:
    """Compute the Spectral Arc Length (SPARC) of a velocity signal as a smoothness measure.

    The velocity signal is mean-centred, transformed to the frequency domain,
    and the normalised magnitude spectrum arc length is computed.  More negative
    values indicate jerkier motion.  Returns 0.0 for signals shorter than 4 frames.

    Reference: Balasubramanian et al. (2012) doi:10.1109/TBME.2011.2179545
    """
    if len(velocity) < 4:
        return 0.0
    velocity = velocity - np.mean(velocity)
    n = len(velocity)
    freq = np.fft.rfftfreq(n, d=1 / fs)
    magnitude = np.abs(np.fft.rfft(velocity))
    if np.max(magnitude) > 0:
        magnitude = magnitude / np.max(magnitude)
    return -float(np.sum(np.sqrt(np.diff(freq) ** 2 + np.diff(magnitude) ** 2)))


def compute_log_dimensionless_jerk(displacement: np.ndarray, fs: float = 30.0) -> float:
    """Compute Log Dimensionless Jerk (LDJ) for a discrete single-movement signal.

    Preferred over SPARC for short single-event signals (word production tasks)
    where SPARC is insensitive because all single-cycle velocity spectra share
    the same shape.  LDJ scales with movement complexity (more syllables produce
    more negative values) and with jerkiness (tremor or hesitation also produces
    more negative values), giving meaningful discrimination between healthy and
    impaired word production.

    Formula: LDJ = -log(T^3 * integral(jerk^2 dt) / A^2)

    More negative values mean jerkier motion.  Rough calibration at 30 fps:
      Smooth 1-syllable word (~0.5 s):   LDJ approx -8 to -10
      Smooth 4-syllable word (~2.0 s):   LDJ approx -13 to -16
      Impaired (tremor or hesitation):   LDJ approx -20 to -50

    Returns -50.0 on degenerate input (fewer than 6 frames or zero amplitude).

    Reference: Balasubramanian et al. (2012) A robust and sensitive metric for
    quantifying movement smoothness. IEEE Trans Biomed Eng 59(8):2126-2136.
    doi:10.1109/TBME.2011.2179545
    """
    if len(displacement) < 6:
        return -50.0
    dt = 1.0 / fs
    T = len(displacement) * dt
    A = float(np.ptp(displacement))
    if A < 1e-6:
        return -50.0
    jerk = np.gradient(np.gradient(np.gradient(displacement, dt), dt), dt)
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    jerk_power = float(_trapz(jerk ** 2, dx=dt))
    if jerk_power <= 0.0:
        return -10.0
    dj = (T ** 3 * jerk_power) / (A ** 2)
    return -float(np.log(dj))


def create_feature_extractor(features_config: Dict[str, Any], tasks_config: Dict[str, Any]) -> FeatureExtractor:
    """Factory: build a FeatureExtractor from configuration dicts."""
    return FeatureExtractor(features_config, tasks_config)
