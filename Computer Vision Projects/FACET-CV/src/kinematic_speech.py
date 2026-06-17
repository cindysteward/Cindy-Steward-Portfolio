"""
Kinematic speech feature extraction based on Palmer et al. (2024).

Computes the 13 clinical inter-landmark distance measurements described in:
    Palmer R. et al. "Facial Movements Extracted from Video for the
    Kinematic Classification of Speech." Sensors 2024, 24, 7235.
    https://doi.org/10.3390/s24227235

These measurements are derived from MediaPipe FaceLandmarker 3D landmark
positions (478 points) and are intended for task groups B and C.

The primacy of lower lip centre and bilateral lip corners as kinematically
independent regions during speech is supported by Lucero and Munhall (2008),
who showed via matrix factorisation that the lower lip centre (landmark ~199,
"pogonion") consistently has the largest displacement component across
speaking tasks:
    Lucero J.C. and Munhall K.G. "Analysis of Facial Motion Patterns During
    Speech Using a Matrix Factorization Algorithm." JASA 2008, 124, 2283.
    https://doi.org/10.1121/1.2973196

Group A extension
-----------------
Facial expression tasks (Group A) do not involve cyclical speech movements,
so the Palmer et al. speech-kinematics model does not apply directly.
For Group A, this module computes blendshape-derived kinematic descriptors:

    kin_a_peak_amplitude      -- maximum blendshape activation reached per repetition
    kin_a_time_to_peak_s      -- seconds from onset to peak activation
    kin_a_return_time_s       -- seconds from peak back to near-baseline
    kin_a_onset_time_s        -- time to first reach 25 % of peak (expression onset marker)
    kin_a_peak_asymmetry      -- left-minus-right activation ratio at peak frame
    kin_a_mean_velocity       -- mean absolute rate-of-change of activation signal
    kin_a_movement_smoothness -- spectral arc-length proxy for movement quality

The onset / apex / offset temporal segmentation of expressions follows the
framework described in Pantic (2009), who showed that the dynamic trajectory
of spontaneous vs. volitional facial expressions differs primarily in the
onset-to-apex slope and the offset symmetry:
    Pantic M. "Machine analysis of facial behaviour: naturalistic and dynamic
    behaviour." Phil Trans R Soc B 2009, 364, 3505-3513.
    https://doi.org/10.1098/rstb.2009.0135

These are extracted by ``extract_group_a_kinematics`` which operates on the
features_df columns (blendshapes and asymmetry ratios) without requiring
3D landmarks, making it robust when landmark storage is disabled.

DDK Clinical Metrics (Group B)
-------------------------------
For Group B diadochokinetic tasks, ``compute_ddk_clinical_metrics`` computes
cycle-level kinematic variables validated in clinical motor-speech research:

    ddk_D_mean              -- mean peak-to-peak lip displacement across cycles
    ddk_D_max               -- maximum peak displacement
    ddk_Tsd                 -- cross-cycle temporal SD (timing variability)
    ddk_STI                 -- spatiotemporal index: SD of normalised displacement at 10 time points
    ddk_Duration_s          -- total articulation time of the DDK sequence
    ddk_Num_Cycles          -- number of detected open-close cycles
    ddk_rate_hz             -- DDK rate (cycles per second)
    ddk_speed_pctN          -- Nth percentile of |velocity| signal (N = 25, 50, 75, 95)

Allison et al. (2022) showed that ddk_Tsd, ddk_STI, ddk_D_mean, ddk_Duration,
and ddk_Num_Cycles each achieved 88 % sensitivity / 88 % specificity for
detecting subtle motor involvement in cerebral palsy despite high intelligibility:
    Allison K.M. et al. "Use of Automated Kinematic Diadochokinesis Analysis to
    Identify Potential Indicators of Speech Motor Involvement in Children with
    Cerebral Palsy." AJSLP 2022, 31, 1682-1696.
    https://doi.org/10.1044/2022_ajslp-21-00241

Speed percentile features (Ls25-Ls95) follow Simmatis et al. (2023), who
demonstrated good-to-strong agreement (ICC-A >= 0.70) between webcam-based
and gold-standard EMA kinematic measures for these features. The same study
found that symmetry features have consistently poor test-retest reliability
across all recording systems and should be treated with caution as primary
indicators:
    Simmatis L.E.R. et al. "Analytical Validation of a Webcam-Based Assessment
    of Speech Kinematics." Folia Phoniatr Logop 2023, 75, 253-265.
    https://doi.org/10.1159/000529685

Speech-specific AU mapping
--------------------------
SPEECH_SPECIFIC_AU_MAPPING documents which FACS Action Units are selectively
active during speech production based on Newby et al. (2025), who isolated
11 speech-specific AUs from 17 phonemes in healthy adults:
    Newby et al. "The Role of Facial Action Units in Investigating Facial
    Movements During Speech." Electronics 2025, 14, 2066.
    https://doi.org/10.3390/electronics14102066

Landmark ID references are from the MediaPipe FaceLandmarker 478-point mesh.
The paper uses "BlazeFace" which is the same mesh.

Significant measurements per the paper (F1 > significance threshold across
all three cohorts): Mouth Opening, Lip Action (Y/Z),
Medial 1/3 Upper Action (Y/Z), Medial 1/3 Lower Action (Y/Z),
Pogonion (Y/Z), Lower Lip from Pogonion (Y), Labial Fissure Width,
Mandibular Angle (velocity/acceleration form).

Automated DDK segmentation precedent
-------------------------------------
Segal et al. (2022) (DDKtor) trained CNN/LSTM models to segment DDK audio
into VOT/vowel/silence at 1 ms resolution, achieving r=0.94-0.99 for DDK
rate vs manual annotation. The cycle-detection approach used here follows
the same principle (detect peaks in the lip-opening signal) adapted to the
visual kinematic domain rather than audio:
    Segal O. et al. "DDKtor: Automatic Diadochokinetic Speech Analysis."
    arXiv 2022, arXiv:2206.14639. https://arxiv.org/abs/2206.14639

Wiltshire et al. (2024) confirmed that kinematic variability across
repetitions distinguishes typical from atypical speech motor control,
supporting the use of Tsd and STI as primary screening features:
    Wiltshire C.E.E. et al. "Characterising Kinematic Variability in the
    Speech of Adults." PLOS ONE 2024.
    https://doi.org/10.1371/journal.pone.0309612
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("pipeline")

SPEECH_SPECIFIC_AU_MAPPING: Dict[int, Dict[str, Any]] = {
    9:  {"name": "Nose Wrinkler / Levator labii",
         "blendshapes": ["noseSneerLeft", "noseSneerRight"],
         "speech_role": "labial retraction during /f/ /v/"},
    11: {"name": "Nasolabial Deepener / Zygomaticus minor",
         "blendshapes": ["mouthSmileLeft", "mouthSmileRight"],
         "speech_role": "lip corner elevation during high front vowels"},
    12: {"name": "Lip Corner Puller / Zygomaticus major",
         "blendshapes": ["mouthSmileLeft", "mouthSmileRight"],
         "speech_role": "dominant for vowels /æ/ /ə/ /ʊ/ and bilabial stops /b/ /p/"},
    14: {"name": "Dimpler / Buccinator",
         "blendshapes": ["mouthDimpleLeft", "mouthDimpleRight"],
         "speech_role": "cheek tension during pressure consonants"},
    15: {"name": "Lip Corner Depressor / Depressor anguli oris",
         "blendshapes": ["mouthFrownLeft", "mouthFrownRight"],
         "speech_role": "lip corner depression during /a/ /ɔ/"},
    17: {"name": "Chin Raiser / Mentalis",
         "blendshapes": ["mouthShrugLower"],
         "speech_role": "dominant for vowel /ɪ/; chin lift in final syllables"},
    20: {"name": "Lip Stretcher / Risorius + Platysma",
         "blendshapes": ["mouthStretchLeft", "mouthStretchRight"],
         "speech_role": "lip retraction during /s/ /z/ and fricatives"},
    23: {"name": "Lip Tightener / Orbicularis oris",
         "blendshapes": ["mouthFunnel", "mouthPucker"],
         "speech_role": "lip rounding and tightening during /w/ /u/"},
    24: {"name": "Lip Pressor / Orbicularis oris (press)",
         "blendshapes": ["mouthClose"],
         "speech_role": "lip closure for bilabial stops /p/ /b/ /m/"},
    25: {"name": "Lip Part / Depressor labii",
         "blendshapes": ["mouthOpen"],
         "speech_role": "lip parting; most consistently active across phonemes"},
    26: {"name": "Jaw Drop / Masseter + pterygoids",
         "blendshapes": ["jawOpen"],
         "speech_role": "jaw opening; primary articulator for vowel height"},
}


def get_speech_au_blendshapes() -> List[str]:
    """Return the unique MediaPipe blendshape names that correspond to speech-specific AUs.

    Useful for targeted feature selection in articulation analysis.
    Based on Newby et al. (2025) Electronics 14, 2066.
    https://doi.org/10.3390/electronics14102066
    """
    seen: set = set()
    names: List[str] = []
    for entry in SPEECH_SPECIFIC_AU_MAPPING.values():
        for bs in entry["blendshapes"]:
            if bs not in seen:
                seen.add(bs)
                names.append(bs)
    return names


def compute_ddk_clinical_metrics(
    signal: np.ndarray,
    fps: float = 30.0,
    min_cycles: int = 2,
) -> Dict[str, float]:
    """Compute cycle-level DDK kinematic variables from a 1-D lip-opening signal.

    Detects open-close cycles by finding local maxima (open peaks) and
    computing the spatiotemporal metrics validated in Allison et al. (2022)
    and the speed percentile features from Simmatis et al. (2023).

    Parameters
    ----------
    signal : 1-D array of kin_mouth_opening (or similar) values over time
    fps    : frame rate used to convert frame counts to seconds
    min_cycles : minimum number of detected cycles to produce valid metrics
                 (below this returns NaN-filled dict)

    Returns
    -------
    Dict with keys:
        ddk_D_mean      : mean peak displacement across cycles
        ddk_D_max       : maximum peak displacement
        ddk_Tsd         : cross-cycle temporal SD (timing variability, seconds)
        ddk_STI         : spatiotemporal index (SD at 10 time-normalised points)
        ddk_Duration_s  : total articulation duration (seconds)
        ddk_Num_Cycles  : number of detected open-close cycles
        ddk_rate_hz     : DDK rate (cycles / second)
        ddk_speed_pct25 : 25th percentile of |velocity| signal
        ddk_speed_pct50 : 50th percentile of |velocity| signal
        ddk_speed_pct75 : 75th percentile of |velocity| signal
        ddk_speed_pct95 : 95th percentile of |velocity| signal

    References
    ----------
    Allison et al. (2022) AJSLP 31, 1682.  https://doi.org/10.1044/2022_ajslp-21-00241
        STI, Tsd, D_mean, Duration, Num_Cycles achieve 88 % sensitivity /
        specificity for subtle motor involvement in cerebral palsy.
    Simmatis et al. (2023) Folia Phoniatr Logop 75, 253.
        https://doi.org/10.1159/000529685
        Speed percentiles Ls25-Ls95 show ICC-A >= 0.70 vs. EMA gold standard.
    Segal et al. (2022) DDKtor. arXiv:2206.14639. https://arxiv.org/abs/2206.14639
        Automated DDK cycle segmentation (CNN/LSTM), r=0.94-0.99 vs manual.
        The peak-detection cycle-finding approach here adapts this to lip
        kinematics from video.
    Wiltshire et al. (2024) PLOS ONE. https://doi.org/10.1371/journal.pone.0309612
        Kinematic variability across repetitions distinguishes typical from
        atypical speech motor control, supporting Tsd and STI as key metrics.
    """
    _nan: Dict[str, float] = {
        "ddk_D_mean": float("nan"),
        "ddk_D_max": float("nan"),
        "ddk_Tsd": float("nan"),
        "ddk_STI": float("nan"),
        "ddk_Duration_s": float("nan"),
        "ddk_Num_Cycles": float("nan"),
        "ddk_rate_hz": float("nan"),
        "ddk_speed_pct25": float("nan"),
        "ddk_speed_pct50": float("nan"),
        "ddk_speed_pct75": float("nan"),
        "ddk_speed_pct95": float("nan"),
    }

    if signal is None or len(signal) < 6:
        return _nan

    sig = np.array(signal, dtype=float)
    dt = 1.0 / fps
    duration_s = float(len(sig)) * dt

    vel = np.gradient(sig, dt)
    abs_vel = np.abs(vel)
    pct25, pct50, pct75, pct95 = (
        float(np.percentile(abs_vel, p)) for p in (25, 50, 75, 95)
    )

    half_range = float((np.max(sig) - np.min(sig)) / 2.0)
    if half_range < 1e-6:
        return {
            **_nan,
            "ddk_Duration_s": duration_s,
            "ddk_speed_pct25": pct25,
            "ddk_speed_pct50": pct50,
            "ddk_speed_pct75": pct75,
            "ddk_speed_pct95": pct95,
        }

    threshold = float(np.min(sig)) + half_range * 0.5
    peaks: List[int] = []
    for i in range(1, len(sig) - 1):
        if sig[i] > sig[i - 1] and sig[i] >= sig[i + 1] and sig[i] >= threshold:
            if peaks and (i - peaks[-1]) < max(2, int(fps / 4)):
                if sig[i] > sig[peaks[-1]]:
                    peaks[-1] = i
            else:
                peaks.append(i)

    n_cycles = len(peaks)
    if n_cycles < min_cycles:
        return {
            **_nan,
            "ddk_Duration_s": duration_s,
            "ddk_speed_pct25": pct25,
            "ddk_speed_pct50": pct50,
            "ddk_speed_pct75": pct75,
            "ddk_speed_pct95": pct95,
        }

    peak_vals = np.array([sig[p] for p in peaks], dtype=float)
    D_mean = float(np.mean(peak_vals))
    D_max = float(np.max(peak_vals))

    if n_cycles >= 2:
        intervals = np.diff(peaks).astype(float) * dt
        Tsd = float(np.std(intervals))
    else:
        Tsd = 0.0

    _n_points = 10
    cycle_profiles: List[np.ndarray] = []
    for k in range(n_cycles - 1):
        start_f = peaks[k]
        end_f = peaks[k + 1]
        if end_f <= start_f:
            continue
        cycle_sig = sig[start_f:end_f]
        interp_grid = np.linspace(0, 1, _n_points)
        x_orig = np.linspace(0, 1, len(cycle_sig))
        cycle_profiles.append(np.interp(interp_grid, x_orig, cycle_sig))

    if len(cycle_profiles) >= 2:
        profile_arr = np.stack(cycle_profiles)
        STI = float(np.mean(np.std(profile_arr, axis=0)))
    else:
        STI = float("nan")

    ddk_rate = float(n_cycles) / duration_s if duration_s > 0 else float("nan")

    return {
        "ddk_D_mean": D_mean,
        "ddk_D_max": D_max,
        "ddk_Tsd": Tsd,
        "ddk_STI": STI,
        "ddk_Duration_s": duration_s,
        "ddk_Num_Cycles": float(n_cycles),
        "ddk_rate_hz": ddk_rate,
        "ddk_speed_pct25": pct25,
        "ddk_speed_pct50": pct50,
        "ddk_speed_pct75": pct75,
        "ddk_speed_pct95": pct95,
    }


_LM = {
    "pogonion":          199,
    "stomion_sup":        13,
    "stomion_inf":        14,
    "labrale_inf":        17,
    "subnasale":           2,
    "mid_labial_sup_L":   81,
    "mid_labial_sup_R":  311,
    "mid_labial_inf_L":  178,
    "mid_labial_inf_R":  402,
    "cheilion_L":         61,
    "cheilion_R":        291,
    "gonion_L":          172,
    "gonion_R":          397,
    "tragion_L":          93,
    "tragion_R":         323,
    "glabella":          168,
    "sellion":           227,
    "zygion_L":          447,
    "inner_canthus_L":   133,
    "inner_canthus_R":   362,
    "frontotemporale_L":   9,
}

_FACE_SIZE_PAIRS = [
    ("tragion_L", "tragion_R"),
    ("glabella", "tragion_L"),
    ("glabella", "tragion_R"),
    ("inner_canthus_L", "inner_canthus_R"),
    ("frontotemporale_L", "tragion_L"),
]


def _lm(lm_arr: np.ndarray, name: str) -> np.ndarray:
    """Return the 3D position of a named landmark. Shape: (3,)"""
    return lm_arr[_LM[name]]


def _avg(*names: str, lm_arr: np.ndarray) -> np.ndarray:
    """Return the mean 3D position of multiple landmarks."""
    return np.mean([lm_arr[_LM[n]] for n in names], axis=0)


def _project(vec: np.ndarray, head_axes: np.ndarray) -> np.ndarray:
    """Project a 3D vector onto head axes. Returns (x_proj, y_proj, z_proj)."""
    return head_axes @ vec


def _dist(p1: np.ndarray, p2: np.ndarray) -> float:
    """Return Euclidean distance between two landmark points."""
    return float(np.linalg.norm(p1 - p2))


def _triangle_area_2d(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Area of triangle in the Y-Z plane (frontal view after head-pose correction)."""
    v1 = p2 - p1
    v2 = p3 - p1
    return abs(v1[0] * v2[1] - v1[1] * v2[0]) / 2.0


def compute_face_size(lm_arr: np.ndarray) -> float:
    """
    Compute the Face Size normalisation scalar.

    Defined as the weighted mean of several inter-landmark distances around the
    forehead and eyes, designed to vary minimally during speech (see paper §2.3.1
    and Table 3). Weights are equal here; the paper derived them empirically from
    adults but equal weights are a robust approximation.
    """
    dists = []
    for n1, n2 in _FACE_SIZE_PAIRS:
        try:
            dists.append(_dist(_lm(lm_arr, n1), _lm(lm_arr, n2)))
        except (IndexError, KeyError):
            pass
    return float(np.mean(dists)) if dists else 1.0


def estimate_head_axes(lm_arr: np.ndarray) -> np.ndarray:
    """
    Estimate head coordinate axes from stable upper-face landmarks.

    Returns a (3, 3) matrix where:
      row 0 = lateral (left→right, X)
      row 1 = vertical (inferior→superior, Y, inverted from image coords)
      row 2 = depth (posterior→anterior, Z)

    Uses tragion (temples) for X, and glabella-to-midpoint-of-tragions for Y.
    Z = X × Y (right-hand rule).
    """
    try:
        t_L = _lm(lm_arr, "tragion_L")
        t_R = _lm(lm_arr, "tragion_R")
        glab = _lm(lm_arr, "glabella")

        x_axis = t_R - t_L
        norm_x = np.linalg.norm(x_axis)
        if norm_x < 1e-6:
            return np.eye(3, dtype=np.float32)
        x_axis = x_axis / norm_x

        mid_t = (t_L + t_R) / 2.0
        y_axis = glab - mid_t
        y_axis = y_axis - np.dot(y_axis, x_axis) * x_axis
        norm_y = np.linalg.norm(y_axis)
        if norm_y < 1e-6:
            y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            y_axis = y_axis / norm_y

        z_axis = np.cross(x_axis, y_axis)
        norm_z = np.linalg.norm(z_axis)
        if norm_z > 1e-6:
            z_axis = z_axis / norm_z

        return np.stack([x_axis, y_axis, z_axis]).astype(np.float32)
    except (IndexError, KeyError):
        return np.eye(3, dtype=np.float32)


def compute_kinematic_frame(
    lm_arr: np.ndarray,
    face_size: float,
    head_axes: np.ndarray,
    origin: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute all 13 clinical kinematic measurements for one frame.

    Returns a flat dict with keys:
      kin_face_size
      kin_mouth_opening                   (scalar, ratio, NOT normalised)
      kin_labial_fissure_width            (scalar, normalised)
      kin_mouth_height                    (scalar, normalised)
      kin_lip_action_{x,y,z}             (3 values, normalised)
      kin_medial_sym_{x,y,z}             (3 values, normalised)
      kin_medial_action_{x,y,z}          (3 values, normalised)
      kin_medial_lower_action_{x,y,z}    (3 values, normalised)
      kin_medial_upper_action_{x,y,z}    (3 values, normalised)
      kin_mouth_area_symmetry             (scalar, normalised by face_size²)
      kin_pogonion_{x,y,z}              (3 values, normalised)
      kin_gonion_{x,y,z}                (3 values, normalised)
      kin_mandibular_angle               (scalar, degrees, NOT normalised)
      kin_lower_lip_from_pogonion_{x,y,z} (3 values, normalised)
    """
    fs = max(face_size, 1e-6)
    result: Dict[str, float] = {"kin_face_size": fs}

    if origin is None:
        try:
            origin = (_lm(lm_arr, "tragion_L") + _lm(lm_arr, "tragion_R")) / 2.0
        except (IndexError, KeyError):
            origin = np.zeros(3, dtype=np.float32)

    def proj_norm(vec):
        """Project vector onto head axes and normalise by face size."""
        p = _project(vec, head_axes) / fs
        return p[0], p[1], p[2]

    try:
        stom_sup = _lm(lm_arr, "stomion_sup")
        stom_inf = _lm(lm_arr, "stomion_inf")
        ch_L = _lm(lm_arr, "cheilion_L")
        ch_R = _lm(lm_arr, "cheilion_R")
        vertical_opening = abs(_project(stom_inf - stom_sup, head_axes)[1])
        fissure_width    = abs(_project(ch_R - ch_L, head_axes)[0]) + 1e-6
        result["kin_mouth_opening"] = float(np.clip(vertical_opening / fissure_width, 0.0, 5.0))
    except (IndexError, KeyError):
        result["kin_mouth_opening"] = 0.0

    try:
        lfw = abs(_project(_lm(lm_arr, "cheilion_R") - _lm(lm_arr, "cheilion_L"), head_axes)[0]) / fs
        result["kin_labial_fissure_width"] = float(lfw)
    except (IndexError, KeyError):
        result["kin_labial_fissure_width"] = 0.0

    try:
        stom_mid = (_lm(lm_arr, "stomion_sup") + _lm(lm_arr, "stomion_inf")) / 2.0
        mh = abs(_project(stom_mid - origin, head_axes)[1]) / fs
        result["kin_mouth_height"] = float(mh)
    except (IndexError, KeyError):
        result["kin_mouth_height"] = 0.0

    try:
        sup_mid = _avg("mid_labial_sup_L", "mid_labial_sup_R", lm_arr=lm_arr)
        inf_mid = _avg("mid_labial_inf_L", "mid_labial_inf_R", lm_arr=lm_arr)
        x, y, z = proj_norm(inf_mid - sup_mid)
        result["kin_lip_action_x"] = x
        result["kin_lip_action_y"] = y
        result["kin_lip_action_z"] = z
    except (IndexError, KeyError):
        for ax in "xyz":
            result[f"kin_lip_action_{ax}"] = 0.0

    try:
        sup_L = _lm(lm_arr, "mid_labial_sup_L")
        sup_R = _lm(lm_arr, "mid_labial_sup_R")
        x, y, z = proj_norm(sup_R - sup_L)
        result["kin_medial_sym_x"] = x
        result["kin_medial_sym_y"] = y
        result["kin_medial_sym_z"] = z
    except (IndexError, KeyError):
        for ax in "xyz":
            result[f"kin_medial_sym_{ax}"] = 0.0

    try:
        mid_mean = _avg("mid_labial_sup_L", "mid_labial_sup_R",
                        "mid_labial_inf_L", "mid_labial_inf_R", lm_arr=lm_arr)
        ch_mean = _avg("cheilion_L", "cheilion_R", lm_arr=lm_arr)
        x, y, z = proj_norm(mid_mean - ch_mean)
        result["kin_medial_action_x"] = x
        result["kin_medial_action_y"] = y
        result["kin_medial_action_z"] = z
    except (IndexError, KeyError):
        for ax in "xyz":
            result[f"kin_medial_action_{ax}"] = 0.0

    try:
        inf_mid2 = _avg("mid_labial_inf_L", "mid_labial_inf_R", lm_arr=lm_arr)
        ch_mean2 = _avg("cheilion_L", "cheilion_R", lm_arr=lm_arr)
        x, y, z = proj_norm(inf_mid2 - ch_mean2)
        result["kin_medial_lower_action_x"] = x
        result["kin_medial_lower_action_y"] = y
        result["kin_medial_lower_action_z"] = z
    except (IndexError, KeyError):
        for ax in "xyz":
            result[f"kin_medial_lower_action_{ax}"] = 0.0

    try:
        sup_mid2 = _avg("mid_labial_sup_L", "mid_labial_sup_R", lm_arr=lm_arr)
        ch_mean3 = _avg("cheilion_L", "cheilion_R", lm_arr=lm_arr)
        x, y, z = proj_norm(sup_mid2 - ch_mean3)
        result["kin_medial_upper_action_x"] = x
        result["kin_medial_upper_action_y"] = y
        result["kin_medial_upper_action_z"] = z
    except (IndexError, KeyError):
        for ax in "xyz":
            result[f"kin_medial_upper_action_{ax}"] = 0.0

    try:
        ss = _lm(lm_arr, "stomion_sup")
        si = _lm(lm_arr, "stomion_inf")
        cL = _lm(lm_arr, "cheilion_L")
        cR = _lm(lm_arr, "cheilion_R")
        def to2d(p):
            """Project a 3D landmark to the 2D frontal plane after head-pose correction."""
            proj = _project(p - origin, head_axes)
            return proj[1:3]
        area_L = _triangle_area_2d(to2d(ss), to2d(si), to2d(cL))
        area_R = _triangle_area_2d(to2d(ss), to2d(si), to2d(cR))
        result["kin_mouth_area_symmetry"] = float(abs(area_L - area_R) / (fs * fs + 1e-8))
    except (IndexError, KeyError):
        result["kin_mouth_area_symmetry"] = 0.0

    try:
        x, y, z = proj_norm(_lm(lm_arr, "pogonion") - origin)
        result["kin_pogonion_x"] = x
        result["kin_pogonion_y"] = y
        result["kin_pogonion_z"] = z
    except (IndexError, KeyError):
        for ax in "xyz":
            result[f"kin_pogonion_{ax}"] = 0.0

    try:
        gon_mean = _avg("gonion_L", "gonion_R", lm_arr=lm_arr)
        x, y, z = proj_norm(gon_mean - origin)
        result["kin_gonion_x"] = x
        result["kin_gonion_y"] = y
        result["kin_gonion_z"] = z
    except (IndexError, KeyError):
        for ax in "xyz":
            result[f"kin_gonion_{ax}"] = 0.0

    try:
        pog = _lm(lm_arr, "pogonion") - origin
        gon = _avg("gonion_L", "gonion_R", lm_arr=lm_arr) - origin
        diff = gon - pog
        diff_proj = _project(diff, head_axes)
        angle_deg = float(np.degrees(np.arctan2(diff_proj[1], diff_proj[2] + 1e-9)))
        result["kin_mandibular_angle"] = angle_deg
    except (IndexError, KeyError):
        result["kin_mandibular_angle"] = 0.0

    try:
        pog2 = _lm(lm_arr, "pogonion")
        lab_inf = _lm(lm_arr, "labrale_inf")
        x, y, z = proj_norm(lab_inf - pog2)
        result["kin_lower_lip_from_pog_x"] = x
        result["kin_lower_lip_from_pog_y"] = y
        result["kin_lower_lip_from_pog_z"] = z
    except (IndexError, KeyError):
        for ax in "xyz":
            result[f"kin_lower_lip_from_pog_{ax}"] = 0.0

    return result


def extract_kinematic_features(
    features_df: pd.DataFrame,
    task_groups: Optional[List[str]] = None,
    neutral_face_size: Optional[float] = None,
) -> pd.DataFrame:
    """
    Compute kinematic features for every frame that has '_landmarks_3d'.

    task_groups: if given (e.g. ['B', 'C']), only process frames in those groups.
    neutral_face_size: if provided, normalise all frames by the neutral-segment
        face size rather than per-frame face size (more stable, matches paper).

    Returns a DataFrame with the same index as features_df, columns 'kin_*'.
    """
    if "_landmarks_3d" not in features_df.columns:
        logger.warning(
            "kinematic_speech: '_landmarks_3d' column not found — "
            "was 3D landmark storage enabled in multi_camera_processor?"
        )
        return pd.DataFrame(index=features_df.index)

    if task_groups:
        mask = features_df.get("task_group", pd.Series("0", index=features_df.index)).isin(task_groups)
    else:
        mask = pd.Series(True, index=features_df.index)

    if neutral_face_size is None and "segment" in features_df.columns:
        neutral_rows = features_df[features_df["segment"] == "neutral"]
        if len(neutral_rows) > 0:
            fs_vals = []
            for raw in neutral_rows["_landmarks_3d"]:
                lm = _parse_landmarks(raw)
                if lm is not None:
                    fs_vals.append(compute_face_size(lm))
            if fs_vals:
                neutral_face_size = float(np.median(fs_vals))

    rows = []
    for idx, row in features_df[mask].iterrows():
        lm = _parse_landmarks(row["_landmarks_3d"])
        if lm is None:
            rows.append((idx, {}))
            continue
        fs = neutral_face_size if neutral_face_size else compute_face_size(lm)
        axes = estimate_head_axes(lm)
        kin = compute_kinematic_frame(lm, fs, axes)
        rows.append((idx, kin))

    if not rows:
        return pd.DataFrame(index=features_df.index)

    kin_df = pd.DataFrame.from_records(
        [r for _, r in rows],
        index=[i for i, _ in rows],
    ).reindex(features_df.index)

    return kin_df


def _parse_landmarks(raw) -> Optional[np.ndarray]:
    """Parse a stored '_landmarks_3d' value back to a (478, 3) float32 array."""
    try:
        if isinstance(raw, np.ndarray):
            return raw.reshape(478, 3).astype(np.float32)
        if isinstance(raw, (list, tuple)):
            arr = np.array(raw, dtype=np.float32)
            if arr.size == 478 * 3:
                return arr.reshape(478, 3)
        if isinstance(raw, str) and raw.startswith("["):
            import json as _json
            arr = np.array(_json.loads(raw), dtype=np.float32)
            if arr.size == 478 * 3:
                return arr.reshape(478, 3)
        return None
    except Exception:
        return None


def add_kinematic_derivatives(kin_df: pd.DataFrame, fps: float = 30.0) -> pd.DataFrame:
    """
    Add first and second-order temporal derivatives (velocity, acceleration)
    for each kinematic measurement column.

    This matches the paper's approach: displacement, velocity, and acceleration
    were each evaluated independently for classification.
    Column naming: kin_<name>_vel, kin_<name>_acc
    """
    kin_df = kin_df.copy()
    base_cols = [c for c in kin_df.columns if c.startswith("kin_")]
    dt = 1.0 / fps
    for col in base_cols:
        vals = kin_df[col].fillna(0.0).to_numpy()
        vel = np.gradient(vals, dt)
        acc = np.gradient(vel, dt)
        kin_df[f"{col}_vel"] = vel
        kin_df[f"{col}_acc"] = acc
    return kin_df


_GROUP_A_PRIMARY_BLENDSHAPES = {
    1: ["mouthPucker", "mouthFunnel"],
    2: ["mouthSmileLeft", "mouthSmileRight"],
    3: ["mouthSmileLeft", "mouthSmileRight", "jawOpen"],
    4: ["tongueOut"],
    5: ["tongueOut"],
    6: ["tongueOut"],
    7: ["browDownLeft", "browDownRight", "mouthFrownLeft", "mouthFrownRight"],
    8: ["cheekPuff"],
    9: ["browOuterUpLeft", "browOuterUpRight", "browInnerUp"],
}

_GROUP_A_SYMMETRY_PAIRS = {
    1: ("mouthLeft", "mouthRight"),
    2: ("mouthSmileLeft", "mouthSmileRight"),
    3: ("mouthSmileLeft", "mouthSmileRight"),
    7: ("browDownLeft", "browDownRight"),
    8: ("cheekSquintLeft", "cheekSquintRight"),
    9: ("browOuterUpLeft", "browOuterUpRight"),
}

_FALLBACK_ASYM_RATIO_COLS = (
    "asymmetry_ratio_mouthSmile",
    "asymmetry_ratio_mouthStretch",
    "asymmetry_ratio_mouthFrown",
    "asymmetry_ratio_mouthLowerDown",
    "asymmetry_ratio_browDown",
    "asymmetry_ratio_browOuterUp",
    "asymmetry_ratio_cheekSquint",
    "asymmetry_ratio_eyeBlink",
    "asymmetry_ratio_eyeSquint",
)


def extract_group_a_kinematics(
    features_df: pd.DataFrame,
    fps: float = 30.0,
) -> pd.DataFrame:
    """
    Compute blendshape-derived kinematic descriptors for Group A (facial expression) tasks.

    Operates on the blendshape activation columns already present in features_df,
    so no 3D landmark storage is required.  Returns a DataFrame with the same index
    containing per-frame Group A kinematic columns (prefix 'kin_a_').

    Columns produced (per frame, relative to the task segment):
        kin_a_mean_activation    : mean of primary blendshapes for this task
        kin_a_peak_amplitude     : running maximum of mean_activation within segment
        kin_a_velocity           : first derivative of mean_activation (activation/s)
        kin_a_acceleration       : second derivative
        kin_a_asymmetry          : signed left-right asymmetry of primary pair (if defined)
        kin_a_abs_asymmetry      : absolute asymmetry magnitude
    """
    result_rows = []

    group_a_mask = (
        features_df.get("task_group", pd.Series("0", index=features_df.index)) == "A"
    )
    a_df = features_df[group_a_mask].copy()

    if a_df.empty:
        return pd.DataFrame(index=features_df.index)

    dt = 1.0 / fps

    for (tid, rep), seg_df in a_df.groupby(["task_id", "repetition"], sort=True):
        tid_int = int(tid) if pd.notna(tid) else 0
        primary = _GROUP_A_PRIMARY_BLENDSHAPES.get(tid_int, [])
        present = [c for c in primary if c in seg_df.columns]

        if present:
            mean_act = seg_df[present].mean(axis=1).fillna(0.0).to_numpy()
        elif "mean_activation" in seg_df.columns:
            mean_act = seg_df["mean_activation"].fillna(0.0).to_numpy()
        else:
            mean_act = np.zeros(len(seg_df))

        vel = np.gradient(mean_act, dt) if len(mean_act) > 1 else np.zeros_like(mean_act)
        acc = np.gradient(vel, dt) if len(vel) > 1 else np.zeros_like(vel)

        running_peak = np.maximum.accumulate(mean_act)

        sym_pair = _GROUP_A_SYMMETRY_PAIRS.get(tid_int)
        if sym_pair and sym_pair[0] in seg_df.columns and sym_pair[1] in seg_df.columns:
            left_vals = seg_df[sym_pair[0]].fillna(0.0).to_numpy()
            right_vals = seg_df[sym_pair[1]].fillna(0.0).to_numpy()
            denom = np.maximum(np.abs(left_vals) + np.abs(right_vals), 1e-6)
            asym = (left_vals - right_vals) / denom
            abs_asym = np.abs(asym)
        else:
            avail = [c for c in _FALLBACK_ASYM_RATIO_COLS if c in seg_df.columns]
            if avail:
                asym_vals = seg_df[avail].fillna(0.0).mean(axis=1).to_numpy()
                asym = asym_vals
                abs_asym = asym_vals
            else:
                asym = np.zeros(len(seg_df))
                abs_asym = np.zeros(len(seg_df))

        for i, idx in enumerate(seg_df.index):
            result_rows.append({
                "_orig_idx": idx,
                "kin_a_mean_activation": float(mean_act[i]),
                "kin_a_peak_amplitude": float(running_peak[i]),
                "kin_a_velocity": float(vel[i]),
                "kin_a_acceleration": float(acc[i]),
                "kin_a_asymmetry": float(asym[i]),
                "kin_a_abs_asymmetry": float(abs_asym[i]),
            })

    if not result_rows:
        return pd.DataFrame(index=features_df.index)

    result_df = pd.DataFrame(result_rows).set_index("_orig_idx")
    return result_df.reindex(features_df.index)


def compute_group_a_task_summary(
    features_df: pd.DataFrame,
    fps: float = 30.0,
) -> Dict[str, Any]:
    """
    Compute per-task summary statistics for Group A kinematics.

    For each (task_id, repetition) in Group A, derives:
        peak_amplitude, time_to_peak_s, return_time_s, mean_abs_asymmetry,
        peak_asymmetry, movement_consistency (coefficient of variation over reps)

    Returns a dict keyed by 'A_{task_id}' with per-repetition sub-dicts plus
    a cross-repetition summary.  Cross-repetition consistency is defined as
    1 - CV(peak_amplitude_across_reps) and serves as a proxy for execution
    reproducibility.
    """
    group_a_mask = (
        features_df.get("task_group", pd.Series("0", index=features_df.index)) == "A"
    )
    a_df = features_df[group_a_mask].copy()
    if a_df.empty:
        return {}

    dt = 1.0 / fps
    results: Dict[str, Any] = {}

    for tid, tid_df in a_df.groupby("task_id", sort=True):
        tid_int = int(tid) if pd.notna(tid) else 0
        key = f"A_{tid_int}"
        primary = _GROUP_A_PRIMARY_BLENDSHAPES.get(tid_int, [])
        present = [c for c in primary if c in tid_df.columns]
        sym_pair = _GROUP_A_SYMMETRY_PAIRS.get(tid_int)

        rep_summaries: List[Dict[str, float]] = []
        for rep, rep_df in tid_df.groupby("repetition", sort=True):
            if int(rep) == 0 or len(rep_df) < 3:
                continue
            if present:
                act = rep_df[present].mean(axis=1).fillna(0.0).to_numpy()
            elif "mean_activation" in rep_df.columns:
                act = rep_df["mean_activation"].fillna(0.0).to_numpy()
            else:
                act = np.zeros(len(rep_df))

            peak_idx = int(np.argmax(act))
            peak_amp = float(act[peak_idx])
            time_to_peak = peak_idx * dt

            above_half = np.where(act >= 0.5 * peak_amp)[0]
            if len(above_half) > 0 and peak_idx < len(act) - 1:
                post_peak = act[peak_idx:]
                below_half_post = np.where(post_peak < 0.5 * peak_amp)[0]
                return_time = float(below_half_post[0]) * dt if len(below_half_post) > 0 else float(len(post_peak)) * dt
            else:
                return_time = 0.0

            if sym_pair and sym_pair[0] in rep_df.columns and sym_pair[1] in rep_df.columns:
                left_vals = rep_df[sym_pair[0]].fillna(0.0).to_numpy()
                right_vals = rep_df[sym_pair[1]].fillna(0.0).to_numpy()
                denom = np.maximum(np.abs(left_vals) + np.abs(right_vals), 1e-6)
                asym = (left_vals - right_vals) / denom
                peak_asym = float(asym[peak_idx]) if peak_idx < len(asym) else 0.0
                mean_abs_asym = float(np.mean(np.abs(asym)))
            else:
                avail = [c for c in _FALLBACK_ASYM_RATIO_COLS if c in rep_df.columns]
                if avail:
                    asym_col = rep_df[avail].fillna(0.0).mean(axis=1).to_numpy()
                    peak_asym = float(asym_col[peak_idx]) if peak_idx < len(asym_col) else 0.0
                    mean_abs_asym = float(np.mean(asym_col))
                else:
                    peak_asym = 0.0
                    mean_abs_asym = 0.0

            onset_threshold = 0.25 * peak_amp
            onset_frames = np.where(act >= onset_threshold)[0]
            onset_time = float(onset_frames[0]) * dt if len(onset_frames) > 0 else 0.0

            rep_summaries.append({
                "repetition": int(rep),
                "peak_amplitude": peak_amp,
                "time_to_peak_s": time_to_peak,
                "return_time_s": return_time,
                "onset_time_s": onset_time,
                "mean_abs_asymmetry": mean_abs_asym,
                "peak_asymmetry": peak_asym,
            })

        if not rep_summaries:
            continue

        peaks = [r["peak_amplitude"] for r in rep_summaries]
        ttps = [r["time_to_peak_s"] for r in rep_summaries]
        onsets = [r["onset_time_s"] for r in rep_summaries]
        asyms = [r["mean_abs_asymmetry"] for r in rep_summaries]

        mean_peak = float(np.mean(peaks))
        cv_peak = float(np.std(peaks) / (mean_peak + 1e-6)) if len(peaks) > 1 else 0.0
        consistency = float(max(0.0, 1.0 - cv_peak))

        results[key] = {
            "task_id": tid_int,
            "n_reps": len(rep_summaries),
            "repetitions": rep_summaries,
            "mean_peak_amplitude": mean_peak,
            "cv_peak_amplitude": cv_peak,
            "movement_consistency": consistency,
            "mean_time_to_peak_s": float(np.mean(ttps)),
            "mean_onset_time_s": float(np.mean(onsets)),
            "mean_abs_asymmetry": float(np.mean(asyms)),
            "max_abs_asymmetry": float(np.max(asyms)) if asyms else 0.0,
        }

    return results

N_INTERP = 1000


def build_reference_profile(
    kinematic_series_list: List[np.ndarray],
    measurement_col: str,
) -> Dict:
    """
    Build a mean spatiotemporal profile from multiple time-normalised series.

    Each series in kinematic_series_list should be a 1D array of measurement
    values for one repetition/participant (already duration-normalised or will
    be interpolated here to N_INTERP points).

    Returns: {'mean': np.ndarray, 'std': np.ndarray, 'n': int, 'col': str}
    """
    interp_grid = np.linspace(0, 1, N_INTERP)
    resampled = []
    for s in kinematic_series_list:
        if len(s) < 2:
            continue
        x = np.linspace(0, 1, len(s))
        resampled.append(np.interp(interp_grid, x, s))
    if not resampled:
        return {"mean": np.zeros(N_INTERP), "std": np.ones(N_INTERP), "n": 0, "col": measurement_col}
    arr = np.stack(resampled)
    return {
        "mean": arr.mean(axis=0),
        "std": arr.std(axis=0) + 1e-8,
        "n": len(resampled),
        "col": measurement_col,
    }


def score_against_profile(
    test_series: np.ndarray, profile: Dict
) -> float:
    """
    Compute the profile similarity score ψ = mean squared difference
    over N_INTERP interpolation points (paper eq. 1).
    Lower = more similar to the reference profile.
    """
    if len(test_series) < 2:
        return float("inf")
    x = np.linspace(0, 1, len(test_series))
    test_interp = np.interp(np.linspace(0, 1, N_INTERP), x, test_series)
    diff = test_interp - profile["mean"]
    return float(np.mean(diff ** 2))


def compute_kinematic_deviation_score(
    test_series: np.ndarray, profile: Dict
) -> float:
    """
    Normalised deviation score (0 = matches profile, 1 = maximally deviant).

    Normalised by the variance of the reference profile so scores are
    comparable across measurements with very different absolute ranges.
    """
    psi = score_against_profile(test_series, profile)
    profile_variance = float(np.mean(profile["std"] ** 2))
    if profile_variance < 1e-10:
        return 0.0
    return float(np.clip(psi / (profile_variance * 10.0), 0.0, 1.0))


def compute_task_kinematic_summary(
    kin_df: pd.DataFrame,
    features_df: pd.DataFrame,
    task_group: str,
    reference_profiles: Optional[Dict[str, Dict]] = None,
) -> Dict:
    """
    For a given task group (B or C), compute kinematic summary statistics.

    Returns a dict with:
      - Per-measurement mean, std, range across all repetitions
      - Depth (Z) velocity summaries (most informative per paper §4.1)
      - Deviation scores against reference_profiles if provided
      - Asymmetry summary (Medial 1/3 Symmetry, Mouth Area Symmetry)
    """
    mask = features_df.get("task_group", pd.Series("0", index=features_df.index)) == task_group
    if not mask.any():
        return {}

    kin_sub = kin_df.loc[mask]
    summary = {"task_group": task_group, "n_frames": int(mask.sum())}

    primary_cols = [
        "kin_mouth_opening",
        "kin_lip_action_y", "kin_lip_action_z",
        "kin_medial_upper_action_y", "kin_medial_upper_action_z",
        "kin_medial_lower_action_y", "kin_medial_lower_action_z",
        "kin_pogonion_y", "kin_pogonion_z",
        "kin_lower_lip_from_pog_y",
        "kin_labial_fissure_width",
        "kin_mandibular_angle",
    ]
    asymmetry_cols = ["kin_medial_sym_x", "kin_medial_sym_y", "kin_medial_sym_z",
                      "kin_mouth_area_symmetry"]

    for col in primary_cols:
        if col not in kin_sub.columns:
            continue
        vals = kin_sub[col].dropna().to_numpy()
        if len(vals) == 0:
            continue
        summary[f"{col}_mean"] = float(np.mean(vals))
        summary[f"{col}_std"] = float(np.std(vals))
        summary[f"{col}_range"] = float(np.ptp(vals))

        vel_col = f"{col}_vel"
        if vel_col in kin_sub.columns:
            vel_vals = kin_sub[vel_col].dropna().to_numpy()
            if len(vel_vals) > 0:
                summary[f"{col}_vel_mean"] = float(np.mean(np.abs(vel_vals)))
                summary[f"{col}_vel_max"] = float(np.max(np.abs(vel_vals)))

        if reference_profiles and col in reference_profiles:
            dev_score = compute_kinematic_deviation_score(vals, reference_profiles[col])
            summary[f"{col}_profile_deviation"] = round(dev_score, 4)

    for col in asymmetry_cols:
        if col not in kin_sub.columns:
            continue
        vals = kin_sub[col].dropna().to_numpy()
        if len(vals) > 0:
            summary[f"{col}_mean"] = float(np.mean(np.abs(vals)))
            summary[f"{col}_max"] = float(np.max(np.abs(vals)))

    if task_group == "B" and "kin_mouth_opening" in kin_sub.columns:
        mouth_sig = kin_sub["kin_mouth_opening"].dropna().to_numpy()
        fps_est = 30.0
        if "timestamp_abs" in features_df.columns:
            ts = features_df.loc[mask, "timestamp_abs"].dropna().to_numpy()
            if len(ts) > 1:
                diffs = np.diff(ts)
                diffs = diffs[diffs > 0]
                if len(diffs) > 0:
                    fps_est = float(1.0 / np.median(diffs))
        ddk_metrics = compute_ddk_clinical_metrics(mouth_sig, fps=fps_est)
        summary.update(ddk_metrics)

    return summary
