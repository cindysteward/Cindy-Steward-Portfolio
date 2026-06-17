"""
Anatomical muscle group mapping for FACET-CV facial motor analysis.

Maps MediaPipe blendshape features to cranial nerve innervation zones and
anatomical muscle groups. Provides regional aggregation, clinical report
generation, and frame-level muscle group activation summaries.

Feature name resolution handles direct blendshape names, aggregated metric
suffixes (_mean, _max, _std, _min, _range, _time_to_peak), asymmetry
features (asymmetry_<base>, asymmetry_ratio_<base>), head-pose features,
and global intensity/temporal features.

The module also renders a publication-quality anatomical face schematic via
generate_3d_anatomical_visualization(), showing each muscle group as a
clinically proportionate polygon region colour-coded by activation level.

Key public symbols
------------------
MUSCLE_GROUP_MAP
    Dict mapping muscle group ID to blendshape list, cranial nerve, and
    clinical relevance text.
aggregate_by_muscle_group(feature_deviations)
    Aggregate per-feature deviation scores to per-group summaries.
generate_anatomical_report(feature_deviations)
    Full structured report with affected groups, nerves, and laterality hint.
generate_3d_anatomical_visualization(...)
    Render and save a facial musculature activation figure.

References
----------
May M, Schaitkin BM (2000) The Facial Nerve: May's Second Edition.
  Thieme, New York.
  Definitive reference for CN VII peripheral branch anatomy (frontal,
  zygomatic, buccal, marginal mandibular, cervical) used to construct
  MUSCLE_GROUP_MAP cranial_nerve entries and the upper/lower face
  paresis distinction.

Standring S (ed) (2020) Gray's Anatomy, 42nd ed. Elsevier, Philadelphia.
  Chapters on cranial nerves and the face: detailed CN VII muscle
  innervation used for orbicularis oris / orbicularis oculi / frontalis /
  zygomaticus / platysma mapping.

Rinn WE (1984) The neuropsychology of facial expression: a review of the
  neurological and psychological mechanisms for producing facial
  expressions. Psychol Bull 95(1):52-77.
  Upper-face bilateral vs lower-face contralateral innervation distinction
  underlying 'spared in peripheral palsy' clinical_relevance entries.
"""

import re
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger("pipeline")

MUSCLE_GROUP_MAP: Dict[str, Dict[str, Any]] = {
    "orbicularis_oculi": {
        "description": "Eye closure (orbicularis oculi, CN VII upper)",
        "cranial_nerve": "CN VII (upper division)",
        "blendshapes": [
            "eyeBlinkLeft", "eyeBlinkRight",
            "eyeSquintLeft", "eyeSquintRight",
            "eyeWideLeft", "eyeWideRight",
            "eyeLookDownLeft", "eyeLookDownRight",
            "eyeLookUpLeft", "eyeLookUpRight",
            "eyeLookInLeft", "eyeLookInRight",
            "eyeLookOutLeft", "eyeLookOutRight",
        ],
        "clinical_relevance": "Upper face - spared in peripheral facial palsy distinction",
    },
    "frontalis": {
        "description": "Forehead elevation (frontalis, CN VII upper)",
        "cranial_nerve": "CN VII (upper division)",
        "blendshapes": [
            "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
            "browDownLeft", "browDownRight",
        ],
        "clinical_relevance": "Eyebrow raise - key for central vs peripheral palsy",
    },
    "orbicularis_oris": {
        "description": "Lip closure and pursing (orbicularis oris, CN VII lower)",
        "cranial_nerve": "CN VII (lower division)",
        "blendshapes": [
            "mouthPucker", "mouthFunnel",
            "mouthClose", "mouthPressLeft", "mouthPressRight",
            "mouthShrugLower", "mouthShrugUpper",
            "mouthUpperUpLeft", "mouthUpperUpRight",
            "mouthRollLower", "mouthRollUpper",
        ],
        "clinical_relevance": "Lip seal - affected in both central and peripheral palsy",
    },
    "zygomaticus": {
        "description": "Smile (zygomaticus major/minor, CN VII lower)",
        "cranial_nerve": "CN VII (lower division)",
        "blendshapes": [
            "mouthSmileLeft", "mouthSmileRight",
            "mouthDimpleLeft", "mouthDimpleRight",
            "cheekSquintLeft", "cheekSquintRight",
        ],
        "clinical_relevance": "Smile symmetry - primary indicator for facial nerve function",
    },
    "depressor": {
        "description": "Lower lip and mouth corner depression (CN VII lower)",
        "cranial_nerve": "CN VII (lower division)",
        "blendshapes": [
            "mouthFrownLeft", "mouthFrownRight",
            "mouthLowerDownLeft", "mouthLowerDownRight",
            "mouthStretchLeft", "mouthStretchRight",
        ],
        "clinical_relevance": "Mouth depression - relevant for marginal mandibular branch",
    },
    "tongue": {
        "description": "Tongue movements (genioglossus/styloglossus, CN XII)",
        "cranial_nerve": "CN XII",
        "blendshapes": ["tongueOut"],
        "clinical_relevance": "Tongue protrusion/deviation - hypoglossal nerve function",
    },
    "buccinator": {
        "description": "Cheek puffing (buccinator, CN VII lower)",
        "cranial_nerve": "CN VII (lower division)",
        "blendshapes": ["cheekPuff"],
        "clinical_relevance": "Cheek inflation - tests buccinator function",
    },
    "jaw": {
        "description": "Jaw opening and movement (masseter/pterygoids, CN V3)",
        "cranial_nerve": "CN V (motor)",
        "blendshapes": [
            "jawOpen", "jawForward", "jawLeft", "jawRight",
        ],
        "clinical_relevance": "Jaw function - trigeminal motor branch",
    },
    "nasal": {
        "description": "Nasal dilation (nasalis, CN VII)",
        "cranial_nerve": "CN VII",
        "blendshapes": [
            "noseSneerLeft", "noseSneerRight",
        ],
        "clinical_relevance": "Nasal flaring - peripheral facial nerve test",
    },
}


_SYMMETRY_REGION_TO_GROUPS: Dict[str, List[str]] = {
    "brow":       ["frontalis"],
    "browDown":   ["frontalis"],
    "browOuterUp": ["frontalis"],
    "browInnerUp": ["frontalis"],
    "eye":        ["orbicularis_oculi"],
    "eyeBlink":   ["orbicularis_oculi"],
    "eyeSquint":  ["orbicularis_oculi"],
    "eyeWide":    ["orbicularis_oculi"],
    "eyeLookDown": ["orbicularis_oculi"],
    "eyeLookUp":  ["orbicularis_oculi"],
    "eyeLookIn":  ["orbicularis_oculi"],
    "eyeLookOut": ["orbicularis_oculi"],
    "cheek":      ["zygomaticus", "buccinator"],
    "cheekSquint": ["zygomaticus"],
    "cheekPuff":  ["buccinator"],
    "nose":       ["nasal"],
    "noseSneer":  ["nasal"],
    "mouth":      ["orbicularis_oris", "zygomaticus", "depressor"],
    "mouthSmile": ["zygomaticus"],
    "mouthDimple": ["zygomaticus"],
    "mouthFrown": ["depressor"],
    "mouthLowerDown": ["depressor"],
    "mouthStretch": ["depressor"],
    "mouthPress": ["orbicularis_oris"],
    "mouthPucker": ["orbicularis_oris"],
    "mouthFunnel": ["orbicularis_oris"],
    "mouthClose": ["orbicularis_oris"],
    "mouthShrugLower": ["orbicularis_oris"],
    "mouthShrugUpper": ["orbicularis_oris"],
    "mouthUpperUp": ["orbicularis_oris"],
    "mouthRollLower": ["orbicularis_oris"],
    "mouthRollUpper": ["orbicularis_oris"],
    "jaw":        ["jaw"],
    "jawOpen":    ["jaw"],
    "jawForward": ["jaw"],
    "tongue":     ["tongue"],
    "tongueOut":  ["tongue"],
}

_METRIC_SUFFIXES = re.compile(
    r"(_mean|_max|_std|_min|_range|_time_to_peak|_across_reps_mean|_across_reps_std|_session_mean)$"
)

_GLOBAL_FEATURES = {
    "mean_activation", "max_activation", "activation_range",
    "activation_velocity", "activation_acceleration",
}

_HEAD_POSE_PREFIXES = ("head_yaw", "head_pitch", "head_roll", "head_pose_deviation")


def _strip_metric_suffix(feature_name: str) -> str:
    """Remove aggregation suffixes from a feature name to recover the base signal name.

    Strips up to three nested suffix layers (e.g. _mean_across_reps_mean).
    Returns the shortest stable form that no longer matches _METRIC_SUFFIXES.
    """
    result = feature_name
    for _ in range(3):
        stripped = _METRIC_SUFFIXES.sub("", result)
        if stripped == result:
            break
        result = stripped
    return result


def _resolve_muscle_group(feature_name: str) -> Optional[str]:
    """Map an arbitrary pipeline feature name to its anatomical muscle group.

    Resolution strategy (tried in order):
      1. Strip metric suffixes and look up the base blendshape directly.
      2. Parse asymmetry/asymmetry_ratio prefixes and resolve the body-region
         token via the symmetry-region-to-group table.
      3. Try progressively shorter camelCase-aware prefixes of the base name
         against every known blendshape (substring match).
      4. Classify global intensity/temporal features (mean_activation, etc.)
         and head-pose features so they don't fall through to 'other'.

    Returns the group name string or None if the feature is a pipeline-global
    measure that does not map to a single anatomical region.
    """
    base = _strip_metric_suffix(feature_name)

    for g_name, g_info in MUSCLE_GROUP_MAP.items():
        if base in g_info["blendshapes"]:
            return g_name

    asym_base = None
    if base.startswith("asymmetry_ratio_"):
        asym_base = base[len("asymmetry_ratio_"):]
    elif base.startswith("asymmetry_"):
        asym_base = base[len("asymmetry_"):]

    if asym_base is not None:
        if asym_base in _SYMMETRY_REGION_TO_GROUPS:
            return _SYMMETRY_REGION_TO_GROUPS[asym_base][0]

        for region, groups in _SYMMETRY_REGION_TO_GROUPS.items():
            if asym_base.lower() == region.lower():
                return groups[0]
            if asym_base.lower().startswith(region.lower()):
                return groups[0]

        for g_name, g_info in MUSCLE_GROUP_MAP.items():
            for bs in g_info["blendshapes"]:
                cleaned = bs.replace("Left", "").replace("Right", "")
                if cleaned.lower() == asym_base.lower():
                    return g_name

        return None

    if base in _GLOBAL_FEATURES:
        return None

    for prefix in _HEAD_POSE_PREFIXES:
        if base.startswith(prefix):
            return None

    if base in _SYMMETRY_REGION_TO_GROUPS:
        return _SYMMETRY_REGION_TO_GROUPS[base][0]

    for g_name, g_info in MUSCLE_GROUP_MAP.items():
        for bs in g_info["blendshapes"]:
            if base.lower() == bs.lower():
                return g_name

    for g_name, g_info in MUSCLE_GROUP_MAP.items():
        for bs in g_info["blendshapes"]:
            cleaned = bs.replace("Left", "").replace("Right", "")
            if base.lower() == cleaned.lower():
                return g_name

    for g_name, g_info in MUSCLE_GROUP_MAP.items():
        for bs in g_info["blendshapes"]:
            if bs.lower().startswith(base.lower()) or base.lower().startswith(bs.lower()):
                return g_name

    return None


def get_muscle_group(blendshape_name: str) -> Optional[str]:
    """Return the muscle group name for a given blendshape name, or None if not found.

    Performs a direct membership lookup in MUSCLE_GROUP_MAP. This is the
    fast path for known blendshape names. For derived or aggregated feature
    names use _resolve_muscle_group instead.
    """
    for group_name, info in MUSCLE_GROUP_MAP.items():
        if blendshape_name in info["blendshapes"]:
            return group_name
    return None


def get_cranial_nerve(blendshape_name: str) -> str:
    """Return the cranial nerve innervation string for a given blendshape name.

    Returns 'unknown' when the blendshape is not in any known muscle group.
    """
    group = get_muscle_group(blendshape_name)
    if group and group in MUSCLE_GROUP_MAP:
        return MUSCLE_GROUP_MAP[group]["cranial_nerve"]
    return "unknown"


def aggregate_by_muscle_group(
    feature_deviations: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Aggregate per-feature deviations into anatomical muscle groups.

    *feature_deviations* is the anomaly_results['feature_deviations'] dict.
    Each feature name is resolved through a multi-stage pipeline that handles
    raw blendshapes, asymmetry features, metric suffixes, and region tokens.
    Global and head-pose features are excluded (they do not map to a single
    anatomical region).  Returns per-group summary with mean deviation,
    max deviation, and number of contributing features, sorted by severity.
    """
    group_data: Dict[str, List[float]] = {}

    for feat, dev_info in feature_deviations.items():
        group = _resolve_muscle_group(feat)

        if group is None:
            continue

        if group not in group_data:
            group_data[group] = []

        mean_dev = dev_info.get("mean_range_dev", 0.0)
        group_data[group].append(mean_dev)

    result: Dict[str, Dict[str, Any]] = {}
    for group, devs in group_data.items():
        arr = np.array(devs)
        info = MUSCLE_GROUP_MAP.get(group, {})
        result[group] = {
            "mean_deviation": float(np.mean(arr)),
            "max_deviation": float(np.max(arr)),
            "n_features": len(devs),
            "n_deviant": int(np.sum(arr > 1.0)),
            "cranial_nerve": info.get("cranial_nerve", "unknown"),
            "clinical_relevance": info.get("clinical_relevance", ""),
            "description": info.get("description", group),
        }

    return dict(sorted(result.items(), key=lambda x: x[1]["mean_deviation"], reverse=True))


def generate_anatomical_report(
    feature_deviations: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Generate a full anatomical analysis report from per-feature deviation scores.

    Aggregates deviations by muscle group, identifies which groups and cranial
    nerves are affected (n_deviant > 0), and produces a laterality hint based
    on whether upper-face and lower-face groups are involved.

    Returns a dict with the following keys:
      muscle_groups     -- full per-group dict from aggregate_by_muscle_group
      affected_groups   -- list of group names with at least one deviant feature
      affected_nerves   -- dict mapping cranial nerve label to list of group names
      laterality_hint   -- plain-text clinical interpretation of upper/lower pattern
      n_groups_affected -- count of groups with at least one deviant feature
      n_groups_total    -- total count of groups in the result
    """
    grouped = aggregate_by_muscle_group(feature_deviations)

    affected_groups = {
        g: info for g, info in grouped.items()
        if info["n_deviant"] > 0
    }

    affected_nerves: Dict[str, List[str]] = {}
    for g, info in affected_groups.items():
        nerve = info.get("cranial_nerve", "unknown")
        if nerve not in affected_nerves:
            affected_nerves[nerve] = []
        affected_nerves[nerve].append(g)

    upper_face = any(
        g in affected_groups for g in ("frontalis", "orbicularis_oculi")
    )
    lower_face = any(
        g in affected_groups
        for g in ("orbicularis_oris", "zygomaticus", "depressor", "buccinator")
    )

    if lower_face and not upper_face:
        laterality_hint = "Lower face involvement with upper face sparing - pattern consistent with central facial palsy"
    elif lower_face and upper_face:
        laterality_hint = "Both upper and lower face involvement - pattern consistent with peripheral facial palsy"
    elif upper_face and not lower_face:
        laterality_hint = "Isolated upper face findings - consider supranuclear etiology"
    else:
        laterality_hint = "No clear lateralization pattern identified"

    return {
        "muscle_groups": grouped,
        "affected_groups": list(affected_groups.keys()),
        "affected_nerves": affected_nerves,
        "laterality_hint": laterality_hint,
        "n_groups_affected": len(affected_groups),
        "n_groups_total": len(grouped),
    }


def generate_per_repetition_anatomical_reports(
    anomaly_results: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    """Build per-task lists of per-repetition anatomical reports.

    Returns a dict keyed by task label (e.g. 'A_3') whose values are lists of
    dicts, one per repetition. Each dict contains 'repetition' (int) and
    'muscle_groups' (same structure as aggregate_by_muscle_group output).

    This enables per-task visualisations where each repetition is plotted as a
    separate data series. The input anomaly_results is expected to contain
    'deviations', 'repetitions', 'task_groups', and 'task_ids' parallel lists,
    as produced by the anomaly detection module.
    """
    deviations_list = anomaly_results.get("deviations", [])
    repetitions = anomaly_results.get("repetitions", list(range(len(deviations_list))))
    task_groups = anomaly_results.get("task_groups", ["0"] * len(deviations_list))
    task_ids = anomaly_results.get("task_ids", [0] * len(deviations_list))

    task_reps: Dict[str, List[Dict[str, Any]]] = {}

    for idx, rep_devs in enumerate(deviations_list):
        tg = str(task_groups[idx]) if idx < len(task_groups) else "0"
        tid = int(task_ids[idx]) if idx < len(task_ids) else 0
        rep = int(repetitions[idx]) if idx < len(repetitions) else idx

        task_key = f"{tg}_{tid}" if tg not in ("0", "nan", "None", "") else "session"

        feature_devs = {}
        for feat, dev_info in rep_devs.items():
            feature_devs[feat] = {
                "mean_range_dev": dev_info.get("range_dev", 0.0),
                "max_range_dev": dev_info.get("range_dev", 0.0),
                "n_deviant": 1 if dev_info.get("is_deviant", False) else 0,
            }

        grouped = aggregate_by_muscle_group(feature_devs)

        if task_key not in task_reps:
            task_reps[task_key] = []

        task_reps[task_key].append({
            "repetition": rep,
            "muscle_groups": grouped,
        })

    return task_reps


def aggregate_activations_by_muscle_group(
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate frame-level blendshape activations into per-muscle-group activations.

    Takes a corrected features DataFrame and computes the mean activation across
    all blendshapes belonging to each muscle group, per frame. Returns a
    DataFrame with one column per muscle group (prefixed 'muscle_') plus
    any frame_index, timestamp_abs, segment, and repetition columns present
    in the input.

    Groups whose blendshapes are all absent from features_df receive a zero
    column rather than being dropped, so the output always has the same
    schema regardless of which features were extracted.
    """
    result_cols = ['frame_index', 'timestamp_abs', 'segment', 'repetition']
    result = (
        features_df[result_cols].copy()
        if all(c in features_df.columns for c in result_cols)
        else pd.DataFrame(index=features_df.index)
    )

    for group_name, group_info in MUSCLE_GROUP_MAP.items():
        blendshapes = group_info['blendshapes']
        present = [bs for bs in blendshapes if bs in features_df.columns]

        if present:
            result[f'muscle_{group_name}'] = features_df[present].mean(axis=1)
        else:
            result[f'muscle_{group_name}'] = 0.0

    return result


def get_muscle_group_summary(
    features_df: pd.DataFrame,
    by_repetition: bool = True,
) -> pd.DataFrame:
    """Compute muscle group activation summary statistics from a features DataFrame.

    Filters to measurement-segment frames before computing statistics. When
    by_repetition=True and a 'repetition' column is present, returns one row
    per repetition with columns for each muscle group's mean, max, and
    standard deviation. When by_repetition=False, returns a single summary
    row across all measurement frames.
    """
    muscle_df = aggregate_activations_by_muscle_group(features_df)

    if 'segment' in muscle_df.columns:
        measurement = muscle_df[muscle_df['segment'] == 'measurement']
        if len(measurement) == 0:
            measurement = muscle_df
    else:
        measurement = muscle_df

    muscle_cols = [c for c in measurement.columns if c.startswith('muscle_')]

    if not by_repetition or 'repetition' not in measurement.columns:
        summary = {}
        for col in muscle_cols:
            values = measurement[col].values
            summary[f'{col}_mean'] = float(np.mean(values))
            summary[f'{col}_max'] = float(np.max(values))
            summary[f'{col}_std'] = float(np.std(values))
        return pd.DataFrame([summary])

    summary_rows = []
    for rep, group_df in measurement.groupby('repetition', dropna=False):
        row = {'repetition': rep}
        for muscle_col in muscle_cols:
            values = group_df[muscle_col].values
            row[f'{muscle_col}_mean'] = float(np.mean(values))
            row[f'{muscle_col}_max'] = float(np.max(values))
            row[f'{muscle_col}_std'] = float(np.std(values))
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def generate_3d_anatomical_visualization(
    muscle_activation_scores: Dict[str, float],
    output_path: Path,
    title: str = "Facial Muscle Activation Map",
    reference_activation_scores: Optional[Dict[str, float]] = None,
    anomaly_flags: Optional[Dict[str, bool]] = None,
    decision_label: Optional[str] = None,
    decision_confidence: Optional[float] = None,
) -> None:
    """Publication-quality anatomical facial muscle activation figure.

    Renders a frontal-view anatomical face schematic with each muscle group
    represented as a clinically proportionate polygon region, colour-coded by
    activation level.  Anomalous muscles are overlaid with hatch-fill and a
    warning border.  An optional decision-screening banner shows classification
    outcome and confidence.  A companion panel provides a ranked horizontal bar
    chart and a vertical colorbar. The full layout is suitable for inclusion
    in a journal figure at 300 DPI.

    All geometry uses a canonical 10×12 normalised coordinate system derived
    from average MediaPipe landmark proportions.  No individual is identifiable.

    Parameters
    ----------
    muscle_activation_scores:
        Mapping muscle group name → [0, 1] normalised activation.
    output_path:
        Destination file path (PDF or PNG).
    title:
        Figure title (appears as bold suptitle).
    reference_activation_scores:
        When provided, colour encodes Δ activation (session − reference).
    anomaly_flags:
        Mapping muscle group name → bool; True marks the muscle as anomalous
        (adds hatch overlay and warning border).
    decision_label:
        Optional classification string shown in the decision-screening banner
        (e.g. "Peripheral Palsy - Moderate", "Within Normal Limits").
    decision_confidence:
        Confidence in [0, 1] shown alongside the decision label.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors
    import matplotlib.patheffects as mpe
    from matplotlib.gridspec import GridSpec
    import matplotlib.ticker as mticker
    import numpy as np

    _anomaly = anomaly_flags or {}
    show_deviation = reference_activation_scores is not None

    def _epts(cx, cy, rx, ry, t0=0.0, t1=2 * np.pi, n=48):
        """Return a list of (x, y) points tracing an ellipse arc."""
        t = np.linspace(t0, t1, n)
        return list(zip(cx + rx * np.cos(t), cy + ry * np.sin(t)))

    MUSCLE_REGIONS: Dict[str, Any] = {
        "frontalis": {
            "left": [
                (2.6, 8.20), (4.85, 8.20), (4.85, 9.60),
                (3.80, 9.85), (2.6, 9.50),
            ],
            "right": [
                (5.15, 8.20), (7.40, 8.20), (7.40, 9.50),
                (6.20, 9.85), (5.15, 9.60),
            ],
        },
        "orbicularis_oculi": {
            "left":  _epts(3.35, 7.30, 1.20, 0.65, n=36),
            "right": _epts(6.65, 7.30, 1.20, 0.65, n=36),
        },
        "nasal": [
            (4.30, 5.70), (5.70, 5.70),
            (5.55, 6.50), (5.00, 6.75), (4.45, 6.50),
        ],
        "zygomaticus": {
            "left": [
                (1.50, 6.55), (3.60, 6.80),
                (4.05, 6.00), (3.75, 4.50),
                (3.20, 4.35), (1.30, 6.00),
            ],
            "right": [
                (8.50, 6.55), (6.40, 6.80),
                (5.95, 6.00), (6.25, 4.50),
                (6.80, 4.35), (8.70, 6.00),
            ],
        },
        "orbicularis_oris": _epts(5.00, 4.00, 1.55, 0.82, n=44),
        "buccinator": {
            "left": [
                (1.40, 3.90), (3.45, 4.10),
                (3.45, 5.80), (1.40, 5.60),
            ],
            "right": [
                (8.60, 3.90), (6.55, 4.10),
                (6.55, 5.80), (8.60, 5.60),
            ],
        },
        "depressor": {
            "left": [
                (3.50, 2.60), (4.75, 3.15),
                (4.55, 2.00), (3.20, 1.85),
            ],
            "right": [
                (6.50, 2.60), (5.25, 3.15),
                (5.45, 2.00), (6.80, 1.85),
            ],
        },
        "jaw": {
            "left":  _epts(3.10, 2.20, 1.35, 0.90, t0=np.pi * 0.55, t1=np.pi * 1.45, n=28),
            "right": _epts(6.90, 2.20, 1.35, 0.90, t0=np.pi * 1.55, t1=np.pi * 0.45 + 2 * np.pi, n=28),
        },
        "tongue": _epts(5.00, 3.92, 0.70, 0.38, n=28),
        "buccinator_mid": None,
    }
    MUSCLE_REGIONS.pop("buccinator_mid", None)

    LATIN_NAMES: Dict[str, str] = {
        "frontalis":         "m. frontalis",
        "orbicularis_oculi": "m. orbicularis oculi",
        "nasal":             "m. nasalis",
        "zygomaticus":       "m. zygomaticus maj./min.",
        "orbicularis_oris":  "m. orbicularis oris",
        "buccinator":        "m. buccinator",
        "depressor":         "m. depressor ang. oris",
        "jaw":               "m. masseter / pterygoid",
        "tongue":            "m. genioglossus (CN XII)",
    }
    CN_LABELS: Dict[str, str] = {
        "frontalis":         "CN VII (temporal)",
        "orbicularis_oculi": "CN VII (zygomatic)",
        "nasal":             "CN VII (buccal)",
        "zygomaticus":       "CN VII (zygomatic/buccal)",
        "orbicularis_oris":  "CN VII (buccal/marginal)",
        "buccinator":        "CN VII (buccal)",
        "depressor":         "CN VII (marginal mand.)",
        "jaw":               "CN V3 (motor)",
        "tongue":            "CN XII",
    }

    if show_deviation:
        _cmap = plt.cm.RdBu_r
        _abs_max = max(
            0.30,
            max(abs(v - float((reference_activation_scores or {}).get(k, v)))
                for k, v in muscle_activation_scores.items()),
        )
        _norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-_abs_max, vmax=_abs_max)
        cb_label = "Δ Activation (session − reference)"
    else:
        _cmap = mcolors.LinearSegmentedColormap.from_list('anat_act', [
            (0.00, '#F5E6D3'),
            (0.10, '#FFDDB0'),
            (0.35, '#FF9A2E'),
            (0.65, '#D83010'),
            (0.88, '#920000'),
            (1.00, '#4A0000'),
        ])
        _norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
        cb_label = "Normalised Muscle Activation"

    def _score(name: str) -> float:
        """Return normalised activation (absolute mode) or deviation from reference (deviation mode)."""
        val = float(muscle_activation_scores.get(name, 0.0))
        if show_deviation:
            ref = float((reference_activation_scores or {}).get(name, val))
            return val - ref
        return val

    has_decision = decision_label is not None
    fig_h = 11.5 if has_decision else 10.5
    fig = plt.figure(figsize=(15, fig_h), dpi=300)

    top_frac = 0.92 if has_decision else 0.95
    gs = GridSpec(
        1, 2,
        figure=fig,
        left=0.02, right=0.97,
        top=top_frac, bottom=0.06,
        wspace=0.06,
        width_ratios=[1.05, 0.65],
    )
    ax_face = fig.add_subplot(gs[0])
    ax_right = fig.add_subplot(gs[1])
    ax_right.axis("off")

    cranium = _epts(5.00, 6.20, 3.80, 5.40, t0=np.pi * 0.08, t1=np.pi * 0.92, n=80)
    jaw_l   = _epts(3.10, 2.20, 1.35, 0.90, t0=np.pi * 0.55, t1=np.pi, n=20)
    jaw_r   = _epts(6.90, 2.20, 1.35, 0.90, t0=0.0, t1=np.pi * 0.45, n=20)
    chin    = _epts(5.00, 1.20, 0.80, 0.65, t0=0.0, t1=np.pi, n=20)
    face_pts = (
        cranium
        + list(reversed(jaw_r))
        + list(reversed(chin))
        + jaw_l
    )
    skin_poly = mpatches.Polygon(
        face_pts, closed=True,
        facecolor="#F5E6D3", edgecolor="#7A6652",
        linewidth=2.2, alpha=1.0, zorder=1,
    )
    ax_face.add_patch(skin_poly)

    for ex, esign in ((1.25, -1), (8.75, 1)):
        ear = mpatches.Ellipse(
            (ex, 5.80), 0.70, 1.30,
            facecolor="#EDD5B5", edgecolor="#7A6652", linewidth=1.5, zorder=0,
        )
        ax_face.add_patch(ear)

    label_positions: Dict[str, tuple] = {}

    _shapely_avail = False
    try:
        from shapely.geometry import Polygon as _ShapelyPoly
        from matplotlib.path import Path as _MplPath
        from matplotlib.patches import PathPatch as _PathPatch
        _shapely_avail = True
    except ImportError:
        pass

    def _smooth_path(pts: list):
        """Return a smooth rounded matplotlib Path via Shapely, or None."""
        if not (_shapely_avail and len(pts) >= 3):
            return None
        try:
            smooth = _ShapelyPoly(pts).buffer(
                0.13, cap_style='round', join_style='round', resolution=10
            )
            if smooth.is_valid and not smooth.is_empty:
                coords = np.array(smooth.exterior.coords)
                n = len(coords)
                codes = ([_MplPath.MOVETO]
                         + [_MplPath.LINETO] * (n - 2)
                         + [_MplPath.CLOSEPOLY])
                return _MplPath(coords, codes)
        except Exception:
            pass
        return None

    def _draw_patch(pts: list, sc: float, is_anomalous: bool) -> None:
        """Draw a filled polygon patch for one muscle region, coloured by activation or deviation score."""
        rgba        = _cmap(_norm(sc))
        fill_alpha  = 0.15 + 0.67 * min(abs(sc), 1.0)
        edge_col    = "#CC0000" if is_anomalous else "#9E7B5A"
        edge_lw     = 1.6 if is_anomalous else 0.8

        mpl_path = _smooth_path(pts)
        if mpl_path is not None:
            ax_face.add_patch(_PathPatch(
                mpl_path, fill=False,
                edgecolor="#A07050", linewidth=0.75, alpha=0.45, zorder=2,
            ))
            ax_face.add_patch(_PathPatch(
                mpl_path,
                facecolor=rgba[:3], edgecolor=edge_col,
                linewidth=edge_lw, alpha=fill_alpha, zorder=2,
            ))
            if is_anomalous:
                ax_face.add_patch(_PathPatch(
                    mpl_path, fill=False, hatch="////",
                    edgecolor="#CC0000", linewidth=1.4, alpha=0.55, zorder=3,
                ))
        else:
            ax_face.add_patch(mpatches.Polygon(
                pts, closed=True, facecolor=rgba[:3],
                edgecolor=edge_col, linewidth=edge_lw,
                alpha=fill_alpha, zorder=2,
            ))
            if is_anomalous:
                ax_face.add_patch(mpatches.Polygon(
                    pts, closed=True, fill=False, hatch="////",
                    edgecolor="#CC0000", linewidth=1.4, alpha=0.55, zorder=3,
                ))

    for gname, region in MUSCLE_REGIONS.items():
        sc = _score(gname)
        is_anom = bool(_anomaly.get(gname, False))
        if region is None:
            continue
        if isinstance(region, dict):
            all_pts: list = []
            for pts in region.values():
                _draw_patch(pts, sc, is_anom)
                all_pts.extend(pts)
            cx = float(np.mean([p[0] for p in all_pts]))
            cy = float(np.mean([p[1] for p in all_pts]))
        else:
            _draw_patch(region, sc, is_anom)
            cx = float(np.mean([p[0] for p in region]))
            cy = float(np.mean([p[1] for p in region]))
        label_positions[gname] = (cx, cy, sc)

    for ex in (3.35, 6.65):
        ax_face.add_patch(mpatches.Ellipse(
            (ex, 7.30), 2.10, 0.92,
            facecolor="white", edgecolor="#555555", linewidth=1.4, zorder=4,
        ))
        ax_face.add_patch(mpatches.Circle(
            (ex, 7.30), 0.38,
            facecolor="#3D5A80", edgecolor="#1A1A2E", linewidth=0.8, zorder=5,
        ))
        ax_face.add_patch(mpatches.Circle(
            (ex, 7.30), 0.16,
            facecolor="black", zorder=6,
        ))

    for bx0, bx1 in ((2.30, 4.40), (5.60, 7.70)):
        xs = np.linspace(bx0, bx1, 40)
        ys = 8.40 + 0.25 * np.sin(np.linspace(0, np.pi, 40))
        ax_face.plot(xs, ys, color="#3B2F2F", linewidth=2.8, zorder=7,
                     solid_capstyle="round")

    nose_x = [5.00, 4.60, 4.50, 4.60, 5.00, 5.40, 5.50, 5.40, 5.00]
    nose_y = [6.80, 6.35, 5.85, 5.60, 5.70, 5.60, 5.85, 6.35, 6.80]
    ax_face.plot(nose_x, nose_y, color="#7A6652", linewidth=1.3, zorder=7)
    ax_face.plot([4.60, 5.40], [5.60, 5.60], color="#7A6652", linewidth=1.3, zorder=7)

    mx = np.linspace(3.65, 6.35, 60)
    ax_face.plot(mx, 4.00 - 0.22 * np.sin(np.linspace(0, np.pi, 60)),
                 color="#5C3D2E", linewidth=2.0, zorder=7, solid_capstyle="round")
    upper_lip_x = np.linspace(3.65, 6.35, 60)
    upper_lip_y = 4.20 + 0.14 * np.sin(np.linspace(0, np.pi, 60))
    ax_face.plot(upper_lip_x, upper_lip_y, color="#5C3D2E", linewidth=1.6, zorder=7)

    ANNOT_LAYOUT: Dict[str, tuple] = {
        "frontalis":         (-1.2, 10.0, "left"),
        "orbicularis_oculi": (-1.2,  8.0, "left"),
        "nasal":             (11.2,  6.6, "right"),
        "zygomaticus":       (11.2,  5.6, "right"),
        "orbicularis_oris":  (-1.2,  4.2, "left"),
        "buccinator":        (11.2,  4.8, "right"),
        "depressor":         (-1.2,  2.5, "left"),
        "jaw":               (11.2,  2.0, "right"),
        "tongue":            (11.2,  3.7, "right"),
    }
    for gname, (tx, ty, ha) in ANNOT_LAYOUT.items():
        if gname not in label_positions:
            continue
        cx, cy, sc = label_positions[gname]
        latin = LATIN_NAMES.get(gname, gname)
        latin_math = latin.replace(" ", r"\ ")
        cn = CN_LABELS.get(gname, "")
        score_str = f"{sc:+.3f}" if show_deviation else f"{sc:.3f}"
        anom_marker = "  \u26a0" if _anomaly.get(gname) else ""
        rgba = _cmap(_norm(sc))
        edge_col = "#CC0000" if _anomaly.get(gname) else tuple(rgba[:3])
        ax_face.annotate(
            f"$\\it{{{latin_math}}}${anom_marker}\n{cn}\nact = {score_str}",
            xy=(cx, cy), xytext=(tx, ty),
            fontsize=7.2, ha=ha, va="center",
            arrowprops=dict(
                arrowstyle="-|>",
                color="#555555",
                linewidth=0.85,
                connectionstyle="arc3,rad=0.08",
            ),
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="#FAFAFA",
                edgecolor=edge_col,
                alpha=0.95,
                linewidth=1.2 if _anomaly.get(gname) else 0.8,
            ),
            zorder=8,
        )

    ax_face.set_xlim(-2.5, 12.5)
    ax_face.set_ylim(-0.8, 11.8)
    ax_face.set_aspect("equal")
    ax_face.axis("off")
    ax_face.set_title(
        "Facial Musculature - Activation Map\n"
        "$\\it{Schematic,\\ frontal\\ view.\\ No\\ individual\\ identifiable.}$",
        fontsize=11, fontweight="bold", pad=8, loc="center",
    )

    cb_ax = fig.add_axes([0.670, 0.12, 0.022, 0.68])
    sm = plt.cm.ScalarMappable(cmap=_cmap, norm=_norm)
    sm.set_array([])
    cb = plt.colorbar(sm, cax=cb_ax)
    cb.set_label(cb_label, fontsize=8.5, labelpad=6)
    cb.ax.tick_params(labelsize=7.5)
    cb.outline.set_linewidth(0.6)

    sorted_groups = sorted(
        muscle_activation_scores.items(),
        key=lambda x: abs(x[1]), reverse=True,
    )
    bar_y   = np.arange(len(sorted_groups))
    bar_vals   = [_score(k) for k, _ in sorted_groups]
    bar_labels = [LATIN_NAMES.get(k, k) for k, _ in sorted_groups]
    bar_cn     = [CN_LABELS.get(k, "") for k, _ in sorted_groups]
    bar_colors = [_cmap(_norm(v)) for v in bar_vals]
    bar_anom   = [bool(_anomaly.get(k, False)) for k, _ in sorted_groups]

    ax_bar = fig.add_axes([0.710, 0.12, 0.245, 0.68])
    bars = ax_bar.barh(
        bar_y, bar_vals, color=bar_colors, alpha=0.88,
        edgecolor="#2C2C2C", linewidth=0.6, height=0.65,
    )
    for idx, (bar, is_anom) in enumerate(zip(bars, bar_anom)):
        if is_anom:
            bar.set_edgecolor("#CC0000")
            bar.set_linewidth(1.8)
            bar.set_hatch("////")

    ax_bar.set_yticks(bar_y)
    _bar_tick_labels = [
        "$\\it{" + lb.replace(" ", r"\ ") + "}$"
        for lb in bar_labels
    ]
    ax_bar.set_yticklabels(_bar_tick_labels, fontsize=7.8)
    ax_bar.set_xlabel(
        "Δ Activation" if show_deviation else "Normalised Activation",
        fontsize=8.5,
    )
    ax_bar.set_title("Ranked by Activation", fontsize=9, fontweight="bold", pad=6)
    if not show_deviation:
        ax_bar.set_xlim(0, 1.05)
    ax_bar.axvline(x=0, color="#333333", linewidth=0.9, linestyle="-")
    if not show_deviation:
        ax_bar.axvline(x=0.5, color="#888888", linewidth=0.7,
                       linestyle="--", alpha=0.6, label="50% activation")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.tick_params(axis="x", labelsize=7.5)
    ax_bar.invert_yaxis()

    ax2 = ax_bar.twinx()
    ax2.set_ylim(ax_bar.get_ylim())
    ax2.set_yticks(bar_y)
    ax2.set_yticklabels(bar_cn, fontsize=6.2, color="#666666", style="italic")
    ax2.invert_yaxis()
    ax2.tick_params(axis="y", length=0)
    ax2.spines["top"].set_visible(False)

    if bar_anom:
        anom_patch = mpatches.Patch(
            facecolor="none", edgecolor="#CC0000",
            hatch="////", linewidth=1.4, label="Anomalous activation",
        )
        ax_bar.legend(
            handles=[anom_patch], fontsize=7, loc="lower right",
            framealpha=0.9, edgecolor="#CCCCCC",
        )

    if has_decision:
        conf_str = f"  (confidence: {decision_confidence:.0%})" if decision_confidence is not None else ""
        banner_col = "#FFEBEE" if "palsy" in (decision_label or "").lower() or "anomal" in (decision_label or "").lower() else "#E8F5E9"
        border_col = "#C62828" if banner_col == "#FFEBEE" else "#2E7D32"
        fig.text(
            0.50, 0.028,
            f"Decision Screening:  {decision_label}{conf_str}",
            ha="center", va="center",
            fontsize=10.5, fontweight="bold",
            color=border_col,
            bbox=dict(
                boxstyle="round,pad=0.55",
                facecolor=banner_col,
                edgecolor=border_col,
                linewidth=1.8,
                alpha=0.97,
            ),
        )

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.975)

    try:
        fig.savefig(str(output_path), dpi=300, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
    except Exception as exc:
        logger.warning("Could not save anatomical figure: %s", exc)
    finally:
        plt.close(fig)
