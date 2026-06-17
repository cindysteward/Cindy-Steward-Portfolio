"""
Orchestrator for study-prompter-sourced analysis sessions.

Drives the full end-to-end analysis pipeline for recordings produced by the
study-prompter HTML tool.  For single-profile sessions it runs one Pipeline
pass.  For COMBINED sessions it runs one Pipeline pass per disorder profile,
reusing the single expensive MediaPipe frame-extraction pass and resegmenting
features per disorder before computing metrics.

Additional computed fields added in anomaly_results (April/May 2026)
===============================================================

``b4_dtw_summary`` (in ``anomaly_results``):

  b4_dtw_vs_ref : float or None
    Test session B4 mean DTW divided by the participant's own reference (baseline)
    B4 mean DTW.  Values > 2.0 indicate that pa-ta-ka kinematics have changed
    dramatically from the participant's own healthy baseline — the canonical
    reference-relative apraxia kinematic marker.  Requires the baseline
    ``dtw_pattern_analysis.json`` to be present for the matched reference session.
    Returns None if the reference DTW cannot be loaded or if B4 mean DTW is
    near-zero (< 0.005).

  b4_rep_dtw_cv : float or None
    Coefficient of variation (std / mean) across individual B4 repetition DTW
    values.  CV > 0.30 indicates high trial-to-trial variability in B4 DTW,
    consistent with apraxic groping behaviour (inconsistent articulatory search
    across repetitions).  Requires ≥ 2 B4 repetitions; returns None otherwise.

``c_dtw_summary`` (in ``anomaly_results``):

  max_c_task_dtw : float
    Maximum DTW value across all 8 Group C word-production tasks.  Complements
    the mean-DTW summary for detecting single-word phonological substitution:
    a single word with a dramatically elevated DTW (> 0.15) while others remain
    normal is the expected kinematic pattern for a phonological substitution
    error affecting one specific word.

``session_metrics``:

  n_a_reps_evaluated : int
    Total number of canonical A-task (A1–A9) repetitions evaluated during
    cross-task substitution-rate computation.  Used downstream to gate
    substitution evidence: a substitution rate derived from ≥ 2 repetitions is
    required before the substitution signal can contribute to buccofacial
    apraxia detection.  Prevents false-positive buccofacial apraxia from
    recordings with sparse A-task coverage (e.g. n = 1 repetition from a
    truncated session).
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np
import shutil
import pandas as pd

from .io_manager import IOManager
from .study_prompter_reader import load_prompter_session, PrompterSession, _NORMAL_PROFILE_MARKERS as _NRM_MARKERS
from .multi_camera_processor import create_multi_camera_processor
from .baseline import create_baseline_constructor, create_baseline_corrector
from .feature_extraction import create_feature_extractor
from .metrics import create_metrics_computer
from .anomaly import create_anomaly_detector
try:
    from .articulation import create_articulation_scorer as _create_artic
except Exception as _e_art:
    logger = logging.getLogger("pipeline")
    logger.warning("articulation module unavailable at import: %s", _e_art)
    _create_artic = None

def create_articulation_scorer_safe(tasks_config):
    """Return an ArticulationScorer if the module is available, otherwise None."""
    if _create_artic is None:
        return None
    return _create_artic(tasks_config)
from .decision_support import create_decision_support
from .visualization import create_visualizer
from .task_profile import TaskProfile, load_task_profile, _BUCCOFACIAL_EXPECTED_REF
from .consolidate import consolidate_subject
from .utils import (
    setup_logging,
    save_json,
    load_json,
    get_pipeline_version,
    resolve_dominant_task,
    MODEL_PATH,
    _FRAME_META_COLUMNS,
    sanitize_events_df,
)

logger = logging.getLogger("pipeline")

_MIN_BASELINE_FRAMES = 30


def _reassign_segments_for_disorder(
    features_df: pd.DataFrame,
    disorder_events_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return a copy of features_df with segment labels reassigned using
    a vectorised interval join against disorder_events_df.

    For each frame timestamp, finds the most recent opening event
    (neutral or measurement) that has not yet been closed by a segment_end.
    Rows outside any valid window are set to inter_trial.
    """
    try:
        disorder_events_df = sanitize_events_df(disorder_events_df)
    except Exception:
        pass

    result = features_df.copy()
    timestamps = result["timestamp_abs"].to_numpy()

    open_events = disorder_events_df[
        disorder_events_df["event_type"].isin(("neutral", "measurement"))
    ].sort_values("timestamp_abs").reset_index(drop=True)

    close_events = disorder_events_df[
        disorder_events_df["event_type"] == "segment_end"
    ].sort_values("timestamp_abs").reset_index(drop=True)

    close_ts = close_events["timestamp_abs"].to_numpy()

    segment_col = np.full(len(timestamps), "inter_trial", dtype=object)
    repetition_col = np.zeros(len(timestamps), dtype=int)
    task_group_col = np.full(len(timestamps), "0", dtype=object)
    task_id_col = np.zeros(len(timestamps), dtype=int)
    task_name_col = np.full(len(timestamps), "(no task selected)", dtype=object)

    for _, open_row in open_events.iterrows():
        ots = float(open_row["timestamp_abs"])
        next_close = close_ts[close_ts > ots]
        window_end = float(next_close[0]) if len(next_close) > 0 else float("inf")

        mask = (timestamps >= ots) & (timestamps < window_end)
        if not mask.any():
            continue

        evt = str(open_row["event_type"])
        seg = "neutral" if evt == "neutral" else "measurement"
        tg = str(open_row.get("task_group") or "0")
        raw_tid = open_row.get("task_id", 0)
        try:
            tid = int(raw_tid)
        except (ValueError, TypeError):
            tid = 0
        tname = str(open_row.get("task_name") or "")
        raw_rep = open_row.get("repetition", 1)
        try:
            rep = int(raw_rep) if raw_rep else 1
        except (ValueError, TypeError):
            rep = 1

        segment_col[mask] = seg
        repetition_col[mask] = rep
        task_group_col[mask] = tg
        task_id_col[mask] = tid
        task_name_col[mask] = tname

    result["segment"] = segment_col
    result["repetition"] = repetition_col
    result["task_group"] = task_group_col
    result["task_id"] = task_id_col
    result["task_name"] = task_name_col
    return result


def _assign_segment_static(
    timestamp_abs: float,
    events_df: pd.DataFrame,
) -> tuple:
    """Stateless segment assignment matching MultiCameraProcessor._assign_frame_segment.

    Returns (segment, repetition, task_group, task_id, task_name).
    Scans events_df in timestamp order, tracking the most recently opened event
    window.  A segment_end closes the current window.  When no open window
    spans the query timestamp, returns inter_trial defaults.
    """
    active_event = None
    active_closed = False

    for _, row in events_df.iterrows():
        row_ts = float(row["timestamp_abs"])
        if row_ts > timestamp_abs:
            break
        evt = str(row["event_type"])
        if evt in ("neutral", "measurement"):
            active_event = row
            active_closed = False
        elif evt == "segment_end":
            if active_event is not None:
                active_closed = True

    if active_event is None or active_closed:
        return ("inter_trial", 0, "0", 0, "(no task selected)")

    evt_type = str(active_event["event_type"])
    task_group = str(active_event.get("task_group", "0") or "0")
    raw_task_id = active_event.get("task_id", 0)
    try:
        task_id = int(raw_task_id)
    except (ValueError, TypeError):
        task_id = 0
    task_name = str(active_event.get("task_name", "") or "")
    raw_rep = active_event.get("repetition", 1)
    try:
        repetition = int(raw_rep) if raw_rep else 1
    except (ValueError, TypeError):
        repetition = 1

    segment = "neutral" if evt_type == "neutral" else "measurement"
    return (segment, repetition, task_group, task_id, task_name)


def _discover_reference_session(
    data_dir: Path,
    study_mode: str,
    subject_id: str,
    session_logger: logging.Logger,
    session_label: str = "",
) -> Optional[str]:
    """Search for the most recently created baseline or NORMAL profile session.

    Scans data/raw/<mode>/<subject>/ for session directories containing
    session_metadata.json whose session_label contains 'baseline' or 'normal'
    (case-insensitive).  Returns the session_id of the most recent match,
    or None if no candidate is found.

    When *session_label* is provided, the search first tries to find a baseline
    whose session_id contains the same condition keywords (e.g. 'upright',
    'ors', 'rotated') so that an upright test session is compared against an
    upright baseline rather than an ORS or rotated baseline.
    """
    raw_subj_dir = data_dir / "raw" / study_mode / subject_id
    if not raw_subj_dir.exists():
        return None

    candidates: List[tuple] = []
    for session_dir in raw_subj_dir.iterdir():
        if not session_dir.is_dir():
            continue
        meta_path = session_dir / "session_metadata.json"
        if not meta_path.exists():
            continue
        try:
            meta = load_json(meta_path)
        except Exception:
            continue
        label = meta.get("session_label", "").lower()
        _sp_up = meta.get("session_profile", "").strip().upper()
        if (
            "baseline" in label
            or "normal" in label
            or "basislijn" in label
            or any(m in _sp_up for m in {"NORMAL", "NORMAAL"})
        ):
            candidates.append((meta.get("created_at", ""), meta.get("session_id", session_dir.name)))

    if not candidates:
        return None

    if session_label:
        _lbl = session_label.lower()
        _cond_keys = []
        if "ors" in _lbl:
            _cond_keys.append("ors")
        if "upright" in _lbl:
            _cond_keys.append("upright")
        if "rotated" in _lbl:
            _cond_keys.append("rotated")
        if _cond_keys:
            matched = [
                (ts, sid) for ts, sid in candidates
                if all(k in sid.lower() for k in _cond_keys)
            ]
            if not matched:
                matched = [
                    (ts, sid) for ts, sid in candidates
                    if _cond_keys[0] in sid.lower()
                ]
            if matched:
                matched.sort(reverse=True)
                return matched[0][1]

    candidates.sort(reverse=True)
    return candidates[0][1]


def _detect_intra_op_sequences(timestamps_path: Path) -> List[int]:
    """Return sorted unique sequence numbers found in the timestamps CSV.

    Reads the 'sequence' column and collects integer values.  Returns an
    empty list when the column is absent or contains only a single value —
    i.e. when no multi-sequence split is needed.
    """
    try:
        df = pd.read_csv(timestamps_path, dtype=str, keep_default_na=False, quotechar='"')
    except Exception:
        return []
    if "sequence" not in df.columns:
        return []
    seen: set = set()
    for val in df["sequence"]:
        s = str(val).strip()
        if s and s.lower() not in ("nan", ""):
            try:
                seen.add(int(float(s)))
            except (ValueError, TypeError):
                pass
    return sorted(seen) if len(seen) > 1 else []


def _make_sequence_csv(timestamps_path: Path, seq_num: int) -> Path:
    """Write a per-sequence filtered timestamps CSV to a temp file.

    Keeps rows whose 'sequence' column matches *seq_num* plus rows whose
    'sequence' is blank/missing (e.g. the shared neutral/baseline section
    recorded before the first battery run begins).

    Also normalises 'sequence_rep' to '1' for any task row where that
    column is empty: within a single battery run each task is performed
    once, so the fallback value should be 1 regardless of the run number
    stored in the 'sequence' column.

    Returns the path to the temp CSV file.  Caller is responsible for
    deleting it when done.
    """
    import tempfile as _tmp

    df = pd.read_csv(timestamps_path, dtype=str, keep_default_na=False, quotechar='"')
    seq_str = str(seq_num)

    def _is_blank(v: str) -> bool:
        """Return True if the string value is empty, 'nan', or 'None'."""
        return str(v).strip() in ("", "nan", "None")

    mask = df["sequence"].apply(lambda x: str(x).strip() == seq_str or _is_blank(str(x)))
    filtered = df[mask].reset_index(drop=True)

    if "sequence_rep" in filtered.columns:
        filtered["sequence_rep"] = filtered["sequence_rep"].apply(
            lambda x: x if not _is_blank(str(x)) else "1"
        )

    fh = _tmp.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False,
        encoding="utf-8", newline="",
        prefix=f"intra_op_seq{seq_num}_",
    )
    filtered.to_csv(fh.name, index=False)
    fh.close()
    return Path(fh.name)


def _run_intra_op_from_frames(
    features_df: "pd.DataFrame",
    video_paths: List[Path],
    timestamps_path: Path,
    subject_id: str,
    session_label: str,
    study_mode: str,
    project_root: Path,
    meta_path: Optional[Path],
    reference_session: Optional[List[str]],
    progress_callback: Optional[Callable[[str, int], None]],
    sequence_nums: List[int],
    primary_fps: float,
    camera_offsets: Dict[int, float],
    annotated_video_tmps: list,
    landmark_video_tmps: list,
    tasks_config: Dict[str, Any],
    features_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Orchestrate per-sequence intra-op analysis after frames are extracted.

    Sequence 1 is saved as *intra_op_baseline_sequence1* and treated as the
    intra-operative baseline for this patient.  Sequences 2, 3, … are run
    as test sessions that reference both sequence 1 and any pre-operative
    baseline found in the subject's data directory.  All sequences share the
    single frame-extraction pass already performed by the caller.
    """
    def _progress(step: str, pct: int) -> None:
        """Forward a progress update to the caller's callback if one was provided."""
        if progress_callback is not None:
            try:
                progress_callback(step, pct)
            except Exception:
                pass

    n = len(sequence_nums)
    sequence_results: List[Dict] = []
    seq1_session_id: Optional[str] = None

    _tmp_io = IOManager(project_root, subject_id, session_label, study_mode, list_only=True)
    pre_op_ref_id = _discover_reference_session(
        _tmp_io.data_dir, study_mode, subject_id, logger,
        session_label=session_label,
    )
    if pre_op_ref_id:
        logger.info("Intra-op: discovered pre-op reference session: %s", pre_op_ref_id)
    del _tmp_io

    for i, seq_num in enumerate(sequence_nums):
        base_pct = 60 + int(i / n * 35)
        is_first = i == 0

        seq_label = (
            f"intra_op_baseline_sequence{seq_num}" if is_first
            else f"intra_op_sequence{seq_num}"
        )
        _progress(f"Intra-op sequence {seq_num}/{sequence_nums[-1]}: {seq_label}", base_pct)
        logger.info("Intra-op sequence %d → session_label='%s'", seq_num, seq_label)

        seq_csv_path: Optional[Path] = None
        try:
            seq_csv_path = _make_sequence_csv(timestamps_path, seq_num)

            seq_session = load_prompter_session(
                timestamps_path=seq_csv_path,
                meta_path=meta_path,
                assembly_path=None,
            )

            seq_features_df = _reassign_segments_for_disorder(features_df, seq_session.events_df)

            if is_first:
                seq_references: Optional[List[str]] = None
            else:
                seq_refs: List[str] = []
                if seq1_session_id:
                    seq_refs.append(seq1_session_id)
                for r in (reference_session or []):
                    if r not in seq_refs:
                        seq_refs.append(r)
                seq_references = seq_refs if seq_refs else None

            def _seq_sub_progress(
                step: str, pct: int,
                _base: int = base_pct, _slot: int = max(1, 35 // n),
            ) -> None:
                """Map per-sequence progress onto the parent progress slot."""
                _progress(f"Seq {seq_num}: {step}", _base + int(pct / 100 * _slot))

            seq_summary = _run_single_profile_analysis(
                subject_id=subject_id,
                session_label=seq_label,
                study_mode=study_mode,
                project_root=project_root,
                features_df=seq_features_df,
                events_df=seq_session.events_df,
                tasks_config=tasks_config,
                features_config=features_config,
                reference_session=seq_references,
                session_profile=seq_session.profile,
                fps=primary_fps,
                annotated_video_srcs=annotated_video_tmps if is_first else None,
                landmark_video_srcs=landmark_video_tmps if is_first else None,
                sub_progress=_seq_sub_progress,
            )

            seq_summary["sequence_number"] = seq_num
            seq_summary["is_intra_op_baseline"] = is_first
            sequence_results.append(seq_summary)

            if is_first:
                seq1_session_id = seq_summary.get("session_id")
                logger.info("Intra-op baseline session_id captured: %s", seq1_session_id)

        except Exception as exc:
            logger.error("Intra-op sequence %d analysis failed: %s", seq_num, exc)
            import traceback as _tb
            logger.debug("Traceback:\n%s", _tb.format_exc())
            sequence_results.append({
                "sequence_number": seq_num,
                "is_intra_op_baseline": is_first,
                "session_label": seq_label,
                "error": str(exc),
            })
        finally:
            if seq_csv_path is not None:
                try:
                    seq_csv_path.unlink()
                except Exception:
                    pass

    try:
        consolidate_subject(project_root, subject_id, study_mode)
    except Exception as exc:
        logger.warning("Could not consolidate after intra-op sequences: %s", exc)

    for seq_res in sequence_results:
        if "output_paths" not in seq_res:
            continue
        try:
            import sys as _sys
            _tools_dir = str(Path(project_root) / "tools")
            if _tools_dir not in _sys.path:
                _sys.path.insert(0, _tools_dir)
            from session_summary_figure import generate_session_summary as _gen_summary
            _gen_summary(Path(seq_res["output_paths"]["results"]).parent)
        except Exception as _exc:
            logger.debug("Could not generate session summary for intra-op seq: %s", _exc)

    try:
        import sys as _sys
        _tools_dir = str(Path(project_root) / "tools")
        if _tools_dir not in _sys.path:
            _sys.path.insert(0, _tools_dir)
        from session_summary_figure import generate_participant_summary as _gen_participant
        _participant_results = Path(project_root) / "data" / "results" / study_mode / subject_id
        _gen_participant(_participant_results)
    except Exception as _exc:
        logger.debug("Could not generate participant summary after intra-op: %s", _exc)

    _progress("Done", 100)

    first_result = next((r for r in sequence_results if r.get("is_intra_op_baseline")), {})
    return {
        "participant_id": subject_id,
        "profile": first_result.get("session_profile", "intra_op"),
        "session_date": first_result.get("session_date", ""),
        "is_combined": False,
        "is_intra_op": True,
        "is_baseline_session": False,
        "disorder_results": sequence_results,
        "output_root": str(project_root / "data" / "results" / study_mode / subject_id),
    }


def _merge_dicts_recursive(dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Recursively merge a list of dicts by averaging numeric scalars and recursing into nested dicts.

    Non-numeric, non-dict values are taken from the first dict that has them.
    This preserves nested structures such as per_task_scores and per_word_scores.
    """
    if not dicts:
        return {}
    if len(dicts) == 1:
        return dict(dicts[0])
    merged: Dict[str, Any] = {}
    all_keys: set = set()
    for d in dicts:
        all_keys.update(d.keys())
    for key in all_keys:
        scalars: List[float] = []
        nested: List[Dict[str, Any]] = []
        fallback = None
        for d in dicts:
            if key not in d:
                continue
            val = d[key]
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                scalars.append(float(val))
            elif isinstance(val, dict):
                nested.append(val)
            elif fallback is None:
                fallback = val
        if scalars:
            merged[key] = float(np.mean(scalars))
        elif nested:
            merged[key] = _merge_dicts_recursive(nested)
        elif fallback is not None:
            merged[key] = fallback
    return merged


def _load_reference_data(
    data_dir: Path,
    study_mode: str,
    subject_id: str,
    reference_session_ids: List[str],
    session_logger: logging.Logger,
) -> Dict[str, Any]:
    """Load and merge baseline stats and repetition metrics from reference sessions.

    For each reference session ID, attempts to load:
    - data/raw/<mode>/<subject>/<session_id>/baseline.json
    - data/processed/<mode>/<subject>/<session_id>/repetition_metrics.csv
    - data/results/<mode>/<subject>/<session_id>/articulation_scores.json

    The subject prefix is extracted as session_id.split("_")[0] with fallback
    to subject_id.

    Merges across sessions by:
    - baseline_stats: per-blendshape equal-weight average of all numeric fields
    - reference_metrics_df: pd.concat of all loaded repetition metrics DataFrames
    - reference_articulation: equal-weight average of all numeric scalar fields

    Missing files are logged as warnings and skipped.  Returns a dict with keys:
    - "baseline_stats": merged dict or None if nothing loaded
    - "reference_metrics_df": DataFrame (may be empty)
    - "reference_articulation": merged dict or None
    - "session_ids_used": list of successfully loaded session IDs
    """
    all_baseline_stats: List[Dict[str, Any]] = []
    all_metrics_dfs: List[pd.DataFrame] = []
    all_articulation: List[Dict[str, Any]] = []
    all_kinematic_summaries: Dict[str, List[Dict[str, Any]]] = {}
    session_ids_used: List[str] = []

    _pipeline_meta_cols = {"_subject_id", "_session_id", "_pipeline_version", "_config_hash"}

    for ref_id in reference_session_ids:
        ref_subject = ref_id.split("_")[0] if ref_id else subject_id
        if not ref_subject:
            ref_subject = subject_id

        baseline_path = data_dir / "raw" / study_mode / ref_subject / ref_id / "baseline.json"
        metrics_path = data_dir / "processed" / study_mode / ref_subject / ref_id / "repetition_metrics.csv"
        artic_path = data_dir / "results" / study_mode / ref_subject / ref_id / "articulation_scores.json"

        loaded_any = False

        if baseline_path.exists():
            try:
                bdata = load_json(baseline_path)
                stats = bdata.get("statistics", bdata)
                if stats:
                    all_baseline_stats.append(stats)
                    loaded_any = True
            except Exception as exc:
                session_logger.warning(
                    "Could not load baseline from %s: %s", ref_id, exc
                )

        if metrics_path.exists():
            try:
                df = pd.read_csv(metrics_path)
                drop_cols = [c for c in _pipeline_meta_cols if c in df.columns]
                if drop_cols:
                    df = df.drop(columns=drop_cols)
                if len(df) > 0:
                    all_metrics_dfs.append(df)
                    loaded_any = True
            except Exception as exc:
                session_logger.warning(
                    "Could not load repetition_metrics from %s: %s", ref_id, exc
                )

        if artic_path.exists():
            try:
                artic = load_json(artic_path)
                if artic:
                    all_articulation.append(artic)
                    loaded_any = True
            except Exception as exc:
                session_logger.warning(
                    "Could not load articulation_scores from %s: %s", ref_id, exc
                )

        for tg in ("B", "C"):
            kin_path = (
                data_dir / "results" / study_mode / ref_subject / ref_id
                / f"kinematic_summary_group_{tg}.json"
            )
            if kin_path.exists():
                try:
                    kin_data = load_json(kin_path)
                    if kin_data:
                        all_kinematic_summaries.setdefault(tg, []).append(kin_data)
                        loaded_any = True
                except Exception as exc:
                    session_logger.warning(
                        "Could not load kinematic_summary_%s from %s: %s", tg, ref_id, exc
                    )

        if loaded_any:
            session_ids_used.append(ref_id)

    merged_baseline: Optional[Dict[str, Any]] = None
    if all_baseline_stats:
        merged_baseline = {}
        all_blendshapes: set = set()
        for stat_dict in all_baseline_stats:
            all_blendshapes.update(stat_dict.keys())
        for bs in all_blendshapes:
            field_accum: Dict[str, List[float]] = {}
            for stat_dict in all_baseline_stats:
                if bs not in stat_dict:
                    continue
                for field_name, val in stat_dict[bs].items():
                    if isinstance(val, (int, float)):
                        field_accum.setdefault(field_name, []).append(float(val))
            merged_baseline[bs] = {f: float(np.mean(vals)) for f, vals in field_accum.items()}

    reference_metrics_df = (
        pd.concat(all_metrics_dfs, ignore_index=True) if all_metrics_dfs else pd.DataFrame()
    )

    merged_articulation: Optional[Dict[str, Any]] = None
    if all_articulation:
        merged_articulation = _merge_dicts_recursive(all_articulation)

    merged_kinematic_summaries: Dict[str, Dict[str, Any]] = {}
    for tg, summaries in all_kinematic_summaries.items():
        if summaries:
            merged_kin: Dict[str, Any] = {}
            all_kin_keys: set = set()
            for s in summaries:
                all_kin_keys.update(s.keys())
            for key in all_kin_keys:
                vals = [
                    s[key] for s in summaries
                    if key in s and isinstance(s[key], (int, float))
                ]
                if vals:
                    merged_kin[key] = float(np.mean(vals))
            if merged_kin:
                merged_kinematic_summaries[tg] = merged_kin

    return {
        "baseline_stats": merged_baseline,
        "reference_metrics_df": reference_metrics_df,
        "reference_articulation": merged_articulation,
        "reference_kinematic_summaries": merged_kinematic_summaries,
        "session_ids_used": session_ids_used,
    }


def _run_single_profile_analysis(
    subject_id: str,
    session_label: str,
    study_mode: str,
    project_root: Path,
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    tasks_config: Dict[str, Any],
    features_config: Dict[str, Any],
    reference_session: Optional[List[str]],
    session_profile: str = "",
    fps: float = 30.0,
    annotated_video_srcs: Optional[List[Optional[Path]]] = None,
    landmark_video_srcs: Optional[List[Optional[Path]]] = None,
    parent_session_id: Optional[str] = None,
    sub_progress: Optional[Callable[[str, int], None]] = None,
    skip_kin_figures: bool = False,
    skip_all_figures: bool = False,
) -> Dict[str, Any]:
    """Run baseline correction, metrics, anomaly detection, decision support,
    and visualization for a single profile's events subset.

    Uses IOManager to create the session directory.  When parent_session_id
    is provided the outputs are placed under
    data/{type}/{mode}/{subject}/{parent_session_id}/{session_label}/ so that
    all disorder profiles from a COMBINED session share a common parent.
    Saves frame_data CSV, blendshapes CSV, corrected_features CSV,
    repetition_metrics CSV, task_metrics CSV, session_metrics JSON,
    anomaly_results JSON, screening_results JSON, confidence_summary JSON,
    and all visualizations.
    Returns the pipeline summary dict.
    """
    io = IOManager(
        project_root, subject_id, session_label, study_mode,
        parent_session_id=parent_session_id,
    )
    session_logger = setup_logging(io.logs_dir, io.session_id)

    def _sp(step: str, pct: int) -> None:
        """Forward a progress update to the sub-progress callback if provided."""
        if sub_progress is not None:
            try:
                sub_progress(step, pct)
            except Exception:
                pass
    session_logger.info(
        "Starting single-profile analysis: subject=%s, session=%s, mode=%s",
        subject_id, session_label, study_mode,
    )

    if annotated_video_srcs:
        for cam_idx, src_path in enumerate(annotated_video_srcs):
            if src_path is None:
                continue
            try:
                suffix = f"_cam{cam_idx + 1}"
                dest_path = io.get_annotated_video_path(suffix=suffix)
                shutil.copy2(str(src_path), str(dest_path))
                session_logger.info("Annotated video (cam%d) saved to %s", cam_idx + 1, dest_path)
            except Exception as _ann_exc:
                session_logger.warning("Could not save annotated video cam%d: %s", cam_idx + 1, _ann_exc)

    if landmark_video_srcs:
        for cam_idx, src_path in enumerate(landmark_video_srcs):
            if src_path is None:
                continue
            try:
                suffix = f"_cam{cam_idx + 1}"
                dest_path = io.get_landmarks_video_path(suffix=suffix)
                shutil.copy2(str(src_path), str(dest_path))
                session_logger.info("Landmark video (cam%d) saved to %s", cam_idx + 1, dest_path)
            except Exception as _lm_exc:
                session_logger.warning("Could not save landmark video cam%d: %s", cam_idx + 1, _lm_exc)

    try:
        events_df = sanitize_events_df(events_df)
    except Exception:
        session_logger.warning("Failed to sanitize events_df — proceeding with original DataFrame")

    decision_rules_config = io.load_config("decision_rules")
    plotting_config = io.load_config("plotting")

    features_df_to_save = features_df.drop(
        columns=[c for c in ["_landmarks_3d"] if c in features_df.columns],
        errors="ignore",
    )
    io.save_dataframe(features_df_to_save, io.get_frame_data_path())
    io.save_dataframe(events_df, io.get_events_path())

    baseline_constructor = create_baseline_constructor(features_config)
    feature_extractor = create_feature_extractor(features_config, tasks_config)
    metrics_computer = create_metrics_computer(features_config, tasks_config)
    articulation_scorer = create_articulation_scorer_safe(tasks_config)
    decision_support = create_decision_support(decision_rules_config)
    visualizer = create_visualizer(plotting_config)

    _profile_up = session_profile.strip().upper()
    _label_low = session_label.lower()
    _TEST_SESSION_MARKERS = frozenset({
        "test", "postop", "post_op", "followup", "follow_up", "retest",
    })
    _label_is_test = any(m in _label_low for m in _TEST_SESSION_MARKERS)
    is_baseline_session = parent_session_id is None and not _label_is_test and (
        "baseline" in _label_low
        or "normal" in _label_low
        or "basislijn" in _label_low
        or "basislijn" in _profile_up
        or any(m in _profile_up for m in _NRM_MARKERS)
    )

    auto_reference_id: Optional[str] = None
    effective_reference_ids: List[str] = list(reference_session or [])

    _BLENDSHAPES_EXCLUDE = _FRAME_META_COLUMNS | {
        "task_group", "task_id", "task_name", "brightness",
    }
    underscore_keep = [
        c for c in features_df.columns
        if c.startswith("_") and c in ("_landmarks_3d",)
    ]
    blendshape_cols = [
        c for c in features_df.columns
        if c not in _BLENDSHAPES_EXCLUDE
        and not c.startswith("_")
        and not any(c.endswith(s) for s in ("_x", "_y", "_z"))
    ]
    landmark_pos_cols = [
        c for c in features_df.columns
        if any(c.endswith(s) for s in ("_x", "_y", "_z"))
        and not c.startswith("_")
    ]
    meta_cols_present = [
        c for c in [
            "frame_index", "timestamp_abs", "segment", "repetition",
            "detection_success", "detection_confidence", "task_group",
            "task_id", "task_name", "brightness",
        ]
        if c in features_df.columns
    ]
    selected_cols = meta_cols_present + underscore_keep + blendshape_cols
    blendshapes_df = features_df[selected_cols].copy()
    if underscore_keep:
        logger.info("Preserving underscore columns for downstream processing: %s", underscore_keep)

    io.save_dataframe(blendshapes_df, io.get_blendshapes_path())

    neutral_df = blendshapes_df[blendshapes_df["segment"] == "neutral"]
    if len(neutral_df) == 0:
        n_baseline = max(_MIN_BASELINE_FRAMES, int(len(blendshapes_df) * 0.1))
        neutral_df = blendshapes_df.head(n_baseline)
        session_logger.warning(
            "No neutral segment found — using first %d frames as baseline", n_baseline
        )

    baseline_constructor.compute_baseline(blendshapes_df, neutral_df)
    baseline_constructor.compute_observed_ranges(blendshapes_df)
    _sp("Baseline correction", 10)

    if not effective_reference_ids and not is_baseline_session:
        auto_reference_id = _discover_reference_session(
            io.data_dir, study_mode, subject_id, session_logger,
            session_label=session_label,
        )
        if auto_reference_id:
            effective_reference_ids = [auto_reference_id]
            session_logger.info(
                "Auto-discovered reference session: %s", auto_reference_id
            )

    reference_data = _load_reference_data(
        io.data_dir, study_mode, subject_id, effective_reference_ids, session_logger
    )
    used_reference_ids: List[str] = reference_data["session_ids_used"]
    reference_baseline_stats: Optional[Dict] = reference_data["baseline_stats"]
    reference_metrics_df: pd.DataFrame = reference_data["reference_metrics_df"]
    reference_articulation: Optional[Dict] = reference_data["reference_articulation"]
    reference_kinematic_summaries: Dict[str, Dict] = reference_data.get("reference_kinematic_summaries", {})

    if reference_baseline_stats and not is_baseline_session:
        baseline_constructor.merge_external_baseline(reference_baseline_stats)
        session_logger.info(
            "Merged reference baseline from %d session(s): %s",
            len(used_reference_ids), used_reference_ids,
        )
    elif reference_baseline_stats and is_baseline_session:
        session_logger.info(
            "Baseline session: skipping merge of reference stats from %s "
            "(keeping this session's own measurements uncontaminated).",
            used_reference_ids,
        )

    try:
        baseline_constructor.save_baseline(io.get_baseline_path())
    except Exception as exc:
        session_logger.warning("Could not save baseline: %s", exc)

    corrector = create_baseline_corrector(baseline_constructor)
    try:
        standardized_df = corrector.standardize_features(blendshapes_df)
    except Exception as _std_exc:
        session_logger.warning("Baseline standardisation failed — using raw blendshapes: %s", _std_exc)
        standardized_df = blendshapes_df.copy()

    if landmark_pos_cols:
        for _lc in landmark_pos_cols:
            if _lc in features_df.columns and _lc not in standardized_df.columns:
                standardized_df[_lc] = features_df[_lc].values

    try:
        extracted_features_df = feature_extractor.extract_features(
            standardized_df, events_df,
            baseline_stats=baseline_constructor.baseline_stats,
            observed_ranges=baseline_constructor.observed_ranges,
        )
    except Exception as _feat_exc:
        session_logger.warning("Feature extraction failed — using standardised frame data as features: %s", _feat_exc)
        extracted_features_df = standardized_df.copy()

    try:
        io.save_dataframe(extracted_features_df, io.get_corrected_features_path())
    except Exception as _save_exc:
        session_logger.warning("Could not save corrected features: %s", _save_exc)

    if "_landmarks_3d" in features_df.columns:
        try:
            extracted_features_df = extracted_features_df.copy()
            extracted_features_df["_landmarks_3d"] = (
                features_df["_landmarks_3d"].reindex(extracted_features_df.index).values
            )
        except Exception as _lm_exc:
            session_logger.warning(
                "Could not carry forward _landmarks_3d: %s", _lm_exc
            )

    cont_detector = None
    try:
        from .anomaly import ContinuousBaselineEstimator, ContinuousAnomalyDetector

        _ref_feat_dfs = []
        for _ref_id in used_reference_ids:
            _ref_subj = _ref_id.split("_")[0] if _ref_id else subject_id
            _ref_feat_path = (
                io.data_dir / "processed" / study_mode / _ref_subj / _ref_id / "corrected_features.csv"
            )
            if _ref_feat_path.exists():
                try:
                    _rdf = pd.read_csv(_ref_feat_path)
                    if "timestamp_abs" in _rdf.columns and len(_rdf) > 5:
                        _ref_feat_dfs.append(_rdf)
                except Exception as _rfe:
                    session_logger.debug("Could not load ref features %s: %s", _ref_id, _rfe)

        if _ref_feat_dfs:
            _ref_combined = pd.concat(_ref_feat_dfs, ignore_index=True)
            cont_baseline = ContinuousBaselineEstimator(
                baseline_duration_s=9999.0,
                min_baseline_frames=10,
            ).fit(_ref_combined)
            session_logger.info(
                "Continuous anomaly: baseline fitted from %d reference session(s) "
                "(%d frames).", len(_ref_feat_dfs), len(_ref_combined)
            )
        else:
            cont_baseline = ContinuousBaselineEstimator(
                baseline_duration_s=30.0,
                min_baseline_frames=30,
            ).fit(extracted_features_df)
            session_logger.info(
                "Continuous anomaly: no reference features found, "
                "using first 30 s of session as baseline."
            )

        normative_stats = None
        normative_path = io.data_dir / "normative_reference.json"
        if normative_path.exists():
            from .baseline import load_normative_reference
            normative_stats = load_normative_reference(normative_path)

        cont_detector = ContinuousAnomalyDetector(
            baseline_estimator=cont_baseline,
            normative_stats=normative_stats,
            window_size_s=2.0,
            step_size_s=0.5,
        )
    except Exception as exc:
        session_logger.warning("Continuous anomaly detection failed: %s", exc)

    try:
        from .anomaly import FatigueDriftMonitor

        fatigue_monitor = FatigueDriftMonitor(
            baseline_duration_s=120.0,
            window_size_s=60.0,
            step_size_s=10.0,
        ).fit(extracted_features_df)

        fatigue_report = fatigue_monitor.analyze(extracted_features_df)
        save_json(fatigue_report, io.results_dir / "fatigue_drift_report.json")
        session_logger.info(
            "Fatigue drift monitor: %d windows, %d flagged (%.0f%% session, "
            "flags: %s).",
            fatigue_report["summary"]["n_windows"],
            fatigue_report["summary"]["n_flagged"],
            fatigue_report["summary"]["flag_fraction"] * 100,
            fatigue_report["summary"].get("flag_counts_by_type", {}),
        )
        if visualizer is not None and not skip_all_figures:
            try:
                visualizer.plot_fatigue_drift_report(
                    fatigue_report,
                    io.results_dir / "fatigue_drift_analysis.png",
                    title=f"Fatigue & Motor Drift — {io.session_id}",
                )
                session_logger.info("Fatigue drift figure saved.")
            except Exception as _vis_exc:
                session_logger.warning("Fatigue drift figure failed: %s", _vis_exc)
    except Exception as exc:
        session_logger.warning("Fatigue drift monitoring failed: %s", exc)

    kin_df = pd.DataFrame()
    kinematic_profiles_ref = None
    kinematic_summaries = {}
    fps_estimate = fps

    try:
        from .kinematic_speech import (
            extract_kinematic_features,
            add_kinematic_derivatives,
            compute_task_kinematic_summary,
            extract_group_a_kinematics,
            compute_group_a_task_summary,
        )

        kin_df = extract_kinematic_features(
            extracted_features_df,
            task_groups=["A", "B", "C"],
            neutral_face_size=None,
        )
        if not kin_df.empty:
            kin_df = add_kinematic_derivatives(kin_df, fps=fps_estimate)
            kin_df_aligned = kin_df.reindex(extracted_features_df.index)
            new_kin_cols = [c for c in kin_df_aligned.columns if c not in extracted_features_df.columns]
            if new_kin_cols:
                extracted_features_df = pd.concat([extracted_features_df, kin_df_aligned[new_kin_cols]], axis=1)
                session_logger.info("Kinematic speech features added: %d columns.", len(new_kin_cols))

        group_a_present = (
            "task_group" in extracted_features_df.columns
            and (extracted_features_df["task_group"] == "A").any()
        )
        if group_a_present:
            try:
                kin_a_df = extract_group_a_kinematics(extracted_features_df, fps=fps_estimate)
                if not kin_a_df.empty:
                    new_a_cols = [c for c in kin_a_df.columns if c not in extracted_features_df.columns]
                    if new_a_cols:
                        extracted_features_df = pd.concat(
                            [extracted_features_df, kin_a_df[new_a_cols].reindex(extracted_features_df.index)],
                            axis=1,
                        )
                        session_logger.info("Group A kinematic features added: %d columns.", len(new_a_cols))

                group_a_summary = compute_group_a_task_summary(extracted_features_df, fps=fps_estimate)
                if group_a_summary:
                    save_json(group_a_summary, io.results_dir / "kinematic_summary_group_A.json")
                    kinematic_summaries["A"] = group_a_summary
                    session_logger.info("Group A kinematic summary: %d tasks.", len(group_a_summary))
            except Exception as _a_kin_exc:
                session_logger.warning("Group A kinematic extraction failed: %s", _a_kin_exc)

        kinematic_profiles = None
        profiles_path = (
            io.data_dir / "results" / study_mode / subject_id
            / f"{subject_id}_kinematic_reference_profiles.json"
        )
        if profiles_path.exists():
            kinematic_profiles = load_json(profiles_path).get("profiles")

        if is_baseline_session and not kin_df.empty and "task_group" in extracted_features_df.columns:
            try:
                from .kinematic_speech import build_reference_profile
                new_profiles: Dict[str, Any] = {} if kinematic_profiles is None else dict(kinematic_profiles)
                for tg_ref in ["A", "B", "C"]:
                    tg_mask = extracted_features_df["task_group"] == tg_ref
                    if not tg_mask.any():
                        continue
                    for tid_ref in extracted_features_df.loc[tg_mask, "task_id"].dropna().unique():
                        task_mask_ref = tg_mask & (extracted_features_df["task_id"] == tid_ref)
                        task_kin = kin_df[task_mask_ref]
                        rep_col = extracted_features_df.loc[task_mask_ref, "repetition"] if "repetition" in extracted_features_df.columns else pd.Series(dtype=float)
                        task_key = f"{tg_ref}_{int(tid_ref)}"
                        existing_task_profile: Dict[str, Any] = new_profiles.get(task_key, {})
                        updated_task_profile: Dict[str, Any] = dict(existing_task_profile)
                        for meas_col in [
                            c for c in kin_df.columns
                            if c.startswith("kin_")
                            and not c.endswith(("_vel", "_acc"))
                            and c != "kin_face_size"
                        ]:
                            series_list = []
                            if len(rep_col) > 0:
                                for rep_id in rep_col.dropna().unique():
                                    rep_mask = rep_col == rep_id
                                    vals = task_kin.loc[rep_mask, meas_col].dropna().to_numpy()
                                    if len(vals) >= 2:
                                        series_list.append(vals)
                            else:
                                vals = task_kin[meas_col].dropna().to_numpy()
                                if len(vals) >= 2:
                                    series_list.append(vals)
                            if series_list:
                                new_prof = build_reference_profile(series_list, meas_col)
                                prev = existing_task_profile.get(meas_col)
                                if prev and "mean" in prev and "n" in prev:
                                    n_old = int(prev["n"])
                                    n_new = int(new_prof["n"])
                                    w_total = n_old + n_new
                                    old_mean = np.array(prev["mean"], dtype=float)
                                    new_mean = new_prof["mean"]
                                    min_len = min(len(old_mean), len(new_mean))
                                    merged_mean = (
                                        old_mean[:min_len] * n_old + new_mean[:min_len] * n_new
                                    ) / w_total
                                    old_std = np.array(prev["std"], dtype=float)
                                    new_std = new_prof["std"]
                                    merged_std = np.sqrt(
                                        (old_std[:min_len] ** 2 * n_old + new_std[:min_len] ** 2 * n_new) / w_total
                                    )
                                    updated_task_profile[meas_col] = {
                                        "mean": merged_mean.tolist(),
                                        "std": merged_std.tolist(),
                                        "n": w_total,
                                    }
                                else:
                                    updated_task_profile[meas_col] = {
                                        "mean": new_prof["mean"].tolist(),
                                        "std": new_prof["std"].tolist(),
                                        "n": new_prof["n"],
                                    }
                        if updated_task_profile:
                            new_profiles[task_key] = updated_task_profile
                profiles_path.parent.mkdir(parents=True, exist_ok=True)
                save_json({"profiles": new_profiles}, profiles_path)
                kinematic_profiles = new_profiles
                session_logger.info(
                    "Kinematic reference profiles saved: %d task(s) at %s",
                    len(new_profiles), profiles_path,
                )
            except Exception as _kp_exc:
                session_logger.warning("Could not build kinematic reference profiles: %s", _kp_exc)

        for tg in ["B", "C"]:
            kin_summary = compute_task_kinematic_summary(
                kin_df, extracted_features_df, tg,
                reference_profiles=kinematic_profiles,
            )
            if kin_summary:
                ref_kin = reference_kinematic_summaries.get(tg, {})
                if ref_kin:
                    rel_devs = []
                    for key, val in kin_summary.items():
                        if not isinstance(val, float) or key not in ref_kin:
                            continue
                        ref_val = ref_kin[key]
                        if not isinstance(ref_val, float):
                            continue
                        scale = max(abs(ref_val), 1e-6)
                        rel_devs.append(abs(val - ref_val) / scale)
                    if rel_devs:
                        kin_summary["overall_deviation"] = float(np.mean(rel_devs) * 3.0)
                        session_logger.info(
                            "Kinematic overall deviation (group %s vs reference): %.3f",
                            tg, kin_summary["overall_deviation"],
                        )

                kinematic_summaries[tg] = kin_summary
                save_json(kin_summary, io.results_dir / f"kinematic_summary_group_{tg}.json")
                session_logger.info("Kinematic summary for group %s: %d metrics.", tg, len(kin_summary))

        kinematic_profiles_ref = kinematic_profiles

        try:
            from .anomaly import ContinuousAnomalyDetector
            if cont_detector is not None:
                kin_df_for_anomaly = kin_df.copy() if not kin_df.empty else kin_df
                if not kin_df_for_anomaly.empty and "timestamp_abs" not in kin_df_for_anomaly.columns:
                    if "timestamp_abs" in extracted_features_df.columns:
                        kin_df_for_anomaly["timestamp_abs"] = extracted_features_df["timestamp_abs"].reindex(kin_df_for_anomaly.index)
                continuous_anomaly_report_with_kin = cont_detector.detect(
                    extracted_features_df,
                    kin_df=kin_df_for_anomaly,
                    kinematic_reference_profiles=kinematic_profiles,
                )
                continuous_anomaly_report = continuous_anomaly_report_with_kin
                save_json(continuous_anomaly_report, io.results_dir / "continuous_anomaly_report.json")
                session_logger.info(
                    "Continuous anomaly detection (with kinematic features): %d periods flagged.",
                    continuous_anomaly_report["summary"]["n_anomalous_periods"],
                )
                try:
                    _cont_plot_path = io.results_dir / "continuous_anomaly_timeline.png"
                    visualizer.plot_continuous_anomaly_timeline(
                        continuous_anomaly_report,
                        _cont_plot_path,
                        session_label=session_label,
                    )
                    session_logger.info("Continuous anomaly timeline saved: %s", _cont_plot_path)
                except Exception as _cplot_exc:
                    session_logger.warning("Could not save continuous anomaly timeline: %s", _cplot_exc)
            else:
                session_logger.debug(
                    "Skipping kinematic-enhanced anomaly detection: cont_detector was not initialized."
                )
        except Exception as kin_exc:
            session_logger.warning("Kinematic-enhanced anomaly detection failed: %s", kin_exc)
    except Exception as exc:
        session_logger.warning("Kinematic feature extraction failed: %s", exc)
        if cont_detector is not None:
            try:
                _fallback_report = cont_detector.detect(
                    extracted_features_df,
                    kin_df=None,
                    kinematic_reference_profiles=None,
                )
                save_json(_fallback_report, io.results_dir / "continuous_anomaly_report.json")
                session_logger.info(
                    "Continuous anomaly detection (fallback, no kinematics): "
                    "%d periods flagged.",
                    _fallback_report["summary"]["n_anomalous_periods"],
                )
                if visualizer is not None:
                    try:
                        visualizer.plot_continuous_anomaly_timeline(
                            _fallback_report,
                            io.results_dir / "continuous_anomaly_timeline.png",
                            session_label=session_label,
                        )
                    except Exception:
                        pass
            except Exception as _fb_exc:
                session_logger.warning("Fallback continuous anomaly detection failed: %s", _fb_exc)
    _sp("Kinematic analysis", 30)

    try:
        repetition_metrics_df = metrics_computer.compute_repetition_metrics(extracted_features_df)
        if len(repetition_metrics_df) > 0:
            reps_to_save = repetition_metrics_df[repetition_metrics_df["repetition"] != 0].copy()
            try:
                io.save_dataframe(reps_to_save, io.get_repetition_metrics_path())
            except Exception as _s: session_logger.warning("Could not save repetition metrics: %s", _s)
    except Exception as _rep_exc:
        session_logger.warning("Repetition metrics computation failed — using empty DataFrame: %s", _rep_exc)
        repetition_metrics_df = pd.DataFrame()

    try:
        task_metrics_df = metrics_computer.compute_task_metrics(repetition_metrics_df)
        if len(task_metrics_df) > 0:
            try:
                io.save_dataframe(task_metrics_df, io.get_task_metrics_path())
            except Exception as _s: session_logger.warning("Could not save task metrics: %s", _s)
    except Exception as _task_exc:
        session_logger.warning("Task metrics computation failed — using empty DataFrame: %s", _task_exc)
        task_metrics_df = pd.DataFrame()

    try:
        session_metrics = metrics_computer.compute_session_metrics(
            task_metrics_df, repetition_metrics_df
        )
        try:
            save_json(session_metrics, io.get_session_metrics_path())
        except Exception as _s: session_logger.warning("Could not save session metrics: %s", _s)
    except Exception as _sess_exc:
        session_logger.warning("Session metrics computation failed — using empty dict: %s", _sess_exc)
        session_metrics = {}
    _sp("Computing metrics", 45)

    profile_path = io.get_task_profile_path()
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    task_profile = load_task_profile(profile_path, subject_id)
    if task_profile is None:
        task_profile = TaskProfile(subject_id)

    if is_baseline_session and len(repetition_metrics_df) > 0:
        try:
            task_profile.update_from_session(
                io.session_id, repetition_metrics_df,
                features_df=extracted_features_df,
                task_metrics_df=task_metrics_df,
            )
            task_profile.save(profile_path)
        except Exception as _prof_exc:
            session_logger.warning("Task profile update failed: %s", _prof_exc)

    frame_data_list = features_df.to_dict("records")
    task_group, task_id = resolve_dominant_task(frame_data_list)
    task_profile_ref = None
    if task_profile.is_loaded():
        task_profile_ref = task_profile.get_task_reference(task_group, task_id)

    anomaly_results: Dict[str, Any] = {"summary": {"n_samples": 0, "n_anomalies": 0}}
    dtw_results: Dict[str, Any] = {}
    if len(repetition_metrics_df) > 0:
        has_multi_task = (
            "task_group" in repetition_metrics_df.columns
            and "task_id" in repetition_metrics_df.columns
        )
        if has_multi_task:
            task_keys = (
                repetition_metrics_df
                .groupby(["task_group", "task_id"])
                .size()
                .index.tolist()
            )
        else:
            task_keys = [(task_group, task_id)]

        per_task_results = []
        for tg, tid in task_keys:
            tid = int(tid)
            if has_multi_task:
                mask = (
                    (repetition_metrics_df["task_group"] == tg)
                    & (repetition_metrics_df["task_id"] == tid)
                )
                task_df = repetition_metrics_df[mask].copy()
            else:
                task_df = repetition_metrics_df.copy()
            if len(task_df) == 0:
                continue
            det = create_anomaly_detector(decision_rules_config, tasks_config)
            det.task_group = tg
            det.task_id = int(tid)
            det._task_config = det._resolve_task_config(tg, int(tid))

            _fit_task_name: Optional[str] = None
            if "task_name" in task_df.columns:
                _fn = task_df["task_name"].dropna().unique()
                _fn = [n for n in _fn if n and str(n) != "(no task selected)"]
                if _fn:
                    _fit_task_name = str(_fn[0])

            _fitted = False
            if len(reference_metrics_df) > 0:
                _DISORDER_CANON_PP = {
                    10: 1, 11: 3, 12: 3, 13: 5, 14: 5, 15: 7, 16: 8, 17: 9,
                }
                _ref_tg = tg
                _ref_tid = _DISORDER_CANON_PP.get(int(tid), int(tid)) if tg == "A" else int(tid)
                ref_task_df = reference_metrics_df.copy()
                if has_multi_task and "task_group" in ref_task_df.columns:
                    ref_mask = (
                        (ref_task_df["task_group"] == _ref_tg)
                        & (ref_task_df["task_id"].astype(int) == _ref_tid)
                    )
                    ref_task_df = ref_task_df[ref_mask]
                if len(ref_task_df) >= 2:
                    det.fit(ref_task_df, task_group=_ref_tg, task_id=_ref_tid)
                    _fitted = True
                elif len(ref_task_df) == 1:
                    if len(task_df) >= 2:
                        session_logger.warning(
                            "Only 1 reference sample for task (%s, %d); fitting on current session instead.",
                            tg, tid,
                        )
                        det.fit(task_df, task_group=tg, task_id=int(tid))
                        _fitted = True

            if not _fitted and task_profile.is_loaded():
                _prof_stats = task_profile.get_task_feature_stats(
                    tg, tid, task_name=_fit_task_name
                )
                if _prof_stats:
                    _task_ref = task_profile.get_task_reference(
                        tg, tid, task_name=_fit_task_name
                    )
                    if _task_ref and _task_ref.get("n_sessions", 0) >= 1:
                        det.set_task_feature_weights(_task_ref)
                        if _task_ref.get("_is_mapped_reference"):
                            session_logger.info(
                                "Task %s_%s → reference %s_%s via task-coupling map (%s)",
                                tg, tid,
                                _task_ref.get("_ref_task_group"),
                                _task_ref.get("_ref_task_id"),
                                _fit_task_name or "?",
                            )
                    _prof_ref_df = task_profile.get_reference_metrics_df(
                        tg, tid, task_name=_fit_task_name
                    )
                    det.fit_from_task_profile(
                        _prof_stats, _prof_ref_df if len(_prof_ref_df) > 0 else None
                    )
                    _fitted = True

            if not _fitted:
                if len(task_df) >= 2:
                    det.fit(task_df, task_group=tg, task_id=int(tid))
                else:
                    session_logger.warning(
                        "Insufficient samples for anomaly fitting for task (%s, %d): "
                        "ref=0, task=%d — skipping.",
                        tg, tid, len(task_df),
                    )
                    per_task_results.append({
                        "anomaly_scores": [],
                        "is_anomaly": [],
                        "summary": {"n_samples": 0, "n_anomalies": 0},
                    })
                    continue
            
            kin_summary_for_task = kinematic_summaries.get(tg)
            _task_result = det.detect_anomalies(
                task_df,
                kin_summary=kin_summary_for_task,
                dtw_results=dtw_results if dtw_results else None,
            )
            _task_display_name = (
                tasks_config.get("task_groups", {})
                .get(tg, {})
                .get("tasks", {})
                .get(int(tid), {})
                .get("display_name", f"{tg}{tid}")
            )
            _n_reps = len(_task_result.get("anomaly_scores", []))
            _task_result["task_names"] = [f"{tg}: {_task_display_name}"] * _n_reps
            per_task_results.append(_task_result)

        if per_task_results:
            if len(per_task_results) == 1:
                anomaly_results = per_task_results[0]
            elif len(per_task_results) > 1:
                _list_fields = [
                    "anomaly_scores", "is_anomaly", "deviation_score",
                    "score_confidence", "anomaly_type", "contributing_features",
                    "mahalanobis_score", "centroid_score", "within_session_score",
                    "method_votes", "weighted_votes", "method_sigmoid_scores",
                    "method_weighted_components", "deviation_ci_lower",
                    "deviation_ci_upper", "mahalanobis_ci_lower", "mahalanobis_ci_upper",
                    "repetitions", "task_groups", "task_ids", "task_names", "deviations",
                ]
                merged: Dict[str, Any] = {f: [] for f in _list_fields}
                merged_n_samples = 0
                merged_n_anomalies = 0
                merged_feat_devs: Dict[str, Any] = {}
                for r in per_task_results:
                    for f in _list_fields:
                        merged[f].extend(r.get(f, []))
                    summary_r = r.get("summary", {})
                    merged_n_samples += int(summary_r.get("n_samples", 0))
                    merged_n_anomalies += int(summary_r.get("n_anomalies", 0))
                    for feat, fdev in r.get("feature_deviations", {}).items():
                        if feat not in merged_feat_devs:
                            merged_feat_devs[feat] = fdev
                        else:
                            for k, v in fdev.items():
                                if isinstance(v, (int, float)):
                                    merged_feat_devs[feat][k] = max(
                                        merged_feat_devs[feat].get(k, 0.0), v
                                    )
                merged["feature_deviations"] = merged_feat_devs
                merged["summary"] = {
                    "n_samples": merged_n_samples,
                    "n_anomalies": merged_n_anomalies,
                    "anomaly_rate": (
                        merged_n_anomalies / merged_n_samples
                        if merged_n_samples > 0 else 0.0
                    ),
                }
                merged["per_task_results"] = per_task_results
                for fld in ("model_type", "n_reference", "n_pca_components",
                            "pca_explained_variance", "effective_threshold", "ml_metadata"):
                    for r in per_task_results:
                        if fld in r:
                            merged[fld] = r[fld]
                            break
                anomaly_results = merged

    save_json(anomaly_results, io.get_anomaly_results_path())
    _sp("Anomaly detection", 60)

    articulation_scores: Dict[str, Any] = {}
    has_speech = False
    if "task_group" in repetition_metrics_df.columns:
        has_speech = repetition_metrics_df["task_group"].isin(["B", "C"]).any()
    if has_speech and len(repetition_metrics_df) > 0 and articulation_scorer is not None:
        articulation_scores = articulation_scorer.compute_scores(
            repetition_metrics_df, extracted_features_df,
            reference_articulation=reference_articulation if not is_baseline_session else None,
        )
        save_json(articulation_scores, io.results_dir / "articulation_scores.json")
        for _artic_key, _artic_val in articulation_scores.items():
            if isinstance(_artic_val, (int, float)):
                session_metrics[_artic_key] = float(_artic_val)

    all_cross_results: Dict[str, Any] = {}
    if (
        not is_baseline_session
        and task_profile.is_loaded()
        and "task_group" in repetition_metrics_df.columns
        and (repetition_metrics_df["task_group"] == "A").any()
    ):
        all_cross_results = {}
        all_sims: List[float] = []
        all_sub_scores: List[float] = []
        n_substitutions = 0
        n_evaluated = 0

        for _atask_id in sorted(
            repetition_metrics_df.loc[
                repetition_metrics_df["task_group"] == "A", "task_id"
            ].dropna().unique()
        ):
            _atask_id = int(_atask_id)
            _amask = (
                (repetition_metrics_df["task_group"] == "A")
                & (repetition_metrics_df["task_id"] == _atask_id)
            )
            _atask_df = repetition_metrics_df[_amask].copy()
            if len(_atask_df) == 0:
                continue
            _atask_name: Optional[str] = None
            if "task_name" in _atask_df.columns:
                _an = _atask_df["task_name"].dropna().unique()
                _an = [n for n in _an if n and str(n) != "(no task selected)"]
                if _an:
                    _atask_name = str(_an[0])
            _cross_expected_id = _atask_id
            _has_bucc_component = _profile_up in ("P2_BUCCOFACIAL", "MIXED_B")
            if _has_bucc_component and _atask_id in _BUCCOFACIAL_EXPECTED_REF:
                _cross_expected_id = _BUCCOFACIAL_EXPECTED_REF[_atask_id]
            try:
                _matching = task_profile.compute_cross_task_matching(
                    _atask_df, "A", _cross_expected_id, task_name=_atask_name
                )
            except Exception as _cm_exc:
                session_logger.debug(
                    "Cross-task matching failed for A/%d: %s", _atask_id, _cm_exc
                )
                _matching = {}
            if _matching:
                all_cross_results[f"A_{_atask_id}"] = _matching
                if 1 <= _atask_id <= 9:
                    all_sims.append(_matching.get("task_profile_similarity", 1.0))
                    all_sub_scores.append(_matching.get("mean_substitution_score", 0.0))
                    n_substitutions += _matching.get("n_substitutions", 0)
                    n_evaluated += _matching.get("n_repetitions_evaluated", 0)

        if all_cross_results:
            save_json(all_cross_results, io.results_dir / "cross_task_matching.json")
        if all_sims:
            session_metrics["task_profile_similarity"] = float(np.mean(all_sims))
            session_metrics["mean_substitution_score"] = float(np.mean(all_sub_scores))
            session_metrics["substitution_rate"] = (
                n_substitutions / n_evaluated if n_evaluated > 0 else 0.0
            )
            session_metrics["n_a_reps_evaluated"] = n_evaluated
            session_logger.info(
                "Cross-task matching: substitution_rate=%.2f, "
                "mean_profile_similarity=%.3f (%d repetitions assessed)",
                session_metrics["substitution_rate"],
                session_metrics["task_profile_similarity"],
                n_evaluated,
            )
            try:
                save_json(session_metrics, io.get_session_metrics_path())
            except Exception as _s:
                session_logger.warning("Could not resave session metrics after cross-task matching: %s", _s)

    if (
        task_profile.is_loaded()
        and "task_group" in extracted_features_df.columns
    ):
        _dtw_feats_for_analysis = extracted_features_df
        _dtw_bare = create_anomaly_detector(decision_rules_config, tasks_config)
        for _dtw_tg, _dtw_tid in (task_keys if has_multi_task else [(task_group, task_id)]):
            _dtw_tid = int(_dtw_tid)
            if _dtw_tg not in ("B", "C"):
                continue
            _dtw_task_name: Optional[str] = None
            if has_multi_task and "task_name" in repetition_metrics_df.columns:
                _dmask = (
                    (repetition_metrics_df["task_group"] == _dtw_tg)
                    & (repetition_metrics_df["task_id"] == _dtw_tid)
                )
                _dns = repetition_metrics_df.loc[_dmask, "task_name"].dropna().unique()
                _dns = [n for n in _dns if n and str(n) != "(no task selected)"]
                if _dns:
                    _dtw_task_name = str(_dns[0])
            _dtw_ref = task_profile.get_task_reference(_dtw_tg, _dtw_tid, task_name=_dtw_task_name)
            if _dtw_ref is None:
                continue
            _dtw_patterns = _dtw_ref.get("activation_pattern", {})
            _dtw_feature = "mean_activation"
            if (
                _dtw_feature not in _dtw_patterns
                or "curves" not in _dtw_patterns.get(_dtw_feature, {})
            ):
                _found = False
                for _f in _dtw_patterns:
                    if "curves" in _dtw_patterns[_f] and len(_dtw_patterns[_f]["curves"]) >= 2:
                        _dtw_feature = _f
                        _found = True
                        break
                if not _found:
                    continue
            _dtw_ref_curves = [
                np.array(c) for c in _dtw_patterns[_dtw_feature]["curves"]
            ]
            if len(_dtw_ref_curves) < 2:
                continue

            _dtw_mask = (
                (_dtw_feats_for_analysis["task_group"] == _dtw_tg)
                & (_dtw_feats_for_analysis["task_id"] == _dtw_tid)
            ) if has_multi_task else pd.Series(True, index=_dtw_feats_for_analysis.index)
            _dtw_feat_df = _dtw_feats_for_analysis[_dtw_mask]
            if len(_dtw_feat_df) == 0 or _dtw_feature not in _dtw_feat_df.columns:
                continue

            _dtw_n_bins = len(_dtw_ref_curves[0])
            _dtw_key = f"{_dtw_tg}_{_dtw_tid}"
            rep_dtw_out: List[Dict[str, Any]] = []
            _rep_col = "repetition" if "repetition" in _dtw_feat_df.columns else None
            _reps = (
                sorted(r for r in _dtw_feat_df[_rep_col].dropna().unique() if r != 0)
                if _rep_col else [1]
            )
            for _rep in _reps:
                _rep_df = _dtw_feat_df[_dtw_feat_df[_rep_col] == _rep] if _rep_col else _dtw_feat_df
                if len(_rep_df) < 5:
                    rep_dtw_out.append(
                        {"repetition": int(_rep), "mean_dtw": 0.0, "min_dtw": 0.0, "is_shape_anomaly": False}
                    )
                    continue
                _vals = _rep_df[_dtw_feature].fillna(0.0).to_numpy()
                _t_norm = np.linspace(0, 1, len(_vals))
                _bins = np.linspace(0, 1, _dtw_n_bins)
                _test_binned = np.interp(_bins, _t_norm, _vals)
                try:
                    _d = _dtw_bare.compute_dtw_pattern_deviation(_test_binned, _dtw_ref_curves)
                except Exception as _exc:
                    session_logger.debug("DTW rep %d task %s/%d: %s", _rep, _dtw_tg, _dtw_tid, _exc)
                    _d = {"mean_dtw": 0.0, "min_dtw": 0.0, "is_shape_anomaly": False}
                _d["repetition"] = int(_rep)
                rep_dtw_out.append(_d)

            if rep_dtw_out:
                dtw_results[_dtw_key] = {
                    "feature": _dtw_feature,
                    "repetitions": rep_dtw_out,
                    "mean_dtw_task": float(np.mean([r["mean_dtw"] for r in rep_dtw_out])),
                    "n_shape_anomalies": int(sum(1 for r in rep_dtw_out if r["is_shape_anomaly"])),
                }

        if dtw_results:
            save_json(dtw_results, io.results_dir / "dtw_pattern_analysis.json")
            n_shape_anom = sum(
                d.get("n_shape_anomalies", 0) for d in dtw_results.values()
            )
            session_metrics["dtw_shape_anomaly_count"] = n_shape_anom
            session_logger.info(
                "DTW pattern analysis: %d tasks assessed, %d shape anomalies",
                len(dtw_results), n_shape_anom,
            )

            _c_dtw_means: List[float] = []
            _c_n_high_dtw = 0
            _c_rep_stds: List[float] = []
            for _dtw_k, _dtw_td in dtw_results.items():
                if not _dtw_k.startswith("C_"):
                    continue
                _dtw_m = float(_dtw_td.get("mean_dtw_task", 0.0))
                _c_dtw_means.append(_dtw_m)
                if _dtw_m > 0.08:
                    _c_n_high_dtw += 1
                _dtw_reps = _dtw_td.get("repetitions", [])
                _rep_dtws = [float(r.get("mean_dtw", 0.0)) for r in _dtw_reps]
                if len(_rep_dtws) > 1:
                    _rm = sum(_rep_dtws) / len(_rep_dtws)
                    _rs = (sum((x - _rm) ** 2 for x in _rep_dtws) / len(_rep_dtws)) ** 0.5
                    _c_rep_stds.append(_rs)
            if _c_dtw_means:
                _c_n_high_relative = 0
                try:
                    if used_reference_ids and not is_baseline_session:
                        _ref_dtw_path = (
                            io.data_dir / "results" / study_mode / subject_id
                            / used_reference_ids[0] / "dtw_pattern_analysis.json"
                        )
                        if _ref_dtw_path.exists():
                            _ref_dtw_data = load_json(_ref_dtw_path)
                            for _dtw_k, _dtw_td in dtw_results.items():
                                if not _dtw_k.startswith("C_"):
                                    continue
                                _dtw_reps_vals = sorted(
                                    float(r.get("mean_dtw", 0.0))
                                    for r in _dtw_td.get("repetitions", [])
                                )
                                if _dtw_reps_vals:
                                    _n = len(_dtw_reps_vals)
                                    _test_m = (
                                        _dtw_reps_vals[_n // 2] if _n % 2 == 1
                                        else (_dtw_reps_vals[_n // 2 - 1] + _dtw_reps_vals[_n // 2]) / 2
                                    )
                                else:
                                    _test_m = float(_dtw_td.get("mean_dtw_task", 0.0))
                                _ref_m  = float(_ref_dtw_data.get(_dtw_k, {}).get("mean_dtw_task", 0.0))
                                if _test_m > max(_ref_m, 0.02) * 3.0:
                                    _c_n_high_relative += 1
                except Exception:
                    pass
                anomaly_results["c_dtw_summary"] = {
                    "c_mean_dtw":         sum(_c_dtw_means) / len(_c_dtw_means),
                    "c_n_high_dtw":       _c_n_high_dtw,
                    "c_n_tasks":          len(_c_dtw_means),
                    "c_mean_rep_std": (
                        sum(_c_rep_stds) / len(_c_rep_stds) if _c_rep_stds else 0.0
                    ),
                    "c_n_high_relative":  _c_n_high_relative,
                    "max_c_task_dtw":     max(_c_dtw_means),
                }

            _b4_dtw_data = dtw_results.get("B_4")
            _b_simple_keys = [k for k in dtw_results if k.startswith("B_") and k != "B_4"]
            if _b4_dtw_data and _b_simple_keys:
                _b4_reps = _b4_dtw_data.get("repetitions", [])
                _b4_rep_dtws = [float(r.get("mean_dtw", 0.0)) for r in _b4_reps]
                _b4_mean = float(_b4_dtw_data.get("mean_dtw_task", 0.0))
                _b4_n_anom = int(_b4_dtw_data.get("n_shape_anomalies", 0))
                _b4_n_reps = len(_b4_reps)

                _b4_peak_ratio = 1.0
                _b4_temporal_type = "none"
                for _br in _b4_reps:
                    _b4_peak_ratio = float(_br.get("peak_time_ratio", 1.0))
                    _b4_temporal_type = str(_br.get("temporal_type", "none"))
                    break

                _b_simple_means: List[float] = []
                for _bk in _b_simple_keys:
                    _bm = float(dtw_results[_bk].get("mean_dtw_task", 0.0))
                    _b_simple_means.append(_bm)
                _b_simple_mean = sum(_b_simple_means) / len(_b_simple_means) if _b_simple_means else 0.0

                _b4_dtw_vs_ref: Optional[float] = None
                try:
                    if used_reference_ids and not is_baseline_session:
                        _b4_ref_dtw_path = (
                            io.data_dir / "results" / study_mode / subject_id
                            / used_reference_ids[0] / "dtw_pattern_analysis.json"
                        )
                        if _b4_ref_dtw_path.exists():
                            _b4_ref_dtw_data = load_json(_b4_ref_dtw_path)
                            _b4_ref_mean = float(
                                _b4_ref_dtw_data.get("B_4", {}).get("mean_dtw_task", 0.0)
                            )
                            if _b4_ref_mean > 0.005:
                                _b4_dtw_vs_ref = _b4_mean / _b4_ref_mean
                except Exception:
                    pass

                _b4_rep_dtw_cv: Optional[float] = None
                if len(_b4_rep_dtws) >= 2:
                    _b4_rep_mean = sum(_b4_rep_dtws) / len(_b4_rep_dtws)
                    if _b4_rep_mean > 0.01:
                        _b4_rep_std = (
                            sum((x - _b4_rep_mean) ** 2 for x in _b4_rep_dtws) / len(_b4_rep_dtws)
                        ) ** 0.5
                        _b4_rep_dtw_cv = _b4_rep_std / _b4_rep_mean

                anomaly_results["b4_dtw_summary"] = {
                    "b4_mean_dtw":         _b4_mean,
                    "b4_n_shape_anom":     _b4_n_anom,
                    "b4_n_reps":           _b4_n_reps,
                    "b_simple_mean_dtw":   _b_simple_mean,
                    "b4_vs_simple_ratio":  _b4_mean / max(_b_simple_mean, 0.001),
                    "b4_peak_time_ratio":  _b4_peak_ratio,
                    "b4_temporal_type":    _b4_temporal_type,
                    "b4_dtw_vs_ref":       _b4_dtw_vs_ref,
                    "b4_rep_dtw_cv":       _b4_rep_dtw_cv,
                }

            if "c_dtw_summary" in anomaly_results or "b4_dtw_summary" in anomaly_results:
                try:
                    save_json(anomaly_results, io.get_anomaly_results_path())
                except Exception:
                    pass

    ref_stats = reference_baseline_stats
    if task_profile_ref and not used_reference_ids:
        ref_stats = task_profile_ref.get("per_feature_stats", ref_stats)

    reference_asymmetry_stats: Optional[Dict[str, float]] = None
    if reference_metrics_df is not None and len(reference_metrics_df) > 0:
        _asym_col = next(
            (c for c in ("mean_asymmetry_ratio", "asymmetry_ratio_mean",
                         "mean_asymmetry", "overall_mean_asymmetry")
             if c in reference_metrics_df.columns),
            None,
        )
        if _asym_col:
            _ref_a = reference_metrics_df
            if "task_group" in reference_metrics_df.columns:
                _ref_a = reference_metrics_df[
                    reference_metrics_df["task_group"].astype(str) == "A"
                ]
            _asym_vals = _ref_a[_asym_col].dropna().values
            if len(_asym_vals) >= 2:
                reference_asymmetry_stats = {
                    "mean": float(np.mean(_asym_vals)),
                    "std":  float(np.std(_asym_vals, ddof=1)),
                    "n":    float(len(_asym_vals)),
                }
            elif len(_asym_vals) == 1:
                reference_asymmetry_stats = {
                    "mean": float(_asym_vals[0]),
                    "std":  0.05,
                    "n":    1.0,
                }

    reference_head_yaw: Optional[float] = None
    if reference_metrics_df is not None and len(reference_metrics_df) > 0:
        if "head_yaw_mean" in reference_metrics_df.columns:
            reference_head_yaw = float(reference_metrics_df["head_yaw_mean"].mean())

    screening_results: Dict[str, Any] = {
        "n_indications": 0, "indication_types": [], "confidence": {},
        "indications": [], "disorder_results": {},
    }
    try:
        _is_ors = (
            "ors" in session_label.lower()
            or "rotated" in session_label.lower()
            or "ors" in io.session_id.lower()
        )
        decision_support.set_session_context(
            is_baseline=is_baseline_session,
            has_reference=len(used_reference_ids) > 0,
            reference_stats=ref_stats,
            task_group=task_group,
            task_id=task_id,
            reference_articulation=reference_articulation,
            reference_asymmetry_stats=reference_asymmetry_stats,
            is_ors_session=_is_ors,
            reference_head_yaw=reference_head_yaw,
        )
        screening_results = decision_support.evaluate(
            session_metrics, task_metrics_df, repetition_metrics_df, anomaly_results
        )
        try:
            save_json(screening_results, io.get_screening_results_path())
        except Exception as _s: session_logger.warning("Could not save screening results: %s", _s)
    except Exception as _scr_exc:
        session_logger.warning("Screening/decision support failed — continuing with empty results: %s", _scr_exc)

    confidence_summary = {
        "confidence": screening_results.get("confidence", {}),
        "n_indications": screening_results.get("n_indications", 0),
        "indication_types": screening_results.get("indication_types", []),
        "is_baseline_session": is_baseline_session,
        "reference_sessions": used_reference_ids,
    }
    try:
        save_json(confidence_summary, io.get_confidence_summary_path())
    except Exception as _s: session_logger.warning("Could not save confidence summary: %s", _s)
    _sp("Screening evaluation", 75)
    session_logger.info("Step: visualization phase starting")

    viz_dir = io.results_dir / "visualizations"
    tables_dir = io.results_dir / "tables"

    neutral_baseline_stats = getattr(baseline_constructor, "baseline_stats", None)
    has_features = len(extracted_features_df) > 0
    has_reps = len(repetition_metrics_df) > 0
    has_anomaly_scores = bool(
        anomaly_results and anomaly_results.get("anomaly_scores")
    )
    _sp("Generating plots", 80)

    _overall_det_rate = session_metrics.get("overall_detection_rate", None)
    if _overall_det_rate is not None and _overall_det_rate < 0.1:
        session_logger.warning(
            "DETECTION WARNING: overall face detection rate is %.1f%% "
            "(MediaPipe could not track the face in most frames). "
            "This is common when the face is sideways or rotated >45°. "
            "All activation/blendshape metrics will reflect zero movement. "
            "Kinematic features will also be unavailable.",
            _overall_det_rate * 100,
        )

    _act_col = next(
        (c for c in ("mean_activation", "max_activation", "activation_range")
         if c in extracted_features_df.columns),
        None,
    )
    _tp_all = task_profile.tasks if task_profile.is_loaded() else None
    if not skip_kin_figures and has_features and _act_col:
        _overlay_kwargs = dict(
            baseline_stats=neutral_baseline_stats,
            reference_baseline_stats=reference_baseline_stats if not is_baseline_session else None,
            task_profile_ref=task_profile_ref,
            all_task_profiles=_tp_all,
        )
        session_logger.info("Plotting: repetition overlay")
        try:
            visualizer.plot_repetition_overlay(
                extracted_features_df, _act_col,
                viz_dir / "activation_overlay",
                title="Activation by Repetition (Overlayed)",
                **_overlay_kwargs,
            )
        except Exception as _exc:
            session_logger.warning("plot_repetition_overlay failed: %s", _exc)
        _sp("Plotting repetition overlay", 81)
        session_logger.info("Plotting: activation per repetition")
        try:
            visualizer.plot_activation_per_repetition(
                extracted_features_df, _act_col,
                viz_dir / "activation_per_repetition",
                title="Activation per Repetition",
                **_overlay_kwargs,
            )
        except Exception as _exc:
            session_logger.warning("plot_activation_per_repetition failed: %s", _exc)
        _sp("Plotting activation per repetition", 82)
        activation_metrics = [c for c in extracted_features_df.columns if "activation" in c][:4]
        if activation_metrics:
            session_logger.info("Plotting: activation overlay by metric")
            try:
                visualizer.plot_activation_overlay_by_metric(
                    extracted_features_df, activation_metrics,
                    viz_dir / "activation_overlay_by_metric",
                    title="Activation Overlay by Metric",
                    **_overlay_kwargs,
                )
            except Exception as _exc:
                session_logger.warning("plot_activation_overlay_by_metric failed: %s", _exc)
        _sp("Plotting activation overlay by metric", 84)

    if not skip_kin_figures and has_reps:
        session_logger.info("Plotting: metrics summary")
        try:
            visualizer.plot_metrics_summary(
                repetition_metrics_df,
                viz_dir / "metrics_summary",
                title="Repetition Metrics Summary",
                baseline_stats=neutral_baseline_stats,
                reference_baseline_stats=reference_baseline_stats,
                task_profile_ref=task_profile_ref,
            )
        except Exception as _exc:
            session_logger.warning("plot_metrics_summary failed: %s", _exc)
    _sp("Plotting metrics summary", 85)

    if not skip_all_figures and screening_results:
        session_logger.info("Plotting: screening summary")
        try:
            visualizer.plot_screening_summary(
                screening_results,
                viz_dir / "screening_summary",
                anomaly_results=anomaly_results,
                title="Clinical Screening Report",
            )
        except Exception as _exc:
            session_logger.warning("plot_screening_summary failed: %s", _exc)
        try:
            visualizer.plot_disorder_evidence(
                screening_results,
                viz_dir / "disorder_evidence",
                title="Disorder Evidence Profile",
            )
        except Exception as _exc:
            session_logger.warning("plot_disorder_evidence failed: %s", _exc)

    if not skip_all_figures and screening_results:
        _brain_title = "Neural Substrates"
        if session_profile:
            _brain_title = f"Neural Substrates \u2014 {session_profile.replace('_', ' ').title()}"
        try:
            visualizer.plot_brain_activation_map(
                screening_results,
                viz_dir / "brain_activation_map",
                title=_brain_title,
            )
        except Exception as _brain_exc:
            session_logger.warning("Brain activation map failed: %s", _brain_exc)

    if not skip_all_figures and has_anomaly_scores:
        session_logger.info("Plotting: anomaly results")
        try:
            visualizer.plot_anomaly_results(
                anomaly_results,
                viz_dir / "anomaly_results",
                title="Anomaly Detection Results",
                baseline_stats=neutral_baseline_stats,
            )
        except Exception as _exc:
            session_logger.warning("plot_anomaly_results failed: %s", _exc)

    if not skip_all_figures and has_anomaly_scores and screening_results:
        session_logger.info("Plotting: anomaly→indication flow")
        try:
            visualizer.plot_anomaly_indication_flow(
                anomaly_results,
                screening_results,
                viz_dir / "anomaly_indication_flow",
                title=f"Anomaly → Indication Flow — {session_label}",
            )
        except Exception as _exc:
            session_logger.warning("plot_anomaly_indication_flow failed: %s", _exc)

    if not skip_all_figures and all_cross_results:
        try:
            visualizer.plot_cross_task_matching(
                all_cross_results,
                viz_dir / "cross_task_matching",
                title="Cross-Task Profile Matching (Group A)",
            )
        except Exception as _exc:
            session_logger.warning("Cross-task matching plot failed: %s", _exc)

    if not skip_kin_figures and dtw_results:
        try:
            visualizer.plot_dtw_pattern_analysis(
                dtw_results,
                viz_dir / "dtw_pattern_analysis",
                title="DTW Temporal Pattern Analysis",
            )
        except Exception as _exc:
            session_logger.warning("DTW plot failed: %s", _exc)
    _sp("Plotting screening results", 87)

    if not skip_kin_figures and has_anomaly_scores:
        try:
            _dr_threshold = decision_rules_config.get("anomaly", {}).get("composite_threshold", 0.45)
            visualizer.plot_deviation_scoring_summary(
                anomaly_results,
                viz_dir / "deviation_scoring_summary",
                title="Deviation Scoring Summary",
                threshold=_dr_threshold,
            )
        except Exception as _exc:
            session_logger.warning("Deviation scoring summary plot failed: %s", _exc)

    if not skip_kin_figures and neutral_baseline_stats:
        try:
            visualizer.plot_baseline_stability(
                neutral_baseline_stats,
                viz_dir / "baseline_stability",
                title="Baseline Blendshape Stability",
            )
        except Exception as _exc:
            session_logger.warning("plot_baseline_stability failed: %s", _exc)

    if not skip_all_figures and has_features:
        session_logger.info("Plotting: anatomical report")
        try:
            from .anatomy import generate_anatomical_report
            anat_feature_devs: Dict[str, Any] = {}
            if len(repetition_metrics_df) > 0 and reference_baseline_stats:
                num_cols = [
                    c for c in repetition_metrics_df.columns
                    if repetition_metrics_df[c].dtype in (float, np.float64)
                    and c in reference_baseline_stats
                ]
                for col in num_cols:
                    ref = reference_baseline_stats[col]
                    ref_mean = ref.get("mean", 0.0)
                    ref_std = max(ref.get("std", 1.0), 1e-6)
                    col_mean = float(repetition_metrics_df[col].mean())
                    anat_feature_devs[col] = {
                        "mean_deviation": abs((col_mean - ref_mean) / ref_std),
                        "value": col_mean,
                    }
            if not anat_feature_devs and len(repetition_metrics_df) > 0:
                num_cols = [c for c in repetition_metrics_df.select_dtypes(include="number").columns]
                for col in num_cols:
                    vals = repetition_metrics_df[col].dropna()
                    if len(vals) > 1:
                        anat_feature_devs[col] = {
                            "mean_deviation": float(vals.std() / max(abs(vals.mean()), 1e-6)),
                            "value": float(vals.mean()),
                        }
            if anat_feature_devs:
                _profile_display = session_profile.replace("_", " ").title() if session_profile else ""
                _anat_title = (
                    f"Anatomical Muscle Group Analysis \u2014 {_profile_display}"
                    if _profile_display else "Anatomical Muscle Group Analysis"
                )
                anat_report = generate_anatomical_report(anat_feature_devs)
                visualizer.plot_anatomical_report(
                    anat_report,
                    viz_dir / "anatomical_report",
                    title=_anat_title,
                )
                try:
                    from .anatomy import generate_3d_anatomical_visualization, MUSCLE_GROUP_MAP
                    muscle_scores: Dict[str, float] = {}
                    anomaly_flags_map: Dict[str, bool] = {}
                    for mg_key, mg_info in anat_report.get("muscle_groups", {}).items():
                        if isinstance(mg_info, dict):
                            dev = float(mg_info.get("mean_deviation",
                                        mg_info.get("normalized_activation", 0.0)))
                            muscle_scores[mg_key] = min(1.0, dev / 5.0)
                            anomaly_flags_map[mg_key] = int(mg_info.get("n_deviant", 0)) > 0
                    if muscle_scores:
                        ref_muscle_scores = None
                        if reference_baseline_stats:
                            try:
                                _ref_by_group: Dict[str, List[float]] = {}
                                for _mg_key, _mg_info in MUSCLE_GROUP_MAP.items():
                                    for _bs in _mg_info.get("blendshapes", []):
                                        for _stat_key in (_bs, f"{_bs}_mean"):
                                            _stat = reference_baseline_stats.get(_stat_key)
                                            if isinstance(_stat, dict):
                                                _val = _stat.get("mean", _stat.get("value"))
                                                if _val is not None:
                                                    _ref_by_group.setdefault(_mg_key, []).append(float(_val))
                                ref_muscle_scores = (
                                    {mg: float(np.mean(vals)) for mg, vals in _ref_by_group.items() if vals}
                                    or None
                                )
                            except Exception:
                                ref_muscle_scores = None
                        _indications = screening_results.get("indications", [])
                        _dec_label: Optional[str] = None
                        _dec_conf: Optional[float] = None
                        if _indications:
                            _top = _indications[0]
                            _dec_label = _top.get("description", _top.get("indication_type", ""))
                            _dec_conf = float(_top.get("confidence", 0.0)) if _top.get("confidence") is not None else None
                        elif screening_results.get("n_indications", 0) == 0:
                            _dec_label = (
                                f"Simulating: {_profile_display}"
                                if _profile_display else "Within Normal Limits"
                            )
                            _dec_conf = float(screening_results.get("confidence", {}).get("overall", 0.0)) or None
                        _anat3d_title = (
                            f"Facial Muscle Activation \u2014 {_profile_display}"
                            if _profile_display else "Facial Muscle Activation \u2014 Anatomical Map"
                        )
                        generate_3d_anatomical_visualization(
                            muscle_scores,
                            viz_dir / "anatomical_3d_activation.pdf",
                            title=_anat3d_title,
                            reference_activation_scores=ref_muscle_scores,
                            anomaly_flags=anomaly_flags_map if any(anomaly_flags_map.values()) else None,
                            decision_label=_dec_label,
                            decision_confidence=_dec_conf,
                        )
                except Exception as _3d_exc:
                    session_logger.warning("3D anatomy visualization failed: %s", _3d_exc)
        except Exception as _anat_exc:
            session_logger.warning("Anatomical report failed: %s", _anat_exc)

    if not skip_kin_figures and has_features:
        session_logger.info("Plotting: muscle group heatmaps")
        try:
            visualizer.plot_muscle_group_activation_heatmap(
                extracted_features_df,
                viz_dir / "muscle_group_heatmap",
            )
        except Exception as _mgh_exc:
            session_logger.warning("Muscle group activation heatmap failed: %s", _mgh_exc)
        try:
            visualizer.plot_muscle_group_temporal_heatmap(
                extracted_features_df,
                viz_dir / "muscle_group_temporal_heatmap",
            )
        except Exception as _mgth_exc:
            session_logger.warning("Muscle group temporal heatmap failed: %s", _mgth_exc)

    if not skip_kin_figures and has_features:
        session_logger.info("Plotting: asymmetry over time")
        try:
            visualizer.plot_asymmetry_over_time(
                extracted_features_df,
                viz_dir / "asymmetry_over_time",
                title="Facial Asymmetry Over Time",
                baseline_stats=neutral_baseline_stats,
                reference_baseline_stats=reference_baseline_stats if not is_baseline_session else None,
                all_task_profiles=_tp_all,
            )
        except Exception as _asym_exc:
            session_logger.warning("Asymmetry plot failed: %s", _asym_exc)

    if not skip_kin_figures and has_speech and articulation_scores:
        session_logger.info("Plotting: speech scores")
        try:
            visualizer.plot_articulation_profile(
                articulation_scores,
                viz_dir / "speech_scores",
                title="Speech Scores",
                reference_scores=reference_articulation,
            )
        except Exception as _art_exc:
            session_logger.warning("Speech scores plot failed: %s", _art_exc)

    if not skip_kin_figures and has_features and "task_group" in extracted_features_df.columns and (extracted_features_df["task_group"] == "A").any() and any(c.startswith("kin_a_") for c in extracted_features_df.columns):
        _kin_a_viz_dir = io.results_dir / "kinematic_profiles"
        session_logger.info("Plotting: Group A kinematic profiles")
        try:
            _kin_a_viz_dir.mkdir(parents=True, exist_ok=True)
            _tp_all_a = task_profile.tasks if task_profile.is_loaded() else None
            visualizer.plot_group_a_kinematics(
                extracted_features_df,
                _kin_a_viz_dir / f"kinematic_group_A_{session_label}.pdf",
                session_label=session_label,
                fps=fps_estimate,
                reference_profiles=kinematic_profiles_ref if not is_baseline_session else None,
                task_name_map=visualizer._build_task_name_map(repetition_metrics_df),
                is_reference_session=is_baseline_session,
                all_task_profiles=_tp_all_a if not is_baseline_session else None,
                task_profile_ref=task_profile_ref if not is_baseline_session else None,
            )
            session_logger.info("Generated Group A kinematic profile PDF")
        except Exception as _ga_plot_exc:
            session_logger.warning("Group A kinematic profile plot failed: %s", _ga_plot_exc)

    _sp("Plotting anatomy and articulation", 88)

    plot_paths: List[str] = []
    for p in sorted(viz_dir.glob("*.png")) if viz_dir.exists() else []:
        plot_paths.append(str(p))

    download_paths: List[str] = []
    for csv_path in sorted(io.processed_dir.glob("*.csv")) if io.processed_dir.exists() else []:
        download_paths.append(str(csv_path))
    if io.results_dir.exists():
        for json_path in sorted(io.results_dir.glob("*.json")):
            download_paths.append(str(json_path))

    if not skip_kin_figures and not kin_df.empty and "task_group" in extracted_features_df.columns:
        kin_viz_dir = io.results_dir / "kinematic_profiles"
        session_logger.info("Plotting: kinematic profiles PDF")
        try:
            kin_pdf_path = visualizer.plot_all_kinematic_tasks(
                kin_df=kin_df,
                features_df=extracted_features_df,
                output_dir=kin_viz_dir,
                session_label=session_label,
                task_groups=["B", "C"],
                reference_profiles=kinematic_profiles_ref if not is_baseline_session else None,
                fps=fps_estimate,
                task_name_map=visualizer._build_task_name_map(repetition_metrics_df),
                is_reference_session=is_baseline_session,
                ddk_summaries=kinematic_summaries.get("B"),
            )
            if kin_pdf_path:
                session_logger.info("Generated kinematic profiles PDF: %s", kin_pdf_path)
                download_paths.append(str(kin_pdf_path))
        except Exception as _kin_exc:
            session_logger.warning("Kinematic profiles PDF failed: %s", _kin_exc)
    else:
        kin_viz_dir = io.results_dir / "kinematic_profiles"

    if not skip_kin_figures and not kin_df.empty and "task_group" in extracted_features_df.columns and (extracted_features_df["task_group"] == "A").any():
        try:
            kin_viz_dir.mkdir(parents=True, exist_ok=True)
            group_a_pdfs = visualizer.plot_group_a_landmark_kinematics(
                kin_df=kin_df,
                features_df=extracted_features_df,
                output_dir=kin_viz_dir,
                session_label=session_label,
                reference_profiles=kinematic_profiles_ref if not is_baseline_session else None,
                fps=fps_estimate,
                task_name_map=visualizer._build_task_name_map(repetition_metrics_df),
                is_reference_session=is_baseline_session,
            )
            for _pdf in group_a_pdfs:
                session_logger.info("Generated Group A kinematic PDF: %s", _pdf)
                download_paths.append(str(_pdf))
        except Exception as _ga_exc:
            session_logger.warning("Group A landmark kinematic PDFs failed: %s", _ga_exc)

    pipeline_summary = {
        "session_id": io.session_id,
        "subject_id": subject_id,
        "session_label": session_label,
        "session_profile": session_profile,
        "study_mode": study_mode,
        "pipeline_version": get_pipeline_version(),
        "config_hash": io.config_hash,
        "timestamp": datetime.now().isoformat(),
        "n_frames": len(features_df),
        "session_metrics": session_metrics,
        "screening_summary": {
            "n_indications": screening_results.get("n_indications", 0),
            "indication_types": screening_results.get("indication_types", []),
            "confidence": screening_results.get("confidence", {}),
            "indications": screening_results.get("indications", []),
        },
        "anomaly_summary": anomaly_results.get("summary", {}),
        "reference_sessions_used": used_reference_ids,
        "is_baseline_session": is_baseline_session,
        "plot_paths": plot_paths,
        "download_paths": download_paths,
        "disorder_results": screening_results.get("disorder_results", {}),
        "output_paths": {
            "raw": str(io.raw_dir),
            "processed": str(io.processed_dir),
            "results": str(io.results_dir),
        },
    }
    save_json(pipeline_summary, io.results_dir / "pipeline_summary.json")
    _sp("Saving results", 95)

    return pipeline_summary


def _validate_inputs(
    video_paths: List[Path],
    session: "PrompterSession",
    logger: logging.Logger,
) -> None:
    """Validate that video files and the parsed session are consistent before
    the expensive MediaPipe pass begins.

    Checks performed:
    1. Every video file exists and is non-empty (size > 0 bytes).
    2. The session has at least one event (events_df is not empty).
    3. The maximum timestamp in events_df does not exceed the duration of the
       longest video by more than 30 seconds.  A larger gap suggests the wrong
       CSV was paired with the video.
    4. If session.participant_id is non-empty and differs from the stem of the
       first video filename (before the first underscore), logs a warning so
       the researcher can catch accidental mismatches.

    Raises ValueError with a descriptive message if checks 1–3 fail.
    Logs a warning for check 4 but does not raise.
    """
    for vp in video_paths:
        if not vp.exists():
            raise ValueError(f"Video file not found: {vp}")
        if vp.stat().st_size == 0:
            raise ValueError(f"Video file is empty: {vp}")

    if session.events_df.empty:
        raise ValueError(
            "Session has no events. The timestamps CSV may be malformed or empty."
        )

    if "timestamp_abs" not in session.events_df.columns:
        raise ValueError("Timestamps CSV missing 'timestamp_abs' column after parsing; check CSV headers and quoting.")
    if session.events_df["timestamp_abs"].isnull().any():
        raise ValueError(
            "Non-numeric timestamps detected in events CSV. Ensure fields are correctly quoted and 'time_from_start_s' contains numeric values."
        )
    earliest_ts = float(session.events_df["timestamp_abs"].min())
    if earliest_ts < -1.0:
        raise ValueError(f"Negative event timestamps found (earliest={earliest_ts}). Check CSV time units and offsets.")

    max_duration = 0.0
    for vp in video_paths:
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            cap.release()
            raise ValueError(f"Cannot open video file: {vp}")
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps > 0 and frame_count > 0:
            duration = frame_count / fps
            max_duration = max(max_duration, duration)

    if max_duration > 0 and not session.events_df.empty:
        max_event_ts = float(session.events_df["timestamp_abs"].max())
        if max_event_ts > max_duration + 30.0:
            raise ValueError(
                f"Maximum event timestamp ({max_event_ts:.1f}s) exceeds the longest "
                f"video duration ({max_duration:.1f}s) by more than 30 seconds. "
                f"Check that the correct timestamps CSV was paired with these videos."
            )
    else:
        raise ValueError("Unable to determine video durations from files; check that video files are readable (ffmpeg/codec support).")

    if session.participant_id and video_paths:
        first_stem = video_paths[0].stem
        vid_pid = first_stem.split("_")[0]
        csv_pid = session.participant_id.split(",")[0].strip()
        if vid_pid != csv_pid:
            logger.warning(
                "Participant ID mismatch: session CSV says '%s' but first video "
                "filename starts with '%s'. Check that the correct files were paired.",
                session.participant_id,
                vid_pid,
            )

    try:
        import mediapipe as _mp
    except Exception:
        raise ValueError(
            "mediapipe is not importable. Install the mediapipe package or ensure your Python environment is configured correctly."
        )
    try:
        from .utils import MODEL_PATH
        if not MODEL_PATH.exists():
            logger.warning(
                "FaceLandmarker model not found at %s — the pipeline will attempt to download it, which may take several minutes and network access.",
                MODEL_PATH,
            )
    except Exception:
        pass


def _load_raw_landmarks_csv(
    csv_path: Path,
    session: "PrompterSession",
    meta_path: Optional[Path],
) -> tuple:
    """Load a pre-extracted raw landmarks CSV and re-label its frames.

    Reads the CSV, estimates the frame rate, then re-assigns segment labels
    using the parsed session events so that task / neutral windows are
    correctly mapped before analysis.

    Returns (features_df, primary_fps, camera_offsets, annotated_video_tmps, landmark_video_tmps).
    """
    features_df = pd.read_csv(str(csv_path))
    features_df = features_df.drop(columns=["_landmarks_3d"], errors="ignore")

    primary_fps = 30.0
    if meta_path is not None:
        try:
            _meta_json = load_json(Path(meta_path))
            _granted = _meta_json.get("cameras", [{}])[0].get("grantedFrameRate")
            if _granted and float(_granted) > 0:
                primary_fps = float(_granted)
                logger.info("fps from recording metadata (raw CSV path): %.1f", primary_fps)
        except Exception:
            pass
    if primary_fps == 30.0 and "frame_index" in features_df.columns and "timestamp_abs" in features_df.columns:
        try:
            _max_frame = float(features_df["frame_index"].max())
            _max_time  = float(features_df["timestamp_abs"].max())
            if _max_time > 0 and _max_frame > 0:
                _derived = _max_frame / _max_time
                if 1.0 <= _derived <= 120.0:
                    primary_fps = _derived
                    logger.info("fps derived from CSV frame/time columns: %.2f", primary_fps)
        except Exception:
            pass

    if len(session.events_df) > 0:
        features_df = _reassign_segments_for_disorder(features_df, session.events_df)
        logger.info(
            "Raw landmarks CSV re-labelled with %d session events.", len(session.events_df)
        )

    return features_df, primary_fps, {}, [], []


def run_prompter_session(
    video_paths: List[Path],
    timestamps_path: Optional[Path],
    subject_id: str,
    session_label: str,
    study_mode: str,
    project_root: Path,
    meta_path: Optional[Path] = None,
    assembly_path: Optional[Path] = None,
    reference_session: Optional[List[str]] = None,
    progress_callback: Optional[Callable[[str, int], None]] = None,
    video_mode: str = "none",
) -> Dict[str, Any]:
    """Run the full analysis pipeline for a study-prompter session.

    Steps:
    1. Load PrompterSession via study_prompter_reader.load_prompter_session.
    2. Load features_config and tasks_config via IOManager.load_config.
    3. Instantiate MultiCameraProcessor with video_paths.
    4. Call align_cameras() if len(video_paths) > 1.
    5. Call process_all_frames(events_df, recording_start_offset_s) — returns
       frame_data_list and events_df.
    6. Build a blendshapes DataFrame and features DataFrame using the existing
       FeatureExtractor — same as the live-capture Pipeline does internally.
    7. If is_combined is False: call _run_single_profile_analysis(...).
    8. If is_combined is True: for each disorder in per_disorder_events,
       call _run_single_profile_analysis(...) with the disorder's filtered
       events and the shared features DataFrame.
    9. Return a summary dict with keys: participant_id, profile, session_date,
       is_combined, disorder_results (list of per-disorder summary dicts),
       output_root (str path).

    progress_callback(step: str, pct: int) is called at key milestones if provided.
    """
    def _progress(step: str, pct: int) -> None:
        """Forward a progress update to the caller's callback if one was provided."""
        if progress_callback is not None:
            try:
                progress_callback(step, pct)
            except Exception as e:
                logging.warning("progress_callback failed: %s", e)

    try:
        _progress("Loading inputs", 5)
        if timestamps_path is None:
            from datetime import datetime as _dt
            import pandas as _pd
            _neutral_end = 5.0
            _synthetic_events = _pd.DataFrame([
                {"timestamp_abs": 0.0, "event_type": "neutral",
                 "task_group": "0", "task_id": 0, "task_name": "continuous", "repetition": 1},
                {"timestamp_abs": _neutral_end, "event_type": "measurement",
                 "task_group": "0", "task_id": 1, "task_name": "continuous", "repetition": 1},
            ])
            session: PrompterSession = PrompterSession(
                participant_id=subject_id,
                profile="continuous",
                session_date=_dt.now().strftime("%Y-%m-%d"),
                recording_start_offset_s=0.0,
                events_df=_synthetic_events,
                is_combined=False,
                disorder_profiles=[],
                per_disorder_events={},
                camera_start_offsets=[],
            )
            for vp in video_paths:
                if not vp.exists():
                    raise ValueError(f"Video file not found: {vp}")
                if vp.stat().st_size == 0:
                    raise ValueError(f"Video file is empty: {vp}")
        else:
            session = load_prompter_session(
                timestamps_path=timestamps_path,
                meta_path=meta_path,
                assembly_path=assembly_path,
            )
            if not (len(video_paths) == 1 and video_paths[0].suffix.lower() == ".csv"):
                _validate_inputs(video_paths, session, logger)

        if session.is_combined and not session.per_disorder_events:
            logger.warning(
                "Session profile is COMBINED but no assembly CSV was provided. "
                "The pipeline will run as a single aggregate session without "
                "per-disorder analysis. Provide --prompter-assembly to enable "
                "per-disorder results."
            )

        config_io = IOManager(
            project_root, subject_id, session_label, study_mode, list_only=True
        )
        features_config = config_io.load_config("features")
        tasks_config = config_io.load_config("tasks")
        del config_io

        _is_raw_csv = (
            len(video_paths) == 1
            and video_paths[0].suffix.lower() == ".csv"
        )

        if _is_raw_csv:
            _progress("Loading pre-extracted landmarks CSV", 20)
            logger.info("Raw landmarks CSV input detected: %s", video_paths[0])
            if not video_paths[0].exists():
                raise ValueError(f"Raw landmarks CSV not found: {video_paths[0]}")
            if video_paths[0].stat().st_size == 0:
                raise ValueError(f"Raw landmarks CSV is empty: {video_paths[0]}")
            features_df, primary_fps, camera_offsets, annotated_video_tmps, landmark_video_tmps = (
                _load_raw_landmarks_csv(video_paths[0], session, meta_path)
            )
            from .utils import sanitize_events_df
            events_df = sanitize_events_df(session.events_df)
            _progress("Landmarks loaded, ready for analysis", 55)
        else:
            from .utils import MODEL_PATH, ensure_model_downloaded
            model_path = ensure_model_downloaded()
            processor = create_multi_camera_processor(
                video_paths=video_paths,
                features_config=features_config,
                model_path=model_path,
            )

            primary_fps = (
                processor.camera_streams[0].fps if getattr(processor, 'camera_streams', None) and len(processor.camera_streams) > 0 else 30.0
            )

            if meta_path is not None:
                try:
                    _meta_json = load_json(Path(meta_path))
                    _cameras = _meta_json.get("cameras", [])
                    if _cameras:
                        _granted = _cameras[0].get("grantedFrameRate")
                        if _granted and float(_granted) > 0:
                            primary_fps = float(_granted)
                            logger.info("fps from recording metadata: %.1f", primary_fps)
                except Exception as _fps_exc:
                    logger.debug("Could not read fps from recording metadata: %s", _fps_exc)

            offsets_from_meta = getattr(session, "camera_start_offsets", [])
            meta_sync_applied = processor.apply_offsets_from_meta(offsets_from_meta)

            import threading
            sync_exception = [None]
            def sync_worker():
                """Run camera alignment in a background thread and capture any exception."""
                try:
                    processor.align_cameras()
                except Exception as exc:
                    sync_exception[0] = exc

            if len(video_paths) > 1 and not meta_sync_applied:
                logger.info(
                    "No per-camera start offsets in recording metadata — "
                    "attempting audio cross-correlation sync."
                )
                sync_thread = threading.Thread(target=sync_worker, daemon=True)
                sync_thread.start()
                sync_thread.join(timeout=60)
                if sync_thread.is_alive():
                    logger.warning("Camera sync step timed out after 60 seconds: skipping sync, all camera offsets set to 0.0")
                    for stream in processor.camera_streams:
                        stream.time_offset_s = 0.0
                elif sync_exception[0] is not None:
                    logger.warning(f"Camera sync step failed: {sync_exception[0]}")
                    for stream in processor.camera_streams:
                        stream.time_offset_s = 0.0
            elif len(video_paths) == 1:
                logger.info("Single camera: no sync needed.")
            else:
                logger.info(
                    "Using per-camera start offsets from recording metadata for sync."
                )

            camera_offsets: Dict[int, float] = {
                stream.camera_index: stream.time_offset_s
                for stream in processor.camera_streams
            }
            _progress("Syncing cameras", 15)

            total_frames_estimate = max((s.total_frames for s in processor.camera_streams), default=1)

            _last_frame_progress_time: list = [0.0]

            def _frame_progress(frame_idx: int, total_frames: int) -> None:
                """Report frame-extraction progress at most every 100 ms."""
                try:
                    now = time.monotonic()
                    if now - _last_frame_progress_time[0] < 0.1:
                        return
                    _last_frame_progress_time[0] = now
                    if total_frames > 0:
                        pct = 15 + int((frame_idx / float(total_frames)) * 40)
                        _progress(f"Processing video frames ({frame_idx}/{total_frames})", min(pct, 55))
                except Exception:
                    pass

            frame_data_list, events_df, annotated_video_tmps, landmark_video_tmps = processor.process_all_frames(
                session.events_df,
                recording_start_offset_s=session.recording_start_offset_s,
                progress_callback=_frame_progress,
                video_mode=video_mode,
            )
            features_df = pd.DataFrame(frame_data_list)

        try:
            _raw_subj_dir = (
                project_root / "data" / "raw" / study_mode / subject_id
            )
            _raw_subj_dir.mkdir(parents=True, exist_ok=True)
            _ts_now = datetime.now().strftime("%Y%m%d_%H%M%S")
            _raw_csv_name = f"{subject_id}_{session_label}_{_ts_now}_raw_frames.csv"
            _raw_csv_path = _raw_subj_dir / _raw_csv_name
            _cols_to_drop = [c for c in ["_landmarks_3d"] if c in features_df.columns]
            features_df.drop(columns=_cols_to_drop, errors="ignore").to_csv(
                str(_raw_csv_path), index=False
            )
            logger.info("Master raw frames CSV saved: %s", _raw_csv_path)
        except Exception as _raw_exc:
            logger.warning("Could not save master raw frames CSV: %s", _raw_exc)

        if "intra_op" in session_label.lower() and timestamps_path is not None:
            _intra_op_seqs = _detect_intra_op_sequences(timestamps_path)
            if len(_intra_op_seqs) > 1:
                logger.info(
                    "Intra-op multi-sequence detected: sequences %s", _intra_op_seqs
                )
                _progress(
                    f"Intra-op: {len(_intra_op_seqs)} sequences detected, processing each", 58
                )
                return _run_intra_op_from_frames(
                    features_df=features_df,
                    video_paths=video_paths,
                    timestamps_path=timestamps_path,
                    subject_id=subject_id,
                    session_label=session_label,
                    study_mode=study_mode,
                    project_root=project_root,
                    meta_path=meta_path,
                    reference_session=reference_session,
                    progress_callback=progress_callback,
                    sequence_nums=_intra_op_seqs,
                    primary_fps=primary_fps,
                    camera_offsets=camera_offsets,
                    annotated_video_tmps=annotated_video_tmps,
                    landmark_video_tmps=landmark_video_tmps,
                    tasks_config=tasks_config,
                    features_config=features_config,
                )

        if session.is_combined and session.per_disorder_events:
            disorder_results = []
            disorder_keys = list(session.per_disorder_events.keys())
            n = len(disorder_keys)
            _combined_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            combined_parent_id = f"{subject_id}_{session_label}_{_combined_ts}"

            if n > 1 and "_landmarks_3d" in features_df.columns:
                try:
                    _tmp_extractor = create_feature_extractor(features_config)
                    features_df = _tmp_extractor._add_landmark_asymmetry(
                        features_df.copy(),
                        add_cheek=True,
                        add_nose=True,
                    )
                    del _tmp_extractor
                    logger.debug("Pre-computed landmark asymmetry for %d combined profiles.", n)
                except Exception as _pre_exc:
                    logger.warning("Could not pre-compute landmark asymmetry: %s", _pre_exc)

            if annotated_video_tmps:
                _parent_raw_vid = (
                    Path(project_root) / "data" / "raw" / study_mode
                    / subject_id / combined_parent_id
                )
                _parent_raw_vid.mkdir(parents=True, exist_ok=True)
                for _cam_idx, _src_path in enumerate(annotated_video_tmps):
                    if _src_path is None:
                        continue
                    try:
                        _vid_suffix = f"_cam{_cam_idx + 1}"
                        _dest = _parent_raw_vid / f"recording{_vid_suffix}_annotated.mp4"
                        shutil.copy2(str(_src_path), str(_dest))
                        logger.info("Annotated video (cam%d) saved to parent session: %s", _cam_idx + 1, _dest)
                    except Exception as _ve:
                        logger.warning("Could not save annotated video cam%d to parent: %s", _cam_idx + 1, _ve)
                for _cam_idx, _src_path in enumerate(landmark_video_tmps or []):
                    if _src_path is None:
                        continue
                    try:
                        _vid_suffix = f"_cam{_cam_idx + 1}"
                        _dest = _parent_raw_vid / f"recording{_vid_suffix}_landmarks_only.mp4"
                        shutil.copy2(str(_src_path), str(_dest))
                        logger.info("Landmark video (cam%d) saved to parent session: %s", _cam_idx + 1, _dest)
                    except Exception as _ve:
                        logger.warning("Could not save landmark video cam%d to parent: %s", _cam_idx + 1, _ve)

            for i, disorder_key in enumerate(disorder_keys):
                base_pct = 60 + int(i / n * 30)
                _progress(
                    f"Analysing profile {i + 1}/{n}: {disorder_key}", base_pct
                )
                disorder_label = disorder_key.lower()

                disorder_events_df = session.per_disorder_events[disorder_key]
                disorder_features_df = _reassign_segments_for_disorder(
                    features_df, disorder_events_df
                )

                try:
                    _slot_size = int(30 / max(n, 1))
                    def _combined_sub_progress(step: str, pct: int, _base=base_pct, _slot=_slot_size) -> None:
                        """Map per-disorder-profile progress onto the COMBINED-session progress slot."""
                        _progress(f"{step}", _base + int(pct / 100 * _slot))
                    disorder_summary = _run_single_profile_analysis(
                        subject_id=subject_id,
                        session_label=disorder_label,
                        study_mode=study_mode,
                        project_root=project_root,
                        features_df=disorder_features_df,
                        events_df=disorder_events_df,
                        tasks_config=tasks_config,
                        features_config=features_config,
                        reference_session=reference_session,
                        session_profile=disorder_key,
                        fps=primary_fps,
                        annotated_video_srcs=None,
                        parent_session_id=combined_parent_id,
                        sub_progress=_combined_sub_progress,
                    )
                except Exception as _disp_exc:
                    logger.error(
                        "Analysis failed for disorder profile '%s': %s — continuing with remaining profiles.",
                        disorder_key, _disp_exc,
                    )
                    import traceback as _tb
                    logger.debug("Traceback for %s:\n%s", disorder_key, _tb.format_exc())
                    disorder_summary = {
                        "disorder_key": disorder_key,
                        "error": str(_disp_exc),
                        "session_label": disorder_label,
                    }
                    disorder_results.append(disorder_summary)
                    _progress(
                        f"Profile {disorder_key} failed (continuing): {str(_disp_exc)[:80]}",
                        base_pct + int(1 / n * 30),
                    )
                    continue

                disorder_summary["disorder_key"] = disorder_key
                disorder_results.append(disorder_summary)

            if disorder_results:
                _parent_raw = (
                    Path(project_root) / "data" / "raw" / study_mode
                    / subject_id / combined_parent_id
                )
                _parent_raw.mkdir(parents=True, exist_ok=True)
                try:
                    _manifest_io = IOManager(
                        project_root, subject_id, session_label, study_mode, list_only=True
                    )
                    _manifest_io.raw_dir = _parent_raw
                    _manifest_io.save_prompter_inputs_manifest(
                        video_paths=[str(p) for p in video_paths],
                        timestamps_path=Path(timestamps_path) if timestamps_path else None,
                        meta_path=Path(meta_path) if meta_path else None,
                        assembly_path=Path(assembly_path) if assembly_path else None,
                        session_offset_s=session.recording_start_offset_s,
                        camera_offsets=camera_offsets,
                    )
                except Exception as exc:
                    logger.warning("Could not save prompter inputs manifest: %s", exc)

            output_root = str(
                project_root / "data" / "results" / study_mode / subject_id / combined_parent_id
            )
            try:
                consolidate_subject(project_root, subject_id, study_mode)
            except Exception as exc:
                logger.warning("Could not consolidate after prompter session: %s", exc)
            try:
                import sys as _sys
                _tools_dir = str(Path(project_root) / "tools")
                if _tools_dir not in _sys.path:
                    _sys.path.insert(0, _tools_dir)
                from session_summary_figure import generate_session_summary as _gen_summary, generate_participant_summary as _gen_participant
                _gen_summary(Path(output_root))
                logger.info("Session summary PDF saved: %s/session_summary.pdf", output_root)
                _participant_results = Path(project_root) / "data" / "results" / study_mode / subject_id
                _gen_participant(_participant_results)
                logger.info("Participant summary PDF saved: %s/session_summary.pdf", _participant_results)
            except Exception as _exc:
                logger.warning("Could not generate session summary figure: %s", _exc)
            _progress("Done", 100)
            return {
                "participant_id": session.participant_id,
                "profile": session.profile,
                "session_date": session.session_date,
                "is_combined": True,
                "disorder_results": disorder_results,
                "output_root": output_root,
            }

        else:
            _progress("Analysing session", 65)
            def _single_sub_progress(step: str, pct: int) -> None:
                """Map per-analysis progress onto the single-profile analysis slot (65-90%)."""
                _progress(f"{step}", 65 + int(pct * 0.25))
            summary = _run_single_profile_analysis(
                subject_id=subject_id,
                session_label=session_label,
                study_mode=study_mode,
                project_root=project_root,
                features_df=features_df,
                events_df=events_df,
                tasks_config=tasks_config,
                features_config=features_config,
                reference_session=reference_session,
                session_profile=session.profile,
                fps=primary_fps,
                annotated_video_srcs=annotated_video_tmps,
                landmark_video_srcs=landmark_video_tmps,
                sub_progress=_single_sub_progress,
            )
            manifest_raw_dir = Path(summary["output_paths"]["raw"])
            try:
                _manifest_io = IOManager(
                    project_root, subject_id, session_label, study_mode, list_only=True
                )
                _manifest_io.raw_dir = manifest_raw_dir
                _manifest_io.save_prompter_inputs_manifest(
                    video_paths=[str(p) for p in video_paths],
                    timestamps_path=Path(timestamps_path) if timestamps_path else None,
                    meta_path=Path(meta_path) if meta_path else None,
                    assembly_path=Path(assembly_path) if assembly_path else None,
                    session_offset_s=session.recording_start_offset_s,
                    camera_offsets=camera_offsets,
                )
            except Exception as exc:
                logger.warning("Could not save prompter inputs manifest: %s", exc)
            try:
                consolidate_subject(project_root, subject_id, study_mode)
            except Exception as exc:
                logger.warning("Could not consolidate after prompter session: %s", exc)
            try:
                import sys as _sys
                _tools_dir = str(Path(project_root) / "tools")
                if _tools_dir not in _sys.path:
                    _sys.path.insert(0, _tools_dir)
                from session_summary_figure import generate_session_summary as _gen_summary, generate_participant_summary as _gen_participant
                _single_results = Path(summary["output_paths"]["results"])
                _session_results_dir = _single_results.parent
                _gen_summary(_session_results_dir)
                logger.info("Session summary PDF saved: %s/session_summary.pdf",
                            _session_results_dir)
                _participant_results = Path(project_root) / "data" / "results" / study_mode / subject_id
                _gen_participant(_participant_results)
                logger.info("Participant summary PDF saved: %s/session_summary.pdf", _participant_results)
            except Exception as _exc:
                logger.warning("Could not generate session summary figure: %s", _exc)
            _progress("Done", 100)
            return {
                "participant_id": session.participant_id,
                "profile": session.profile,
                "session_date": session.session_date,
                "is_combined": False,
                "disorder_results": [summary],
                "output_root": summary["output_paths"]["results"],
            }

    except Exception as exc:
        error_msg = str(exc).split("\n")[0][:200]
        _progress(f"Error: {error_msg}", -1)
        raise
