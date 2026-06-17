"""
Consolidated CSV generator for the facial motor and speech behavior analysis pipeline.

Walks all sessions for a given subject, merges processed metrics, anomaly results,
screening outcomes, and session metadata into two CSV files saved under
``data/results/<study_mode>/<subject_id>/``:

  1. ``<subject_id>_consolidated.csv`` — one row per repetition across all sessions
  2. ``<subject_id>_session_overview.csv`` — one row per session with summary-level info

These files are the single source of truth you can open in Excel / pandas to
trace any data point back to its session, repetition, and pipeline run.

Run standalone (retroactive generation for existing data)::

    python -m src.consolidate --subject T001 --mode pilot

Or import and call from the pipeline::

    from src.consolidate import consolidate_subject
    consolidate_subject(project_root, subject_id, study_mode)
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger("pipeline")

PROJECT_ROOT = Path(__file__).parent.parent

if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_json_safe(path: Path) -> Optional[Dict]:
    """Load a JSON file, returning ``None`` on any failure."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _discover_sessions(data_dir: Path, study_mode: str, subject_id: str) -> List[str]:
    """Return a sorted list of session-id strings that have at least a raw-data directory.

    Only includes session directories whose name starts with subject_id.
    Returns an empty list when the raw directory does not exist.
    """
    raw_dir = data_dir / "raw" / study_mode / subject_id
    if not raw_dir.exists():
        return []
    return sorted(
        d.name
        for d in raw_dir.iterdir()
        if d.is_dir() and d.name.startswith(subject_id)
    )


def _resolve_results_file(results_dir: Path, filename: str) -> Optional[Path]:
    """Find a results file by checking the flat layout first, then profile subdirectories.

    Newer sessions store results under a profile subdirectory such as
    results_dir/normal/anomaly_results.json.  When the top-level path is
    absent, the 'normal' subdirectory is tried first (it is the reference
    profile), then all other subdirectories in sorted order.  Returns None
    when the file cannot be found anywhere.
    """
    flat = results_dir / filename
    if flat.exists():
        return flat

    if results_dir.exists():
        subdirs = sorted(results_dir.iterdir())
        priority = [results_dir / "normal"] + [s for s in subdirs if s.name != "normal"]
        for candidate_subdir in priority:
            if not candidate_subdir.is_dir():
                continue
            candidate = candidate_subdir / filename
            if candidate.exists():
                return candidate

    return None


def _resolve_processed_file(processed_dir: Path, filename: str) -> Optional[Path]:
    """Find a processed file by checking the flat layout first, then profile subdirectories.

    Applies the same search strategy as _resolve_results_file.
    Returns None when the file cannot be found.
    """
    flat = processed_dir / filename
    if flat.exists():
        return flat

    if processed_dir.exists():
        subdirs = sorted(processed_dir.iterdir())
        priority = [processed_dir / "normal"] + [s for s in subdirs if s.name != "normal"]
        for candidate_subdir in priority:
            if not candidate_subdir.is_dir():
                continue
            candidate = candidate_subdir / filename
            if candidate.exists():
                return candidate

    return None


def _build_repetition_rows(
    data_dir: Path, study_mode: str, subject_id: str, session_ids: List[str]
) -> pd.DataFrame:
    """Build a one-row-per-repetition DataFrame spanning all supplied sessions.

    For each session, loads repetition_metrics.csv and enriches it with:
      - session identity columns (subject_id, session_id, session_label, etc.)
      - anomaly_score and is_anomaly from anomaly_results.json
      - session-level indication counts from screening_results.json
      - confidence sub-scores from confidence_summary.json

    Sessions without a repetition_metrics.csv are silently skipped.
    Returns an empty DataFrame when no sessions have valid data.
    """
    all_rows: List[pd.DataFrame] = []

    for sid in session_ids:
        processed_dir = data_dir / "processed" / study_mode / subject_id / sid
        results_dir = data_dir / "results" / study_mode / subject_id / sid
        raw_dir = data_dir / "raw" / study_mode / subject_id / sid

        rep_path_resolved = _resolve_processed_file(processed_dir, "repetition_metrics.csv")
        if rep_path_resolved is None:
            continue
        rep_df = pd.read_csv(rep_path_resolved)

        meta = _load_json_safe(raw_dir / "session_metadata.json") or {}
        new_cols = {}

        new_cols["subject_id"] = subject_id
        new_cols["session_id"] = sid
        new_cols["session_label"] = meta.get(
            "session_label", sid.split("_")[1] if "_" in sid else ""
        )
        new_cols["session_type"] = "baseline" if "baseline" in sid.lower() else "test"
        new_cols["study_mode"] = study_mode
        new_cols["session_timestamp"] = meta.get("timestamp", "")
        new_cols["pipeline_version"] = meta.get("pipeline_version", "")
        new_cols["config_hash"] = meta.get("config_hash", "")

        anomaly_path = _resolve_results_file(results_dir, "anomaly_results.json")
        anomaly = _load_json_safe(anomaly_path) if anomaly_path else None
        if anomaly:
            n = len(rep_df)
            scores = anomaly.get("anomaly_scores", [])
            is_anomaly = anomaly.get("is_anomaly", [])
            scores_padded = (list(scores) + [None] * n)[:n]
            is_anomaly_padded = (
                [int(a) if isinstance(a, bool) else a for a in is_anomaly] + [None] * n
            )[:n]
            new_cols["anomaly_score"] = scores_padded
            new_cols["is_anomaly"] = is_anomaly_padded
        else:
            new_cols["anomaly_score"] = None
            new_cols["is_anomaly"] = None

        screening_path = _resolve_results_file(results_dir, "screening_results.json")
        screening = _load_json_safe(screening_path) if screening_path else None
        if screening:
            indications = screening.get("indications", [])
            new_cols["session_n_indications"] = len(indications)
            new_cols["session_indication_types"] = (
                ", ".join(ind.get("indication_type", "") for ind in indications)
                if indications
                else ""
            )
        else:
            new_cols["session_n_indications"] = 0
            new_cols["session_indication_types"] = ""

        confidence_path = _resolve_results_file(results_dir, "confidence_summary.json")
        confidence = _load_json_safe(confidence_path) if confidence_path else None
        if confidence:
            conf = confidence.get("confidence", {})
            new_cols["confidence_overall"] = conf.get("overall")
            new_cols["confidence_data_quality"] = conf.get("data_quality")
            new_cols["confidence_consistency"] = conf.get("consistency")
        else:
            new_cols["confidence_overall"] = None
            new_cols["confidence_data_quality"] = None
            new_cols["confidence_consistency"] = None

        scalar_cols = {k: v for k, v in new_cols.items() if not isinstance(v, list)}
        list_cols = {k: v for k, v in new_cols.items() if isinstance(v, list)}
        for k, v in scalar_cols.items():
            rep_df[k] = v
        for k, v in list_cols.items():
            rep_df[k] = v

        all_rows.append(rep_df)

    if not all_rows:
        return pd.DataFrame()

    consolidated = pd.concat(all_rows, ignore_index=True)

    id_cols = [
        "subject_id", "session_id", "session_label", "session_type",
        "study_mode", "session_timestamp", "repetition", "task_group", "task_id",
        "n_frames", "duration_sec",
    ]
    outcome_cols = [
        "anomaly_score", "is_anomaly",
        "session_n_indications", "session_indication_types",
        "confidence_overall", "confidence_data_quality", "confidence_consistency",
    ]
    meta_cols = ["pipeline_version", "config_hash"]

    existing_id = [c for c in id_cols if c in consolidated.columns]
    existing_out = [c for c in outcome_cols if c in consolidated.columns]
    existing_meta = [c for c in meta_cols if c in consolidated.columns]
    metric_cols = [
        c
        for c in consolidated.columns
        if c not in existing_id + existing_out + existing_meta
        and c not in ("_subject_id", "_session_id", "_pipeline_version", "_config_hash")
    ]

    ordered = existing_id + existing_out + metric_cols + existing_meta
    consolidated = consolidated.drop(
        columns=[
            c
            for c in ("_subject_id", "_session_id", "_pipeline_version", "_config_hash")
            if c in consolidated.columns
        ],
        errors="ignore",
    )
    ordered = [c for c in ordered if c in consolidated.columns]
    return consolidated[ordered]


def _build_session_overview(
    data_dir: Path, study_mode: str, subject_id: str, session_ids: List[str]
) -> pd.DataFrame:
    """Build a one-row-per-session overview DataFrame spanning all supplied sessions.

    Collects session identity, posture, frame counts, duration, asymmetry
    summaries, anomaly counts, indication types, confidence scores, and
    data-availability flags for each session.  Columns are ordered by
    a canonical schema; any extra columns are appended at the end.

    Returns an empty DataFrame when no sessions are found.
    """
    rows: List[Dict[str, Any]] = []

    for sid in session_ids:
        processed_dir = data_dir / "processed" / study_mode / subject_id / sid
        results_dir = data_dir / "results" / study_mode / subject_id / sid
        raw_dir = data_dir / "raw" / study_mode / subject_id / sid

        row: Dict[str, Any] = {
            "subject_id": subject_id,
            "session_id": sid,
            "study_mode": study_mode,
        }

        meta = _load_json_safe(raw_dir / "session_metadata.json") or {}
        row["session_label"] = meta.get("session_label", "")
        row["session_type"] = "baseline" if "baseline" in sid.lower() else "test"
        row["session_timestamp"] = meta.get("timestamp", "")
        row["pipeline_version"] = meta.get("pipeline_version", "")
        row["config_hash"] = meta.get("config_hash", "")

        _posture_tokens = (sid + " " + row["session_label"]).lower()
        _supine_kws = ("supine", "lying", "or_sim", "or-sim", "orsim", "intraop", "intra_op", "_ors_", "_ors", "ors_")
        row["posture"] = "supine" if any(k in _posture_tokens for k in _supine_kws) else "upright"

        summary_path = _resolve_results_file(results_dir, "pipeline_summary.json")
        summary = _load_json_safe(summary_path) if summary_path else None
        if summary:
            row["n_repetitions"] = summary.get("n_repetitions", 0)
            row["_raw_n_frames_total"] = summary.get("n_frames", 0)

        metrics_path = _resolve_processed_file(processed_dir, "session_metrics.json")
        metrics = _load_json_safe(metrics_path) if metrics_path else None
        if metrics:
            n_analyzed = metrics.get("total_frames", 0)
            det_rate = metrics.get("overall_detection_rate") or 1.0
            n_task_captured = int(round(n_analyzed / det_rate)) if det_rate > 0 else n_analyzed
            row["n_frames_captured"] = n_task_captured
            row["n_frames_analyzed"] = n_analyzed
            row["n_repetitions"] = metrics.get(
                "total_repetitions", row.get("n_repetitions", 0)
            )
            row["total_duration_sec"] = metrics.get("total_duration_sec", 0)
            row["overall_mean_asymmetry"] = metrics.get("overall_mean_asymmetry")
            row["overall_max_asymmetry"] = metrics.get("overall_max_asymmetry")
            row["overall_mean_signed_asymmetry"] = metrics.get("overall_mean_signed_asymmetry")
            row["dominant_side"] = metrics.get("dominant_side")
            row["overall_detection_rate"] = det_rate
            row["mean_duration_per_rep"] = metrics.get("duration_sec_session_mean")
            row["mean_activation_session_mean"] = metrics.get(
                "mean_activation_mean_session_mean"
            )
            row["max_activation_session_mean"] = metrics.get(
                "max_activation_mean_session_mean"
            )

        anomaly_path = _resolve_results_file(results_dir, "anomaly_results.json")
        anomaly = _load_json_safe(anomaly_path) if anomaly_path else None
        if anomaly:
            anomaly_sum = anomaly.get("summary", {})
            is_anomaly = anomaly.get("is_anomaly", [])
            row["n_anomalies"] = anomaly_sum.get(
                "n_anomalies", sum(1 for a in is_anomaly if a)
            )
            row["anomaly_rate"] = (
                row["n_anomalies"] / max(len(is_anomaly), 1) if is_anomaly else None
            )
        else:
            row["n_anomalies"] = None
            row["anomaly_rate"] = None

        screening_path = _resolve_results_file(results_dir, "screening_results.json")
        screening = _load_json_safe(screening_path) if screening_path else None
        if screening:
            indications = screening.get("indications", [])
            row["n_indications"] = len(indications)
            row["indication_types"] = (
                ", ".join(ind.get("indication_type", "") for ind in indications)
                if indications
                else ""
            )
            row["indication_severities"] = (
                ", ".join(ind.get("severity", "") for ind in indications)
                if indications
                else ""
            )
        else:
            row["n_indications"] = 0
            row["indication_types"] = ""
            row["indication_severities"] = ""

        confidence_path = _resolve_results_file(results_dir, "confidence_summary.json")
        confidence = _load_json_safe(confidence_path) if confidence_path else None
        if confidence:
            conf = confidence.get("confidence", {})
            row["confidence_data_quality"] = conf.get("data_quality")
            row["confidence_consistency"] = conf.get("consistency")
            row["confidence_model_rule_agreement"] = conf.get("model_rule_agreement")
            row["confidence_overall"] = conf.get("overall")
            row["is_baseline_session"] = confidence.get("is_baseline_session")
            row["reference_session"] = confidence.get("reference_session")

        _raw_blendshapes = (raw_dir / "blendshapes.csv")
        _raw_blendshapes_subdir = next(
            (p for p in raw_dir.iterdir() if p.is_dir() and (p / "blendshapes.csv").exists()),
            None,
        ) if raw_dir.exists() else None
        row["has_raw_data"] = (raw_dir / "frame_data.csv").exists() or any(
            (raw_dir / sd / "frame_data.csv").exists()
            for sd in (["normal"] if (raw_dir / "normal").is_dir() else [])
        )
        row["has_blendshapes"] = _raw_blendshapes.exists() or _raw_blendshapes_subdir is not None
        row["has_corrected_feat"] = _resolve_processed_file(processed_dir, "corrected_features.csv") is not None
        _viz_dirs = [results_dir / "visualizations"] + [
            sd / "visualizations"
            for sd in (results_dir.iterdir() if results_dir.exists() else [])
            if sd.is_dir()
        ]
        row["has_visualizations"] = any(
            vd.exists() and any(vd.iterdir()) for vd in _viz_dirs
        )

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    overview = pd.DataFrame(rows)

    col_order = [
        "subject_id", "session_id", "session_label", "session_type", "posture", "study_mode",
        "session_timestamp", "pipeline_version", "config_hash",
        "n_frames_captured", "n_frames_analyzed", "n_repetitions", "total_duration_sec",
        "mean_duration_per_rep",
        "overall_mean_asymmetry", "overall_max_asymmetry",
        "overall_mean_signed_asymmetry", "dominant_side",
        "overall_detection_rate",
        "mean_activation_session_mean", "max_activation_session_mean",
        "n_anomalies", "anomaly_rate",
        "n_indications", "indication_types", "indication_severities",
        "confidence_data_quality", "confidence_consistency",
        "confidence_model_rule_agreement", "confidence_overall",
        "is_baseline_session", "reference_session",
        "has_raw_data", "has_blendshapes", "has_corrected_feat", "has_visualizations",
    ]
    col_order = [c for c in col_order if c in overview.columns]
    extras = [c for c in overview.columns if c not in col_order]
    return overview[col_order + extras]


def consolidate_subject(
    project_root: Path,
    subject_id: str,
    study_mode: str = "pilot",
) -> Tuple[Optional[Path], Optional[Path]]:
    """Generate the two consolidated CSV files for a subject and return their paths.

    Discovers all sessions with raw data, then calls _build_repetition_rows
    and _build_session_overview to create:
      - {subject_id}_consolidated.csv  (one row per repetition)
      - {subject_id}_session_overview.csv  (one row per session)

    Both files are written to data/results/{study_mode}/{subject_id}/.
    Numeric columns are rounded before writing.

    Also attempts to generate detection-quality and condition-comparison
    figures using the visualization module; any failure there is logged as a
    warning and does not prevent the CSVs from being saved.

    Returns:
        A tuple (consolidated_path, overview_path).  Either element is None
        when no data was available for that output.
    """
    data_dir = Path(project_root) / "data"
    output_dir = data_dir / "results" / study_mode / subject_id
    output_dir.mkdir(parents=True, exist_ok=True)

    session_ids = _discover_sessions(data_dir, study_mode, subject_id)
    if not session_ids:
        logger.warning("No sessions found for %s in %s mode.", subject_id, study_mode)
        return None, None

    logger.info("Found %d session(s) for %s: %s", len(session_ids), subject_id, session_ids)

    consolidated_df = _build_repetition_rows(data_dir, study_mode, subject_id, session_ids)
    consolidated_path: Optional[Path] = None
    if len(consolidated_df) > 0:
        num_cols = consolidated_df.select_dtypes(include="number").columns
        consolidated_df[num_cols] = consolidated_df[num_cols].round(6)
        consolidated_path = output_dir / f"{subject_id}_consolidated.csv"
        consolidated_df.to_csv(consolidated_path, index=False, float_format="%.6g")
        logger.info(
            "  -> %s  (%d rows x %d cols)",
            consolidated_path, len(consolidated_df), len(consolidated_df.columns),
        )

    overview_df = _build_session_overview(data_dir, study_mode, subject_id, session_ids)
    overview_path: Optional[Path] = None
    if len(overview_df) > 0:
        num_cols = overview_df.select_dtypes(include="number").columns
        overview_df[num_cols] = overview_df[num_cols].round(4)
        overview_path = output_dir / f"{subject_id}_session_overview.csv"
        overview_df.to_csv(overview_path, index=False, float_format="%.4g")
        logger.info(
            "  -> %s  (%d rows x %d cols)",
            overview_path, len(overview_df), len(overview_df.columns),
        )
        try:
            from .visualization import create_visualizer
            from .utils import load_yaml
            plotting_cfg = load_yaml(project_root / "config" / "plotting.yaml")
            viz = create_visualizer(plotting_cfg)
            viz.plot_detection_quality_summary(
                overview_df,
                output_dir / f"{subject_id}_detection_quality_summary.pdf",
                subject_id=subject_id,
            )
            postures = [str(p) for p in (overview_df["posture"].dropna().unique() if "posture" in overview_df.columns else []) if str(p).strip()]
            if len(postures) >= 2:
                viz.plot_condition_comparison(
                    overview_df,
                    output_dir / f"{subject_id}_condition_comparison.pdf",
                    subject_id=subject_id,
                )
        except Exception as _ve:
            logger.warning("Could not generate consolidation figures: %s", _ve)

    return consolidated_path, overview_path


def main():
    """CLI entry point for standalone consolidated CSV generation."""
    parser = argparse.ArgumentParser(
        description="Generate consolidated CSVs for a subject's data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m src.consolidate --subject T001 --mode pilot
    python -m src.consolidate --subject PAT001 --mode patient
        """,
    )
    parser.add_argument(
        "--subject", "-s", required=True, help="Subject ID (e.g., T001)"
    )
    parser.add_argument(
        "--mode", "-m", default="pilot", choices=["pilot", "patient"],
        help="Study mode (default: pilot)",
    )
    parser.add_argument(
        "--project-root", "-p", default=str(PROJECT_ROOT),
        help="Project root directory",
    )

    args = parser.parse_args()

    consolidated_path, overview_path = consolidate_subject(
        Path(args.project_root), args.subject, args.mode
    )

    if consolidated_path or overview_path:
        print("\nDone! Consolidated files saved.")
    else:
        print("\nNo data to consolidate.")


if __name__ == "__main__":
    main()
