"""
I/O manager for the FACET-CV pipeline.

Centralises all file-path construction, directory creation, and data
persistence.  Raw data is kept separate from processed and results data,
and every path method returns a pathlib.Path so callers never need to
assemble paths by hand.

The IOManager is instantiated once per session.  In list_only mode it sets
up all path attributes without creating any directories or writing metadata,
which lets helper functions scan existing data trees without side effects.

Typical usage::

    io = IOManager(project_root, subject_id="T001", session_label="pre_op",
                   study_mode="pilot")
    io.save_dataframe(rep_df, io.get_repetition_metrics_path())
"""

import shutil
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .utils import (
    load_yaml, save_yaml, load_json, save_json,
    compute_config_hash, get_pipeline_version,
)


class IOManager:
    """Manages directory layout, session metadata, and data persistence for one session.

    All output paths follow the convention:
        data/{raw|processed|results}/{study_mode}/{subject_id}/{session_id}/

    Call the various get_*_path() methods to retrieve canonical locations for
    each artifact type without manually constructing paths.
    """

    def __init__(
        self,
        project_root: Union[str, Path],
        subject_id: str,
        session_label: str,
        study_mode: str,
        list_only: bool = False,
        parent_session_id: Optional[str] = None,
    ) -> None:
        """Set up directory paths and optionally create directories and write metadata.

        When list_only is True all path attributes are populated so scanning
        methods such as list_sessions_with_metadata can work, but no directories
        are created and no metadata file is written.

        When parent_session_id is provided the directories nest as
        {type}/{mode}/{subject}/{parent_session_id}/{session_label}/ rather
        than the default flat layout.  This is used for COMBINED profile sessions
        so that each disorder-profile folder sits inside its parent session folder.

        Args:
            project_root: Root of the project tree (contains config/, data/, etc.).
            subject_id: Participant identifier, e.g. 'T001'.
            session_label: Short label for this session, e.g. 'pre_op_baseline'.
            study_mode: Either 'pilot' or 'patient'.
            list_only: If True, skip directory creation and metadata writing.
            parent_session_id: Session ID of a parent COMBINED session, if any.
        """
        self.project_root = Path(project_root)
        self.subject_id = subject_id
        self.session_label = session_label
        self.study_mode = study_mode
        self.parent_session_id = parent_session_id
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = (
            f"{parent_session_id}/{session_label}"
            if parent_session_id
            else f"{subject_id}_{session_label}_{self.timestamp}"
        )

        self.config_dir = self.project_root / "config"
        self.data_dir = self.project_root / "data"
        self.logs_dir = self.project_root / "logs"

        if parent_session_id:
            _leaf = session_label
            self.raw_dir = self.data_dir / "raw" / study_mode / subject_id / parent_session_id / _leaf
            self.processed_dir = self.data_dir / "processed" / study_mode / subject_id / parent_session_id / _leaf
            self.results_dir = self.data_dir / "results" / study_mode / subject_id / parent_session_id / _leaf
        else:
            self.raw_dir = self.data_dir / "raw" / study_mode / subject_id / self.session_id
            self.processed_dir = self.data_dir / "processed" / study_mode / subject_id / self.session_id
            self.results_dir = self.data_dir / "results" / study_mode / subject_id / self.session_id
        self.subject_dir = self.data_dir / study_mode / subject_id

        if not list_only:
            self.config_hash = compute_config_hash([
                self.config_dir / "tasks.yaml",
                self.config_dir / "features.yaml",
                self.config_dir / "decision_rules.yaml",
            ])
            self._create_directories()
            self._save_session_metadata()
        else:
            self.config_hash = ""

    def _create_directories(self) -> None:
        """Ensure all required output directories exist."""
        for directory in (self.raw_dir, self.processed_dir, self.results_dir, self.logs_dir):
            directory.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "visualizations").mkdir(exist_ok=True)

    def _save_session_metadata(self, session_profile: str = "") -> None:
        """Write session_metadata.json to the raw directory.

        Records subject_id, session_label, session_id, study_mode, timestamp,
        pipeline_version, config_hash, created_at, and the optional
        session_profile string.  Called automatically during __init__ unless
        list_only was set.
        """
        metadata = {
            "subject_id": self.subject_id,
            "session_label": self.session_label,
            "session_id": self.session_id,
            "study_mode": self.study_mode,
            "timestamp": self.timestamp,
            "pipeline_version": get_pipeline_version(),
            "config_hash": self.config_hash,
            "created_at": datetime.now().isoformat(),
            "session_profile": session_profile,
        }
        save_json(metadata, self.raw_dir / "session_metadata.json")

    def load_config(self, config_name: str) -> Dict[str, Any]:
        """Load a YAML configuration file from the config directory."""
        return load_yaml(self.config_dir / f"{config_name}.yaml")

    def get_raw_video_path(self, suffix: str = "") -> Path:
        """Return path for a raw recording video file."""
        return self.raw_dir / f"recording{suffix}.mp4"

    def get_annotated_video_path(self, suffix: str = "") -> Path:
        """Return path for an annotated recording video file."""
        return self.raw_dir / f"recording{suffix}_annotated.mp4"

    def get_landmarks_video_path(self, suffix: str = "") -> Path:
        """Return path for a landmarks-only skeleton recording video file."""
        return self.raw_dir / f"recording{suffix}_landmarks_only.mp4"

    def get_normal_speed_video_path(self, suffix: str = "") -> Path:
        """Return path for a normal-speed raw recording video file."""
        return self.raw_dir / f"recording{suffix}_normal.mp4"

    def get_normal_speed_annotated_video_path(self, suffix: str = "") -> Path:
        """Return path for a normal-speed annotated recording video file."""
        return self.raw_dir / f"recording{suffix}_annotated_normal.mp4"

    def get_frame_data_path(self) -> Path:
        """Return path for the per-frame data CSV."""
        return self.raw_dir / "frame_data.csv"

    def get_landmarks_path(self) -> Path:
        """Return path for the landmarks CSV."""
        return self.raw_dir / "landmarks.csv"

    def get_blendshapes_path(self) -> Path:
        """Return path for the blendshapes CSV."""
        return self.raw_dir / "blendshapes.csv"

    def get_events_path(self) -> Path:
        """Return path for the events CSV."""
        return self.raw_dir / "events.csv"

    def get_baseline_path(self) -> Path:
        """Return path for the baseline JSON."""
        return self.raw_dir / "baseline.json"

    def get_corrected_features_path(self) -> Path:
        """Return path for the baseline-corrected features CSV."""
        return self.processed_dir / "corrected_features.csv"

    def get_repetition_metrics_path(self) -> Path:
        """Return path for the repetition-level metrics CSV."""
        return self.processed_dir / "repetition_metrics.csv"

    def get_task_metrics_path(self) -> Path:
        """Return path for the task-level metrics CSV."""
        return self.processed_dir / "task_metrics.csv"

    def get_session_metrics_path(self) -> Path:
        """Return path for the session-level metrics JSON."""
        return self.processed_dir / "session_metrics.json"

    def get_anomaly_results_path(self) -> Path:
        """Return path for anomaly detection results JSON."""
        return self.results_dir / "anomaly_results.json"

    def get_screening_results_path(self) -> Path:
        """Return path for clinical screening results JSON."""
        return self.results_dir / "screening_results.json"

    def get_confidence_summary_path(self) -> Path:
        """Return path for the confidence summary JSON."""
        return self.results_dir / "confidence_summary.json"

    def get_validation_table_path(self) -> Path:
        """Return path for the validation summary CSV."""
        return self.results_dir / "tables" / "validation_summary.csv"

    def get_visualization_path(self, name: str, extension: str = "png") -> Path:
        """Return path for a named visualization output file."""
        return self.results_dir / "visualizations" / f"{name}.{extension}"

    def get_log_path(self) -> Path:
        """Return path for the session log file."""
        return self.logs_dir / f"{self.session_id}.log"

    def get_task_profile_path(self) -> Path:
        """Return path for the subject's accumulated task profile JSON."""
        return (
            self.data_dir / "results" / self.study_mode
            / self.subject_id / f"{self.subject_id}_task_profile.json"
        )

    def get_subject_database_path(self) -> Path:
        """Return path for the subject's consolidated SQLite database."""
        return (
            self.data_dir / "results" / self.study_mode
            / self.subject_id / f"{self.subject_id}_consolidated.db"
        )

    def get_subject_summary_path(self) -> Path:
        """Return path for the subject summary JSON."""
        return (
            self.data_dir / "results" / self.study_mode
            / self.subject_id / f"{self.subject_id}_summary.json"
        )

    def get_consolidated_csv_path(self) -> Path:
        """Return path for the full repetition-level consolidated CSV."""
        return (
            self.data_dir / "results" / self.study_mode
            / self.subject_id / f"{self.subject_id}_consolidated.csv"
        )

    def get_session_overview_csv_path(self) -> Path:
        """Return path for the session-level overview CSV."""
        return (
            self.data_dir / "results" / self.study_mode
            / self.subject_id / f"{self.subject_id}_session_overview.csv"
        )

    def get_decision_trace_path(self) -> Path:
        """Return path for the decision trace table CSV. Creates the tables directory on demand."""
        tables_dir = self.results_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        return tables_dir / "decision_trace.csv"

    def save_dataframe(
        self, df: pd.DataFrame, path: Path, include_metadata: bool = True
    ) -> None:
        """Write a DataFrame to CSV, optionally prepending pipeline provenance columns.

        When include_metadata is True, adds _subject_id, _session_id,
        _pipeline_version, and _config_hash columns to a copy of df before
        writing.  Creates parent directories as needed.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        if include_metadata:
            df = df.copy()
            df["_subject_id"] = self.subject_id
            df["_session_id"] = self.session_id
            df["_pipeline_version"] = get_pipeline_version()
            df["_config_hash"] = self.config_hash
        df.to_csv(path, index=False)

    def load_dataframe(self, path: Path) -> pd.DataFrame:
        """Load a CSV file into a DataFrame."""
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        return pd.read_csv(path)

    def copy_input_video(self, source_path: Union[str, Path]) -> Path:
        """Copy an external video file into the raw directory and return the destination."""
        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Input video not found: {source_path}")
        dest_path = self.raw_dir / f"input_{source_path.name}"
        shutil.copy2(source_path, dest_path)
        return dest_path

    def get_reference_session_path(self, reference_session_id: str) -> Path:
        """Return the processed-data directory for a reference (baseline) session.

        Infers the subject ID from the first underscore-delimited token of
        reference_session_id; falls back to self.subject_id when the ID has
        fewer than two tokens.
        """
        parts = reference_session_id.split("_")
        ref_subject = parts[0] if len(parts) >= 2 else self.subject_id
        return self.data_dir / "processed" / self.study_mode / ref_subject / reference_session_id

    def list_sessions(
        self,
        subject_id: Optional[str] = None,
        study_mode: Optional[str] = None,
    ) -> list:
        """List session directory names for the given subject and study mode."""
        study_mode = study_mode or self.study_mode
        subject_id = subject_id or self.subject_id
        sessions_dir = self.data_dir / "processed" / study_mode / subject_id
        if not sessions_dir.exists():
            return []
        return [d.name for d in sessions_dir.iterdir() if d.is_dir()]

    def list_all_subject_sessions(self) -> list:
        """List all sessions for the current subject across raw, processed, and results."""
        sessions = []
        for data_type in ("raw", "processed", "results"):
            base_dir = self.data_dir / data_type / self.study_mode / self.subject_id
            if not base_dir.exists():
                continue
            for session_dir in base_dir.iterdir():
                if session_dir.is_dir() and session_dir.name.startswith(self.subject_id):
                    sessions.append({
                        "session_id": session_dir.name,
                        "data_type": data_type,
                        "path": session_dir,
                    })

        unique: Dict[str, dict] = {}
        for s in sessions:
            unique.setdefault(s["session_id"], s)
        return list(unique.values())

    def get_baseline_sessions(self) -> list:
        """Return all baseline sessions for the current subject."""
        return [s for s in self.list_all_subject_sessions() if "_baseline_" in s["session_id"]]

    def set_session_id(self, session_id: str) -> None:
        """Re-target this manager at an existing session so reprocessing writes to the same folder.

        Updates raw_dir, processed_dir, results_dir, and subject_dir to match
        session_id, and creates those directories if they do not yet exist.
        Useful when the pipeline is re-run after data was already collected.
        """
        self.session_id = session_id
        parts = session_id.split("_")
        self.timestamp = parts[-1] if len(parts) >= 3 else datetime.now().strftime("%Y%m%d_%H%M%S")

        self.raw_dir = self.data_dir / "raw" / self.study_mode / self.subject_id / self.session_id
        self.processed_dir = self.data_dir / "processed" / self.study_mode / self.subject_id / self.session_id
        self.results_dir = self.data_dir / "results" / self.study_mode / self.subject_id / self.session_id
        self.subject_dir = self.data_dir / self.study_mode / self.subject_id

        for d in (self.raw_dir, self.processed_dir, self.results_dir):
            d.mkdir(parents=True, exist_ok=True)

    def delete_processed_and_results(self, session_id: Optional[str] = None) -> None:
        """Delete the processed and results directories for a session.

        Targets session_id if provided, otherwise targets the current session.
        Errors during deletion are suppressed (ignore_errors=True) so partial
        cleanup does not abort the caller.  Use this when you need to re-run
        processing from raw data without manual folder cleanup.
        """
        target = session_id or self.session_id
        for sub in ("processed", "results"):
            p = self.data_dir / sub / self.study_mode / self.subject_id / target
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)

    def get_prompter_inputs_path(self) -> Path:
        """Return path to the directory where copied study-prompter input files are stored."""
        return self.raw_dir / "prompter_inputs"

    def save_prompter_inputs_manifest(
        self,
        video_paths: list,
        timestamps_path: Path,
        meta_path: Optional[Path],
        assembly_path: Optional[Path],
        session_offset_s: float,
        camera_offsets: Dict[int, float],
    ) -> Path:
        """Copy input files into raw/prompter_inputs/ and write a manifest JSON.

        Copies timestamps_path, meta_path, and assembly_path (when each exists)
        into the prompter_inputs subdirectory alongside the manifest.

        The manifest records:
          - original video file paths
          - number of cameras
          - timestamps CSV name
          - metadata JSON name
          - assembly CSV name
          - session_offset_s (recording_start_offset_s from the meta file)
          - per-camera audio-sync offsets

        Returns the path of the written manifest JSON.
        """
        inputs_dir = self.get_prompter_inputs_path()
        inputs_dir.mkdir(parents=True, exist_ok=True)

        original_video_paths = [str(Path(vp).resolve()) for vp in video_paths]

        if timestamps_path is not None and Path(timestamps_path).exists():
            shutil.copy2(timestamps_path, inputs_dir / Path(timestamps_path).name)

        if meta_path is not None and Path(meta_path).exists():
            shutil.copy2(meta_path, inputs_dir / Path(meta_path).name)

        if assembly_path is not None and Path(assembly_path).exists():
            shutil.copy2(assembly_path, inputs_dir / Path(assembly_path).name)

        manifest = {
            "session_id": self.session_id,
            "n_cameras": len(video_paths),
            "camera_video_paths": original_video_paths,
            "timestamps_csv": Path(timestamps_path).name if timestamps_path else None,
            "meta_json": Path(meta_path).name if meta_path else None,
            "assembly_csv": Path(assembly_path).name if assembly_path else None,
            "session_offset_s": session_offset_s,
            "camera_audio_offsets": {str(k): v for k, v in camera_offsets.items()},
        }
        manifest_path = inputs_dir / "prompter_inputs_manifest.json"
        save_json(manifest, manifest_path)
        return manifest_path

    def list_sessions_with_metadata(
        self,
        subject_id: Optional[str] = None,
        study_mode: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return enriched session metadata for every session found under data/raw.

        Scans data/raw/{mode}/{subject}/ for directories that contain a
        session_metadata.json file.  For each session the method loads:

          - session_metadata.json: session_id, session_label, created_at,
            pipeline_version
          - results/{session_id}/confidence_summary.json (if present):
            n_indications, indication_types, is_baseline_session
          - results/{session_id}/pipeline_summary.json (if present): profile
          - has_reference_data: True when processed/{session_id}/
            repetition_metrics.csv exists and is non-empty

        Results are sorted by created_at descending (newest first).
        Returns an empty list when the subject directory does not exist.
        """
        study_mode = study_mode or self.study_mode
        subject_id = subject_id or self.subject_id
        raw_subj_dir = self.data_dir / "raw" / study_mode / subject_id
        if not raw_subj_dir.exists():
            return []

        sessions: List[Dict[str, Any]] = []
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

            session_id = session_dir.name
            info: Dict[str, Any] = {
                "session_id": meta.get("session_id", session_id),
                "session_label": meta.get("session_label", ""),
                "created_at": meta.get("created_at", ""),
                "pipeline_version": meta.get("pipeline_version", ""),
                "n_indications": 0,
                "indication_types": [],
                "is_baseline_session": False,
                "profile": "",
                "has_reference_data": False,
            }

            conf_path = (
                self.data_dir / "results" / study_mode / subject_id
                / session_id / "confidence_summary.json"
            )
            if conf_path.exists():
                try:
                    conf = load_json(conf_path)
                    info["n_indications"] = conf.get("n_indications", 0)
                    info["indication_types"] = conf.get("indication_types", [])
                    info["is_baseline_session"] = conf.get("is_baseline_session", False)
                except Exception:
                    pass

            pip_path = (
                self.data_dir / "results" / study_mode / subject_id
                / session_id / "pipeline_summary.json"
            )
            if pip_path.exists():
                try:
                    pip = load_json(pip_path)
                    info["profile"] = pip.get("session_profile", pip.get("profile", ""))
                except Exception:
                    pass

            rep_path = (
                self.data_dir / "processed" / study_mode / subject_id
                / session_id / "repetition_metrics.csv"
            )
            if rep_path.exists():
                try:
                    df = pd.read_csv(rep_path, nrows=1)
                    info["has_reference_data"] = len(df) > 0
                except Exception:
                    pass

            sessions.append(info)

        sessions.sort(key=lambda s: s.get("created_at", ""), reverse=True)
        return sessions
