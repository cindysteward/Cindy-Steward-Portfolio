"""
Shared utilities for the FACET-CV pipeline.

Provides configuration I/O (YAML and JSON), session ID generation, logging
setup, data validation, numpy-compatible JSON serialisation, model download
management, and DataFrame column-filtering helpers used by every other module
in the pipeline.

The two main column-filter functions (get_feature_columns,
get_numeric_feature_columns) enforce a consistent exclusion list so that
metadata, private, and raw landmark coordinate columns are never accidentally
treated as features during baseline construction or anomaly scoring.
"""

import sys
import json
import yaml
import hashlib
import logging
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


MODEL_PATH = Path(__file__).parent.parent / "models" / "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

_FRAME_META_COLUMNS = frozenset({
    "frame_index", "timestamp_abs", "segment", "repetition",
    "detection_success", "detection_confidence", "time_rel_sec",
    "n_cameras_contributing",
    "task_name", "task_group", "task_id",
    "brightness", "occluded", "inter_ocular_distance", "psnr",
})


def setup_logging(
    log_dir: Path, session_id: str, level: int = logging.INFO
) -> logging.Logger:
    """Configure and return a logger named 'pipeline' that writes to both file and stdout.

    Creates log_dir if it does not exist, then opens a timestamped log file
    under it.  Both handlers share the same formatter and log level.

    Args:
        log_dir: Directory where the log file will be created.
        session_id: Used as a prefix in the log filename.
        level: Python logging level (default INFO).

    Returns:
        The configured Logger instance.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("pipeline")
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML file and return its contents as a dictionary.

    Raises FileNotFoundError if the path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """Write a dictionary to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a JSON file and return its contents as a dictionary."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """Write a dictionary to a JSON file with indented formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=json_serializer)


def json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for numpy types, datetime, and Path objects.

    Converts ndarray to list, numpy integer/float/bool to their Python
    equivalents, datetime to ISO string, and Path to str.  Raises TypeError
    for any other type so callers know when an unserializable object slips
    through.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def compute_config_hash(config_paths: List[Union[str, Path]]) -> str:
    """Compute a truncated SHA-256 hash over the contents of multiple config files.

    Files are sorted by path before hashing so the result is deterministic
    regardless of the order in which paths are supplied.  Missing files are
    silently skipped.  Returns the first 12 hex characters of the digest.
    """
    hasher = hashlib.sha256()
    for path in sorted(config_paths):
        path = Path(path)
        if path.exists():
            with open(path, "rb") as f:
                hasher.update(f.read())
    return hasher.hexdigest()[:12]


def get_pipeline_version() -> str:
    """Return the current pipeline version string."""
    return "1.0.0"


def validate_session_metadata(metadata: Dict[str, Any]) -> bool:
    """Validate that required session metadata fields are present and non-empty.

    Checks for subject_id, session_label, study_mode, and timestamp.
    Also verifies that study_mode is either 'pilot' or 'patient'.
    Raises ValueError with a descriptive message on the first failing check.
    Returns True when all checks pass.
    """
    for field in ("subject_id", "session_label", "study_mode", "timestamp"):
        if field not in metadata:
            raise ValueError(f"Missing required metadata field: {field}")
    if metadata["study_mode"] not in ("pilot", "patient"):
        raise ValueError(f"Invalid study_mode: {metadata['study_mode']}")
    return True


def create_session_id(subject_id: str, session_label: str) -> str:
    """Generate a timestamped session identifier in the format subject_label_YYYYMMDD_HHMMSS."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{subject_id}_{session_label}_{timestamp}"


def normalize_time_to_relative(
    timestamps: np.ndarray, start_time: Optional[float] = None
) -> np.ndarray:
    """Shift timestamps so the first (or given) value becomes zero."""
    if start_time is None:
        start_time = timestamps[0]
    return timestamps - start_time


def ensure_array(data: Union[List, np.ndarray]) -> np.ndarray:
    """Coerce a list to a numpy array; pass through existing arrays."""
    if isinstance(data, list):
        return np.array(data)
    return data


def safe_divide(
    numerator: np.ndarray, denominator: np.ndarray, default: float = 0.0
) -> np.ndarray:
    """Element-wise division that replaces any non-finite result with default.

    Handles division by zero and NaN propagation silently.  Returns an array
    of the same shape as the inputs.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.divide(numerator, denominator)
        result[~np.isfinite(result)] = default
    return result


def compute_statistics(data: np.ndarray) -> Dict[str, float]:
    """Compute descriptive statistics over a 1-D array.

    Returns a dict with keys: mean, std, median, q25, q75, min, max, n.
    All statistics use nan-safe numpy functions.  Returns zeros for all
    numeric keys and n=0 when the input array is empty.
    """
    if len(data) == 0:
        return {
            "mean": 0.0, "std": 0.0, "median": 0.0,
            "q25": 0.0, "q75": 0.0, "min": 0.0, "max": 0.0, "n": 0,
        }
    return {
        "mean": float(np.nanmean(data)),
        "std": float(np.nanstd(data)),
        "median": float(np.nanmedian(data)),
        "q25": float(np.nanpercentile(data, 25)),
        "q75": float(np.nanpercentile(data, 75)),
        "min": float(np.nanmin(data)),
        "max": float(np.nanmax(data)),
        "n": int(np.sum(~np.isnan(data))),
    }


def get_timestamp() -> str:
    """Return the current time as an ISO-formatted string."""
    return datetime.now().isoformat()


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as ``HH:MM:SS.ss`` or ``MM:SS.ss``."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"
    return f"{minutes:02d}:{secs:05.2f}"


def ensure_model_downloaded() -> Path:
    """Download the MediaPipe FaceLandmarker model if not already present.

    Uses a tiered SSL strategy:
      1. Default SSL context (works on most Linux / Windows).
      2. certifi bundle if installed (common macOS fix).
      3. System CA store via ssl.create_default_context() with purpose set.
      4. Unverified download as last resort with an explicit console warning.
    Streams the download in 64 KB chunks and prints a progress indicator.
    Raises RuntimeError if all strategies fail.
    """
    import ssl
    import urllib.error

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        return MODEL_PATH

    print(f"Downloading FaceLandmarker model → {MODEL_PATH} …")

    def _try_download(ctx: Optional[ssl.SSLContext]) -> bool:
        """Attempt to download the model file using the given SSL context; return True on success."""
        tmp = MODEL_PATH.with_suffix(".tmp")
        try:
            req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Mozilla/5.0"})
            opener_args = {"context": ctx} if ctx is not None else {}
            with urllib.request.urlopen(req, **opener_args, timeout=120) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                with open(tmp, "wb") as fh:
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        fh.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded / total * 100
                            print(f"\r  {pct:5.1f}%  ({downloaded // 1024} / {total // 1024} KB)", end="", flush=True)
            tmp.replace(MODEL_PATH)
            print("\nModel downloaded successfully.")
            return True
        except Exception as exc:
            if tmp.exists():
                tmp.unlink()
            print(f"\n  [warn] Download attempt failed: {exc}")
            return False

    if _try_download(None):
        return MODEL_PATH

    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
        if _try_download(ctx):
            return MODEL_PATH
    except ImportError:
        pass

    try:
        ctx = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH)
        ctx.load_default_certs()
        if _try_download(ctx):
            return MODEL_PATH
    except Exception:
        pass

    print("\n[WARN] All verified SSL strategies failed. Attempting unverified download.")
    print("       To fix permanently: pip install certifi   OR   run /Applications/Python*/Install\\ Certificates.command (macOS)")
    ctx_unverified = ssl._create_unverified_context()
    if _try_download(ctx_unverified):
        return MODEL_PATH

    raise RuntimeError(
        f"Failed to download FaceLandmarker model from {MODEL_URL}.\n"
        "Check your network connection or download the file manually and place it at:\n"
        f"  {MODEL_PATH}"
    )


def get_feature_columns(
    df: pd.DataFrame, extra_excludes: frozenset = frozenset()
) -> List[str]:
    """Return all non-metadata, non-private columns from df.

    Useful when every remaining column (numeric or otherwise) should be treated
    as a feature, for example during baseline construction.

    Raw landmark position columns (ending in _x, _y, or _z) are excluded
    because they are normalised image coordinates, not blendshape features.
    Z-scoring them would corrupt the values used by HeadPoseNormalizer.

    Args:
        df: Input DataFrame.
        extra_excludes: Additional column names to exclude beyond the default
            metadata set.

    Returns:
        List of column name strings.
    """
    excludes = _FRAME_META_COLUMNS | set(extra_excludes)
    return [
        c for c in df.columns
        if c not in excludes
        and not c.startswith("_")
        and not any(c.endswith(s) for s in ("_x", "_y", "_z"))
    ]


def get_numeric_feature_columns(
    df: pd.DataFrame, extra_excludes: frozenset = frozenset()
) -> List[str]:
    """Return numeric, non-metadata feature columns from df.

    Equivalent to get_feature_columns but restricted to dtype numeric.  The
    base metadata exclusion set is extended by extra_excludes so callers can
    tailor the filter for their specific DataFrame schema.

    Raw landmark position columns (ending in _x, _y, or _z) are excluded for
    the same reason as in get_feature_columns.

    Args:
        df: Input DataFrame.
        extra_excludes: Additional column names to exclude beyond the default
            metadata set.

    Returns:
        List of numeric column name strings.
    """
    excludes = _FRAME_META_COLUMNS | set(extra_excludes)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return [
        c for c in numeric_cols
        if c not in excludes
        and not c.startswith("_")
        and not any(c.endswith(s) for s in ("_x", "_y", "_z"))
    ]


def resolve_dominant_task(frame_data: List[Dict]) -> Tuple[str, int]:
    """Determine the most common task_group and task_id from frame-level data.

    Uses a Counter majority-vote over all non-null values in the list.
    Returns ('0', 0) when the list is empty or contains no task annotations.
    """
    task_group: str = "0"
    task_id: int = 0
    if not frame_data:
        return task_group, task_id

    tg_vals = [f.get("task_group") for f in frame_data if f.get("task_group")]
    tid_vals = [f.get("task_id") for f in frame_data if f.get("task_id")]

    if tg_vals:
        task_group = Counter(tg_vals).most_common(1)[0][0] or "0"
    if tid_vals:
        counts = Counter(tid_vals).most_common(1)
        task_id = counts[0][0] if counts else 0

    return task_group, task_id


def sanitize_events_df(events_df: pd.DataFrame) -> pd.DataFrame:
    """Enforce correct dtypes on an events DataFrame that came from CSV parsing.

    Coerces timestamp_abs to float and drops any rows where that conversion
    fails.  Enforces int on task_id and repetition (filling NaN with 0 and 1
    respectively) and str on task_group and task_name.

    Call this at every function boundary that receives an events_df to avoid
    dtype surprises from mixed-type CSV reads.

    Returns an empty DataFrame with canonical columns when the input is None
    or has zero rows.
    """
    if events_df is None or len(events_df) == 0:
        return pd.DataFrame(columns=[
            "timestamp_abs", "event_type", "task_group",
            "task_id", "task_name", "repetition",
        ])
    df = events_df.copy()
    df["timestamp_abs"] = pd.to_numeric(df["timestamp_abs"], errors="coerce")
    df = df.dropna(subset=["timestamp_abs"]).reset_index(drop=True)
    if "task_id" in df.columns:
        df["task_id"] = pd.to_numeric(df["task_id"], errors="coerce").fillna(0).astype(int)
    if "task_group" in df.columns:
        df["task_group"] = df["task_group"].fillna("0").astype(str)
    if "task_name" in df.columns:
        df["task_name"] = df["task_name"].fillna("").astype(str)
    if "repetition" in df.columns:
        df["repetition"] = (
            pd.to_numeric(df["repetition"], errors="coerce").fillna(1).astype(int)
        )
    return df
