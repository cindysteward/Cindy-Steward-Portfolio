"""
Intake helper for study-prompter session files.

Scans a source folder (default: the current directory) for study-prompter
output files, groups them by participant ID and session timestamp, and moves
each group into a structured destination folder ready for pipeline processing.

Usage:
    python intake.py                          # scan current folder
    python intake.py --src ~/Downloads        # scan a specific folder
    python intake.py --src ~/Downloads --dst ~/study_data/raw_sessions
    python intake.py --src ~/Downloads --dry-run   # preview without moving
    python intake.py --src ~/Downloads --audio-shift-ms 80
                                           apply a fixed 80 ms audio delay to cam files

Output structure (one folder per session):
    <dst>/<participant_id>/<participant_id>_<timestamp>/
        <participant_id>_cam1_<timestamp>.webm
        <participant_id>_cam2_<timestamp>.webm   (if present)
        <participant_id>_screen_<timestamp>.webm (if present)
        <participant_id>_timestamps_<timestamp>.csv
        <participant_id>_assembly_<timestamp>.csv (if present)
        <participant_id>_recording_meta_<timestamp>.json

After running, the script prints the full pipeline command for each session.
"""

import argparse
import json
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


_PATTERN = re.compile(
    r'^(?P<pid>.+?)_(?P<type>cam\d*|screen|timestamps|assembly|recording_meta)'
    r'_(?P<ts>\d{4}-\d{2}-\d{2}_\d{6})\.'
    r'(?P<ext>webm|mp4|csv|json)$'
)


def _check_ffmpeg() -> bool:
    """Return True if ffmpeg is available on PATH."""
    return shutil.which("ffmpeg") is not None


def fix_session_videos(
    session_dir: Path, dry_run: bool, extra_audio_delay_ms: int = 0
) -> None:
    """Apply ffmpeg audio corrections to every video file in session_dir.

    Runs two filters in sequence:
    - adelay=N:all=1  (only when extra_audio_delay_ms != 0): shifts audio by a
      fixed number of milliseconds to correct a constant lead/lag offset.
    - aresample=async=1: dynamically resamples audio to match the video
      container timestamps, correcting clock-rate drift that accumulates over
      the recording duration.
    The video stream is copied as-is; only the audio stream is re-encoded.
    """
    if not session_dir.exists():
        if dry_run:
            print(f"    [dry-run] skipping video fix — session dir not yet created")
        return

    cam_files = sorted(
        f for f in session_dir.iterdir()
        if f.is_file()
        and f.suffix.lower() in {".webm", ".mp4"}
        and re.search(r'_cam\d*_', f.name)
    )
    other_files = sorted(
        f for f in session_dir.iterdir()
        if f.is_file()
        and f.suffix.lower() in {".webm", ".mp4"}
        and not re.search(r'_cam\d*_', f.name)
    )

    cam_filter = (
        f"adelay={extra_audio_delay_ms}:all=1,aresample=async=1"
        if extra_audio_delay_ms
        else "aresample=async=1"
    )
    other_filter = "aresample=async=1"

    def _apply_filter(vf: Path, audio_filter: str) -> None:
        if dry_run:
            print(f"    [dry-run] ffmpeg {audio_filter}: {vf.name}")
            return
        tmp = vf.with_suffix(".aresample_tmp" + vf.suffix)
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
            "-i", str(vf), "-c:v", "copy", "-af", audio_filter, str(tmp),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            if tmp.exists():
                tmp.unlink()
            print(f"    [warn] ffmpeg fix failed for {vf.name}: {result.stderr[:300]}")
        else:
            tmp.replace(vf)
            print(f"    [sync-fix] {vf.name}")

    for vf in cam_files:
        _apply_filter(vf, cam_filter)
    for vf in other_files:
        _apply_filter(vf, other_filter)


def _read_session_meta(session_dir: Path) -> Optional[dict]:
    """Read the recording_meta JSON (if present) and print useful fields.

    Returns the parsed JSON dict or None if no meta file is found.
    """
    meta_files = list(session_dir.glob("*_recording_meta_*.json"))
    if not meta_files:
        return None
    try:
        with meta_files[0].open("r", encoding="utf8") as fh:
            data = json.load(fh)
    except Exception:
        print(f"  [warn] failed to read recording meta: {meta_files[0].name}")
        return None
    # Print any audio delay compensation metadata we care about
    adc = data.get("audio_delay_compensation_s")
    base_lat = data.get("audio_ctx_base_latency_s")
    if adc is not None:
        print(f"  recording_meta: audio_delay_compensation_s = {adc}")
    if base_lat is not None:
        print(f"  recording_meta: audio_ctx_base_latency_s = {base_lat}")
    return data


def scan_folder(src: Path) -> Dict[Tuple[str, str], Dict[str, List[Path]]]:
    """Scan src for study-prompter files and group them by (pid, timestamp).

    Returns a dict mapping (pid, timestamp) to a dict of file-type to list of
    paths.  File types are: 'cam', 'screen', 'timestamps', 'assembly',
    'recording_meta'.  Only files at the top level of src are examined
    (non-recursive).
    """
    groups: Dict[Tuple[str, str], Dict[str, List[Path]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for f in src.iterdir():
        if not f.is_file():
            continue
        m = _PATTERN.match(f.name)
        if m is None:
            continue
        pid = m.group("pid")
        file_type = m.group("type")
        ts = m.group("ts")
        category = "cam" if re.match(r"^cam\d*$", file_type) else file_type
        groups[(pid, ts)][category].append(f)
    return {k: dict(v) for k, v in groups.items()}


def move_group(
    pid: str,
    ts: str,
    files: Dict[str, List[Path]],
    dst: Path,
    dry_run: bool,
) -> Optional[Path]:
    """Move all files for one session group into dst/<pid>/<pid>_<ts>/.

    Returns the destination session directory path, or None if no files were
    moved.  In dry_run mode, prints what would happen without moving anything.
    """
    session_dir = dst / pid / f"{pid}_{ts}"
    all_files: List[Path] = []
    for file_list in files.values():
        all_files.extend(file_list)
    if not all_files:
        return None
    if dry_run:
        print(f"  [dry-run] Would create: {session_dir}")
        for f in sorted(all_files, key=lambda p: p.name):
            print(f"    {f.name}")
        return session_dir
    try:
        session_dir.mkdir(parents=True, exist_ok=True)
        for f in all_files:
            shutil.move(str(f), session_dir / f.name)
        return session_dir
    except OSError as e:
        print(f"  [error] Failed to create/move session files: {e}")
        return None


def print_pipeline_command(pid: str, ts: str, session_dir: Path) -> None:
    """Print the run_pipeline.py command for the session, using the actual
    filenames found in session_dir.

    Uses --prompter-videos for all cam files, --prompter-timestamps for the
    timestamps CSV, --prompter-assembly for the assembly CSV if present, and
    --prompter-meta for the recording meta JSON if present.  The --subject is
    the pid and --session is left as <SESSION_LABEL> for the researcher to
    fill in.
    """
    cam_files = sorted(session_dir.glob(f"{pid}_cam*_{ts}.*"))
    timestamps_files = list(session_dir.glob(f"{pid}_timestamps_{ts}.csv"))
    assembly_files = list(session_dir.glob(f"{pid}_assembly_{ts}.csv"))
    meta_files = list(session_dir.glob(f"{pid}_recording_meta_{ts}.json"))

    parts = ["python run_pipeline.py"]
    parts.append(f"  --subject {pid}")
    parts.append("  --session <SESSION_LABEL>")
    if timestamps_files:
        parts.append(f"  --prompter-timestamps \"{timestamps_files[0]}\"")
    if cam_files:
        videos_arg = " ".join(f'"{f}"' for f in cam_files)
        parts.append(f"  --prompter-videos {videos_arg}")
    if assembly_files:
        parts.append(f"  --prompter-assembly \"{assembly_files[0]}\"")
    if meta_files:
        parts.append(f"  --prompter-meta \"{meta_files[0]}\"")

    print("\n".join(parts))


def main() -> None:
    """Parse arguments, scan, group, move, and print pipeline commands."""
    parser = argparse.ArgumentParser(
        description="Group and move study-prompter session files into a structured folder."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("."),
        help="Folder to scan for session files (default: current directory).",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=None,
        help=(
            "Destination root folder.  Defaults to a 'sessions' subfolder "
            "inside --src."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without moving any files.",
    )
    parser.add_argument(
        "--audio-shift-ms",
        type=int,
        default=0,
        help=("Optional initial audio shift (milliseconds) to apply via ffmpeg adelay "
              "before running aresample. Useful to correct constant audio lead/lag."),
    )
    args = parser.parse_args()

    src: Path = args.src.expanduser().resolve()
    dst: Path = (args.dst.expanduser().resolve() if args.dst else src / "sessions")

    if not src.is_dir():
        print(f"Error: source folder does not exist: {src}")
        raise SystemExit(1)

    groups = scan_folder(src)

    ffmpeg_available = _check_ffmpeg()
    if not ffmpeg_available:
        print("[warn] ffmpeg not found on PATH — audio drift fix will be skipped.")
        print("       Install ffmpeg (e.g. brew install ffmpeg) and re-run intake to apply.")

    unmatched: List[Path] = []
    for f in src.iterdir():
        if f.is_file() and _PATTERN.match(f.name) is None:
            unmatched.append(f)

    total_files_moved = 0
    sessions_processed = 0

    for (pid, ts), files in sorted(groups.items()):
        file_count = sum(len(v) for v in files.values())
        print(f"\nSession  pid={pid}  ts={ts}  ({file_count} file(s))")
        session_dir = move_group(pid, ts, files, dst, args.dry_run)
        if session_dir is not None and (args.dry_run or session_dir.exists()):
            sessions_processed += 1
            total_files_moved += file_count
            print(f"  -> {session_dir}")
            if ffmpeg_available and not args.dry_run:
                meta = _read_session_meta(session_dir)
                shift_ms = args.audio_shift_ms
                fix_session_videos(session_dir, args.dry_run, shift_ms)
            print_pipeline_command(pid, ts, session_dir)

    print(f"\n{'='*60}")
    print(f"Sessions found : {len(groups)}")
    print(f"Sessions processed : {sessions_processed}")
    print(f"Files moved    : {total_files_moved}")
    if unmatched:
        print(f"Unmatched files ({len(unmatched)} skipped):")
        for f in sorted(unmatched, key=lambda p: p.name):
            print(f"  {f.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
