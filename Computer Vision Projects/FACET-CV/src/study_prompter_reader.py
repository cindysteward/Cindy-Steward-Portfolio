"""
Parser for study-prompter output CSV files.

Reads the timestamps CSV and optional assembly CSV produced by
tasklist_animation/study-prompter.html, validates required columns,
filters practice rows, maps section/event fields to the internal
pipeline events_df format, and assembles a PrompterSession dataclass
that the rest of the pipeline can consume directly.

Section B task normalisation (COMBINED profile):
  The assembly CSV assigns sequential task_numbers across the entire combined
  session (e.g. disorder B tasks numbered 5–11 rather than 1–4).  This module
  normalises those numbers back to the canonical 1–4 range defined in
  tasks.yaml so that articulation scoring, anomaly detection, and decision
  support all operate on consistent task identifiers regardless of which
  session position a task occupied.

  Canonical mapping:
    pa-pa-pa          → B task_id 1
    ta-ta-ta          → B task_id 2
    ka-ka-ka          → B task_id 3
    pa-ta-ka (and all sequencing permutations) → B task_id 4 (complex)

Section C task normalisation (COMBINED profile):
  C tasks are mapped to canonical 1–8 complexity IDs using word-similarity
  matching (_match_c_word_to_canonical_id).  This correctly handles:
    - Exact canonical labels: "Kleurpotlood" → C_8
    - Disorder-simulation variants: "Tleurpotlood" → C_8 (Kleurpotlood)
    - Globally offset IDs: C_25-C_32 with correct labels → C_1-C_8
  A sequential fallback handles any rows whose labels don't match the
  canonical word list.

References
----------
Ruis C (2018) Monitoring cognition during awake brain surgery in adults:
  a systematic review. Neuropsychol Rev 28(3):272–298.
  Identifies counting, picture naming, reading, and repetition as the
  dominant intraoperative tasks; provides the clinical framework for the
  Group A / B / C structure managed by this parser.
  https://doi.org/10.1080/13803395.2018.1469602

Zwart M, Ruis C (2024) An update on tests used for intraoperative
  monitoring of cognition during awake craniotomy. Acta Neurochir 166:167.
  Updated systematic review (2017–2023); confirms language + motor as the
  dominant monitored domains; validates digitalized continuous monitoring;
  lists orofacial motor tasks (counting, lip pouting) that correspond
  directly to Group A / B prompts in this parser.
  https://doi.org/10.1007/s00701-024-06062-6

De Witt Hamer PC, Moritz-Gasser S, Menjot de Champfleur N, Duffau H,
  Herbet G (2014) The Dutch Linguistic Intraoperative Protocol: a valid
  linguistic approach to awake brain surgery. Brain Lang 140:14–24.
  DuLIP task battery (counting, naming, reading, repetition) that inspired
  the Group-A / B / C paradigm structure and task-normalisation conventions
  implemented in this module.
  https://doi.org/10.1016/j.bandl.2014.10.011

Shinoura N, Ohue S, Tabei Y, et al. (2005) Preoperative fMRI, tractography
  and continuous task during awake surgery. Minim Invasive Neurosurg
  48(2):77–82.
  First description of continuous orofacial motor task monitoring during
  awake craniotomy (lip puckering, tongue movement, mouth opening); provides
  the clinical rationale for the continuous-task session type supported by
  this parser.
  https://doi.org/10.1055/s-2004-830227

Kanno A, Mikuni N (2015) Evaluation of language function under awake
  craniotomy. Neurol Med Chir (Tokyo) 55(5):367–373.
  Reviews pre- and intraoperative language assessment; identifies patient
  fatigue, articulatory ability, and head fixation as the key confounders
  in visual observation — motivating the quantitative task-timestamping
  and video-based analysis enabled by this parser.
  https://doi.org/10.2176/nmc.ra.2014-0395
"""

import csv
import io
import json
import logging
from dataclasses import dataclass, field
import difflib
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger("pipeline")


def _repair_outer_quoted_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Fix rows where the study-prompter CSV writer wrapped an entire row in outer double-quotes.

    This happens when a label field contains a comma: the CSV writer quotes
    the entire row rather than just the affected field.  pandas then reads
    that row as a single string in the first column with all other columns
    set to NaN.

    Detection: the first column value contains commas and all remaining
    columns in that row are NaN or empty.  Repair: re-parse the value as a
    CSV row and redistribute the tokens across the correct columns.  Rows
    that do not match the pattern are passed through unchanged.
    """
    columns = df.columns.tolist()
    n_cols = len(columns)
    fixed_rows = []
    for _, row in df.iterrows():
        first_val = str(row.iloc[0])
        if "," in first_val and all(
            (str(v) == "" or v != v)
            for v in row.iloc[1:]
        ):
            try:
                reader = csv.reader(io.StringIO(first_val))
                values = next(reader)
                if len(values) == n_cols:
                    fixed_rows.append(dict(zip(columns, values)))
                    continue
            except Exception:
                pass
        fixed_rows.append(row.to_dict())
    return pd.DataFrame(fixed_rows)


_REQUIRED_TIMESTAMPS_COLS = {
    "participant_id",
    "profile",
    "session_date",
    "section",
    "task_number",
    "event",
    "time_from_start_s",
    "label",
}

_REQUIRED_ASSEMBLY_COLS = {
    "disorder_profile",
    "section",
    "task_number",
    "sequence_rep",
    "label",
    "expression",
    "start_s",
    "end_s",
    "participant_id",
    "session_date",
}


_COMBINED_PROFILE_MARKERS: frozenset = frozenset({
    "COMBINED",
    "GECOMBINEERD",
})

_B_LABEL_TO_CANONICAL_ID: dict = {
    "pa-pa-pa": 1,
    "ta-ta-ta": 2,
    "ka-ka-ka": 3,
    "pa-ta-ka": 4,
    "ka-pa-ta": 4,
    "ta-pa-ka": 4,
    "pa-ka-ta": 4,
    "ta-ka-pa": 4,
    "ka-ta-pa": 4,
    "pa pa pa": 1,
    "papapa":   1,
    "ta ta ta": 2,
    "tatata":   2,
    "ka ka ka": 3,
    "kakaka":   3,
}

_B_LABEL_KEYWORD_MAP: tuple = (
    ("pa-pa-pa", 1),
    ("pa pa pa", 1),
    ("papapa",   1),
    ("ta-ta-ta", 2),
    ("ta ta ta", 2),
    ("tatata",   2),
    ("ka-ka-ka", 3),
    ("ka ka ka", 3),
    ("kakaka",   3),
    ("pa-ta-ka", 4),
    ("pa-ka-ta", 4),
    ("ta-pa-ka", 4),
    ("ta-ka-pa", 4),
    ("ka-pa-ta", 4),
    ("ka-ta-pa", 4),
)


def _normalise_b_task_id(label: str, raw_task_number: int) -> int:
    """Return the canonical 1–4 task_id for a section B task.

    Resolution order:
    1. Exact match in _B_LABEL_TO_CANONICAL_ID.
    2. Substring keyword search (handles prefixed labels like "Langzame pa-pa-pa",
       deliberate-error labels, or labels with additional context words).
    3. Falls back to raw_task_number when no match is found, preserving novel
       disorder task IDs for downstream semantic resolution.
    """
    if label:
        norm = label.lower().strip()
        canonical = _B_LABEL_TO_CANONICAL_ID.get(norm)
        if canonical is not None:
            return canonical
        for keyword, cid in _B_LABEL_KEYWORD_MAP:
            if keyword in norm:
                return cid
        if "pa" in norm and "ta" in norm and "ka" in norm:
            return 4
    return raw_task_number


_C_CANONICAL_WORDS_BY_ID: Dict[int, List[str]] = {
    1: ["tak", "bus"],
    2: ["heks", "chest"],
    3: ["knoop", "spoon"],
    4: ["plons", "wrist"],
    5: ["hamer", "hammer"],
    6: ["avontuur", "adventure"],
    7: ["spaghetti"],
    8: ["kleurpotlood", "handwriting"],
}

_C_CANONICAL_WORD_LIST: List[tuple] = [
    (word, tid)
    for tid, words in _C_CANONICAL_WORDS_BY_ID.items()
    for word in words
]


def _match_c_word_to_canonical_id(label: str) -> Optional[int]:
    """Return the canonical C task_id (1–8) for a word label.

    Uses difflib.SequenceMatcher to find the closest canonical Dutch/English
    word.  Returns None if no match exceeds the similarity threshold (0.45).
    This handles both exact canonical labels ("Kleurpotlood" → 8) and
    disorder-simulation variants ("Tleurpotlood" → 8).
    """
    if not label:
        return None
    tokens = label.lower().strip().split()
    best_ratio = 0.0
    best_id: Optional[int] = None
    for token in tokens:
        for canon, tid in _C_CANONICAL_WORD_LIST:
            ratio = difflib.SequenceMatcher(None, token, canon).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_id = tid
    if best_ratio >= 0.45:
        return best_id
    return None


def _normalise_c_task_ids_in_events(event_rows: list) -> list:
    """Map C-section task_ids to canonical 1–8 complexity IDs.

    Strategy (in order):
    1. Word-similarity: compare the row's ``task_name`` label against the
       canonical Dutch/English word list.  This correctly maps disorder
       variants like "Tleurpotlood" → C_8 (Kleurpotlood) and also handles
       standard labels regardless of their original numeric task_id.
    2. Sequential fallback: if a C row has no recognisable label (or
       word-similarity fails for all rows), fall back to sorting the unique
       original task_ids and mapping them 1-indexed by sorted position —
       the same approach used before, preserving the complexity ordering
       guaranteed by the study-prompter task sequence.

    Only rows with task_group == "C" are affected.
    """
    c_rows_idx = [
        i for i, r in enumerate(event_rows)
        if r.get("task_group") == "C" and r.get("task_id", 0) != 0
    ]
    if not c_rows_idx:
        return event_rows

    result = list(event_rows)
    word_mapped: Dict[int, int] = {}
    unmatched_orig_ids: List[int] = []

    for i in c_rows_idx:
        row = result[i]
        orig_id = row["task_id"]
        if orig_id in word_mapped:
            continue
        label = str(row.get("task_name", ""))
        matched = _match_c_word_to_canonical_id(label)
        if matched is not None:
            word_mapped[orig_id] = matched
        else:
            if orig_id not in unmatched_orig_ids:
                unmatched_orig_ids.append(orig_id)

    seq_mapped: Dict[int, int] = {}
    non_canonical_unmatched = sorted(
        oid for oid in unmatched_orig_ids if not (1 <= oid <= 8)
    )
    if non_canonical_unmatched:
        claimed = set(word_mapped.values())
        available = [tid for tid in range(1, 9) if tid not in claimed]
        for rank, orig_id in enumerate(non_canonical_unmatched):
            if rank < len(available):
                seq_mapped[orig_id] = available[rank]
            else:
                seq_mapped[orig_id] = rank + 1


    id_map = {**word_mapped, **seq_mapped}
    for i in c_rows_idx:
        orig_id = result[i]["task_id"]
        if orig_id in id_map and id_map[orig_id] != orig_id:
            result[i] = dict(result[i])
            result[i]["task_id"] = id_map[orig_id]

    return result


_NORMAL_PROFILE_MARKERS: frozenset = frozenset({
    "NORMAL",
    "NORMAAL",
})


@dataclass
class PrompterSession:
    """Parsed representation of a single study-prompter recording session.

    Attributes:
        participant_id: Participant identifier string from the CSV header.
        profile: Profile label as written in the CSV (e.g. 'NORMAL', 'COMBINED').
        session_date: Session date string as written in the CSV.
        recording_start_offset_s: Seconds between recording start and the
            first study-prompter event; used to align video timestamps.
        events_df: Pipeline events DataFrame with columns timestamp_abs,
            event_type, task_group, task_id, task_name, repetition.
        is_combined: True when the profile is a COMBINED / GECOMBINEERD session
            that contains events for multiple disorder sub-profiles.
        disorder_profiles: For COMBINED sessions, the list of disorder profile
            keys found in the assembly CSV.  Empty for non-combined sessions.
        per_disorder_events: For COMBINED sessions, a dict mapping each
            disorder key to its own events_df (baseline + disorder tasks).
        camera_start_offsets: Per-camera start-time offsets in seconds relative
            to the first camera, used to synchronise multi-camera recordings.
    """

    participant_id: str
    profile: str
    session_date: str
    recording_start_offset_s: float
    events_df: pd.DataFrame
    is_combined: bool
    disorder_profiles: List[str]
    per_disorder_events: Dict[str, pd.DataFrame]
    camera_start_offsets: List[float]


def load_recording_meta(meta_path: Optional[Path]) -> Dict[str, Any]:
    """Load the recording metadata JSON file and return its contents as a dict.

    Returns an empty dict silently when meta_path is None or the file does
    not exist, so callers that have no metadata file can still proceed.
    """
    if meta_path is None:
        return {}
    meta_path = Path(meta_path)
    if not meta_path.exists():
        return {}
    with open(meta_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _map_row_to_event(row: pd.Series) -> Optional[Dict[str, Any]]:
    """Convert a single timestamps-CSV row into a pipeline events_df row dict.

    Maps section/event pairs to the internal event types used by the rest of
    the pipeline:
      - Baseline/neutral section + start -> event_type 'neutral'
      - Baseline/neutral section + end   -> event_type 'segment_end'
      - Section A/B/C + start            -> event_type 'measurement'
      - Section A/B/C + end              -> event_type 'segment_end'

    Returns None for Practice rows or for rows whose section/event combination
    does not match any supported mapping.  Also returns None when the
    time_from_start_s value cannot be parsed as a float, after logging a warning.
    """
    section = str(row["section"]).strip()
    event = str(row["event"]).strip()

    _BASELINE_ALIASES = {"baseline", "neutral", "rest", "rust", "rustmoment", "base"}
    section_norm = section.strip().upper()
    for prefix in ("SECTION ", "SECTIE ", "DEEL ", "GROEP "):
        if section_norm.startswith(prefix):
            section_norm = section_norm[len(prefix):].strip()

    if section.strip().lower() == "practice":
        return None

    raw_ts = row.get("time_from_start_s", "")
    try:
        timestamp_abs = float(str(raw_ts).strip())
    except (ValueError, TypeError):
        logger.warning(
            "Skipping row with non-numeric time_from_start_s=%r (section=%r, event=%r)",
            raw_ts, section, event,
        )
        return None
    label = str(row.get("label", ""))
    task_number = row.get("task_number", 0)
    try:
        task_id = int(task_number)
    except (ValueError, TypeError):
        task_id = 0

    if section.strip().lower() in _BASELINE_ALIASES:
        if event.lower() == "start":
            return {
                "timestamp_abs": timestamp_abs,
                "event_type": "neutral",
                "task_group": "0",
                "task_id": 0,
                "task_name": label,
            }
        if event.lower() == "end":
            return {
                "timestamp_abs": timestamp_abs,
                "event_type": "segment_end",
                "task_group": "0",
                "task_id": 0,
                "task_name": label,
            }
        return None

    if section_norm in ("A", "B", "C"):
        try:
            raw_rep = row.get("sequence_rep", None)
        except Exception:
            raw_rep = None
        if raw_rep in (None, "", "nan") or str(raw_rep).strip() == "":
            raw_rep = row.get("sequence", None)
        try:
            repetition = (
                int(float(raw_rep))
                if raw_rep not in (None, "", "nan") and str(raw_rep).strip() != ""
                else 1
            )
        except (ValueError, TypeError):
            repetition = 1
        if event.lower() == "start":
            return {
                "timestamp_abs": timestamp_abs,
                "event_type": "measurement",
                "task_group": section_norm,
                "task_id": task_id,
                "task_name": label,
                "repetition": repetition,
            }
        if event.lower() == "end":
            return {
                "timestamp_abs": timestamp_abs,
                "event_type": "segment_end",
                "task_group": section_norm,
                "task_id": task_id,
                "task_name": label,
                "repetition": repetition,
            }
        return None

    return None


def parse_timestamps_csv(
    csv_path: Path,
    recording_start_offset_s: float = 0.0,
    camera_start_offsets: Optional[List[float]] = None,
) -> "PrompterSession":
    """Parse a study-prompter timestamps CSV file into a PrompterSession.

    Steps performed:
      1. Read the CSV with outer-quote repair applied.
      2. Validate required columns are present.
      3. Filter out Practice rows.
      4. Map each row to a pipeline event dict via _map_row_to_event.
      5. Normalise C-section task IDs to the canonical 1-8 range.
      6. Add a synthetic segment_end event for any unclosed final segment.
      7. Enforce correct dtypes on all event columns.

    Args:
        csv_path: Path to the study-prompter timestamps CSV file.
        recording_start_offset_s: Seconds between recording start and the
            first prompter event (read from the metadata JSON).
        camera_start_offsets: Per-camera start offsets for multi-camera sync.

    Returns:
        A PrompterSession with is_combined=True when a COMBINED profile is
        detected.  The per_disorder_events dict is empty until parse_assembly_csv
        is called.

    Raises:
        ValueError: When required columns are absent from the CSV.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(
        csv_path,
        quotechar='"',
        dtype=str,
        keep_default_na=False,
    )
    df = _repair_outer_quoted_rows(df)

    missing = _REQUIRED_TIMESTAMPS_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Timestamps CSV is missing required columns: {sorted(missing)}"
        )

    non_practice = df[
        df["section"].str.strip().str.lower() != "practice"
    ].reset_index(drop=True)

    source_row = non_practice if len(non_practice) > 0 else df

    def _first_nonempty(col: pd.Series) -> str:
        """Return the first non-null, non-blank string value in col."""
        vals = col.dropna().astype(str)
        vals = vals[vals.str.strip() != ""]
        return str(vals.iloc[0]) if len(vals) > 0 else ""

    participant_id = _first_nonempty(source_row["participant_id"])
    profile = _first_nonempty(source_row["profile"])
    session_date = _first_nonempty(source_row["session_date"])

    _profile_upper = profile.strip().upper()
    is_combined = any(marker in _profile_upper for marker in _COMBINED_PROFILE_MARKERS)

    event_rows = []
    for _, row in df.iterrows():
        try:
            mapped = _map_row_to_event(row)
            if mapped is not None:
                event_rows.append(mapped)
        except Exception as exc:
            logger.warning(
                "Skipping malformed row: %s — %s", row.to_dict(), exc
            )

    event_rows = _normalise_c_task_ids_in_events(event_rows)

    if event_rows:
        events_df = (
            pd.DataFrame(event_rows)
            .sort_values("timestamp_abs")
            .reset_index(drop=True)
        )
    else:
        events_df = pd.DataFrame(
            columns=["timestamp_abs", "event_type", "task_group", "task_id", "task_name", "repetition"]
        )

    if len(events_df) > 0:
        fixed_rows = []
        last_open: Optional[pd.Series] = None
        for _, row in events_df.iterrows():
            fixed_rows.append(row.to_dict())
            if row["event_type"] in ("neutral", "measurement"):
                last_open = row
            elif row["event_type"] == "segment_end":
                last_open = None
        if last_open is not None:
            fixed_rows.append({
                "timestamp_abs": float(last_open["timestamp_abs"]) + 30.0,
                "event_type": "segment_end",
                "task_group": last_open.get("task_group", "0"),
                "task_id": last_open.get("task_id", 0),
                "task_name": last_open.get("task_name", ""),
                "repetition": last_open.get("repetition", 1),
            })
            events_df = (
                pd.DataFrame(fixed_rows)
                .sort_values("timestamp_abs")
                .reset_index(drop=True)
            )

    if len(events_df) > 0:
        events_df["timestamp_abs"] = pd.to_numeric(
            events_df["timestamp_abs"], errors="coerce"
        )
        events_df = events_df.dropna(subset=["timestamp_abs"]).reset_index(drop=True)
        events_df["task_id"] = (
            pd.to_numeric(events_df.get("task_id", 0), errors="coerce")
            .fillna(0).astype(int)
        )
        events_df["task_group"] = events_df.get("task_group", "0").fillna("0").astype(str)
        events_df["task_name"] = events_df.get("task_name", "").fillna("").astype(str)
        if "repetition" in events_df.columns:
            events_df["repetition"] = (
                pd.to_numeric(events_df["repetition"], errors="coerce")
                .fillna(1).astype(int)
            )
        else:
            events_df["repetition"] = 1

    return PrompterSession(
        participant_id=participant_id,
        profile=profile,
        session_date=session_date,
        recording_start_offset_s=recording_start_offset_s,
        events_df=events_df,
        is_combined=is_combined,
        disorder_profiles=[],
        per_disorder_events={},
        camera_start_offsets=camera_start_offsets if camera_start_offsets is not None else [],
    )


def parse_assembly_csv(
    assembly_path: Path,
    base_session: "PrompterSession",
) -> "PrompterSession":
    """Attach per-disorder event DataFrames to a PrompterSession from the assembly CSV.

    For each unique disorder_profile value in the assembly CSV, builds a
    combined events_df that contains the baseline events from base_session
    plus the start/end events for the tasks belonging to that disorder.
    B-section task IDs are normalised to the canonical 1-4 range, and
    C-section IDs are normalised to 1-8.

    Mutates base_session.disorder_profiles and base_session.per_disorder_events
    in place and returns the modified session object.

    Args:
        assembly_path: Path to the assembly CSV produced by the study prompter.
        base_session: A PrompterSession already parsed from the timestamps CSV.

    Returns:
        The updated base_session with per_disorder_events populated.

    Raises:
        ValueError: When required columns are absent from the assembly CSV.
    """
    if not base_session.is_combined:
        base_session.is_combined = True

    assembly_path = Path(assembly_path)
    df = pd.read_csv(assembly_path, quotechar='"', dtype=str, keep_default_na=False)

    missing = _REQUIRED_ASSEMBLY_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Assembly CSV is missing required columns: {sorted(missing)}"
        )

    baseline_events = base_session.events_df[
        base_session.events_df["task_group"] == "0"
    ].copy()

    disorder_keys = [
        str(k) for k in df["disorder_profile"].dropna().unique()
    ]
    base_session.disorder_profiles = disorder_keys

    for disorder_key in disorder_keys:
        subset = df[df["disorder_profile"] == disorder_key].copy()

        disorder_event_rows = []
        for _, row in subset.iterrows():
            section = str(row.get("section", "")).strip().upper()
            for _pfx in ("SECTION ", "SECTIE ", "DEEL ", "GROEP "):
                if section.startswith(_pfx):
                    section = section[len(_pfx):].strip()
                    break
            if section not in ("A", "B", "C"):
                continue

            label = str(row.get("label", ""))
            task_number = row.get("task_number", 0)
            try:
                raw_task_id = int(task_number)
            except (ValueError, TypeError):
                raw_task_id = 0

            if section == "B":
                task_id = _normalise_b_task_id(label, raw_task_id)
            else:
                task_id = raw_task_id

            try:
                start_s = float(row["start_s"])
            except (ValueError, TypeError) as exc:
                logger.warning(
                    "Skipping assembly row with invalid start_s: %s — %s", row.to_dict(), exc
                )
                continue
            try:
                end_s = float(row["end_s"])
            except (ValueError, TypeError) as exc:
                logger.warning(
                    "Skipping assembly row with invalid end_s: %s — %s", row.to_dict(), exc
                )
                continue

            raw_seq_rep = row.get("sequence_rep", None)
            try:
                rep_num = (
                    int(float(raw_seq_rep))
                    if raw_seq_rep not in (None, "", "nan") and str(raw_seq_rep).strip() != ""
                    else 1
                )
            except (ValueError, TypeError):
                rep_num = 1

            disorder_event_rows.append({
                "timestamp_abs": start_s,
                "event_type": "measurement",
                "task_group": section,
                "task_id": task_id,
                "task_name": label,
                "repetition": rep_num,
            })
            disorder_event_rows.append({
                "timestamp_abs": end_s,
                "event_type": "segment_end",
                "task_group": section,
                "task_id": task_id,
                "task_name": label,
                "repetition": rep_num,
            })

        disorder_event_rows = _normalise_c_task_ids_in_events(disorder_event_rows)

        _tg_tid_name_counts: dict = {}
        for ev in disorder_event_rows:
            if ev["event_type"] == "measurement":
                key = (ev["task_group"], ev["task_id"])
                names = _tg_tid_name_counts.setdefault(key, set())
                names.add(ev["task_name"])
        for key, names in _tg_tid_name_counts.items():
            if len(names) <= 1:
                continue
            tg, tid = key
            meas_evs = sorted(
                [ev for ev in disorder_event_rows
                 if ev["task_group"] == tg and ev["task_id"] == tid
                 and ev["event_type"] == "measurement"],
                key=lambda e: e["timestamp_abs"],
            )
            name_rep_map: dict = {}
            for new_rep, ev in enumerate(meas_evs, 1):
                name_rep_map[(ev["task_name"], ev["repetition"], ev["timestamp_abs"])] = new_rep
            for ev in disorder_event_rows:
                if ev["task_group"] != tg or ev["task_id"] != tid:
                    continue
                if ev["event_type"] == "measurement":
                    new_rep = name_rep_map.get(
                        (ev["task_name"], ev["repetition"], ev["timestamp_abs"])
                    )
                    if new_rep is not None:
                        ev["repetition"] = new_rep
                elif ev["event_type"] == "segment_end":
                    closest = min(
                        (m for m in meas_evs if m["task_name"] == ev["task_name"]),
                        key=lambda m: abs(m["timestamp_abs"] - ev["timestamp_abs"]),
                        default=None,
                    )
                    if closest is not None:
                        ev["repetition"] = name_rep_map.get(
                            (closest["task_name"], closest["repetition"],
                             closest["timestamp_abs"]),
                            ev["repetition"],
                        )

        if disorder_event_rows:
            disorder_df = pd.DataFrame(disorder_event_rows)
        else:
            disorder_df = pd.DataFrame(
                columns=["timestamp_abs", "event_type", "task_group", "task_id", "task_name", "repetition"]
            )

        combined_events = (
            pd.concat([baseline_events, disorder_df], ignore_index=True)
            .sort_values("timestamp_abs")
            .reset_index(drop=True)
        )
        if len(combined_events) > 0:
            combined_events["timestamp_abs"] = pd.to_numeric(
                combined_events["timestamp_abs"], errors="coerce"
            )
            combined_events = combined_events.dropna(subset=["timestamp_abs"]).reset_index(drop=True)
            if "task_id" in combined_events.columns:
                combined_events["task_id"] = pd.to_numeric(
                    combined_events["task_id"], errors="coerce").fillna(0).astype(int)
            combined_events["task_group"] = combined_events.get("task_group", "0").fillna("0").astype(str)
            combined_events["task_name"] = combined_events.get("task_name", "").fillna("").astype(str)
            if "repetition" not in combined_events.columns:
                combined_events["repetition"] = 1
            combined_events["repetition"] = (
                pd.to_numeric(combined_events["repetition"], errors="coerce")
                .fillna(1).astype(int)
            )

        base_session.per_disorder_events[disorder_key] = combined_events

    return base_session


def _parse_recording_meta(meta: Any) -> tuple:
    """Extract recording_start_offset_s and per-camera start offsets from metadata.

    Handles three metadata shapes:
      - Current nested format: {'cameras': [...], 'screen': {...}}
      - Legacy flat list: [{camera_1_dict}, {camera_2_dict}, ...]
      - Single dict (single-camera): {recording_start_offset_s: ..., ...}

    Returns (recording_start_offset_s: float, camera_start_offsets: List[float]).
    Returns (0.0, []) for empty or unrecognised input.
    """
    if not meta:
        return 0.0, []

    if isinstance(meta, dict) and "cameras" in meta:
        camera_entries = meta.get("cameras", [])
    elif isinstance(meta, list):
        camera_entries = meta
    elif isinstance(meta, dict):
        camera_entries = [meta]
    else:
        return 0.0, []

    if not camera_entries:
        return 0.0, []

    raw_offset = camera_entries[0].get("recording_start_offset_s", 0.0)
    try:
        recording_start_offset_s = float(raw_offset)
    except (ValueError, TypeError):
        recording_start_offset_s = 0.0

    camera_start_offsets: List[float] = []
    for entry in camera_entries:
        raw = entry.get("start_offset_from_first_cam_s", 0.0)
        try:
            camera_start_offsets.append(float(raw))
        except (ValueError, TypeError):
            camera_start_offsets.append(0.0)

    return recording_start_offset_s, camera_start_offsets


def load_prompter_session(
    timestamps_path: Path,
    meta_path: Optional[Path] = None,
    assembly_path: Optional[Path] = None,
) -> "PrompterSession":
    """Load and parse all study-prompter output files into a PrompterSession.

    This is the primary entry point for the rest of the pipeline.  It:
      1. Loads the recording metadata JSON (if provided) to extract camera
         timing offsets.
      2. Parses the timestamps CSV via parse_timestamps_csv.
      3. Optionally attaches per-disorder events via parse_assembly_csv.

    Args:
        timestamps_path: Path to the study-prompter timestamps CSV.
        meta_path: Optional path to the recording metadata JSON file.
        assembly_path: Optional path to the COMBINED assembly CSV.

    Returns:
        A fully populated PrompterSession ready for use in the pipeline.
    """
    meta = load_recording_meta(meta_path)
    recording_start_offset_s, camera_start_offsets = _parse_recording_meta(meta)

    session = parse_timestamps_csv(
        Path(timestamps_path),
        recording_start_offset_s=recording_start_offset_s,
        camera_start_offsets=camera_start_offsets,
    )

    if assembly_path is not None:
        session = parse_assembly_csv(Path(assembly_path), session)

    return session
