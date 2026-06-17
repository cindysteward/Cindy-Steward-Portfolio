"""
Re-run decision support screening on already-processed pilot data.

Reads the features_used dict from an existing screening_results.json and the
anomaly_results.json produced by an earlier full pipeline run, then re-evaluates
them with the current DecisionSupport logic. This lets you update decision rules
and threshold parameters without re-processing video, re-running MediaPipe, or
recomputing anomaly detection.

Only the decision tree evaluation step is repeated. All upstream data (blendshape
features, kinematic metrics, anomaly scores) remain unchanged from the original run.

Usage:
    python src/rescreen.py           # re-screen all configured pilot test sessions
    python src/rescreen.py --dry-run # print results without writing back to disk
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.decision_support import DecisionSupport


def _load_json(path: Path) -> Dict:
    """Load a JSON file and return its contents as a dict. Returns an empty dict if not found."""
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    """Load a CSV file and return it as a DataFrame. Returns None if not found or unreadable."""
    if path.exists():
        try:
            return pd.read_csv(path, low_memory=False)
        except Exception:
            return None
    return None


def _build_reference_asymmetry_stats(ref_rep_csv: Path) -> Optional[Dict[str, float]]:
    """Compute mean and std of Group A asymmetry ratios from a reference repetition_metrics.csv.

    Returns a dict with keys 'mean', 'std', and 'n', or None if the file is missing
    or contains no valid asymmetry data.
    """
    df = _load_csv(ref_rep_csv)
    if df is None or df.empty:
        return None
    asym_col = next(
        (c for c in ("mean_asymmetry_ratio", "asymmetry_ratio_mean",
                     "mean_asymmetry", "overall_mean_asymmetry")
         if c in df.columns),
        None,
    )
    if asym_col is None:
        return None
    sub = df[df["task_group"].astype(str) == "A"] if "task_group" in df.columns else df
    vals = sub[asym_col].dropna().values
    if len(vals) >= 2:
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)), "n": float(len(vals))}
    if len(vals) == 1:
        return {"mean": float(vals[0]), "std": 0.05, "n": 1.0}
    return None


def _build_reference_articulation(artic_json: Path) -> Optional[Dict[str, Any]]:
    """Load reference articulation from pre-computed articulation_scores.json."""
    if artic_json.exists():
        d = _load_json(artic_json)
        if d and "mean_articulation_score" in d:
            return d
    return None


def _build_reference_head_yaw(ref_processed_dir: Path) -> Optional[float]:
    """Return mean head yaw (degrees) from the reference session's session_metrics.json.

    Used to compute the between-session yaw offset for asymmetry correction in
    the direct paresis detection path.  A yaw rotation of Δθ degrees creates
    an apparent asymmetry of |sin(Δθ)| in landmark-derived features; this offset
    is subtracted before the paresis threshold comparison.
    """
    sm = ref_processed_dir / "session_metrics.json"
    if sm.exists():
        d = _load_json(sm)
        yaw = d.get("head_yaw_mean_session_mean")
        if yaw is not None:
            return float(yaw)
    rep = ref_processed_dir / "repetition_metrics.csv"
    df = _load_csv(rep)
    if df is not None and "head_yaw_mean" in df.columns:
        return float(df["head_yaw_mean"].mean())
    return None


def _build_reference_baseline_stats(ref_rep_csv: Path) -> Dict[str, Any]:
    """Build a per-feature mean/std stats dict from a reference repetition_metrics.csv.

    Returns an empty dict if the file is missing or contains no numeric columns.
    """
    df = _load_csv(ref_rep_csv)
    if df is None or df.empty:
        return {}
    stats: Dict[str, Any] = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        v = df[col].dropna()
        if not v.empty:
            stats[col] = {"mean": float(v.mean()), "std": float(v.std(ddof=1) if len(v) > 1 else 0.0)}
    return stats


def rescreen_session(
    results_dir: Path,
    reference_rep_csv: Path,
    reference_artic_json: Path,
    session_label: str,
    decision_support: DecisionSupport,
    reference_processed_dir: Optional[Path] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Re-run decision support for one processed test session using pre-computed features."""
    existing_screening = _load_json(results_dir / "screening_results.json")
    anomaly_results = _load_json(results_dir / "anomaly_results.json")

    session_metrics: Dict[str, Any] = existing_screening.get("features_used", {})
    if not session_metrics:
        return {}

    ref_asym_stats = _build_reference_asymmetry_stats(reference_rep_csv)
    ref_articulation = _build_reference_articulation(reference_artic_json)
    ref_baseline_stats = _build_reference_baseline_stats(reference_rep_csv)
    ref_head_yaw = _build_reference_head_yaw(reference_processed_dir) if reference_processed_dir else None

    is_ors = "ors" in session_label.lower() or "rotated" in session_label.lower()

    decision_support.set_session_context(
        is_baseline=False,
        has_reference=True,
        reference_stats=ref_baseline_stats,
        task_group="0",
        task_id=0,
        reference_articulation=ref_articulation,
        reference_asymmetry_stats=ref_asym_stats,
        is_ors_session=is_ors,
        reference_head_yaw=ref_head_yaw,
    )

    result = decision_support.evaluate(
        session_metrics=session_metrics,
        task_metrics_df=pd.DataFrame(),
        repetition_metrics_df=pd.DataFrame(),
        anomaly_results=anomaly_results,
    )

    if not dry_run and result:
        output_json = results_dir / "screening_results.json"
        result["features_used"] = session_metrics
        with open(output_json, "w") as f:
            json.dump(result, f, indent=2, default=str)

    return result


def main() -> None:
    """Entry point: re-screen configured pilot test sessions and print accuracy summary."""
    parser = argparse.ArgumentParser(description="Re-run screening on existing processed data.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(ROOT / "config" / "decision_rules.yaml") as f:
        rules = yaml.safe_load(f)
    ds = DecisionSupport(rules)

    TRUTH = {
        "normal": [],
        "p1_paresis": ["facial_paresis"],
        "p2_buccofacial": ["buccofacial_apraxia"],
        "p3_dysarthria": ["dysarthria"],
        "p4_apraxia": ["speech_apraxia"],
        "p5_phono": ["phonological_disorder"],
        "mixed_a": ["speech_apraxia", "facial_paresis"],
        "mixed_b": ["dysarthria", "buccofacial_apraxia"],
        "mixed_c": ["facial_paresis", "phonological_disorder"],
    }

    REFERENCES = {
        "PAC3_upright": (
            ROOT / "data/processed/pilot/PAC3/PAC3_baseline_upright_20260426_213446/repetition_metrics.csv",
            ROOT / "data/results/pilot/PAC3/PAC3_baseline_upright_20260426_213446/articulation_scores.json",
            ROOT / "data/processed/pilot/PAC3/PAC3_baseline_upright_20260426_213446",
        ),
        "PAC3_ORS": (
            ROOT / "data/processed/pilot/PAC3/PAC3_baseline_ORS_rotated_20260426_213728/repetition_metrics.csv",
            ROOT / "data/results/pilot/PAC3/PAC3_baseline_ORS_rotated_20260426_213728/articulation_scores.json",
            ROOT / "data/processed/pilot/PAC3/PAC3_baseline_ORS_rotated_20260426_213728",
        ),
        "PAC7_upright": (
            ROOT / "data/processed/pilot/PAC7/PAC7_baseline_upright_20260426_211706/repetition_metrics.csv",
            ROOT / "data/results/pilot/PAC7/PAC7_baseline_upright_20260426_211706/articulation_scores.json",
            ROOT / "data/processed/pilot/PAC7/PAC7_baseline_upright_20260426_211706",
        ),
        "PAC7_ORS": (
            ROOT / "data/processed/pilot/PAC7/PAC7_baseline_ORS_rotated_20260426_213245/repetition_metrics.csv",
            ROOT / "data/results/pilot/PAC7/PAC7_baseline_ORS_rotated_20260426_213245/articulation_scores.json",
            ROOT / "data/processed/pilot/PAC7/PAC7_baseline_ORS_rotated_20260426_213245",
        ),
        "PAC16_upright": (
            ROOT / "data/processed/pilot/PAC16/PAC16_baseline_upright_20260426_193711/repetition_metrics.csv",
            ROOT / "data/results/pilot/PAC16/PAC16_baseline_upright_20260426_193711/articulation_scores.json",
            ROOT / "data/processed/pilot/PAC16/PAC16_baseline_upright_20260426_193711",
        ),
    }

    TEST_SESSIONS = [
        ("PAC3",  "PAC3_test_upright_20260426_233643",      "PAC3_upright",  "upright"),
        ("PAC3",  "PAC3_test_ORS_rotated_20260426_234237",  "PAC3_ORS",      "ORS"),
        ("PAC7",  "PAC7_test_upright_20260426_222155",       "PAC7_upright",  "upright"),
        ("PAC7",  "PAC7_test_ORS_rotated_20260426_224541",   "PAC7_ORS",      "ORS"),
        ("PAC16", "PAC16_test_upright_20260426_195918",      "PAC16_upright", "upright"),
    ]

    total, correct = 0, 0

    for pac, sess_id, ref_key, position in TEST_SESSIONS:
        ref_csv, ref_artic, ref_proc_dir = REFERENCES[ref_key]
        sess_results = ROOT / "data/results/pilot" / pac / sess_id
        tag = f"{pac}_{position}"
        print(f"\n=== {tag} ===")

        if not sess_results.exists():
            print(f"  [skip] results dir missing: {sess_results}")
            continue

        for profile_dir in sorted(sess_results.iterdir()):
            if not profile_dir.is_dir():
                continue
            profile = profile_dir.name
            if profile not in TRUTH:
                continue

            result = rescreen_session(
                results_dir=profile_dir,
                reference_rep_csv=ref_csv,
                reference_artic_json=ref_artic,
                session_label=sess_id,
                decision_support=ds,
                reference_processed_dir=ref_proc_dir,
                dry_run=args.dry_run,
            )

            if not result:
                print(f"  [skip] no features for {profile}")
                continue

            detected = sorted(set(i.get("indication_type", "") for i in result.get("indications", [])))
            expected = sorted(TRUTH[profile])
            ok = detected == expected
            total += 1
            if ok:
                correct += 1
            status = "OK  " if ok else "FAIL"
            print(f"  {status} {profile}: expected={expected} got={detected}")

    pct = 100 * correct / total if total else 0
    print(f"\n{'='*60}")
    print(f"TOTAL: {correct}/{total} = {pct:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
