"""
Cross-participant analysis for FACET-CV facial motor and speech behaviour analysis.

Aggregates subject-level session overview and consolidated repetition CSVs across
multiple participants to produce group-level summary statistics, box plots,
and an optional correlation matrix against demographic/clinical variables.

This module can be used as a library or run as a standalone script. When run as
a script it accepts --subjects, --mode, --demographics, --output_dir, and
--group_col arguments.

Usage (standalone)::

    python -m src.cross_participant \\
        --subjects P001 P002 P003 \\
        --mode pilot \\
        --demographics demographics.csv \\
        --output_dir data/results/group

Or import directly::

    from src.cross_participant import compare_participants
    compare_participants(project_root, subject_ids, study_mode,
                         demographics_path=Path("demographics.csv"),
                         output_dir=Path("data/results/group"))

Output files written to output_dir:

  group_session_overview.csv    - merged session-level summary across subjects
  group_aggregated.csv          - per-subject aggregated deviation/anomaly scores
  group_boxplots_overview.pdf   - box plots of key session metrics
  group_boxplots_deviation.pdf  - box plots of anomaly/deviation metrics
  group_correlation_matrix.pdf  - Pearson feature correlation heatmap
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("pipeline")

PROJECT_ROOT = Path(__file__).parent.parent

if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))


_KEY_OVERVIEW_COLS = [
    "mean_deviation_score",
    "anomaly_rate",
    "mean_detection_confidence",
    "n_repetitions",
    "articulation_score_pataka",
    "simple_syllable_mean",
    "mean_articulation_score",
    "articulation_impairment_consistency",
]

_KEY_REP_COLS = [
    "deviation_score",
    "is_anomaly",
    "score_confidence",
]


def _load_overview(data_dir: Path, study_mode: str, subject_id: str) -> Optional[pd.DataFrame]:
    """Load the session overview CSV for a single subject.

    Searches data_dir/results/{study_mode}/{subject_id}/ for a file matching
    *_session_overview.csv. Returns a DataFrame with a 'subject_id' column
    added, or None if no matching file exists or the file cannot be read.
    """
    subject_dir = data_dir / "results" / study_mode / subject_id
    paths = list(subject_dir.glob("*_session_overview.csv"))
    if not paths:
        return None
    try:
        df = pd.read_csv(paths[0])
        df["subject_id"] = subject_id
        return df
    except Exception as exc:
        logger.warning("Could not load overview for %s: %s", subject_id, exc)
        return None


def _load_consolidated(data_dir: Path, study_mode: str, subject_id: str) -> Optional[pd.DataFrame]:
    """Load the consolidated repetition CSV for a single subject.

    Searches data_dir/results/{study_mode}/{subject_id}/ for a file matching
    *_consolidated.csv. Returns a DataFrame with a 'subject_id' column added,
    or None if no matching file exists or the file cannot be read.
    """
    subject_dir = data_dir / "results" / study_mode / subject_id
    paths = list(subject_dir.glob("*_consolidated.csv"))
    if not paths:
        return None
    try:
        df = pd.read_csv(paths[0])
        df["subject_id"] = subject_id
        return df
    except Exception as exc:
        logger.warning("Could not load consolidated CSV for %s: %s", subject_id, exc)
        return None


def _merge_demographics(
    df: pd.DataFrame,
    demo_df: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join demographics onto aggregated data on subject_id."""
    if "subject_id" not in demo_df.columns:
        logger.warning("Demographics file has no 'subject_id' column - skipping merge.")
        return df
    try:
        return df.merge(demo_df, on="subject_id", how="left", suffixes=("", "_demo"))
    except Exception as exc:
        logger.warning("Could not merge demographics: %s", exc)
        return df


def _subject_level_aggregates(consolidated_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-repetition consolidated data to one row per subject+session."""
    agg_cols: Dict[str, Any] = {}
    if "deviation_score" in consolidated_df.columns:
        agg_cols["mean_deviation_score"] = ("deviation_score", "mean")
        agg_cols["std_deviation_score"] = ("deviation_score", "std")
        agg_cols["max_deviation_score"] = ("deviation_score", "max")
    if "is_anomaly" in consolidated_df.columns:
        agg_cols["anomaly_rate"] = ("is_anomaly", "mean")
    if "score_confidence" in consolidated_df.columns:
        agg_cols["mean_score_confidence"] = ("score_confidence", "mean")

    group_keys = [c for c in ["subject_id", "session_id", "study_mode"] if c in consolidated_df.columns]
    if not group_keys or not agg_cols:
        return consolidated_df

    agg_list = {k: v for k, v in agg_cols.items()}
    try:
        result = consolidated_df.groupby(group_keys).agg(**agg_list).reset_index()
        return result
    except Exception as exc:
        logger.warning("Aggregation failed: %s", exc)
        return pd.DataFrame()


def _plot_group_boxplots(
    group_df: pd.DataFrame,
    group_col: Optional[str],
    metrics: List[str],
    output_path: Path,
    title: str = "Group Comparison",
) -> None:
    """Generate a grid of box plots for each metric, optionally grouped by group_col."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available - skipping group box plots.")
        return

    avail = [m for m in metrics if m in group_df.columns]
    if not avail:
        return

    n_cols = min(3, len(avail))
    n_rows = (len(avail) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for idx, metric in enumerate(avail):
        ax = axes[idx // n_cols][idx % n_cols]
        ax.set_title(metric.replace("_", " ").title(), fontsize=10)
        ax.set_xlabel("")

        if group_col and group_col in group_df.columns:
            groups = sorted(group_df[group_col].dropna().unique())
            data_per_group = [group_df[group_df[group_col] == g][metric].dropna().values for g in groups]
            valid = [(g, d) for g, d in zip(groups, data_per_group) if len(d) > 0]
            if valid:
                labels, data_lists = zip(*valid)
                bp = ax.boxplot(data_lists, labels=labels, patch_artist=True)
                colors = plt.cm.Set2(np.linspace(0, 0.8, len(labels)))
                for patch, color in zip(bp["boxes"], colors):
                    patch.set_facecolor(color)
                ax.tick_params(axis="x", labelsize=8)
        else:
            vals = group_df[metric].dropna().values
            if len(vals) > 0:
                ax.boxplot(vals, patch_artist=True)
                ax.set_xticklabels(["All subjects"])

        ax.set_ylabel(metric.split("_")[-1], fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for idx in range(len(avail), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    plt.tight_layout()
    try:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    except Exception as exc:
        logger.warning("Could not save group box plots: %s", exc)
    finally:
        plt.close(fig)


def _plot_correlation_matrix(
    df: pd.DataFrame,
    numeric_cols: List[str],
    output_path: Path,
    title: str = "Feature Correlation Matrix",
) -> None:
    """Plot a Pearson correlation matrix heatmap for the specified columns."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available - skipping correlation matrix.")
        return

    avail = [c for c in numeric_cols if c in df.columns]
    if len(avail) < 2:
        return

    corr = df[avail].corr(method="pearson")
    n = len(avail)
    fig_size = max(8, n * 0.7 + 2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.7, label="Pearson r")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    short_labels = [c.replace("_", "\n") for c in avail]
    ax.set_xticklabels(short_labels, fontsize=max(6, 9 - n // 5), rotation=45, ha="right")
    ax.set_yticklabels(short_labels, fontsize=max(6, 9 - n // 5))
    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=max(5, 8 - n // 5),
                        color="white" if abs(val) > 0.6 else "#333333",
                        fontweight="bold" if abs(val) > 0.6 else "normal")
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    try:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    except Exception as exc:
        logger.warning("Could not save correlation matrix: %s", exc)
    finally:
        plt.close(fig)


def compare_participants(
    project_root: Path,
    subject_ids: List[str],
    study_mode: str = "pilot",
    demographics_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    group_col: Optional[str] = None,
) -> Dict[str, Any]:
    """Run cross-participant analysis across *subject_ids*.

    Loads ``*_session_overview.csv`` and ``*_consolidated.csv`` for each subject,
    merges them with optional demographic data, then produces:

      - ``group_session_overview.csv`` - merged session-level summary
      - ``group_boxplots_overview.pdf`` - box plots of key session metrics
      - ``group_boxplots_deviation.pdf`` - box plots of anomaly / deviation metrics
      - ``group_correlation_matrix.pdf`` - feature correlation heatmap

    Parameters
    ----------
    project_root:
        Root directory of the ``master_project`` package.
    subject_ids:
        List of subject IDs to include.
    study_mode:
        Data sub-folder (e.g. ``"pilot"``).
    demographics_path:
        Optional path to a CSV / Excel file with a ``subject_id`` column.
    output_dir:
        Directory for output files.  Defaults to
        ``data/results/{study_mode}/group_analysis/``.
    group_col:
        Column name in demographics (or session overview) to use as the
        grouping variable for box plots (e.g. ``"diagnosis"``).

    Returns
    -------
    dict with keys: overview_df, aggregated_df, n_subjects, output_paths
    """
    data_dir = project_root / "data"
    if output_dir is None:
        output_dir = data_dir / "results" / study_mode / "group_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_overviews: List[pd.DataFrame] = []
    all_consolidated: List[pd.DataFrame] = []

    for subj in subject_ids:
        ov = _load_overview(data_dir, study_mode, subj)
        if ov is not None and len(ov) > 0:
            all_overviews.append(ov)
        cn = _load_consolidated(data_dir, study_mode, subj)
        if cn is not None and len(cn) > 0:
            all_consolidated.append(cn)

    if not all_overviews and not all_consolidated:
        logger.warning("No data found for any of the specified subjects.")
        return {"overview_df": None, "aggregated_df": None, "n_subjects": 0, "output_paths": {}}

    overview_df = pd.concat(all_overviews, ignore_index=True) if all_overviews else pd.DataFrame()
    consolidated_df = pd.concat(all_consolidated, ignore_index=True) if all_consolidated else pd.DataFrame()

    demo_df: Optional[pd.DataFrame] = None
    if demographics_path is not None and demographics_path.exists():
        try:
            if demographics_path.suffix in (".xlsx", ".xls"):
                demo_df = pd.read_excel(demographics_path)
            else:
                demo_df = pd.read_csv(demographics_path)
            if "subject_id" in demo_df.columns:
                if len(overview_df) > 0:
                    overview_df = _merge_demographics(overview_df, demo_df)
                if len(consolidated_df) > 0:
                    consolidated_df = _merge_demographics(consolidated_df, demo_df)
                logger.info("Demographics merged from %s", demographics_path)
            else:
                logger.warning("Demographics file missing 'subject_id' column - not merged.")
        except Exception as exc:
            logger.warning("Could not load demographics from %s: %s", demographics_path, exc)

    aggregated_df = _subject_level_aggregates(consolidated_df) if len(consolidated_df) > 0 else pd.DataFrame()

    effective_group_col = group_col
    if effective_group_col is None and demo_df is not None and len(overview_df) > 0:
        candidate_cols = [c for c in demo_df.columns
                          if c not in ("subject_id",) and overview_df[c].nunique() <= 10
                          ] if "subject_id" in demo_df.columns else []
        effective_group_col = candidate_cols[0] if candidate_cols else None

    output_paths: Dict[str, Path] = {}

    if len(overview_df) > 0:
        ov_out = output_dir / "group_session_overview.csv"
        overview_df.to_csv(str(ov_out), index=False)
        output_paths["session_overview_csv"] = ov_out
        logger.info("Group session overview CSV saved: %s", ov_out)

    if len(aggregated_df) > 0:
        agg_out = output_dir / "group_aggregated.csv"
        aggregated_df.to_csv(str(agg_out), index=False)
        output_paths["aggregated_csv"] = agg_out
        logger.info("Group aggregated CSV saved: %s", agg_out)

    source_df = overview_df if len(overview_df) > 0 else aggregated_df
    if len(source_df) > 0:
        box_out = output_dir / "group_boxplots_overview.pdf"
        ov_metrics = [c for c in _KEY_OVERVIEW_COLS if c in source_df.columns]
        _plot_group_boxplots(source_df, effective_group_col, ov_metrics, box_out,
                             title="Session Overview - Group Comparison")
        output_paths["boxplots_overview_pdf"] = box_out

    if len(aggregated_df) > 0:
        dev_out = output_dir / "group_boxplots_deviation.pdf"
        dev_metrics = [c for c in ["mean_deviation_score", "anomaly_rate", "std_deviation_score",
                                    "max_deviation_score", "mean_score_confidence"]
                       if c in aggregated_df.columns]
        _plot_group_boxplots(aggregated_df, effective_group_col, dev_metrics, dev_out,
                             title="Deviation & Anomaly - Group Comparison")
        output_paths["boxplots_deviation_pdf"] = dev_out

    corr_source = overview_df if len(overview_df) >= 3 else aggregated_df
    if len(corr_source) >= 3:
        num_cols = corr_source.select_dtypes(include="number").columns.tolist()
        num_cols = [c for c in num_cols if corr_source[c].std() > 1e-9 and c != "subject_id"][:20]
        if len(num_cols) >= 2:
            corr_out = output_dir / "group_correlation_matrix.pdf"
            _plot_correlation_matrix(corr_source, num_cols, corr_out,
                                     title="Cross-Participant Feature Correlation")
            output_paths["correlation_matrix_pdf"] = corr_out

    summary_stats: Dict[str, Any] = {}
    if len(overview_df) > 0:
        for col in _KEY_OVERVIEW_COLS:
            if col in overview_df.columns:
                vals = pd.to_numeric(overview_df[col], errors="coerce").dropna()
                if len(vals) > 0:
                    summary_stats[col] = {
                        "mean": float(vals.mean()),
                        "std": float(vals.std()),
                        "median": float(vals.median()),
                        "n": int(len(vals)),
                    }

    return {
        "overview_df": overview_df if len(overview_df) > 0 else None,
        "aggregated_df": aggregated_df if len(aggregated_df) > 0 else None,
        "n_subjects": len(subject_ids),
        "n_subjects_with_data": len(all_overviews) + len(all_consolidated),
        "summary_stats": summary_stats,
        "output_paths": {k: str(v) for k, v in output_paths.items()},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-participant analysis")
    parser.add_argument("--subjects", nargs="+", required=True,
                        help="Subject IDs to include")
    parser.add_argument("--mode", default="pilot",
                        help="Study mode (default: pilot)")
    parser.add_argument("--demographics", default=None,
                        help="Path to demographics CSV or Excel file")
    parser.add_argument("--output_dir", default=None,
                        help="Directory for output files")
    parser.add_argument("--group_col", default=None,
                        help="Column name to use for grouping in box plots")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    result = compare_participants(
        project_root=PROJECT_ROOT,
        subject_ids=args.subjects,
        study_mode=args.mode,
        demographics_path=Path(args.demographics) if args.demographics else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        group_col=args.group_col,
    )
    print(f"Processed {result['n_subjects_with_data']} subjects with data.")
    for k, v in result.get("output_paths", {}).items():
        print(f"  {k}: {v}")
