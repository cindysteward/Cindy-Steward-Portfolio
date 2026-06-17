"""
Session summary figure: five-panel PDF for one test session.

Reads all per-profile result JSON files from a single test session results
directory and produces a single PDF with five panels:

  A  Detection matrix       -- profiles x disorders, coloured by severity
  B  Task-group evidence    -- forest plot: mean deviation +/-1 SD per group/profile
  C  Deviation confidence   -- per-repetition scatter in deviation x log-Mahalanobis
                               space with classical (dashed) and robust (solid) ellipses
                               anchored on the normal reference profile
  D  Anomaly fraction       -- percentage of flagged frames per profile with overall
                               confidence score
  E  Frame landmark quality -- camera/MediaPipe detection quality per profile:
                               box plots of per-frame detection confidence, plus scatter
                               in (head yaw, confidence) space with reference ellipses

Usage:
    cd master_project
    source venv/bin/activate
    python tools/session_summary_figure.py \\
        data/results/pilot/PAC1/PAC1_test_upright_20260101_120000 \\

Output: session_summary.pdf written into the provided results directory.
"""

import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as mgridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

warnings.filterwarnings("ignore")

try:
    from sklearn.covariance import MinCovDet
    _HAS_MCD = True
except ImportError:
    _HAS_MCD = False

try:
    from mpl_toolkits.mplot3d import Axes3D
    _HAS_3D = True
except ImportError:
    _HAS_3D = False

C = dict(
    blue    = "#0072B2",
    orange  = "#E69F00",
    green   = "#009E73",
    red     = "#D55E00",
    pink    = "#CC79A7",
    cyan    = "#56B4E9",
    gray    = "#999999",
    lgray   = "#E8E8E8",
    dgray   = "#444444",
    bg      = "#FAFAFA",
)

DISORDER_COLOR = {
    "facial_paresis":        C["orange"],
    "buccofacial_apraxia":   C["blue"],
    "dysarthria":            C["red"],
    "speech_apraxia":        C["pink"],
    "phonological_disorder": C["green"],
}
ALL_DISORDERS = list(DISORDER_COLOR.keys())

DISORDER_SHORT = {
    "facial_paresis":        "Facial\nParesis",
    "buccofacial_apraxia":   "Buccofacial\nApraxia",
    "dysarthria":            "Dysarthria",
    "speech_apraxia":        "Speech\nApraxia",
    "phonological_disorder": "Phonol.\nDisorder",
}

SEV_ALPHA = {"severe": 1.0, "moderate": 0.70, "mild": 0.42, "none": 0.0}
SEV_LABEL = {"severe": "S", "moderate": "M", "mild": "m"}

PROFILE_LABEL = {
    "normal":          "Normal",
    "p1_paresis":      "P1  Facial paresis",
    "p2_buccofacial":  "P2  Buccofacial apraxia",
    "p3_dysarthria":   "P3  Dysarthria",
    "p4_apraxia":      "P4  Speech apraxia",
    "p5_phono":        "P5  Phonol. disorder",
    "mixed_a":         "Mixed A",
    "mixed_b":         "Mixed B",
    "mixed_c":         "Mixed C",
}

PROFILE_COLOR = {
    "normal":          C["gray"],
    "p1_paresis":      C["orange"],
    "p2_buccofacial":  C["blue"],
    "p3_dysarthria":   C["red"],
    "p4_apraxia":      C["pink"],
    "p5_phono":        C["green"],
    "mixed_a":         "#7B68EE",
    "mixed_b":         "#BFAE00",
    "mixed_c":         "#00B4CC",
}

TG = {"A": C["blue"], "B": C["orange"], "C": C["green"]}

SESSION_COLORS = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#D55E00",
    "#56B4E9",
    "#F0E442",
    "#7B68EE",
    "#00B4CC",
]


def _session_color(ordered_names: list, name: str) -> str:
    """Return a colour for *name* based on its position in *ordered_names*.

    Falls back to PROFILE_COLOR (for per-session/disorder views) if the name
    is recognised there; otherwise assigns a colour by index so every session
    in a participant-level view gets a distinct hue.
    """
    if name in PROFILE_COLOR:
        return PROFILE_COLOR[name]
    try:
        idx = ordered_names.index(name)
    except ValueError:
        idx = 0
    return SESSION_COLORS[idx % len(SESSION_COLORS)]


ASYM_PAIRS = {
    "browDown":     ("browDownLeft",     "browDownRight"),
    "browOuterUp":  ("browOuterUpLeft",  "browOuterUpRight"),
    "eyeBlink":     ("eyeBlinkLeft",     "eyeBlinkRight"),
    "eyeSquint":    ("eyeSquintLeft",    "eyeSquintRight"),
    "eyeWide":      ("eyeWideLeft",      "eyeWideRight"),
    "cheekSquint":  ("cheekSquintLeft",  "cheekSquintRight"),
    "noseSneer":    ("noseSneerLeft",    "noseSneerRight"),
    "mouthSmile":   ("mouthSmileLeft",   "mouthSmileRight"),
    "mouthFrown":   ("mouthFrownLeft",   "mouthFrownRight"),
    "mouthDimple":  ("mouthDimpleLeft",  "mouthDimpleRight"),
    "mouthStretch": ("mouthStretchLeft", "mouthStretchRight"),
    "mouthPress":   ("mouthPressLeft",   "mouthPressRight"),
    "mouthLowerDown": ("mouthLowerDownLeft", "mouthLowerDownRight"),
    "mouthUpperUp": ("mouthUpperUpLeft", "mouthUpperUpRight"),
}

ASYM_REGIONS = {
    "Brow":           ["browDown", "browOuterUp"],
    "Eye":            ["eyeBlink", "eyeSquint", "eyeWide"],
    "Cheek":          ["cheekSquint"],
    "Nose":           ["noseSneer"],
    "Mouth (upper)":  ["mouthSmile", "mouthDimple", "mouthStretch",
                       "mouthPress", "mouthUpperUp"],
    "Mouth (lower)":  ["mouthFrown", "mouthLowerDown"],
}

_ASYM_COLORS = [C["blue"], C["orange"], C["green"], C["pink"], C["red"]]


def _load(d: Path):
    """Load all required result JSON files from a single profile subdirectory.

    Returns a dict with keys: name, label, sr, ar, cf, car.
    Returns None if any required file is missing.
    """
    need = ["screening_results.json", "anomaly_results.json",
            "confidence_summary.json", "continuous_anomaly_report.json"]
    if not all((d / f).exists() for f in need):
        return None
    return dict(
        name  = d.name,
        label = PROFILE_LABEL.get(d.name, d.name),
        sr    = json.loads((d / "screening_results.json").read_text()),
        ar    = json.loads((d / "anomaly_results.json").read_text()),
        cf    = json.loads((d / "confidence_summary.json").read_text()),
        car   = json.loads((d / "continuous_anomaly_report.json").read_text()),
    )


def _group_stats(ar):
    """Return {group: (mean, sd, n)} for groups A, B, C."""
    devs = np.array(ar.get("deviation_score", []), dtype=float)
    tgs  = ar.get("task_groups", [])
    out  = {}
    for g in ("A", "B", "C"):
        idx  = [i for i, t in enumerate(tgs) if t == g]
        vals = devs[idx] if idx else np.array([])
        out[g] = (
            float(np.mean(vals)) if len(vals) else np.nan,
            float(np.std(vals))  if len(vals) > 1 else 0.0,
            len(vals),
        )
    return out


def _rep_xy(ar):
    """Return (x, y) arrays for the confidence-ellipse scatter.

    x = deviation_score  (0-1)
    y = log1p(mahalanobis_score) clipped at 6 for readability
    """
    devs = np.array(ar.get("deviation_score",    []), dtype=float)
    mah  = np.array(ar.get("mahalanobis_score",  []), dtype=float)
    if len(devs) == 0 or len(mah) == 0:
        return np.array([]), np.array([])
    y = np.clip(np.log1p(mah), 0, 7)
    return devs, y


def _cov_ellipse(mean, cov, ax, n_std, **kw):
    """Draw a covariance ellipse centred at *mean* with radius *n_std* std devs."""
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    w = 2 * n_std * np.sqrt(np.abs(eigvals[0]))
    h = 2 * n_std * np.sqrt(np.abs(eigvals[1]))
    ax.add_patch(Ellipse(xy=mean, width=w, height=h, angle=angle, **kw))


def _panel_A(ax, profiles):
    """Detection matrix: profiles as rows, disorder types as columns.

    Each cell is coloured by disorder type and shaded by severity. The severity
    label (S/M/m) is printed in each detected cell. Overall confidence score is
    shown to the right of each row.
    """
    n_p, n_d = len(profiles), len(ALL_DISORDERS)
    ax.set_facecolor(C["bg"])

    for i, p in enumerate(profiles):
        inds = {x["indication_type"]: x.get("severity", "mild")
                for x in p["sr"].get("indications", [])}
        row  = n_p - 1 - i
        for j, d in enumerate(ALL_DISORDERS):
            sev = inds.get(d, "none")
            col = DISORDER_COLOR[d] if sev != "none" else C["lgray"]
            alp = SEV_ALPHA[sev]    if sev != "none" else 0.50
            ax.add_patch(plt.Rectangle(
                [j + 0.05, row + 0.10], 0.90, 0.80,
                facecolor=col, alpha=alp,
                edgecolor="white", linewidth=2.0, zorder=2,
            ))
            if sev != "none":
                ax.text(j + 0.50, row + 0.50, SEV_LABEL[sev],
                        ha="center", va="center",
                        fontsize=8, fontweight="bold", color="white", zorder=3)

        conf = p["cf"].get("confidence", {}).get("overall")
        conf_str = f"{conf:.2f}" if conf is not None else "—"
        ax.text(n_d + 0.10, row + 0.50, conf_str,
                va="center", ha="left",
                fontsize=7.5, color=C["blue"], fontweight="semibold")

    ax.set_yticks(np.arange(n_p) + 0.5)
    ax.set_yticklabels([p["label"] for p in reversed(profiles)],
                        fontsize=8.5, color=C["dgray"])
    ax.set_xticks(np.arange(n_d) + 0.5)
    ax.set_xticklabels([DISORDER_SHORT[d] for d in ALL_DISORDERS],
                        rotation=30, ha="right", fontsize=7.5, color=C["dgray"])
    ax.set_xlim(0, n_d + 0.80)
    ax.set_ylim(0, n_p)
    ax.tick_params(left=False, bottom=False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("A   Detection Matrix", loc="left",
                 fontsize=10, fontweight="bold", pad=6)
    ax.text(n_d + 0.10, n_p + 0.10, "conf.",
            fontsize=6.5, color=C["blue"], va="bottom")

    sev_h = [
        mpatches.Patch(facecolor=C["lgray"], alpha=0.80, label="None"),
        mpatches.Patch(facecolor=C["blue"],  alpha=0.42, label="Mild (m)"),
        mpatches.Patch(facecolor=C["blue"],  alpha=0.70, label="Moderate (M)"),
        mpatches.Patch(facecolor=C["blue"],  alpha=1.00, label="Severe (S)"),
    ]
    ax.legend(handles=sev_h, ncol=4, fontsize=6.5,
              loc="upper center", bbox_to_anchor=(0.45, -0.22),
              framealpha=0.95, title="Severity", title_fontsize=6.5)


def _panel_B(ax, profiles):
    """Forest plot of mean deviation score per task group per profile.

    Each profile gets a row. Within each row, task groups A, B, and C are shown
    as separate coloured dots with +/-1 SD error bars. Dots above the anomaly
    threshold (0.40) are highlighted. A vertical band from the normal reference
    profile's mean is drawn as a visual anchor.
    """
    n_p   = len(profiles)
    row_h = 1.3
    grp_y = {"A": +0.33, "B": 0.00, "C": -0.33}
    cap_h = 0.09
    ax.set_facecolor(C["bg"])

    norm = next((p for p in profiles if p["name"] == "normal"), None)
    if norm:
        norm_stats = _group_stats(norm["ar"])
        ref_vals   = [v for g in ("A","B","C") for v,_,n in [norm_stats[g]]
                      if not np.isnan(v)]
        if ref_vals:
            ref_mean = np.mean(ref_vals)
            ax.axvspan(ref_mean - 0.05, ref_mean + 0.05,
                       alpha=0.10, color=C["gray"], zorder=0)
            ax.axvline(ref_mean, color=C["gray"], linewidth=0.9,
                       linestyle=":", alpha=0.7, zorder=1)

    ax.axvline(0.40, color=C["red"], linewidth=1.2,
               linestyle="--", alpha=0.80, zorder=1)
    ax.text(0.41, n_p * row_h + 0.10, "threshold",
            fontsize=6.5, color=C["red"], va="bottom")

    for i, p in enumerate(profiles):
        stats  = _group_stats(p["ar"])
        y_cent = (n_p - 1 - i) * row_h + row_h / 2
        ax.axhline(y_cent, color=C["lgray"], linewidth=0.6, zorder=0)

        for g in ("A", "B", "C"):
            mean, sd, n = stats[g]
            if np.isnan(mean):
                continue
            y   = y_cent + grp_y[g]
            col = TG[g]
            if sd > 0:
                ax.plot([mean - sd, mean + sd], [y, y],
                        color=col, linewidth=1.8, alpha=0.55, zorder=2,
                        solid_capstyle="butt")
                for cap in [mean - sd, mean + sd]:
                    ax.plot([cap, cap], [y - cap_h, y + cap_h],
                            color=col, linewidth=1.3, alpha=0.55, zorder=2)
            above = mean > 0.40
            ax.scatter(mean, y,
                       s=80 if above else 50,
                       c=col,
                       marker="o",
                       alpha=0.95 if above else 0.65,
                       edgecolors="white" if above else "none",
                       linewidths=0.9,
                       zorder=4)

    ax.set_yticks([(n_p - 1 - i) * row_h + row_h / 2 for i in range(n_p)])
    ax.set_yticklabels([p["label"] for p in profiles],
                        fontsize=8.5, color=C["dgray"])
    ax.set_xlim(0.00, 0.90)
    ax.set_ylim(-0.45, n_p * row_h + 0.10)
    ax.set_xlabel("Mean deviation score (±1 SD)", fontsize=9)
    ax.set_title("B   Task-Group Deviation Evidence", loc="left",
                 fontsize=10, fontweight="bold", pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(left=False)
    ax.grid(axis="x", alpha=0.15, zorder=0)

    ax.legend(handles=[
        Line2D([0],[0], marker="o", color="w", markerfacecolor=TG["A"],
               markersize=8, label="Group A  (facial expression)"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=TG["B"],
               markersize=8, label="Group B  (DDK / motor speech)"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=TG["C"],
               markersize=8, label="Group C  (word production)"),
        Line2D([0],[0], color=C["red"], linestyle="--",
               linewidth=1.2, label="Anomaly threshold (0.40)"),
    ], fontsize=7.5, loc="lower right", framealpha=0.95)


def _panel_C(ax, profiles):
    """Per-repetition scatter in (deviation_score, log-Mahalanobis) space.

    The normal reference profile defines the baseline cluster.
    Classical confidence ellipses (from sample covariance) are drawn as dashed.
    Robust ellipses (from Minimum Covariance Determinant) are drawn as solid.
    All other profiles' repetitions are plotted as small coloured dots so the
    reader can see which profiles fall inside or outside the normal region.
    """
    ax.set_facecolor(C["bg"])

    all_data = {}
    for p in profiles:
        xs, ys = _rep_xy(p["ar"])
        if len(xs):
            all_data[p["name"]] = (xs, ys, p["label"])

    ref_key = "normal"
    if ref_key not in all_data:
        ref_key = next(iter(all_data), None)
    if ref_key is None:
        ax.set_title("C  Confidence Ellipses\n(no data)", loc="left",
                     fontsize=10, fontweight="bold")
        return

    ref_x, ref_y, _ = all_data[ref_key]
    ref_pts = np.column_stack([ref_x, ref_y])
    ref_mean_cls = ref_pts.mean(axis=0)
    ref_cov_cls  = np.cov(ref_pts.T)

    if _HAS_MCD and len(ref_pts) >= 6:
        try:
            mcd = MinCovDet(support_fraction=0.85, random_state=0).fit(ref_pts)
            ref_mean_rob = mcd.location_
            ref_cov_rob  = mcd.covariance_
        except Exception:
            ref_mean_rob = ref_mean_cls
            ref_cov_rob  = ref_cov_cls
    else:
        ref_mean_rob = ref_mean_cls
        ref_cov_rob  = ref_cov_cls

    for n_std, alpha in [(1.0, 0.50), (2.0, 0.28)]:
        _cov_ellipse(ref_mean_cls, ref_cov_cls, ax, n_std,
                     facecolor="none", edgecolor=C["cyan"],
                     linewidth=1.6, linestyle="--", alpha=alpha, zorder=3)
    for n_std, alpha in [(1.0, 0.55), (2.0, 0.30)]:
        _cov_ellipse(ref_mean_rob, ref_cov_rob, ax, n_std,
                     facecolor="none", edgecolor=C["blue"],
                     linewidth=1.6, linestyle="-", alpha=alpha, zorder=3)

    rng = np.random.default_rng(42)
    for name, (xs, ys, label) in all_data.items():
        col   = PROFILE_COLOR.get(name, C["gray"])
        is_ref = name == ref_key
        jx = rng.uniform(-0.008, 0.008, size=len(xs)) if not is_ref else 0
        jy = rng.uniform(-0.04, 0.04,   size=len(ys)) if not is_ref else 0

        ax.scatter(xs + jx, ys + jy,
                   s=18 if not is_ref else 22,
                   c=col,
                   alpha=0.55 if not is_ref else 0.40,
                   edgecolors="none",
                   linewidths=0,
                   zorder=2,
                   label=label)

    ax.plot(*ref_mean_cls, marker="+", color=C["blue"],
            markersize=10, markeredgewidth=1.5, zorder=5)
    ax.plot(*ref_mean_rob, marker="x", color=C["cyan"],
            markersize=8, markeredgewidth=1.5, zorder=5)

    ax.set_xlabel("Deviation score", fontsize=9)
    ax.set_ylabel("log(1 + Mahalanobis distance)", fontsize=9)
    ax.set_ylim(0, 7.5)
    ax.set_title("C   Repetition-level Confidence Ellipses", loc="left",
                 fontsize=10, fontweight="bold", pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3)
    ax.grid(alpha=0.15, zorder=0)

    profile_handles = [
        Line2D([0],[0], marker="o", color="w",
               markerfacecolor=PROFILE_COLOR.get(p["name"], C["gray"]),
               markersize=7, alpha=0.8, label=p["label"])
        for p in profiles
    ]
    ellipse_handles = [
        Line2D([0],[0], color=C["cyan"],  linestyle="--", linewidth=1.5,
               label="Classical CI (1σ, 2σ)"),
        Line2D([0],[0], color=C["blue"],  linestyle="-",  linewidth=1.5,
               label="Robust CI  (1σ, 2σ)"),
        Line2D([0],[0], marker="+", color=C["blue"],  markersize=8,
               linestyle="none", label="Normal centroid (classical)"),
        Line2D([0],[0], marker="x", color=C["cyan"],  markersize=8,
               linestyle="none", label="Normal centroid (robust)"),
    ]
    ax.legend(
        handles=profile_handles + ellipse_handles,
        fontsize=6.8, loc="upper left",
        framealpha=0.92, ncol=2,
        title="Profiles  /  Ellipse type", title_fontsize=6.8,
    )


def _panel_D(ax, profiles):
    """Dual x-axis panel.

    Bottom axis (cyan):  % anomalous windows, natural scale 0-100 %.
    Top axis (orange):   Overall detection confidence, natural scale 0-1.
    Both axes are independent so no scaling hack is needed.
    """
    n_p    = len(profiles)
    y_pos  = np.arange(n_p)
    labels = [p["label"] for p in reversed(profiles)]

    fracs, oconfs = [], []
    for p in reversed(profiles):
        car  = p["car"]
        nw   = max(car.get("n_windows", 1), 1)
        na   = car.get("n_anomalous_windows", 0)
        fracs.append(100.0 * na / nw)
        oconfs.append(float(p["cf"].get("confidence", {}).get("overall") or 0.0))

    ax.set_facecolor(C["bg"])
    bars = ax.barh(y_pos - 0.18, fracs, height=0.34,
                   color=C["cyan"], alpha=0.80,
                   edgecolor="none", label="% anomalous windows")
    ax.axvline(30, color=C["gray"], linewidth=0.8, linestyle=":", alpha=0.60,
               zorder=0)
    for bar, val in zip(bars, fracs):
        col = C["red"] if val > 38 else C["dgray"]
        ax.text(val + 0.8, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}%", va="center", fontsize=7.0, color=col)

    ax.set_xlabel("% anomalous windows", fontsize=9, color=C["blue"])
    ax.tick_params(axis="x", labelcolor=C["blue"])
    ax.set_xlim(0, 100)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8.5, color=C["dgray"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(left=False)
    ax.grid(axis="x", alpha=0.12, zorder=0)

    ax2 = ax.twiny()
    conf_bars = ax2.barh(y_pos + 0.18, oconfs, height=0.34,
                         color=C["orange"], alpha=0.80,
                         edgecolor="none", label="Detection confidence")
    for bar, val in zip(conf_bars, oconfs):
        ax2.text(val + 0.008, bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}", va="center", fontsize=7.0, color=C["orange"])

    ax2.set_xlabel("Detection confidence", fontsize=9, color=C["orange"])
    ax2.tick_params(axis="x", labelcolor=C["orange"])
    ax2.set_xlim(0, 1.05)
    ax2.spines["bottom"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    ax.set_title("D   Anomaly Fraction & Detection Confidence", loc="left",
                 fontsize=10, fontweight="bold", pad=20)

    h1 = mpatches.Patch(facecolor=C["cyan"],   alpha=0.80, label="% anomalous windows")
    h2 = mpatches.Patch(facecolor=C["orange"], alpha=0.80, label="Detection confidence")
    ax.legend(handles=[h1, h2], fontsize=7.5, loc="lower right", framealpha=0.95)


def _find_processed_dir(session_dir: Path) -> Path | None:
    """Derive the processed-data directory from a results session directory.

    results/.../pilot/PAC1/{session_id}  →  processed/.../pilot/PAC1/{session_id}
    """
    parts = session_dir.parts
    try:
        idx = parts.index("results")
        proc_parts = parts[:idx] + ("processed",) + parts[idx + 1:]
        return Path(*proc_parts)
    except ValueError:
        return None


def _find_raw_dir(session_dir: Path) -> Path | None:
    """Derive the raw-data directory from a results session directory.

    results/.../patient/PA1  →  raw/.../patient/PA1
    """
    parts = session_dir.parts
    try:
        idx = parts.index("results")
        raw_parts = parts[:idx] + ("raw",) + parts[idx + 1:]
        return Path(*raw_parts)
    except ValueError:
        return None


def _short_session_label(name: str) -> str:
    """Strip subject-ID prefix and trailing timestamp from a session folder name.

    'PA1_postop_test_20260407_230743' → 'postop_test\n20260407'
    """
    parts = name.split("_")
    if parts and parts[0] and parts[0][0].isalpha() and any(c.isdigit() for c in parts[0]):
        parts = parts[1:]
    label_parts: list[str] = []
    for p in parts:
        label_parts.append(p)
        if len(p) == 8 and p.isdigit():
            break
    return "_".join(label_parts)


def _load_frame_quality(profile: dict, session_dir: Path,
                        max_frames: int = 800) -> pd.DataFrame | None:
    """Load per-frame quality columns from corrected_features.csv.

    When the profile contains a ``_feat_csv`` key (used by the participant-level
    summary) that path is used directly, bypassing the session_dir lookup.

    Returns a DataFrame with columns:
      detection_confidence  — MediaPipe per-frame score (always present if available)
      head_yaw              — estimated yaw angle in degrees (0 = frontal)
      head_pose_deviation   — total angular deviation from reference pose in degrees
    Returns None when the file is missing or unreadable.
    """
    override = profile.get("_feat_csv")
    if override:
        csv_path = Path(override)
    else:
        proc_dir = _find_processed_dir(session_dir)
        if proc_dir is None:
            return None
        csv_path = proc_dir / profile["name"] / "corrected_features.csv"
    if not csv_path.exists():
        return None
    WANT = {"detection_confidence", "detection_success", "head_yaw",
            "head_pose_deviation", "head_roll", "occluded"}
    try:
        df = pd.read_csv(csv_path, usecols=lambda c: c in WANT, low_memory=False)
        if "detection_confidence" not in df.columns and "detection_success" not in df.columns:
            return None
        if "detection_success" in df.columns:
            df["detection_success_score"] = df["detection_success"].astype(float)
        if "detection_confidence" not in df.columns:
            df["detection_confidence"] = np.nan
        if "occluded" not in df.columns:
            df["occluded"] = False
        else:
            df["occluded"] = df["occluded"].fillna(False).astype(bool)
        if len(df) > max_frames:
            df = df.sample(max_frames, random_state=42)
        return df.reset_index(drop=True)
    except Exception:
        return None


def _panel_E(ax, profiles, session_dir: Path):
    """2D scatter: detection_confidence (X) vs detection_success (Y).

    Each session is plotted as a single point at its per-session mean, with
    error bars showing ±1 SD.  Raw per-frame data shown as faint background
    scatter so density is visible.  Occluded frames in orange.
    """
    all_fq = {}
    for p in profiles:
        fq = _load_frame_quality(p, session_dir)
        if fq is not None and len(fq):
            all_fq[p["name"]] = (fq, p["label"])

    ax.set_facecolor(C["bg"])

    if not all_fq:
        ax.text(0.5, 0.5, "Frame quality data not available\n"
                "(corrected_features.csv not found)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=8.5, color=C["gray"])
        ax.set_title("E   Detection Confidence vs Success", loc="left",
                     fontsize=10, fontweight="bold", pad=5)
        return

    ordered = [(p["name"], p["label"]) for p in profiles if p["name"] in all_fq]
    _MARKERS = ["o", "^", "s", "D", "P", "X", "v", "*", "h"]
    _ordered_names = [n for n, _ in ordered]

    def _occluded_mask(fq: pd.DataFrame) -> np.ndarray:
        """Return a boolean mask of frames marked as occluded, or all-False if no such column."""
        return fq["occluded"].values.astype(bool) if "occluded" in fq.columns else np.zeros(len(fq), dtype=bool)

    _use_success = any("detection_success_score" in all_fq[name][0].columns for name, _ in ordered)
    _y_col = "detection_success_score" if _use_success else "detection_confidence"

    rng = np.random.default_rng(42)
    legend_handles = []
    first_occ = True

    for idx, (name, label) in enumerate(ordered):
        fq   = all_fq[name][0]
        col  = _session_color(_ordered_names, name)
        mk   = _MARKERS[idx % len(_MARKERS)]
        occ  = _occluded_mask(fq)

        xs_all = fq["detection_confidence"].values if "detection_confidence" in fq.columns else np.array([])
        ys_all = fq[_y_col].values if _y_col in fq.columns else np.array([])

        if len(xs_all) == 0 or len(ys_all) == 0:
            continue

        n = len(xs_all)
        idx_samp = rng.choice(n, size=min(n, 1500), replace=False)
        jx = rng.uniform(-0.003, 0.003, size=len(idx_samp))
        jy = rng.uniform(-0.03,  0.03,  size=len(idx_samp))
        xs_s = xs_all[idx_samp]
        ys_s = ys_all[idx_samp]
        occ_s = occ[idx_samp]

        ax.scatter(xs_s[~occ_s] + jx[~occ_s], ys_s[~occ_s] + jy[~occ_s],
                   s=6, c=col, alpha=0.15, marker=mk, edgecolors="none", zorder=1)
        if occ_s.any():
            ax.scatter(xs_s[occ_s] + jx[occ_s], ys_s[occ_s] + jy[occ_s],
                       s=6, c="#E67E22", alpha=0.35, marker=mk, edgecolors="none",
                       zorder=2, label="Occluded (raw)" if first_occ else None)
            first_occ = False

        x_mean = float(np.nanmean(xs_all))
        y_mean = float(np.nanmean(ys_all))
        x_sd   = float(np.nanstd(xs_all))
        y_sd   = float(np.nanstd(ys_all))

        ax.errorbar(x_mean, y_mean, xerr=x_sd, yerr=y_sd,
                    fmt=mk, color=col, markersize=8, markeredgewidth=1.2,
                    elinewidth=1.2, capsize=3, capthick=1.0,
                    alpha=0.92, zorder=5)
        h = ax.scatter([], [], s=50, c=col, marker=mk, label=label)
        legend_handles.append(h)

    ax.axvline(0.5, color=C["red"], linewidth=1.0, linestyle="--",
               alpha=0.55, label="Conf. threshold (0.5)")

    if _use_success:
        ax.set_ylim(-0.12, 1.18)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Failed (0)", "Success (1)"], fontsize=8)
        ax.set_ylabel("Detection success", fontsize=9)
    else:
        ax.set_ylabel("Detection confidence", fontsize=9)

    ax.set_xlabel("Detection confidence", fontsize=9)
    ax.set_title("E   Detection Confidence vs Success\n"
                 "       (mean ± SD per session; faint = raw frames; orange = occluded)",
                 loc="left", fontsize=9, fontweight="bold", pad=5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.15, zorder=0)

    legend_handles.append(
        Line2D([0], [0], color=C["red"], linestyle="--", linewidth=1.0,
               label="Conf. threshold (0.5)")
    )
    if not first_occ:
        legend_handles.append(
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#E67E22",
                   markersize=5, alpha=0.7, label="Occluded frames")
        )
    ax.legend(handles=legend_handles, fontsize=7, loc="lower right", framealpha=0.92)


def _panel_F(ax, profiles, session_dir: Path):
    """3D scatter: detection_confidence (X), detection_success (Y), |head_yaw| (Z).

    Each session in a distinct colour.  Occluded frames in orange.
    Gives an intuitive view of how yaw angle relates to tracking quality.
    """
    all_fq = {}
    for p in profiles:
        fq = _load_frame_quality(p, session_dir)
        if fq is not None and len(fq) and "head_yaw" in fq.columns:
            all_fq[p["name"]] = (fq, p["label"])

    if not all_fq or not _HAS_3D:
        ax.text(0.5, 0.5, "(head_yaw / 3D not available)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=8, color=C["gray"])
        ax.set_title("F   3D: Confidence × Success × Yaw",
                     loc="left", fontsize=9, fontweight="bold", pad=5)
        return

    ordered = [(p["name"], p["label"]) for p in profiles if p["name"] in all_fq]
    _MARKERS = ["o", "^", "s", "D", "P", "X", "v", "*", "h"]
    _ordered_names_F = [n for n, _ in ordered]
    _use_success = any("detection_success_score" in all_fq[name][0].columns for name, _ in ordered)
    _y_col = "detection_success_score" if _use_success else "detection_confidence"

    def _occluded_mask(fq: pd.DataFrame) -> np.ndarray:
        """Return a boolean mask of frames marked as occluded, or all-False if no such column."""
        return fq["occluded"].values.astype(bool) if "occluded" in fq.columns else np.zeros(len(fq), dtype=bool)

    rng = np.random.default_rng(7)
    first_occ = True

    for idx, (name, label) in enumerate(ordered):
        fq  = all_fq[name][0]
        col = _session_color(_ordered_names_F, name)
        mk  = _MARKERS[idx % len(_MARKERS)]
        occ = _occluded_mask(fq)

        if "detection_confidence" not in fq.columns or _y_col not in fq.columns:
            continue

        xs = fq["detection_confidence"].values
        ys = fq[_y_col].values
        zs = np.abs(fq["head_yaw"].values)

        n = len(xs)
        idx_samp = rng.choice(n, size=min(n, 1200), replace=False)
        xs, ys, zs, occ = xs[idx_samp], ys[idx_samp], zs[idx_samp], occ[idx_samp]

        ax.scatter(xs[~occ], ys[~occ], zs[~occ], s=5, c=col,
                   alpha=0.30, marker=mk, edgecolors="none", depthshade=True,
                   label=label, zorder=2)
        if occ.any():
            ax.scatter(xs[occ], ys[occ], zs[occ], s=5, c="#E67E22",
                       alpha=0.55, marker=mk, edgecolors="none", depthshade=True,
                       label="Occluded" if first_occ else None, zorder=3)
            first_occ = False

    ax.set_xlabel("Confidence", fontsize=7, labelpad=4)
    ax.set_ylabel("Success" if _use_success else "Confidence", fontsize=7, labelpad=4)
    ax.set_zlabel("|Yaw| (°)", fontsize=7, labelpad=4)
    ax.tick_params(axis="both", labelsize=6)
    ax.set_title("F   3D: Confidence × Success × Yaw",
                 loc="left", fontsize=9, fontweight="bold", pad=5)
    if _use_success:
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["0", "1"], fontsize=6)
    ax.legend(fontsize=5, bbox_to_anchor=(0.0, 0.5), loc="center right",
              framealpha=0.92, markerscale=1.4, ncol=1,
              borderpad=0.5, handletextpad=0.4)


def _panel_G(ax, profiles, session_dir: Path):
    """3D scatter: detection_confidence (X), detection_success (Y), |head_roll| (Z).

    Head roll measures the tilt of the inter-eye line:
      ~0°  → upright frontal recording
      ~90° → participant lying on their side (ORS / supine)

    Each session in a distinct colour.  Occluded frames in orange.
    """
    all_fq = {}
    for p in profiles:
        fq = _load_frame_quality(p, session_dir)
        if fq is not None and len(fq) and "head_roll" in fq.columns:
            all_fq[p["name"]] = (fq, p["label"])

    if not all_fq or not _HAS_3D:
        ax.text(0.5, 0.5, "(head_roll / 3D not available)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=8, color=C["gray"])
        ax.set_title("G   3D: Confidence × Success × Roll",
                     loc="left", fontsize=9, fontweight="bold", pad=5)
        return

    ordered = [(p["name"], p["label"]) for p in profiles if p["name"] in all_fq]
    _MARKERS = ["o", "^", "s", "D", "P", "X", "v", "*", "h"]
    _ordered_names_G = [n for n, _ in ordered]
    _use_success = any("detection_success_score" in all_fq[name][0].columns for name, _ in ordered)
    _y_col = "detection_success_score" if _use_success else "detection_confidence"

    def _occluded_mask(fq: pd.DataFrame) -> np.ndarray:
        """Return a boolean mask of frames marked as occluded, or all-False if no such column."""
        return fq["occluded"].values.astype(bool) if "occluded" in fq.columns else np.zeros(len(fq), dtype=bool)

    rng = np.random.default_rng(13)
    first_occ = True

    for idx, (name, label) in enumerate(ordered):
        fq  = all_fq[name][0]
        col = _session_color(_ordered_names_G, name)
        mk  = _MARKERS[idx % len(_MARKERS)]
        occ = _occluded_mask(fq)

        if "detection_confidence" not in fq.columns or _y_col not in fq.columns:
            continue

        xs = fq["detection_confidence"].values
        ys = fq[_y_col].values
        zs = np.abs(fq["head_roll"].values)

        n = len(xs)
        idx_samp = rng.choice(n, size=min(n, 1200), replace=False)
        xs, ys, zs, occ = xs[idx_samp], ys[idx_samp], zs[idx_samp], occ[idx_samp]

        ax.scatter(xs[~occ], ys[~occ], zs[~occ], s=5, c=col,
                   alpha=0.30, marker=mk, edgecolors="none", depthshade=True,
                   label=label, zorder=2)
        if occ.any():
            ax.scatter(xs[occ], ys[occ], zs[occ], s=5, c="#E67E22",
                       alpha=0.55, marker=mk, edgecolors="none", depthshade=True,
                       label="Occluded" if first_occ else None, zorder=3)
            first_occ = False

    ax.set_xlabel("Confidence", fontsize=7, labelpad=4)
    ax.set_ylabel("Success" if _use_success else "Confidence", fontsize=7, labelpad=4)
    ax.set_zlabel("|Roll| (°)", fontsize=7, labelpad=4)
    ax.tick_params(axis="both", labelsize=6)
    ax.set_title("G   3D: Confidence × Success × Roll",
                 loc="left", fontsize=9, fontweight="bold", pad=5)
    if _use_success:
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["0", "1"], fontsize=6)
    ax.legend(fontsize=5, bbox_to_anchor=(0.0, 0.5), loc="center right",
              framealpha=0.92, markerscale=1.4, ncol=1,
              borderpad=0.5, handletextpad=0.4)


_LM_NOSE_TIP = 1

_LM_CHEEK_L = [116, 123, 147]
_LM_CHEEK_R = [345, 352, 376]
_LM_NOSE_ALA_L = [129, 49]
_LM_NOSE_ALA_R = [358, 279]
_LM_INNER_CANTHUS_L = 133
_LM_INNER_CANTHUS_R = 362

_LM_LANDMARK_MAX = max(
    _LM_NOSE_TIP,
    max(_LM_CHEEK_L), max(_LM_CHEEK_R),
    max(_LM_NOSE_ALA_L), max(_LM_NOSE_ALA_R),
    _LM_INNER_CANTHUS_L, _LM_INNER_CANTHUS_R,
)

_BLENDSHAPE_NOISE_THRESHOLD = 1e-4


def _landmark_region_asymmetry(
    lm_series: "pd.Series",
    lm_left: list,
    lm_right: list,
    max_samples: int = 500,
) -> float:
    """Compute mean landmark-based L-R asymmetry from a _landmarks_3d column.

    Uses the nose tip (landmark 1) as the facial midline x-reference.
    All distances are normalized by interocular distance (inner canthus L to R)
    following the s41598-2025 facial symmetry paper, making the result invariant
    to camera distance changes across sessions.
    Returns the mean |dist_L − dist_R| / (dist_L + dist_R + ε) over sampled
    frames, yielding a value in [0, 1] where 0 = perfectly symmetric.
    """
    series = lm_series.dropna()
    if len(series) == 0:
        return 0.0

    if len(series) > max_samples:
        series = series.sample(max_samples, random_state=42)

    asym_vals = []
    for lm_str in series:
        try:
            flat = json.loads(str(lm_str))
            needed = (_LM_LANDMARK_MAX + 1) * 3
            if len(flat) < needed:
                continue
            lm = np.array(flat, dtype=np.float32).reshape(-1, 3)
            nose_x = lm[_LM_NOSE_TIP, 0]
            iod = float(np.linalg.norm(lm[_LM_INNER_CANTHUS_L] - lm[_LM_INNER_CANTHUS_R]))
            iod = max(iod, 1e-6)
            x_L = float(np.mean([lm[i, 0] for i in lm_left]))
            x_R = float(np.mean([lm[i, 0] for i in lm_right]))
            dist_L = abs(x_L - nose_x) / iod
            dist_R = abs(x_R - nose_x) / iod
            asym_vals.append(abs(dist_L - dist_R) / (dist_L + dist_R + 1e-6))
        except Exception:
            continue

    return float(np.mean(asym_vals)) if asym_vals else 0.0


def _load_asymmetry(profile: dict, session_dir: Path,
                    max_frames: int = 3000) -> "pd.DataFrame | None":
    """Compute per-frame absolute L–R asymmetry from raw blendshapes.

    Loads the pre-z-score blendshapes.csv (or frame_data.csv as fallback) so
    that asymmetry values are in the original [0, 1] blendshape scale.  Using
    corrected_features.csv is intentionally avoided: z-scoring features with
    near-zero neutral variance inflates asymmetry values into the thousands.

    For Cheek (cheekSquint) and Nose (noseSneer) regions whose blendshape
    values are essentially at the noise floor, landmark-based asymmetry is
    computed instead: the x-distance of symmetric malar/alar landmarks from
    the nose-tip midline is used to detect structural facial asymmetry that
    the blendshape activations miss.

    Returns a DataFrame with one column per ASYM_PAIRS key, values = |L − R|.
    """
    override_raw = profile.get("_raw_dir")
    if override_raw:
        for fname in ("blendshapes.csv", "frame_data.csv"):
            csv_path = Path(override_raw) / fname
            if csv_path.exists():
                break
        else:
            return None
    else:
        raw_dir = _find_raw_dir(session_dir)
        if raw_dir is None:
            return None
        for fname in ("blendshapes.csv", "frame_data.csv"):
            csv_path = raw_dir / profile["name"] / fname
            if csv_path.exists():
                break
        else:
            return None

    needed = set()
    for L, R in ASYM_PAIRS.values():
        needed.add(L)
        needed.add(R)

    try:
        df_full = pd.read_csv(csv_path, low_memory=False)
        if df_full.empty:
            return None
        if len(df_full) > max_frames:
            df_full = df_full.sample(max_frames, random_state=42).reset_index(drop=True)

        result = {}
        for key, (L, R) in ASYM_PAIRS.items():
            if L in df_full.columns and R in df_full.columns:
                result[key] = (df_full[L] - df_full[R]).abs().reset_index(drop=True)

        cheek_blendshape_mean = float(result["cheekSquint"].mean()) if "cheekSquint" in result else 0.0
        nose_blendshape_mean = float(result["noseSneer"].mean()) if "noseSneer" in result else 0.0

        _skip_lm_fallback = profile.get("_is_ors", False)

        if "_landmarks_3d" in df_full.columns and not _skip_lm_fallback:
            lm_series = df_full["_landmarks_3d"]
            if cheek_blendshape_mean < _BLENDSHAPE_NOISE_THRESHOLD:
                cheek_lm_asym = _landmark_region_asymmetry(lm_series, _LM_CHEEK_L, _LM_CHEEK_R)
                if cheek_lm_asym > 0.0:
                    result["cheekSquint"] = pd.Series(
                        [cheek_lm_asym] * len(df_full), dtype=float
                    )

            if nose_blendshape_mean < _BLENDSHAPE_NOISE_THRESHOLD:
                nose_lm_asym = _landmark_region_asymmetry(lm_series, _LM_NOSE_ALA_L, _LM_NOSE_ALA_R)
                if nose_lm_asym > 0.0:
                    result["noseSneer"] = pd.Series(
                        [nose_lm_asym] * len(df_full), dtype=float
                    )

        return pd.DataFrame(result) if result else None
    except Exception:
        return None


def _panel_H(ax, profiles, session_dir: Path):
    """Panel H: mean absolute asymmetry per facial region, compared across sessions.

    Each facial region (Brow, Eye, Cheek, Nose, Mouth upper/lower) is shown as
    a grouped horizontal bar.  One bar per session, colour-coded.  When exactly
    two sessions are present, a Δ column on the right shows direction and
    magnitude of change (green ▼ = less asymmetry, red ▲ = more asymmetry).

    Values are computed from raw (pre-z-score) blendshapes so the scale stays
    in [0, 1].
    """
    session_asym: dict = {}
    for p in profiles:
        df = _load_asymmetry(p, session_dir)
        if df is not None and len(df):
            lbl = _short_session_label(p["name"])
            session_asym[p["name"]] = (df, lbl)

    ax.set_facecolor(C["bg"])

    if not session_asym:
        ax.text(0.5, 0.5, "Asymmetry data not available\n(blendshapes.csv not found)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=8.5, color=C["gray"])
        ax.set_title("H   Facial Asymmetry by Region  (raw blendshapes, |L − R|)",
                     loc="left", fontsize=9, fontweight="bold", pad=5)
        return

    regions = list(ASYM_REGIONS.keys())
    n_regions = len(regions)
    session_names = list(session_asym.keys())
    n_sess = len(session_names)

    region_vals: dict = {}
    for sname, (df, _) in session_asym.items():
        region_vals[sname] = {}
        for region, keys in ASYM_REGIONS.items():
            avail = [k for k in keys if k in df.columns]
            if avail:
                val = float(df[avail].mean().mean())
                region_vals[sname][region] = val if not np.isnan(val) else 0.0
            else:
                region_vals[sname][region] = 0.0

    bar_h = max(0.10, min(0.28, 0.80 / max(n_sess, 1)))
    gap   = max(0.02, bar_h * 0.18)
    y_pos = np.arange(n_regions, dtype=float)
    x_max_data = max(
        v for rd in region_vals.values() for v in rd.values()
    ) or 0.1

    _label_thresh = max(x_max_data * 0.04, 0.005)
    _ann_fs = max(5.0, 7.0 - 0.4 * n_sess)

    for si, sname in enumerate(session_names):
        _, label = session_asym[sname]
        y_off = (si - (n_sess - 1) / 2) * (bar_h + gap)
        vals  = [region_vals[sname].get(r, 0.0) for r in regions]
        color = _ASYM_COLORS[si % len(_ASYM_COLORS)]
        bars  = ax.barh(
            y_pos + y_off, vals, height=bar_h,
            color=color, alpha=0.78, label=label,
            edgecolor="white", linewidth=0.5, zorder=2,
        )
        for bar, val in zip(bars, vals):
            if val >= _label_thresh:
                ax.text(
                    bar.get_width() + x_max_data * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", fontsize=_ann_fs, color=C["dgray"],
                )

    if n_sess == 2:
        _ref_markers = ("baseline", "preop", "normal", "normaal", "basislijn")
        ref_name  = next(
            (n for n in session_names
             if any(m in n.lower() for m in _ref_markers)),
            session_names[0],
        )
        test_name = next(n for n in session_names if n != ref_name)
        s0, s1 = ref_name, test_name
        lbl0 = session_asym[s0][1]
        lbl1 = session_asym[s1][1]
        x_lim_right = x_max_data * 1.30
        ax.set_xlim(right=x_lim_right)
        x_delta_ax = 0.88

        ax.text(x_delta_ax, 1.02,
                f"Δ test − ref  ({lbl1} − {lbl0})",
                transform=ax.transAxes, fontsize=6, color=C["gray"],
                style="italic", ha="left", va="bottom")

        for ri, region in enumerate(regions):
            v0 = region_vals[s0].get(region, 0.0)
            v1 = region_vals[s1].get(region, 0.0)
            diff = v1 - v0
            arrow = "▼" if diff < 0 else "▲"
            clr   = C["green"] if diff < 0 else C["red"]
            y_ax = 1.0 - (ri + 0.5) / n_regions
            ax.text(x_delta_ax, y_ax,
                    f"{arrow} {abs(diff):.3f}",
                    transform=ax.transAxes,
                    va="center", ha="left", fontsize=7, color=clr, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(regions, fontsize=8.5)
    ax.invert_yaxis()

    for i in range(n_regions - 1):
        ax.axhline(y=i + 0.5, color=C["gray"], linewidth=0.8, alpha=0.45,
                   linestyle="-", zorder=1)

    ax.set_xlabel("Mean |L − R| blendshape activation (raw, 0 – 1 scale)", fontsize=9)
    ax.set_title("H   Facial Asymmetry by Region  (raw blendshapes, |L − R|)",
                 loc="left", fontsize=9, fontweight="bold", pad=5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.15, axis="x", zorder=0)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.92)


def generate_session_summary(session_dir: Path, output_path=None):
    """Generate the five-panel session summary PDF for a single test session.

    Loads per-profile result JSON files from session_dir, builds eight panels
    (A through H), and saves the figure as a PDF.

    Args:
        session_dir: Path to the session results directory containing per-profile
            subdirectories, each with screening_results.json and related files.
        output_path: Destination PDF path. Defaults to session_dir/session_summary.pdf.
    """
    session_dir = Path(session_dir)
    if output_path is None:
        output_path = session_dir / "session_summary.pdf"

    profiles = [_load(d) for d in sorted(session_dir.iterdir()) if d.is_dir()]
    profiles = [p for p in profiles if p is not None]
    if not profiles:
        print(f"No profiles found in {session_dir}")
        return

    def _key(p):
        """Sort profiles so 'normal' comes first, disorder profiles second, others third."""
        n = p["name"]
        if n == "normal":     return (0, n)
        if n.startswith("p"): return (1, n)
        return (2, n)
    profiles.sort(key=_key)
    print(f"Loaded {len(profiles)} profiles: {[p['name'] for p in profiles]}")

    fig = plt.figure(figsize=(20, 19), facecolor="white")

    gs_top = mgridspec.GridSpec(
        2, 2,
        figure        = fig,
        left          = 0.07,
        right         = 0.98,
        top           = 0.95,
        bottom        = 0.40,
        hspace        = 0.42,
        wspace        = 0.28,
        height_ratios = [1.0, 1.4],
        width_ratios  = [1.0, 1.3],
    )
    gs_mid = mgridspec.GridSpec(
        1, 3,
        figure       = fig,
        left         = 0.07,
        right        = 0.98,
        top          = 0.36,
        bottom       = 0.22,
        wspace       = 0.28,
        width_ratios = [1.1, 1.0, 1.0],
    )
    gs_bot = mgridspec.GridSpec(
        1, 1,
        figure = fig,
        left   = 0.07,
        right  = 0.85,
        top    = 0.19,
        bottom = 0.04,
    )

    ax_A = fig.add_subplot(gs_top[0, 0])
    ax_B = fig.add_subplot(gs_top[0, 1])
    ax_C = fig.add_subplot(gs_top[1, 0])
    ax_D = fig.add_subplot(gs_top[1, 1])
    ax_E = fig.add_subplot(gs_mid[0, 0])
    ax_F = fig.add_subplot(gs_mid[0, 1], projection="3d") if _HAS_3D else fig.add_subplot(gs_mid[0, 1])
    ax_G = fig.add_subplot(gs_mid[0, 2], projection="3d") if _HAS_3D else fig.add_subplot(gs_mid[0, 2])
    ax_H = fig.add_subplot(gs_bot[0, 0])

    _panel_A(ax_A, profiles)
    _panel_B(ax_B, profiles)
    _panel_C(ax_C, profiles)
    _panel_D(ax_D, profiles)
    _panel_E(ax_E, profiles, session_dir)
    _panel_F(ax_F, profiles, session_dir)
    _panel_G(ax_G, profiles, session_dir)
    _panel_H(ax_H, profiles, session_dir)

    fig.suptitle(
        f"Session Summary  ·  {session_dir.name}",
        fontsize=12, fontweight="bold", y=0.972, color=C["dgray"],
    )
    fig.text(
        0.50, 0.010,
        "Deviation: OC-SVM + Mahalanobis ensemble vs. participant reference baseline.  "
        "Threshold 0.40.  Deviation/quality ellipses anchored on normal reference profile.  "
        "Classical CI (dashed cyan) vs. robust MCD CI (solid blue).  "
        "Frame confidence from MediaPipe FaceLandmarker per-frame detection score.",
        ha="center", va="bottom", fontsize=6.0, color=C["gray"], style="italic",
    )

    fig.savefig(output_path, format="pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def _build_participant_profiles(participant_results_dir: Path) -> list[dict]:
    """Build a list of per-session profile dicts that are compatible with the
    existing per-session panel functions (A–H).

    Each entry mirrors a regular profile dict but represents one session.
    The ``_feat_csv`` and ``_raw_dir`` override keys allow the shared data
    loaders to resolve the correct per-session paths.
    """
    subject_id = participant_results_dir.name
    session_dirs = sorted(
        d for d in participant_results_dir.iterdir()
        if d.is_dir() and d.name.startswith(subject_id)
    )

    _NEEDED = [
        "screening_results.json",
        "anomaly_results.json",
        "confidence_summary.json",
        "continuous_anomaly_report.json",
    ]

    profiles: list[dict] = []
    for sd in session_dirs:
        results_cand = None
        for cand in [sd / "normal"] + sorted(sd.iterdir() if sd.exists() else []):
            if cand.is_dir() and all((cand / f).exists() for f in _NEEDED):
                results_cand = cand
                break
        if results_cand is None and all((sd / f).exists() for f in _NEEDED):
            results_cand = sd
        if results_cand is None:
            continue

        try:
            prof: dict = {
                "name":  sd.name,
                "label": _session_display_label(sd.name),
                "sr":  json.loads((results_cand / "screening_results.json").read_text()),
                "ar":  json.loads((results_cand / "anomaly_results.json").read_text()),
                "cf":  json.loads((results_cand / "confidence_summary.json").read_text()),
                "car": json.loads((results_cand / "continuous_anomaly_report.json").read_text()),
                "_is_test": "test" in sd.name.lower(),
                "_is_ors":  any(k in sd.name.lower()
                                for k in ("_ors_", "_ors", "ors_", "supine")),
            }
        except Exception:
            continue

        parts = list(sd.parts)
        try:
            ri = parts.index("results")
        except ValueError:
            continue

        proc_sd = Path(*parts[:ri]) / "processed" / Path(*parts[ri + 1:])
        for cand in [proc_sd / "normal"] + sorted(proc_sd.iterdir() if proc_sd.exists() else []):
            if cand.is_dir() and (cand / "corrected_features.csv").exists():
                prof["_feat_csv"] = cand / "corrected_features.csv"
                break

        raw_sd = Path(*parts[:ri]) / "raw" / Path(*parts[ri + 1:])
        for cand in [raw_sd / "normal"] + sorted(raw_sd.iterdir() if raw_sd.exists() else []):
            if cand.is_dir() and any((cand / f).exists()
                                     for f in ("blendshapes.csv", "frame_data.csv")):
                prof["_raw_dir"] = cand
                break

        profiles.append(prof)

    from collections import Counter
    label_count = Counter(p["label"] for p in profiles)
    label_seen: dict = {}
    for p in profiles:
        lbl = p["label"]
        if label_count[lbl] > 1:
            label_seen[lbl] = label_seen.get(lbl, 0) + 1
            p["label"] = f"{lbl} ({label_seen[lbl]})"

    return profiles


def _session_display_label(session_name: str) -> str:
    """Human-readable label for a session, used as the legend entry."""
    parts = session_name.split("_")
    if parts and parts[0][0].isalpha() and any(c.isdigit() for c in parts[0]):
        parts = parts[1:]
    label_parts, date_added = [], False
    for p in parts:
        if len(p) == 8 and p.isdigit():
            label_parts.append(p)
            date_added = True
            break
        label_parts.append(p)
    return " ".join(label_parts)


def generate_participant_summary(participant_results_dir: Path, output_path=None):
    """Generate a participant-level PDF summarising ALL sessions (baseline + test).

    Each session is treated as a *profile* and rendered using the same panel
    functions (A–H) as the per-session summary, so the visual design is
    identical.  Panels compare sessions instead of disorder-simulation profiles.

    ``_feat_csv`` / ``_raw_dir`` overrides on each profile dict let the shared
    data loaders resolve the correct per-session paths without needing a common
    session directory.
    """
    participant_results_dir = Path(participant_results_dir)
    if output_path is None:
        output_path = participant_results_dir / "session_summary.pdf"

    subject_id = participant_results_dir.name
    profiles = _build_participant_profiles(participant_results_dir)

    if not profiles:
        print(f"No sessions with results found under {participant_results_dir}")
        return

    print(f"Participant summary: {len(profiles)} sessions for {subject_id}")

    _SESSION_PALETTE = [
        C["gray"], C["blue"], C["orange"], C["red"],
        C["green"], C["pink"], "#7B68EE", "#BFAE00", "#00B4CC",
    ]
    _saved_colors: dict = {}
    for i, prof in enumerate(profiles):
        name = prof["name"]
        _saved_colors[name] = PROFILE_COLOR.get(name)
        PROFILE_COLOR[name] = _SESSION_PALETTE[i % len(_SESSION_PALETTE)]

    try:
        fig = plt.figure(figsize=(20, 19), facecolor="white")

        gs_top = mgridspec.GridSpec(
            2, 2,
            figure        = fig,
            left          = 0.07,
            right         = 0.98,
            top           = 0.95,
            bottom        = 0.40,
            hspace        = 0.42,
            wspace        = 0.28,
            height_ratios = [1.0, 1.4],
            width_ratios  = [1.0, 1.3],
        )
        gs_mid = mgridspec.GridSpec(
            1, 3,
            figure       = fig,
            left         = 0.07,
            right        = 0.98,
            top          = 0.36,
            bottom       = 0.22,
            wspace       = 0.28,
            width_ratios = [1.1, 1.0, 1.0],
        )
        gs_bot = mgridspec.GridSpec(
            1, 1,
            figure = fig,
            left   = 0.07,
            right  = 0.85,
            top    = 0.19,
            bottom = 0.04,
        )

        ax_A = fig.add_subplot(gs_top[0, 0])
        ax_B = fig.add_subplot(gs_top[0, 1])
        ax_C = fig.add_subplot(gs_top[1, 0])
        ax_D = fig.add_subplot(gs_top[1, 1])
        ax_E = fig.add_subplot(gs_mid[0, 0])
        ax_F = fig.add_subplot(gs_mid[0, 1], projection="3d") if _HAS_3D else fig.add_subplot(gs_mid[0, 1])
        ax_G = fig.add_subplot(gs_mid[0, 2], projection="3d") if _HAS_3D else fig.add_subplot(gs_mid[0, 2])
        ax_H = fig.add_subplot(gs_bot[0, 0])

        dummy_dir = participant_results_dir

        _panel_A(ax_A, profiles)
        _panel_B(ax_B, profiles)
        _panel_C(ax_C, profiles)
        _panel_D(ax_D, profiles)
        _panel_E(ax_E, profiles, dummy_dir)
        _panel_F(ax_F, profiles, dummy_dir)
        _panel_G(ax_G, profiles, dummy_dir)
        _panel_H(ax_H, profiles, dummy_dir)

        fig.suptitle(
            f"Participant Overview  ·  {subject_id}  ({len(profiles)} sessions)",
            fontsize=12, fontweight="bold", y=0.972, color=C["dgray"],
        )
        fig.text(
            0.50, 0.010,
            "Each session treated as a profile.  Deviation: OC-SVM + Mahalanobis ensemble.  "
            "Threshold 0.40.  Representative (normal) profile loaded per session.  "
            "Ellipses anchored on the first baseline session.",
            ha="center", va="bottom", fontsize=6.0, color=C["gray"], style="italic",
        )

        fig.savefig(output_path, format="pdf", bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved participant summary: {output_path}")

    finally:
        for name, original_color in _saved_colors.items():
            if original_color is None:
                PROFILE_COLOR.pop(name, None)
            else:
                PROFILE_COLOR[name] = original_color


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Generate academic session summary PDF"
    )
    p.add_argument("session_dir", help="Path to session results directory")
    p.add_argument("--output", "-o", default=None,
                   help="Output PDF path (default: <session_dir>/session_summary.pdf)")
    p.add_argument("--participant", action="store_true",
                   help="Generate participant-level summary instead of session-level")
    args = p.parse_args()
    if args.participant:
        generate_participant_summary(
            Path(args.session_dir),
            Path(args.output) if args.output else None,
        )
    else:
        generate_session_summary(
            Path(args.session_dir),
            Path(args.output) if args.output else None,
        )
