"""
Visualization module for the FACET-CV facial motor and speech behavior analysis pipeline.

Generates all clinical and analytical visual outputs for a recording session, including
time-series overlays, asymmetry plots, anomaly detection summaries, kinematic
spatiotemporal profiles, and multi-panel screening reports.  All outputs use a
colorblind-safe palette and write either PNG images (single-panel plots) or
multi-page PDFs (per-task detail reports).

The central class is :class:`Visualizer`.  The factory function
:func:`create_visualizer` instantiates it from the plotting YAML config loaded
by IOManager.

This module is imported by the pipeline orchestrators (run_pipeline.py,
prompter_pipeline.py) and is not intended to be run directly.

References
----------
Palmer et al. (2024) doi:10.3390/s24227235 - facial kinematics methodology
Kong et al. (2021) doi:10.3758/s13414-020-02199-5 - fatigue monitoring
Di Stasi et al. (2014) doi:10.1097/SLA.0000000000000260 - saccadic fatigue
Sakoe & Chiba (1978) doi:10.1109/TASSP.1978.1163055 - DTW
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler as SkScaler
import logging
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger("pipeline")

COLORBLIND_SAFE_PALETTE = {
    'blue': '#0072B2',
    'orange': '#E69F00',
    'green': '#009E73',
    'pink': '#CC79A7',
    'yellow': '#F0E442',
    'cyan': '#56B4E9',
    'red': '#D55E00',
    'gray': '#999999',
    'coral': '#FF6B6B',
    'peach': '#FFEAA7',
    'mint': '#81ECEC',
    'lavender': '#A29BFE',
    'purple': '#7B68EE'
}

REPETITION_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]


class Visualizer:
    """Generates all visual outputs (PNG plots, PDF tables, multi-page PDFs) for a pipeline session.

    A single Visualizer instance is created per pipeline run from the plotting YAML
    config.  It applies matplotlib RC params on construction so all figures produced
    in a session share the same font, DPI, and grid style.  All save methods call
    the internal :meth:`_save_figure` helper which respects the ``output`` section
    of the config (DPI, bbox_inches, transparency).

    Colour conventions
    ------------------
    - Colorblind-safe palette (``COLORBLIND_SAFE_PALETTE``) is used for all data
      series.  Individual repetitions use ``REPETITION_COLORS`` (tab10 subset).
    - Profile bands: lavender fill for ±1 robust-sigma (MAD-based), light purple
      for ±2 sigma, dark purple for 95 % CI.
    - Reference baseline: coral horizontal line.
    - Anomaly flags: orange stars at peak deviation.

    Grouping / paging helpers
    -------------------------
    :meth:`_build_task_pages_by_task` - groups features_df by (task_group, task_id),
        one entry per task containing ALL repetitions.  Used by most overlay plots.
    :meth:`_build_task_pages` - groups by (task_group, task_id, repetition), giving
        one entry per (task, rep) combination.  Used by heatmaps.

    Selected public methods
    -----------------------
    plot_timeseries:
        Simple multi-panel line plot of one or more feature columns vs time.
    plot_repetition_overlay / plot_activation_per_repetition:
        Multi-page PDFs with overlaid repetitions and profile bands per task.
    plot_asymmetry_over_time:
        Per-task asymmetry analysis (time-series + bar + delta panels).
    plot_anomaly_results:
        Adaptive multi-page anomaly report (heatmap, score distribution,
        feature deviations, method consensus).
    plot_screening_summary:
        Single-image clinical screening report with indications and verdict.
    plot_articulation_profile:
        Speech scores summary with component breakdown.
    plot_kinematic_spatiotemporal / plot_all_kinematic_tasks:
        Spatiotemporal kinematic profiles for Groups A, B, and C tasks.
    plot_fatigue_drift_report:
        Four-panel fatigue and motor drift analysis.
    plot_brain_activation_map:
        Glass-brain neural substrates figure (requires nilearn + nibabel).
    plot_feature_for_task:
        On-demand PNG for a single feature on a single task; ``task_key`` uses
        the ``"TG_TID"`` format (e.g. ``"A_1"``) from :meth:`_build_task_pages_by_task`.
    """


    _KINEMATIC_PRIMARY_COLS = [
        "kin_mouth_opening",
        "kin_labial_fissure_width",
        "kin_mouth_height",
        "kin_lip_action_y", "kin_lip_action_z",
        "kin_medial_lower_action_y", "kin_medial_lower_action_z",
        "kin_medial_upper_action_y", "kin_medial_upper_action_z",
        "kin_medial_sym_x",
        "kin_mouth_area_symmetry",
        "kin_pogonion_y", "kin_pogonion_z",
        "kin_mandibular_angle",
        "kin_lower_lip_from_pog_y",
    ]

    _GROUP_A_LANDMARK_COLS = [
        "kin_mouth_opening",
        "kin_labial_fissure_width",
        "kin_lip_action_y", "kin_lip_action_z",
        "kin_medial_upper_action_y", "kin_medial_upper_action_z",
        "kin_medial_lower_action_y", "kin_medial_lower_action_z",
        "kin_medial_sym_x",
        "kin_mouth_area_symmetry",
        "kin_pogonion_y", "kin_pogonion_z",
        "kin_mandibular_angle",
        "kin_lower_lip_from_pog_y",
    ]

    def __init__(self, plotting_config: Dict[str, Any]):
        """Configure matplotlib RC params from *plotting_config* and store colour maps.

        Applies both the explicit key/value mappings and the optional ``general.rc``
        sub-dict from *plotting_config*, which lets plotting.yaml drive any arbitrary
        matplotlib rcParam (e.g. ``axes.spines.top``, ``pdf.fonttype``).
        """
        import matplotlib as _mpl
        try:
            if _mpl.get_backend().lower() not in (
                'agg', 'pdf', 'svg', 'ps', 'cairo', 'pgf', 'template'
            ):
                try:
                    _mpl.use('Agg')
                except Exception:
                    pass
        except Exception:
            pass
        self.config = plotting_config
        self.colors = plotting_config.get('colors', {})
        self.general = plotting_config.get('general', {})

        rc_base = {
            'figure.dpi': self.general.get('figure_dpi', 150),
            'font.family': self.general.get('font_family', 'sans-serif'),
            'font.size': self.general.get('font_size', 9),
            'axes.titlesize': self.general.get('title_size', 11),
            'axes.labelsize': self.general.get('label_size', 10),
            'xtick.labelsize': self.general.get('tick_size', 8),
            'ytick.labelsize': self.general.get('tick_size', 8),
            'legend.fontsize': self.general.get('legend_size', 8),
            'lines.linewidth': self.general.get('line_width', 1.5),
            'lines.markersize': self.general.get('marker_size', 5),
            'axes.grid': self.general.get('axes_grid', True),
            'grid.alpha': self.general.get('axes_grid_alpha', 0.25),
            'grid.linestyle': self.general.get('axes_grid_linestyle', ':'),
        }

        extra_rc = self.general.get('rc', {})
        rc_base.update(extra_rc)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            plt.rcParams.update(rc_base)

        self.save_dpi = self.general.get('save_dpi', 300)

    def _get_colors(self) -> List[str]:
        """Return a list of colorblind-safe palette colors for consolidation plots."""
        return [
            COLORBLIND_SAFE_PALETTE.get('blue',    '#0072B2'),
            COLORBLIND_SAFE_PALETTE.get('red',     '#D55E00'),
            COLORBLIND_SAFE_PALETTE.get('green',   '#009E73'),
            COLORBLIND_SAFE_PALETTE.get('orange',  '#E69F00'),
            COLORBLIND_SAFE_PALETTE.get('lavender','#CC79A7'),
            COLORBLIND_SAFE_PALETTE.get('cyan',    '#56B4E9'),
            COLORBLIND_SAFE_PALETTE.get('yellow',  '#F0E442'),
        ]

    def _get_derived_baseline_value(self, baseline_stats: Optional[Dict], feature: str,
                                    standardization_stats: Optional[Dict] = None) -> Optional[float]:
        """Get baseline value for any feature, optionally in standardized space.

        When *standardization_stats* is provided (the session's own neutral baseline
        used for z-score standardisation), the returned value is expressed in that
        standardised space:
            z = (raw_baseline_value − session_mean) / session_std

        If *baseline_stats* IS *standardization_stats* the result is always 0.0
        (since the session is standardised against itself).

        When *standardization_stats* is None the raw baseline mean is returned
        (legacy behaviour).
        """
        if baseline_stats is None:
            return None

        raw_val = self._raw_baseline_value(baseline_stats, feature)
        if raw_val is None:
            return None

        if standardization_stats is None:
            return raw_val

        if baseline_stats is standardization_stats:
            return 0.0

        std_raw = self._raw_baseline_value(standardization_stats, feature)
        std_std = self._raw_baseline_std(standardization_stats, feature)
        if std_raw is None:
            return None

        if std_std is not None and std_std > 0:
            return (raw_val - std_raw) / std_std
        else:
            return raw_val - std_raw

    def _raw_baseline_value(self, baseline_stats: Optional[Dict], feature: str) -> Optional[float]:
        """Return the raw baseline mean for *feature* from baseline_stats."""
        if baseline_stats is None:
            return None
        if feature in baseline_stats:
            if isinstance(baseline_stats[feature], dict) and 'mean' in baseline_stats[feature]:
                return baseline_stats[feature]['mean']
            return None

        exclude_cols = {'frame_index', 'timestamp_abs', 'segment', 'repetition',
                       'detection_success', 'time_rel_sec', 'task_group', 'task_id',
                       'task_name', 'asymmetry_ratio_brow', 'asymmetry_ratio_eye',
                       'asymmetry_ratio_mouth', 'asymmetry_ratio_cheek',
                       'inter_ocular_distance', 'brightness', 'occluded',
                       'detection_confidence'}

        blendshape_means = []
        for col, stats in baseline_stats.items():
            if col not in exclude_cols and not col.startswith('asymmetry'):
                if isinstance(stats, dict) and 'mean' in stats:
                    blendshape_means.append(stats['mean'])

        if not blendshape_means:
            return None

        if feature == 'mean_activation':
            return np.mean(blendshape_means)
        elif feature == 'max_activation':
            return np.max(blendshape_means)
        elif feature == 'activation_range':
            return np.max(blendshape_means) - np.min(blendshape_means)
        elif feature == 'activation_velocity':
            return 0.0

        return None

    def _raw_baseline_std(self, baseline_stats: Optional[Dict], feature: str) -> Optional[float]:
        """Return the raw baseline std for *feature* from baseline_stats."""
        if baseline_stats is None:
            return None
        if feature in baseline_stats:
            if isinstance(baseline_stats[feature], dict) and 'std' in baseline_stats[feature]:
                return baseline_stats[feature]['std']
        return None

    @staticmethod
    def _build_task_name_map(repetition_metrics_df: Optional[pd.DataFrame]) -> Dict[str, str]:
        """Build a ``{task_group}_{task_id}`` → short-label lookup from the metrics DataFrame."""
        mapping: Dict[str, str] = {}
        if repetition_metrics_df is None or 'task_name' not in repetition_metrics_df.columns:
            return mapping
        for _, row in repetition_metrics_df.iterrows():
            tk = f"{row.get('task_group', '0')}_{row.get('task_id', 0)}"
            tn = str(row.get('task_name', ''))
            short = tn.split(': ', 1)[-1] if ': ' in tn else tn
            if short:
                mapping[tk] = short[:15]
        return mapping

    def _detect_task_segments(self, features_df: pd.DataFrame,
                              repetitions: List) -> Dict[str, Dict]:
        """Detect average time positions for each task_id across repetitions.

        Returns dict keyed by task_key (e.g. "A_2") with:
            avg_start, avg_end, task_group, task_id, task_name
        """
        if 'task_id' not in features_df.columns:
            return {}

        segments: Dict[str, Dict] = {}
        for rep in repetitions:
            rep_df = features_df[features_df['repetition'] == rep]
            if len(rep_df) == 0:
                continue
            rep_start = rep_df['timestamp_abs'].min()

            for tid, tdf in rep_df.groupby('task_id'):
                tid_int = int(tid) if pd.notna(tid) else 0
                if tid_int == 0:
                    continue
                tg = str(tdf['task_group'].iloc[0]) if 'task_group' in tdf.columns else '0'
                tk = f"{tg}_{tid_int}"
                seg_start = tdf['timestamp_abs'].min() - rep_start
                seg_end = tdf['timestamp_abs'].max() - rep_start

                if tk not in segments:
                    tname = ''
                    if 'task_name' in tdf.columns:
                        names = tdf['task_name'].dropna().unique()
                        names = [n for n in names if n != '(no task selected)']
                        tname = names[0] if names else ''
                    segments[tk] = {
                        'starts': [], 'ends': [],
                        'task_group': tg, 'task_id': tid_int, 'task_name': tname
                    }
                segments[tk]['starts'].append(seg_start)
                segments[tk]['ends'].append(seg_end)

        result = {}
        for tk, seg in segments.items():
            result[tk] = {
                'avg_start': float(np.mean(seg['starts'])),
                'avg_end': float(np.mean(seg['ends'])),
                'task_group': seg['task_group'],
                'task_id': seg['task_id'],
                'task_name': seg['task_name'],
            }
        return result

    def _build_task_pages_by_task(self, features_df: pd.DataFrame):
        """Build a list of (label, task_key, filtered_df) tuples grouped by (task_group, task_id) only.

        Unlike _build_task_pages, repetition is NOT included in the grouping key, so each entry
        contains ALL repetitions for a task.  task_key uses the profile-store format "TG_TID".

        Result is cached by DataFrame object identity — all plot calls within a single analysis
        pass the same DataFrame, so the groupby+sort runs only once per session.
        """
        if getattr(self, '_task_pages_by_task_cache_df', None) is features_df:
            return self._task_pages_by_task_cache_result

        has_task_id = 'task_id' in features_df.columns
        has_task_group = 'task_group' in features_df.columns
        task_pages = []
        if has_task_id:
            group_cols = ['task_group', 'task_id'] if has_task_group else ['task_id']
            for group_key, grp in features_df.groupby(group_cols, sort=True):
                if has_task_group:
                    tg, tid = group_key
                else:
                    tg, tid = '0', group_key
                tg_str = str(tg)
                tid_int = int(tid) if pd.notna(tid) else 0
                if tg_str in ('0', 'nan', 'None', '') and tid_int == 0:
                    continue
                tk = f"{tg_str}_{tid_int}"
                task_label = None
                if 'task_name' in grp.columns:
                    names = grp['task_name'].dropna().unique()
                    names = [n for n in names if n not in ('(no task selected)', '')]
                    task_label = names[0] if names else None
                if task_label is None:
                    task_label = f"Task {tg_str}-{tid_int}"
                task_pages.append((task_label, tk, grp))
        elif has_task_group:
            for tg, grp in features_df.groupby('task_group', sort=True):
                if str(tg) in ('0', 'nan', 'None', ''):
                    continue
                tk = f"{tg}_0"
                task_label = None
                if 'task_name' in grp.columns:
                    names = grp['task_name'].dropna().unique()
                    names = [n for n in names if n not in ('(no task selected)', '')]
                    task_label = names[0] if names else None
                if task_label is None:
                    task_label = f"Task {tg}"
                task_pages.append((task_label, tk, grp))
        if not task_pages:
            task_pages = [("All Tasks", None, features_df)]
        self._task_pages_by_task_cache_df = features_df
        self._task_pages_by_task_cache_result = task_pages
        return task_pages

    def _build_task_pages(self, features_df: pd.DataFrame):
        """Build a list of (label, task_key, filtered_df) tuples grouped by (task_group, task_id, repetition)."""
        has_task_id = 'task_id' in features_df.columns
        has_task_group = 'task_group' in features_df.columns
        has_repetition = 'repetition' in features_df.columns
        task_pages = []
        if has_task_id:
            group_cols = ['task_group', 'task_id'] if has_task_group else ['task_id']
            if has_repetition:
                group_cols = group_cols + ['repetition']
            for group_key, grp in features_df.groupby(group_cols, sort=True):
                if has_task_group and has_repetition:
                    tg, tid, rep = group_key
                elif has_task_group:
                    tg, tid = group_key
                    rep = None
                elif has_repetition:
                    tid, rep = group_key
                    tg = '0'
                else:
                    tg, tid, rep = '0', group_key, None
                tg_str = str(tg)
                tid_int = int(tid) if pd.notna(tid) else 0
                if tg_str in ('0', 'nan', 'None', '') and tid_int == 0:
                    continue
                rep_int = int(rep) if rep is not None and pd.notna(rep) else None
                tk = f"{tg_str}_{tid_int}_rep{rep_int}" if rep_int is not None else f"{tg_str}_{tid_int}"
                task_label = None
                if 'task_name' in grp.columns:
                    names = grp['task_name'].dropna().unique()
                    names = [n for n in names if n != '(no task selected)']
                    task_label = names[0] if names else None
                if task_label is None:
                    task_label = f"Task {tg_str}-{tid_int}"
                if rep_int is not None:
                    task_label = f"{task_label} (Rep {rep_int})"
                task_pages.append((task_label, tk, grp))
        elif has_task_group:
            group_cols = ['task_group']
            if has_repetition:
                group_cols = group_cols + ['repetition']
            for group_key, grp in features_df.groupby(group_cols, sort=True):
                if has_repetition:
                    tg, rep = group_key
                    rep_int = int(rep) if pd.notna(rep) else None
                else:
                    tg = group_key
                    rep_int = None
                if str(tg) in ('0', 'nan', 'None', ''):
                    continue
                task_label = None
                if 'task_name' in grp.columns:
                    names = grp['task_name'].dropna().unique()
                    names = [n for n in names if n != '(no task selected)']
                    task_label = names[0] if names else None
                if task_label is None:
                    task_label = f"Task {tg}"
                if rep_int is not None:
                    task_label = f"{task_label} (Rep {rep_int})"
                tk = f"{tg}_0_rep{rep_int}" if rep_int is not None else f"{tg}_0"
                task_pages.append((task_label, tk, grp))
        if not task_pages:
            task_pages = [("All Tasks", None, features_df)]
        return task_pages

    @staticmethod
    def _robust_sigma(mad_arr):
        """Convert MAD values to robust standard-deviation equivalents (1.4826 * MAD)."""
        return np.asarray(mad_arr, dtype=float) * 1.4826

    def _to_z_space(self, raw_mean, raw_std, feature: str,
                     standardization_stats: Optional[Dict] = None):
        """Convert raw profile values to z-score space using standardization stats.
        Returns (z_mean, z_std). If no conversion possible, returns originals."""
        if standardization_stats and feature in standardization_stats:
            s_mean = standardization_stats[feature].get('mean', 0)
            s_std = standardization_stats[feature].get('std', 1)
            if s_std > 0:
                if isinstance(raw_mean, np.ndarray):
                    return (raw_mean - s_mean) / s_std, raw_std / s_std
                else:
                    return (raw_mean - s_mean) / s_std, raw_std / s_std
        return raw_mean, raw_std

    def _lookup_per_feature_stat(self, task_ref: Dict, feature: str):
        """Look up per_feature_stats for a feature, trying direct key then {feature}_mean suffix."""
        pf = task_ref.get('per_feature_stats', {})
        stats = pf.get(feature)
        if stats is not None and 'mean' in stats:
            return stats
        stats = pf.get(f'{feature}_mean')
        if stats is not None and 'mean' in stats:
            return stats
        return None

    def _overlay_task_profiles(self, ax, feature: str, features_df: pd.DataFrame,
                               repetitions: List,
                               all_task_profiles: Optional[Dict] = None,
                               task_profile_ref: Optional[Dict] = None,
                               max_duration: float = 0,
                               standardization_stats: Optional[Dict] = None) -> None:
        """Overlay robust MAD-based profile bands on an axes object.

        Draws two bands per task segment:
          - Outer band (±2 robust σ): lighter fill
          - Inner band (±1 robust σ): more intense fill
          - 95 % CI strip: tighter, slightly more opaque than ±1 σ so it reads
            as the confidence envelope around the mean
        Falls back to std_pattern / std when mad_pattern / mad is unavailable.
        For features with stored activation_pattern curves the full temporal
        shape is drawn; others get a horizontal band from per_feature_stats.
        """
        OUTER_COLOR = '#E8D5F5'
        INNER_COLOR = COLORBLIND_SAFE_PALETTE['lavender']
        CI_COLOR    = '#7c4fc7'

        if all_task_profiles and 'task_id' in features_df.columns:
            segments = self._detect_task_segments(features_df, repetitions)
            legend_added = False
            for tk, seg_info in sorted(segments.items(), key=lambda x: x[1]['avg_start']):
                task_ref = all_task_profiles.get(tk)
                if task_ref is None:
                    continue

                pattern = task_ref.get('activation_pattern', {}).get(feature)
                has_curve = pattern is not None and 'mean_pattern' in pattern
                prof_n = (pattern or {}).get('n', (pattern or {}).get('n_curves', task_ref.get('n_repetitions_total', task_ref.get('n', 0)))) if has_curve else task_ref.get('n_repetitions_total', task_ref.get('n', 0))

                if has_curve:
                    ref_median = np.array(pattern['mean_pattern'], dtype=float)
                    if 'mad_pattern' in pattern:
                        robust_1s = self._robust_sigma(pattern['mad_pattern'])
                    else:
                        robust_1s = np.array(pattern.get('std_pattern', np.zeros_like(ref_median)), dtype=float)
                    ref_median, robust_1s = self._to_z_space(ref_median, robust_1s, feature, standardization_stats)
                    robust_2s = robust_1s * 2.0
                    t_task = np.linspace(seg_info['avg_start'], seg_info['avg_end'], len(ref_median))

                    ax.fill_between(t_task, ref_median - robust_2s, ref_median + robust_2s,
                                    color=OUTER_COLOR, alpha=0.13,
                                    label='Profile ±2σ' if not legend_added else None,
                                    zorder=0)
                    ax.fill_between(t_task, ref_median - robust_1s, ref_median + robust_1s,
                                    color=INNER_COLOR, alpha=0.28,
                                    label='Profile ±1σ (MAD)' if not legend_added else None,
                                    zorder=1)
                    if prof_n and prof_n > 1:
                        from scipy.stats import t as _t_ov
                        se_arr = robust_1s / np.sqrt(prof_n)
                        tc_ov  = _t_ov.ppf(0.975, df=prof_n - 1)
                        ax.fill_between(t_task,
                                        ref_median - tc_ov * se_arr,
                                        ref_median + tc_ov * se_arr,
                                        color=CI_COLOR, alpha=0.14,
                                        label='Profile 95% CI' if not legend_added else None,
                                        zorder=2)
                    ax.plot(t_task, ref_median, color=INNER_COLOR,
                            linewidth=1.8, linestyle='-', alpha=0.9,
                            label='Task Profile Median' if not legend_added else None,
                            zorder=3)
                else:
                    feat_stats = self._lookup_per_feature_stat(task_ref, feature)
                    if feat_stats is None:
                        continue
                    ref_m = feat_stats.get('median', feat_stats['mean'])
                    mad_val = feat_stats.get('mad', 0.0)
                    robust_1s_val = mad_val * 1.4826 if mad_val > 0 else feat_stats.get('std', 0.0)
                    ref_m, robust_1s_val = self._to_z_space(ref_m, robust_1s_val, feature, standardization_stats)
                    robust_2s_val = robust_1s_val * 2.0
                    t0, t1 = seg_info['avg_start'], seg_info['avg_end']
                    ax.fill_between([t0, t1], ref_m - robust_2s_val, ref_m + robust_2s_val,
                                    color=OUTER_COLOR, alpha=0.13,
                                    label='Profile ±2σ' if not legend_added else None,
                                    zorder=0)
                    ax.fill_between([t0, t1], ref_m - robust_1s_val, ref_m + robust_1s_val,
                                    color=INNER_COLOR, alpha=0.28,
                                    label='Profile ±1σ (MAD)' if not legend_added else None,
                                    zorder=1)
                    if prof_n and prof_n > 1:
                        from scipy.stats import t as _t_ov
                        se_val = robust_1s_val / np.sqrt(prof_n)
                        tc_ov  = _t_ov.ppf(0.975, df=prof_n - 1)
                        ax.fill_between([t0, t1],
                                        ref_m - tc_ov * se_val,
                                        ref_m + tc_ov * se_val,
                                        color=CI_COLOR, alpha=0.14,
                                        label='Profile 95% CI' if not legend_added else None,
                                        zorder=2)
                    ax.hlines(ref_m, t0, t1, colors=INNER_COLOR,
                              linewidth=1.8, linestyle='-', alpha=0.9,
                              label='Task Profile Median' if not legend_added else None,
                              zorder=3)

                legend_added = True
            return

        if task_profile_ref is not None and max_duration > 0:
            pattern = task_profile_ref.get('activation_pattern', {}).get(feature)
            prof_n = (pattern or {}).get('n', (pattern or {}).get('n_curves', task_profile_ref.get('n_repetitions_total', task_profile_ref.get('n', 0))))
            if pattern and 'mean_pattern' in pattern:
                ref_median = np.array(pattern['mean_pattern'], dtype=float)
                if 'mad_pattern' in pattern:
                    robust_1s = self._robust_sigma(pattern['mad_pattern'])
                else:
                    robust_1s = np.array(pattern.get('std_pattern', np.zeros_like(ref_median)), dtype=float)
                ref_median, robust_1s = self._to_z_space(ref_median, robust_1s, feature, standardization_stats)
                robust_2s = robust_1s * 2.0
                n_pts = len(ref_median)
                t_norm = np.linspace(0, max_duration, n_pts)
                ax.fill_between(t_norm, ref_median - robust_2s, ref_median + robust_2s,
                                color=OUTER_COLOR, alpha=0.13,
                                label='Profile ±2σ', zorder=0)
                ax.fill_between(t_norm, ref_median - robust_1s, ref_median + robust_1s,
                                color=INNER_COLOR, alpha=0.28,
                                label='Profile ±1σ (MAD)', zorder=1)
                if prof_n and prof_n > 1:
                    from scipy.stats import t as _t_ov
                    se_arr = robust_1s / np.sqrt(prof_n)
                    tc_ov  = _t_ov.ppf(0.975, df=prof_n - 1)
                    ax.fill_between(t_norm,
                                    ref_median - tc_ov * se_arr,
                                    ref_median + tc_ov * se_arr,
                                    color=CI_COLOR, alpha=0.14,
                                    label='Profile 95% CI', zorder=2)
                ax.plot(t_norm, ref_median, color=INNER_COLOR,
                        linewidth=1.8, linestyle='-', alpha=0.9,
                        label='Task Profile Median', zorder=3)

    def plot_timeseries(self, features_df: pd.DataFrame, feature_columns: List[str],
                       output_path: Path, title: str = "Feature Timeseries",
                       show_confidence: bool = True) -> None:
        """Line plot of one or more feature columns against relative time.

        Produces a vertically stacked set of axes (one per feature), sharing a
        common x-axis, with a dashed zero line and grid.  Prefers the
        ``time_rel_sec`` column for the x-axis; falls back to ``timestamp_abs``
        when absent.  The figure is saved as a PNG via :meth:`_save_figure`.

        Parameters
        ----------
        features_df:
            Frame-level features DataFrame with a time column and the requested
            feature columns.
        feature_columns:
            List of column names to plot.  Columns not present in *features_df*
            produce a "not found" text label rather than raising an error.
        output_path:
            Destination path; the suffix is replaced with ``.png``.
        title:
            Super-title for the figure.
        show_confidence:
            Reserved for future confidence-interval overlay; currently unused.
        """
        fig, axes = plt.subplots(len(feature_columns), 1,
                                figsize=(12, 3 * len(feature_columns)), sharex=True)

        if len(feature_columns) == 1:
            axes = [axes]

        time_col = 'time_rel_sec' if 'time_rel_sec' in features_df.columns else 'timestamp_abs'

        for ax, col in zip(axes, feature_columns):
            if col not in features_df.columns:
                ax.text(0.5, 0.5, f"Feature '{col}' not found", ha='center', va='center')
                continue

            time_data = features_df[time_col].values
            feature_data = features_df[col].values

            valid_mask = ~np.isnan(feature_data)

            ax.plot(time_data[valid_mask], feature_data[valid_mask],
                   color=self.colors.get('primary', '#2E86AB'),
                   linewidth=self.general.get('line_width', 1.5))

            ax.set_ylabel(col)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        axes[-1].set_xlabel(self.config.get('time_axis', {}).get('label', 'Relative Time (s)'))
        axes[0].set_title(title)

        plt.tight_layout()
        self._save_figure(fig, output_path)
        plt.close(fig)

    def plot_repetition_overlay(self, features_df: pd.DataFrame, feature: str,
                                output_path: Path, title: str = "Repetition Overlay",
                                baseline_stats: Optional[Dict] = None,
                                reference_baseline_stats: Optional[Dict] = None,
                                task_profile_ref: Optional[Dict] = None,
                                all_task_profiles: Optional[Dict] = None) -> None:
        """Multi-page PDF: all repetitions of *feature* overlaid per task, up to 4 tasks per page.

        Each task panel shows individual repetition traces (tab10 colours),
        a black dashed mean curve, optional MAD-based profile bands (lavender
        ±1 sigma / light purple ±2 sigma / dark purple 95 % CI), and a coral
        reference-baseline horizontal line.

        Parameters
        ----------
        features_df:
            Frame-level features with ``task_group``, ``task_id``, ``repetition``,
            ``timestamp_abs``, and the requested *feature*.
        feature:
            Column name to plot on each task panel.
        output_path:
            Destination path; the suffix is replaced with ``.pdf``.
        baseline_stats:
            Session neutral-baseline statistics dict (used for z-score conversion
            of profile bands and for the baseline reference line).
        reference_baseline_stats:
            External reference session baseline dict.  Drawn as a coral horizontal
            line showing where the participant's healthy baseline falls.
        task_profile_ref:
            Single-task kinematic reference profile dict.
        all_task_profiles:
            Full cross-task profile dict keyed by ``"TG_TID"``.
        """
        from matplotlib.backends.backend_pdf import PdfPages

        if feature not in features_df.columns:
            return

        task_pages = self._build_task_pages_by_task(features_df)
        tasks_per_page = 4
        pdf_path = output_path.with_suffix('.pdf')
        output_config = self.config.get('output', {})

        with PdfPages(pdf_path) as pdf:
            for batch_start in range(0, len(task_pages), tasks_per_page):
                batch = task_pages[batch_start:batch_start + tasks_per_page]
                n_tasks = len(batch)
                n_cols = min(2, n_tasks)
                n_rows = (n_tasks + n_cols - 1) // n_cols

                fig, axes = plt.subplots(n_rows, n_cols,
                                         figsize=(7 * n_cols, 5 * n_rows), squeeze=False)
                axes_flat = axes.flatten()

                for t_idx, (task_label, task_key, task_df) in enumerate(batch):
                    ax = axes_flat[t_idx]
                    if 'repetition' not in task_df.columns:
                        ax.set_visible(False)
                        continue
                    repetitions = sorted([r for r in task_df['repetition'].unique() if r != 0])
                    all_data = []
                    max_duration = 0

                    for i, rep in enumerate(repetitions):
                        rep_df = task_df[task_df['repetition'] == rep]
                        if len(rep_df) == 0 or feature not in rep_df.columns:
                            continue
                        if 'timestamp_abs' not in rep_df.columns:
                            continue
                        start_time = rep_df['timestamp_abs'].min()
                        time_rel = (rep_df['timestamp_abs'] - start_time).values
                        feature_vals = rep_df[feature].values
                        max_duration = max(max_duration, time_rel.max())
                        color = REPETITION_COLORS[i % len(REPETITION_COLORS)]
                        ax.plot(time_rel, feature_vals, color=color,
                                alpha=0.7, linewidth=2, label=f'Rep {int(rep)}', zorder=4)
                        all_data.append((time_rel, feature_vals))

                    if len(all_data) > 1:
                        min_len = min(len(d[0]) for d in all_data)
                        if min_len > 0:
                            aligned_data = np.array([d[1][:min_len] for d in all_data])
                            mean_curve = np.mean(aligned_data, axis=0)
                            time_axis = all_data[0][0][:min_len]
                            ax.plot(time_axis, mean_curve, color='black',
                                    linewidth=3, label='Mean', linestyle='--', zorder=10)

                    if task_profile_ref is not None or all_task_profiles is not None:
                        self._overlay_task_profiles(ax, feature, task_df, repetitions,
                                                    all_task_profiles=all_task_profiles,
                                                    task_profile_ref=task_profile_ref,
                                                    max_duration=max_duration,
                                                    standardization_stats=baseline_stats)

                    baseline_vals_for_ylim = []
                    ref_val = self._get_derived_baseline_value(reference_baseline_stats, feature,
                                                                standardization_stats=baseline_stats)
                    if ref_val is not None and abs(ref_val) > 1e-6:
                        ref_label = f'{ref_val:.3f}' if abs(ref_val) >= 0.001 else f'{ref_val:.1e}'
                        ax.axhline(y=ref_val, color=COLORBLIND_SAFE_PALETTE['coral'],
                                   linestyle='-', linewidth=2,
                                   label=f'Ref Baseline ({ref_label})', alpha=0.7)
                        baseline_vals_for_ylim.append(ref_val)

                    if all_data:
                        all_vals = np.concatenate([d[1] for d in all_data])
                        if baseline_vals_for_ylim:
                            all_vals = np.concatenate([all_vals, baseline_vals_for_ylim])
                        valid_vals = all_vals[~np.isnan(all_vals) & ~np.isinf(all_vals)]
                        if len(valid_vals) > 0:
                            ymin, ymax = float(np.min(valid_vals)), float(np.max(valid_vals))
                            margin = (ymax - ymin) * 0.1 if ymax != ymin else 0.1
                            ax.set_ylim(ymin - margin, ymax + margin)

                    ax.set_xlim(0, max_duration * 1.02 if max_duration > 0 else 1)
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel(feature.replace('_', ' ').title())
                    short_label = task_label.split(': ', 1)[-1] if ': ' in task_label else task_label
                    ax.set_title(short_label, fontsize=11, fontweight='bold')
                    n_legend_items = len(repetitions) + 3
                    if n_legend_items > 8:
                        ax.legend(loc='upper left', fontsize=6, framealpha=0.85,
                                  bbox_to_anchor=(1.02, 1), borderaxespad=0,
                                  ncol=max(1, n_legend_items // 8))
                    else:
                        ax.legend(loc='upper right', fontsize=7, framealpha=0.85,
                                  bbox_to_anchor=(1.0, 1.0))
                    ax.grid(True, alpha=0.3)

                for j in range(len(batch), len(axes_flat)):
                    axes_flat[j].set_visible(False)

                page_num = batch_start // tasks_per_page + 1
                total_pages = (len(task_pages) + tasks_per_page - 1) // tasks_per_page
                fig.suptitle(f"{title} (Page {page_num}/{total_pages})",
                             fontsize=14, fontweight='bold')
                plt.tight_layout()
                pdf.savefig(fig, dpi=output_config.get('save_dpi', 300), bbox_inches='tight')
                plt.close(fig)

    def plot_activation_per_repetition(self, features_df: pd.DataFrame, feature: str,
                                       output_path: Path, title: str = "Activation per Repetition",
                                       baseline_stats: Optional[Dict] = None,
                                       reference_baseline_stats: Optional[Dict] = None,
                                       task_profile_ref: Optional[Dict] = None,
                                       all_task_profiles: Optional[Dict] = None) -> None:
        """Multi-page PDF: one page per (task_group, task_id), all repetitions overlaid on one plot.

        All repetitions are drawn on a single axes with distinct colours, a mean
        curve in black dashed, optional ±1 sigma / ±2 sigma profile bands, and the
        reference baseline as a horizontal line.  This keeps each task to one
        page regardless of repetition count.  Mean is computed by interpolating all
        repetitions to a common time grid (200 points) before averaging.

        Parameters
        ----------
        features_df:
            Frame-level features DataFrame for the session.
        feature:
            Column name to plot.
        output_path:
            Destination path; the suffix is replaced with ``.pdf``.
        baseline_stats:
            Session neutral-baseline statistics dict (used for z-score conversion).
        reference_baseline_stats:
            External reference session baseline dict.
        task_profile_ref:
            Single-task kinematic reference profile dict.
        all_task_profiles:
            Full cross-task profile dict keyed by ``"TG_TID"``.
        """
        from matplotlib.backends.backend_pdf import PdfPages

        if feature not in features_df.columns:
            return

        task_pages = self._build_task_pages_by_task(features_df)
        pdf_path = output_path.with_suffix('.pdf')
        output_config = self.config.get('output', {})

        with PdfPages(pdf_path) as pdf:
            for task_label, task_key, task_df in task_pages:
                if 'repetition' not in task_df.columns:
                    continue
                repetitions = sorted([r for r in task_df['repetition'].unique() if r != 0])
                if not repetitions:
                    repetitions = sorted(task_df['repetition'].unique())
                if not repetitions:
                    continue

                tp = None
                if all_task_profiles and task_key:
                    tp = all_task_profiles.get(task_key)
                if tp is None and task_profile_ref is not None:
                    tp = task_profile_ref

                fig, ax = plt.subplots(figsize=(9, 5))

                all_data = []
                max_duration = 0.0

                for i, rep in enumerate(repetitions):
                    rep_df = task_df[task_df['repetition'] == rep]
                    if len(rep_df) == 0 or feature not in rep_df.columns:
                        continue
                    if 'timestamp_abs' not in rep_df.columns:
                        continue
                    start_time = rep_df['timestamp_abs'].min()
                    t_rel = (rep_df['timestamp_abs'] - start_time).values
                    vals = rep_df[feature].values
                    max_duration = max(max_duration, t_rel.max() if len(t_rel) > 0 else 0)
                    color = REPETITION_COLORS[i % len(REPETITION_COLORS)]
                    ax.plot(t_rel, vals, color=color, linewidth=1.8, alpha=0.75,
                            label=f'Rep {int(rep)}', zorder=4)
                    all_data.append((t_rel, vals))

                if len(all_data) > 1:
                    n_interp = 200
                    common_t = np.linspace(0, max_duration, n_interp)
                    interp_vals = []
                    for t_arr, v_arr in all_data:
                        if len(t_arr) >= 2:
                            interp_vals.append(np.interp(common_t, t_arr, v_arr))
                    if len(interp_vals) > 1:
                        arr = np.array(interp_vals)
                        mean_curve = np.nanmean(arr, axis=0)
                        if np.any(np.isfinite(mean_curve)):
                            ax.plot(common_t, mean_curve, color='#212121',
                                linewidth=2.5, linestyle='--', label='Mean', zorder=7)
                if tp is not None and max_duration > 0:
                    pattern = tp.get('activation_pattern', {}).get(feature, None)
                    if pattern and 'mean_pattern' in pattern:
                        ref_median = np.array(pattern['mean_pattern'], dtype=float)
                        if 'mad_pattern' in pattern:
                            robust_1s = self._robust_sigma(pattern['mad_pattern'])
                        else:
                            robust_1s = np.array(pattern.get('std_pattern', np.zeros_like(ref_median)), dtype=float)
                        ref_median, robust_1s = self._to_z_space(ref_median, robust_1s, feature, baseline_stats)
                        robust_2s = robust_1s * 2.0
                        t_norm = np.linspace(0, max_duration, len(ref_median))
                        prof_n = pattern.get('n', pattern.get('n_curves', tp.get('n_repetitions_total', tp.get('n', 0))))
                        ax.fill_between(t_norm, ref_median - robust_2s, ref_median + robust_2s,
                                        color='#E8D5F5', alpha=0.13, zorder=0, label='Profile ±2σ')
                        ax.fill_between(t_norm, ref_median - robust_1s, ref_median + robust_1s,
                                        color=COLORBLIND_SAFE_PALETTE['lavender'], alpha=0.28, zorder=1, label='Profile ±1σ')
                        if prof_n and prof_n > 1:
                            from scipy.stats import t as _t_pr
                            se_pr = robust_1s / np.sqrt(prof_n)
                            tc_pr = _t_pr.ppf(0.975, df=prof_n - 1)
                            ax.fill_between(t_norm,
                                            ref_median - tc_pr * se_pr,
                                            ref_median + tc_pr * se_pr,
                                            color='#7c4fc7', alpha=0.14, zorder=2, label='Profile 95% CI')
                        ax.plot(t_norm, ref_median, color=COLORBLIND_SAFE_PALETTE['lavender'],
                                linewidth=1.6, linestyle='-', alpha=0.9, label='Profile Median', zorder=3)
                    else:
                        feat_stats = self._lookup_per_feature_stat(tp, feature)
                        if feat_stats and max_duration > 0:
                            ref_m = feat_stats.get('median', feat_stats['mean'])
                            mad_val = feat_stats.get('mad', 0.0)
                            r1s = mad_val * 1.4826 if mad_val > 0 else feat_stats.get('std', 0.0)
                            ref_m, r1s = self._to_z_space(ref_m, r1s, feature, baseline_stats)
                            r2s = r1s * 2.0
                            prof_n = tp.get('n_repetitions_total', tp.get('n', 0))
                            ax.axhspan(ref_m - r2s, ref_m + r2s, alpha=0.10,
                                       color='#E8D5F5', zorder=0, label='Profile ±2σ')
                            ax.axhspan(ref_m - r1s, ref_m + r1s, alpha=0.22,
                                       color=COLORBLIND_SAFE_PALETTE['lavender'], zorder=1, label='Profile ±1σ')
                            if prof_n and prof_n > 1:
                                from scipy.stats import t as _t_pr
                                se_pr = r1s / np.sqrt(prof_n)
                                tc_pr = _t_pr.ppf(0.975, df=prof_n - 1)
                                ax.axhspan(ref_m - tc_pr * se_pr, ref_m + tc_pr * se_pr,
                                           alpha=0.14, color='#7c4fc7', zorder=2, label='Profile 95% CI')
                            ax.axhline(ref_m, color=COLORBLIND_SAFE_PALETTE['lavender'],
                                       linewidth=1.6, linestyle='-', alpha=0.9,
                                       label='Profile Median', zorder=3)

                ref_val = self._get_derived_baseline_value(reference_baseline_stats, feature,
                                                            standardization_stats=baseline_stats)
                if ref_val is not None and abs(ref_val) > 1e-6:
                    ax.axhline(y=ref_val, color=COLORBLIND_SAFE_PALETTE['coral'],
                               linestyle='-', linewidth=1.5, alpha=0.7, label='Reference Baseline')

                if max_duration > 0:
                    ax.set_xlim(0, max_duration * 1.02)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel(feature.replace('_', ' ').title())
                ax.set_title(f'{title}: {task_label}', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(loc='best', fontsize=8, framealpha=0.85,
                          ncol=min(len(repetitions) + 2, 5))
                fig.tight_layout()
                pdf.savefig(fig, dpi=output_config.get('save_dpi', 300), bbox_inches='tight')
                plt.close(fig)

    def plot_asymmetry_over_time(self, features_df: pd.DataFrame, output_path: Path,
                                title: str = "Asymmetry Over Time",
                                baseline_stats: Optional[Dict] = None,
                                reference_baseline_stats: Optional[Dict] = None,
                                all_task_profiles: Optional[Dict] = None) -> None:
        """Multi-page PDF: per-task asymmetry analysis.

        One page per task.  All repetitions are shown together on that page —
        each rep occupies its own framed cell (box outline) arranged in a
        2-column grid.  Each cell contains two panels:
          Top    — time-series of top-5 most variable asymmetry ratios with
                   smoothed trend lines, absolute severity bands, and (when
                   reference data is available) ±1 SD reference profile bands.
          Middle — signed mean asymmetry bar chart sorted by magnitude, with
                   reference mean dots overlaid when a reference is available.
          Bottom — delta bar chart comparing this repetition's mean to the
                   reference profile (or, if no reference, to the neutral
                   session baseline).
        """
        import math
        from matplotlib.backends.backend_pdf import PdfPages
        from matplotlib.gridspec import GridSpecFromSubplotSpec

        asymmetry_cols = [c for c in features_df.columns if c.startswith('asymmetry_ratio_')]
        if not asymmetry_cols:
            return

        task_pages    = self._build_task_pages_by_task(features_df)
        output_config = self.config.get('output', {})
        pdf_path      = output_path.with_suffix('.pdf')
        time_col      = 'time_rel_sec' if 'time_rel_sec' in features_df.columns else 'timestamp_abs'

        C_MILD  = COLORBLIND_SAFE_PALETTE['orange']
        C_MOD   = '#D32F2F'
        C_OK    = COLORBLIND_SAFE_PALETTE['green']

        with PdfPages(pdf_path) as pdf:
            for task_label, task_key, task_df in task_pages:
                task_asym_cols = [c for c in asymmetry_cols if c in task_df.columns]
                if not task_asym_cols:
                    continue

                repetitions = sorted([r for r in task_df['repetition'].unique() if r != 0])
                if not repetitions:
                    repetitions = [None]

                short_label = task_label.split(': ', 1)[-1] if ': ' in task_label else task_label
                n_reps  = len(repetitions)
                n_cols  = min(2, n_reps)
                n_rows  = math.ceil(n_reps / n_cols)

                fig_w = 16
                fig_h = max(8, 6.5 * n_rows + 1.5)
                fig = plt.figure(figsize=(fig_w, fig_h))

                outer_gs = plt.GridSpec(
                    n_rows, n_cols, figure=fig,
                    hspace=0.70, wspace=0.45,
                    top=0.88, bottom=0.05, left=0.07, right=0.97,
                )

                for rep_idx, rep_id in enumerate(repetitions):
                    row_i = rep_idx // n_cols
                    col_i = rep_idx % n_cols

                    rep_df = task_df[task_df['repetition'] == rep_id] if rep_id is not None else task_df
                    if rep_df.empty:
                        continue

                    rep_asym_cols = [c for c in task_asym_cols if c in rep_df.columns]
                    abs_means = {col: rep_df[col].abs().mean() for col in rep_asym_cols}
                    top_cols  = sorted(abs_means, key=abs_means.get, reverse=True)[:5]

                    has_baseline  = baseline_stats is not None and bool(baseline_stats)
                    has_reference = (
                        (reference_baseline_stats is not None and bool(reference_baseline_stats))
                        or (all_task_profiles is not None and bool(all_task_profiles))
                    )
                    _n_inner = 3 if (has_baseline or has_reference) else 2
                    _hr = [2.2, 1.6, 1.1] if _n_inner == 3 else [2.2, 1.6]
                    inner_gs = GridSpecFromSubplotSpec(
                        _n_inner, 1,
                        subplot_spec=outer_gs[row_i, col_i],
                        height_ratios=_hr,
                        hspace=0.60,
                    )
                    ax_ts  = fig.add_subplot(inner_gs[0])
                    ax_bar = fig.add_subplot(inner_gs[1])
                    ax_delta = fig.add_subplot(inner_gs[2]) if (has_baseline or has_reference) else None

                    bbox = outer_gs[row_i, col_i].get_position(fig)
                    pad = 0.008
                    rect = plt.Rectangle(
                        (bbox.x0 - pad, bbox.y0 - pad),
                        bbox.width + 2 * pad,
                        bbox.height + 2 * pad,
                        linewidth=1.4,
                        edgecolor='#BBBBBB',
                        facecolor='#FAFAFA',
                        transform=fig.transFigure,
                        zorder=0,
                    )
                    fig.add_artist(rect)

                    rep_title = f'Rep {int(rep_id)}' if rep_id is not None else 'All data'
                    ax_ts.set_title(rep_title, fontsize=9, fontweight='bold', pad=10)

                    t = rep_df[time_col].values
                    for i, col in enumerate(top_cols):
                        region = col.replace('asymmetry_ratio_', '').replace('_', ' ').title()
                        color  = REPETITION_COLORS[i % len(REPETITION_COLORS)]
                        y      = rep_df[col].values
                        ax_ts.plot(t, y, color=color, alpha=0.40, linewidth=1.0)
                        win = max(3, len(y) // 20)
                        if len(y) >= win:
                            y_smooth = pd.Series(y).rolling(window=win, center=True, min_periods=1).mean().values
                            ax_ts.plot(t, y_smooth, color=color, alpha=0.90,
                                       linewidth=1.8, label=region)

                    for thresh, col_band in [(0.15, C_MILD), (0.25, C_MOD)]:
                        ax_ts.axhline(y=thresh,  color=col_band, linestyle='--', alpha=0.65, linewidth=1.0)
                        ax_ts.axhline(y=-thresh, color=col_band, linestyle='--', alpha=0.65, linewidth=1.0)
                    ax_ts.axhline(y=0, color='#333333', linewidth=0.9)

                    _ref_means_for_ts: Dict[str, float] = {}
                    _ref_stds_for_ts:  Dict[str, float] = {}
                    for _col in top_cols:
                        _ref_val_ts = None
                        _ref_std_ts = 0.0
                        if all_task_profiles and task_key:
                            _tp_ref = all_task_profiles.get(task_key, {})
                            _feat_stat = self._lookup_per_feature_stat(_tp_ref, _col)
                            if _feat_stat:
                                _ref_val_ts = _feat_stat.get('median', _feat_stat.get('mean'))
                                _ref_std_ts = _feat_stat.get('mad', 0.0) * 1.4826 if _feat_stat.get('mad', 0.0) > 0 else _feat_stat.get('std', 0.0)
                        if _ref_val_ts is None and reference_baseline_stats is not None:
                            _rbstat = reference_baseline_stats.get(_col, {})
                            if isinstance(_rbstat, dict):
                                _ref_val_ts = _rbstat.get('median', _rbstat.get('mean'))
                                _ref_std_ts = _rbstat.get('mad', 0.0) * 1.4826 if _rbstat.get('mad', 0.0) > 0 else _rbstat.get('std', 0.0)
                        if _ref_val_ts is not None:
                            _ref_means_for_ts[_col] = float(_ref_val_ts)
                            _ref_stds_for_ts[_col]  = float(_ref_std_ts)

                    if _ref_means_for_ts:
                        ref_band_plotted = False
                        for i, _col in enumerate(top_cols):
                            if _col not in _ref_means_for_ts:
                                continue
                            _rv = _ref_means_for_ts[_col]
                            _rs = _ref_stds_for_ts.get(_col, 0.0)
                            _rc = REPETITION_COLORS[i % len(REPETITION_COLORS)]
                            lbl = 'Ref ±1 SD' if not ref_band_plotted else None
                            ax_ts.axhspan(_rv - _rs, _rv + _rs, alpha=0.10, color=_rc, zorder=0)
                            ax_ts.axhline(_rv, color=_rc, linestyle=':', linewidth=1.4,
                                          alpha=0.80, label=lbl, zorder=1)
                            ref_band_plotted = True

                    data_abs_max = max(rep_df[rep_asym_cols].abs().max().max(), 0.30)
                    ax_ts.set_ylim(-data_abs_max * 1.1, data_abs_max * 1.1)
                    ax_ts.set_ylabel('Asymmetry Ratio', fontsize=7)
                    ax_ts.set_xlabel(self.config.get('time_axis', {}).get('label', 'Time (s)'), fontsize=7)
                    ax_ts.legend(ncol=1, fontsize=6.5, loc='upper left',
                                 bbox_to_anchor=(1.01, 1.0), borderaxespad=0,
                                 framealpha=0.85)
                    ax_ts.grid(True, alpha=0.22)
                    ax_ts.tick_params(labelsize=7)

                    all_regions_mean = {}
                    for col in rep_asym_cols[:14]:
                        region = col.replace('asymmetry_ratio_', '').replace('_', ' ')[:16]
                        all_regions_mean[region] = float(rep_df[col].mean())

                    sorted_items = sorted(all_regions_mean.items(), key=lambda x: x[1])
                    reg_names = [it[0] for it in sorted_items]
                    reg_means = [it[1] for it in sorted_items]
                    bar_cols  = [C_MOD if abs(v) >= 0.25 else (C_MILD if abs(v) >= 0.15 else C_OK)
                                 for v in reg_means]

                    ax_bar.barh(reg_names, reg_means, color=bar_cols, alpha=0.85,
                                edgecolor='none', height=0.60)
                    ax_bar.axvline(x=0, color='#333333', linewidth=1.0)
                    ax_bar.axvline(x=0.15,  color=C_MILD, linestyle='--', alpha=0.55, linewidth=0.9)
                    ax_bar.axvline(x=-0.15, color=C_MILD, linestyle='--', alpha=0.55, linewidth=0.9)

                    _bar_ref_plotted = False
                    for _bi, (_reg, _mean_v) in enumerate(sorted_items):
                        _orig_col = f'asymmetry_ratio_{_reg.replace(" ", "_")}'
                        _ref_m = _ref_means_for_ts.get(_orig_col)
                        if _ref_m is None:
                            for _c in top_cols:
                                if _c.replace('asymmetry_ratio_', '').replace('_', ' ')[:16] == _reg:
                                    _ref_m = _ref_means_for_ts.get(_c)
                                    break
                        if _ref_m is not None:
                            lbl = 'Ref mean' if not _bar_ref_plotted else None
                            ax_bar.scatter([_ref_m], [_bi], color='black', s=28,
                                           zorder=5, marker='D', label=lbl)
                            _bar_ref_plotted = True

                    lim = max(abs(min(reg_means, default=0)), abs(max(reg_means, default=0)), 0.30) * 1.2
                    ax_bar.set_xlim(-lim, lim)
                    ax_bar.set_title('Mean Signed Asymmetry  (L ← 0 → R)', fontsize=7.5, fontweight='bold', pad=3)
                    ax_bar.set_xlabel('Asymmetry Ratio', fontsize=7)
                    ax_bar.tick_params(axis='y', labelsize=6.5)
                    ax_bar.tick_params(axis='x', labelsize=7)
                    ax_bar.spines['top'].set_visible(False)
                    ax_bar.spines['right'].set_visible(False)
                    if _bar_ref_plotted:
                        ax_bar.legend(fontsize=6.5, loc='upper right')

                    if ax_delta is not None:
                        _compare_stats = None
                        _delta_title   = '\u0394 vs Baseline'
                        _delta_xlabel  = '\u0394 from baseline'
                        if _ref_means_for_ts:
                            _compare_stats = {
                                col: {'mean': _ref_means_for_ts[col]}
                                for col in _ref_means_for_ts
                            }
                            _delta_title  = '\u0394 vs Reference'
                            _delta_xlabel = '\u0394 from reference'
                        elif has_baseline and baseline_stats:
                            _compare_stats = {
                                col: baseline_stats[col]
                                for col in top_cols
                                if col in baseline_stats
                            }

                        delta_vals, delta_labels, delta_colors = [], [], []
                        if _compare_stats:
                            for col in top_cols:
                                _cs = _compare_stats.get(col)
                                if _cs is None:
                                    continue
                                ref_mean_val = float(_cs['mean'])
                                rep_mean_val = float(rep_df[col].mean())
                                delta = rep_mean_val - ref_mean_val
                                short = col.replace('asymmetry_ratio_', '').replace('_', ' ')
                                delta_vals.append(delta)
                                delta_labels.append(short)
                                delta_colors.append(
                                    C_MOD if abs(delta) > 0.20 else (C_MILD if abs(delta) > 0.10 else C_OK)
                                )
                        if delta_vals:
                            y_pos = list(range(len(delta_vals)))
                            ax_delta.barh(y_pos, delta_vals, color=delta_colors, alpha=0.80, height=0.65)
                            ax_delta.set_yticks(y_pos)
                            ax_delta.set_yticklabels(delta_labels, fontsize=6.5)
                            ax_delta.axvline(x=0, color='#333333', linewidth=1.0)
                            d_lim = max(abs(min(delta_vals)), abs(max(delta_vals)), 0.15) * 1.25
                            ax_delta.set_xlim(-d_lim, d_lim)
                            ax_delta.set_xlabel(_delta_xlabel, fontsize=7)
                            ax_delta.set_title(_delta_title, fontsize=8, fontweight='bold', pad=4)
                            ax_delta.spines['top'].set_visible(False)
                            ax_delta.spines['right'].set_visible(False)
                            ax_delta.tick_params(axis='x', labelsize=7)
                        else:
                            ax_delta.axis('off')

                fig.suptitle(
                    f'{title}  —  {short_label}',
                    fontsize=13, fontweight='bold', y=0.96,
                )
                pdf.savefig(fig, dpi=output_config.get('save_dpi', 150), bbox_inches='tight')
                plt.close(fig)


    def plot_metrics_summary(self, repetition_metrics_df: pd.DataFrame, output_path: Path,
                            title: str = "Metrics Summary",
                            baseline_stats: Optional[Dict] = None,
                            reference_baseline_stats: Optional[Dict] = None,
                            task_profile_ref: Optional[Dict] = None) -> None:
        """Bar chart of selected repetition-level metrics with baseline and profile overlays.

        Plots up to four metric columns (preferring ``mean_asymmetry_ratio``,
        ``max_asymmetry_ratio``, ``duration_sec``; falls back to ``_mean``
        columns or any numeric column) as grouped bar charts with one bar per
        repetition.  Baseline and profile reference values are drawn as horizontal
        lines when provided.  Saved as a PNG.

        Parameters
        ----------
        repetition_metrics_df:
            One row per (task, repetition) as produced by the metrics computer.
        output_path:
            Destination path; the suffix is replaced with ``.png``.
        baseline_stats:
            Session neutral-baseline statistics for reference line placement.
        reference_baseline_stats:
            External reference session baseline statistics.
        task_profile_ref:
            Reference task profile dict for optional profile-mean overlay.
        """
        if len(repetition_metrics_df) == 0:
            return

        task_info = ""
        if 'task_group' in repetition_metrics_df.columns or 'task_id' in repetition_metrics_df.columns:
            tg = repetition_metrics_df['task_group'].iloc[0] if 'task_group' in repetition_metrics_df.columns else None
            tid = repetition_metrics_df['task_id'].iloc[0] if 'task_id' in repetition_metrics_df.columns else None
            task_name = repetition_metrics_df['task_name'].iloc[0] if 'task_name' in repetition_metrics_df.columns else None
            if tg and tg != '0' and str(tg) != 'nan' and str(tg) != 'None':
                if task_name and task_name != '(no task selected)':
                    task_info = f" [{task_name}]"
                else:
                    task_info = f" [Task {tg}{tid if tid and tid != 0 else ''}]"

        metric_cols = ['mean_asymmetry_ratio', 'max_asymmetry_ratio', 'duration_sec']
        available_cols = [c for c in metric_cols if c in repetition_metrics_df.columns]

        if not available_cols:
            mean_cols = [c for c in repetition_metrics_df.columns if '_mean' in c][:4]
            available_cols = mean_cols if mean_cols else list(repetition_metrics_df.select_dtypes(include=[np.number]).columns)[:4]

        if not available_cols:
            return

        n_cols = len(available_cols)
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 5))

        if n_cols == 1:
            axes = [axes]

        if 'repetition' in repetition_metrics_df.columns:
            rep_df = repetition_metrics_df[repetition_metrics_df['repetition'] != 0].copy()
            repetitions = rep_df['repetition'].values
        else:
            rep_df = repetition_metrics_df.copy()
            repetitions = np.arange(1, len(rep_df) + 1)

        has_task_cols = (
            'task_group' in rep_df.columns
            and 'task_id' in rep_df.columns
        )
        has_task_name_col = 'task_name' in rep_df.columns

        def _make_x_labels(df: pd.DataFrame, rep_vals: np.ndarray) -> List[str]:
            """Build tick labels for repetition axis using task_name when available."""
            labels = []
            for i, rep in enumerate(rep_vals):
                if has_task_name_col:
                    tname = str(df['task_name'].iloc[i]) if i < len(df) else ''
                    tname = tname[:12] if tname and tname not in ('nan', '(no task selected)') else ''
                    labels.append(f'{tname}\nR{int(rep)}' if tname else f'R{int(rep)}')
                elif has_task_cols:
                    tg = str(df['task_group'].iloc[i]) if i < len(df) else ''
                    tid = str(df['task_id'].iloc[i]) if i < len(df) else ''
                    tg = '' if tg in ('nan', '0', 'None') else tg
                    labels.append(f'{tg}{tid}\nR{int(rep)}' if tg else f'R{int(rep)}')
                else:
                    labels.append(f'R{int(rep)}')
            return labels

        x_labels = _make_x_labels(rep_df, repetitions)

        for ax, col in zip(axes, available_cols):
            values = rep_df[col].values
            n_reps = len(values)

            bar_colors = [REPETITION_COLORS[i % len(REPETITION_COLORS)] for i in range(n_reps)]

            x_positions = np.arange(1, n_reps + 1)
            bars = ax.bar(x_positions, values, color=bar_colors, alpha=0.8, edgecolor='black')

            if len(values) > 0:
                mean_val = np.mean(values)
                ax.axhline(y=mean_val, color='black', linestyle='--', linewidth=2,
                          label=f'Mean: {mean_val:.3f}')

                if reference_baseline_stats and col in reference_baseline_stats:
                    ref_val = reference_baseline_stats[col].get('mean', 0)
                    if abs(ref_val) > 1e-6:
                        ax.axhline(y=ref_val, color=COLORBLIND_SAFE_PALETTE['coral'],
                                  linestyle='-', linewidth=1.5, alpha=0.7,
                                  label=f'Ref Baseline: {ref_val:.3f}')

                comp_stats = reference_baseline_stats if reference_baseline_stats else baseline_stats
                if comp_stats and col in comp_stats:
                    baseline_mean = comp_stats[col].get('mean', 0)
                    baseline_std = comp_stats[col].get('std', 1)
                    for i, (bar, val) in enumerate(zip(bars, values)):
                        if baseline_std > 0 and abs(val - baseline_mean) > 2.5 * baseline_std:
                            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                                   '★', ha='center', fontsize=10, color=COLORBLIND_SAFE_PALETTE['red'])

            if task_profile_ref is not None:
                pf_stats = task_profile_ref.get('per_feature_stats', {})
                if col in pf_stats:
                    pf_mean = pf_stats[col].get('mean', None)
                    pf_std = pf_stats[col].get('std', None)
                    if pf_mean is not None:
                        ax.axhline(y=pf_mean, color=COLORBLIND_SAFE_PALETTE['lavender'],
                                  linestyle='-', linewidth=2, label=f'Task Profile: {pf_mean:.3f}')
                        if pf_std is not None and pf_std > 0:
                            ax.axhspan(pf_mean - pf_std, pf_mean + pf_std,
                                      color=COLORBLIND_SAFE_PALETTE['lavender'], alpha=0.15)

            ax.set_xticks(x_positions)
            ax.set_xticklabels([], fontsize=7)
            ax.set_xlabel('Task / Repetition')
            ax.set_ylabel(col.replace('_', ' ').title())
            ax.set_title(col.replace('_', ' ').title())
            ax.legend(loc='best', fontsize=8, framealpha=0.85)
            ax.grid(True, alpha=0.3, axis='y')

            finite_values = values[np.isfinite(values)] if len(values) > 0 else np.array([])
            if len(finite_values) > 0:
                ymin, ymax = min(0, np.min(finite_values)), np.max(finite_values)
                margin = (ymax - ymin) * 0.15 if ymax != ymin else 0.1
                ax.set_ylim(ymin - margin * 0.5, ymax + margin)

        fig.suptitle(f"{title}{task_info}", fontsize=14, fontweight='bold')

        plt.tight_layout()
        self._save_figure(fig, output_path)
        plt.close(fig)

    def plot_screening_summary(
        self,
        screening_results: Dict[str, Any],
        output_path: Path,
        anomaly_results: Optional[Dict[str, Any]] = None,
        articulation_scores: Optional[Dict[str, Any]] = None,
        title: str = "Clinical Screening Report",
    ) -> None:
        """Single-image clinical screening report.

        Layout (2 rows × 3 cols):
          Row 0, col 0-1 : Task × Repetition anomaly score heatmap (wide)
          Row 0, col 2   : Confidence components gauge bars
          Row 1, col 0-1 : Screening indications, sorted by severity + confidence
          Row 1, col 2   : Overall verdict badge with key stats
        """
        import matplotlib.colors as mcolors
        from matplotlib.gridspec import GridSpec

        C_PASS = '#2E7D32'
        C_FAIL = '#C62828'
        C_MILD = '#F9A825'
        C_MOD  = '#E65100'
        C_SEV  = '#B71C1C'

        indications = screening_results.get('indications', [])
        confidence  = screening_results.get('confidence', {})
        overall_conf= confidence.get('overall', 0.0)
        n_ind       = len(indications)

        _ATYPE_LABEL: Dict[str, str] = {
            "facial_asymmetry":    "Facial Asymmetry",
            "side_amplitude":      "Side Asymmetry",
            "kinematic_profile":   "Kinematic Profile",
            "temporal_distortion": "Temporal Distortion",
            "articulation":        "Articulation",
            "amplitude_reduction": "Amplitude Reduction",
            "task_substitution":   "Task Substitution",
            "pattern_shift":       "Pattern Shift",
            "sustained_elevation": "Sustained Elevation",
            "transient_spike":     "Transient Spike",
            "drift":               "Drift",
            "unknown":             "Unknown",
            "timing_drop":         "Timing Drop",
            "smoothness_drop":     "Smoothness Drop",
            "amplitude_drop":      "Amplitude Drop",
            "component_drop":      "Multi-Component Drop",
        }
        _ATYPE_ABBR: Dict[str, str] = {
            "facial_asymmetry":    "FA",
            "side_amplitude":      "SA",
            "kinematic_profile":   "KP",
            "temporal_distortion": "TD",
            "articulation":        "AR",
            "amplitude_reduction": "AmpR",
            "task_substitution":   "TS",
            "pattern_shift":       "PS",
            "sustained_elevation": "SE",
            "transient_spike":     "TrS",
            "drift":               "Dr",
            "unknown":             "?",
            "timing_drop":         "Tdp",
            "smoothness_drop":     "Sdp",
            "amplitude_drop":      "Adp",
            "component_drop":      "Cdp",
        }
        _ATYPE_COLOR: Dict[str, str] = {
            "facial_asymmetry":    "#CC79A7",
            "side_amplitude":      "#E91E63",
            "kinematic_profile":   "#0072B2",
            "temporal_distortion": "#E69F00",
            "articulation":        "#009E73",
            "amplitude_reduction": "#D55E00",
            "task_substitution":   "#56B4E9",
            "pattern_shift":       "#795548",
            "sustained_elevation": "#9C27B0",
            "transient_spike":     "#FF7043",
            "drift":               "#78909C",
            "unknown":             "#BDBDBD",
            "timing_drop":         "#E69F00",
            "smoothness_drop":     "#0072B2",
            "amplitude_drop":      "#D55E00",
            "component_drop":      "#795548",
        }
        _FEAT_FRIENDLY: Dict[str, str] = {
            "kin_mouth_opening":        "mouth opening",
            "kin_lip_action":           "lip action",
            "kin_lip_action_y":         "lip action (vert)",
            "kin_lip_action_z":         "lip action (depth)",
            "kin_medial_upper_action_y":"upper lip (vert)",
            "kin_medial_lower_action_y":"lower lip (vert)",
            "kin_medial_sym_x":         "lip midline sym",
            "kin_mouth_area_symmetry":  "mouth area sym",
            "kin_labial_fissure_width": "lip fissure width",
            "kin_jaw_excursion":        "jaw excursion",
            "kin_medial_symmetry":      "medial symmetry",
            "mouthSmileLeft":           "smile (left)",
            "mouthSmileRight":          "smile (right)",
            "mouthFrownLeft":           "frown (left)",
            "mouthFrownRight":          "frown (right)",
            "mouthPucker":              "lip pucker",
            "mouthLeft":                "mouth (left)",
            "mouthRight":               "mouth (right)",
            "duration_sec":             "duration",
            "mean_activation":          "mean activation",
        }

        def _feat_label(feat: str) -> str:
            """Short human-readable label for a contributing feature."""
            for k, v in _FEAT_FRIENDLY.items():
                if k in feat:
                    return v
            s = feat.replace("kin_", "").replace("_mean", "").replace("_", " ")
            return s[:22]

        def _side_from_features(feat_counts: Dict[str, int]) -> str:
            """Infer dominant affected side from feature name keyword counts."""
            left  = sum(v for k, v in feat_counts.items()
                        if any(x in k for x in ("Left", "_left", "left_", "LeftEye")))
            right = sum(v for k, v in feat_counts.items()
                        if any(x in k for x in ("Right", "_right", "right_", "RightEye")))
            if left == 0 and right == 0:
                return ""
            if left > right * 1.4:
                return "  ← Left"
            if right > left * 1.4:
                return "  → Right"
            return "  ↔ Both"

        per_task_detail: List[Dict[str, Any]] = []
        if anomaly_results:
            _ptr = anomaly_results.get("per_task_results", [anomaly_results])
            for tr in _ptr:
                n_flag = sum(1 for v in tr.get("is_anomaly", []) if v)
                if n_flag == 0:
                    continue
                n_tot = max(len(tr.get("is_anomaly", [])), 1)
                tn_list = tr.get("task_names", [])
                first_name = tn_list[0] if tn_list else "?"
                dom_type = tr.get("summary", {}).get("dominant_anomaly_type", "unknown")
                type_bk  = tr.get("summary", {}).get("anomaly_type_breakdown", {})
                feat_counts: Dict[str, int] = {}
                for rep_cf in tr.get("contributing_features", []):
                    if isinstance(rep_cf, list):
                        for f in rep_cf:
                            feat_counts[f] = feat_counts.get(f, 0) + 1
                top_feats = sorted(feat_counts, key=feat_counts.get, reverse=True)[:3]
                side_note = _side_from_features(feat_counts)
                dev_vals  = [float(d) for d in tr.get("deviation_score", [])
                             if d is not None and not np.isnan(float(d))]
                mean_dev  = float(np.mean(dev_vals)) if dev_vals else 0.0
                per_task_detail.append({
                    "name":       first_name,
                    "dom_type":   dom_type,
                    "type_bk":    type_bk,
                    "top_feats":  top_feats,
                    "side":       side_note,
                    "n_flag":     n_flag,
                    "n_tot":      n_tot,
                    "mean_dev":   mean_dev,
                })

        if articulation_scores:
            _pt_devs = articulation_scores.get("per_task_deviations", {})
            _pt_sc   = articulation_scores.get("per_task_scores", {})
            _DROP_COMP: Dict[str, float] = {
                "timing":     -0.15,
                "smoothness": -0.07,
                "amplitude":  -0.20,
            }
            _COMP_TYPE: Dict[str, str] = {
                "timing":     "timing_drop",
                "smoothness": "smoothness_drop",
                "amplitude":  "amplitude_drop",
            }
            for _tk in sorted(_pt_devs.keys()):
                _devs    = _pt_devs[_tk]
                _ti      = _pt_sc.get(_tk, {})
                _tname   = _ti.get("task_name", _tk)
                _grp_c   = _tk.split("_")[0] if "_" in _tk else ""
                _disp    = f"{_grp_c}: {_tname}" if _grp_c else _tname
                _dropped = {
                    c: _devs[f"{c}_deviation"]
                    for c, thr in _DROP_COMP.items()
                    if f"{c}_deviation" in _devs and _devs[f"{c}_deviation"] < thr
                }
                if not _dropped:
                    continue
                _dom_c   = min(_dropped, key=lambda c: _dropped[c])
                _dom_ty  = _COMP_TYPE.get(_dom_c, "component_drop") if len(_dropped) == 1 else "component_drop"
                _abs_max = abs(min(_dropped.values()))
                _feats   = [f"Δ{c} {_dropped[c]:+.3f}" for c in _dropped]
                per_task_detail.append({
                    "name":      _disp,
                    "dom_type":  _dom_ty,
                    "type_bk":   {_dom_ty: 1},
                    "top_feats": _feats,
                    "side":      "",
                    "n_flag":    len(_dropped),
                    "n_tot":     3,
                    "mean_dev":  _abs_max,
                })

        n_anom_scores = len(anomaly_results.get('deviation_score', [])) if anomaly_results else 0
        unique_tasks_screen = list(dict.fromkeys(
            anomaly_results.get('task_names', ['?'] * n_anom_scores)
        )) if anomaly_results else ['?']
        n_tasks_screen = max(len(unique_tasks_screen), 1)
        n_detail_rows  = max(len(per_task_detail), 1)

        fig_h = max(11, n_tasks_screen * 0.55 + n_ind * 0.42 + n_detail_rows * 0.70 + 5)

        fig = plt.figure(figsize=(18, fig_h))
        gs  = GridSpec(3, 3, figure=fig,
                       height_ratios=[max(3, n_tasks_screen * 0.55 + 1),
                                      max(3, n_ind * 0.45 + 1.5),
                                      max(2.5, n_detail_rows * 0.62 + 1.0)],
                       hspace=0.50, wspace=0.30)

        cmap_traffic = mcolors.LinearSegmentedColormap.from_list(
            'traffic', [(0, C_PASS), (0.45, '#FDD835'), (1, C_FAIL)]
        )

        ax_mat = fig.add_subplot(gs[0, :2])
        if anomaly_results:
            scores     = anomaly_results.get('deviation_score', [])
            is_anomaly = anomaly_results.get('is_anomaly', [])
            rep_ids    = anomaly_results.get('repetitions', list(range(1, len(scores) + 1)))
            tn_list    = anomaly_results.get('task_names', ['?'] * len(scores))

            task_names = list(dict.fromkeys(tn_list))
            rep_nums   = sorted(set(int(r) for r in rep_ids))

            mat      = np.full((len(task_names), len(rep_nums)), np.nan)
            anom_mat_s = np.zeros((len(task_names), len(rep_nums)), dtype=bool)
            for i, (s, a, tname, rid) in enumerate(
                zip(scores, is_anomaly, tn_list, rep_ids)
            ):
                ti = task_names.index(tname) if tname in task_names else 0
                ri = rep_nums.index(int(rid)) if int(rid) in rep_nums else i % len(rep_nums)
                if ti < mat.shape[0] and ri < mat.shape[1]:
                    mat[ti, ri]       = s
                    anom_mat_s[ti, ri]= a

            masked = np.ma.array(mat, mask=np.isnan(mat))
            im = ax_mat.imshow(masked, aspect='auto', cmap=cmap_traffic, vmin=0, vmax=1,
                               interpolation='nearest')
            n_rn = len(rep_nums)
            n_tn = len(task_names)
            font_sz = max(7, min(10, 70 // max(n_rn, 1)))
            for ti in range(n_tn):
                for ri in range(n_rn):
                    val = mat[ti, ri]
                    if not np.isnan(val):
                        txt_c = 'white' if val > 0.50 else '#222222'
                        cell_txt = f'{val:.2f}' + (' ★' if anom_mat_s[ti, ri] else '')
                        ax_mat.text(ri, ti, cell_txt, ha='center', va='center',
                                    fontsize=font_sz, color=txt_c,
                                    fontweight='bold' if anom_mat_s[ti, ri] else 'normal')
            _task_dom_type: Dict[str, str] = {}
            if anomaly_results:
                _ptrs = anomaly_results.get("per_task_results", [anomaly_results])
                for _ptr in _ptrs:
                    _tn = _ptr.get("task_names", [])
                    if _tn:
                        _dom = _ptr.get("summary", {}).get("dominant_anomaly_type", "")
                        _task_dom_type[_tn[0]] = _dom

            ax_mat.set_xticks(range(n_rn))
            ax_mat.set_xticklabels([f'R{r}' for r in rep_nums], fontsize=max(8, font_sz))
            ax_mat.set_yticks(range(n_tn))
            ytick_labels = []
            for t in task_names:
                short = (t.split(': ', 1)[-1] if ': ' in t else t)[:22]
                dom = _task_dom_type.get(t, "")
                abbr = _ATYPE_ABBR.get(dom, "")
                if abbr:
                    ytick_labels.append(f'{short}  [{abbr}]')
                else:
                    ytick_labels.append(short)
            ytick_fs = max(7, min(10, 180 // max(n_tn, 1)))
            ax_mat.set_yticklabels(ytick_labels, fontsize=ytick_fs)
            for tick, t in zip(ax_mat.get_yticklabels(), task_names):
                dom = _task_dom_type.get(t, "")
                col = _ATYPE_COLOR.get(dom, '#222222') if dom else '#222222'
                tick.set_color(col)
                tick.set_fontweight('bold' if dom else 'normal')
            ax_mat.set_xlabel('Repetition', fontsize=10)
            plt.colorbar(im, ax=ax_mat, label='Deviation Score', shrink=0.7, pad=0.01)
            ax_mat.set_title('Anomaly Score Matrix  (★ = flagged, label colour = anomaly type)',
                             fontsize=11, fontweight='bold')
        else:
            ax_mat.text(0.5, 0.5, 'No anomaly data available',
                        ha='center', va='center', fontsize=12, transform=ax_mat.transAxes)
            ax_mat.axis('off')

        ax_conf = fig.add_subplot(gs[0, 2])
        conf_items = [
            ('Data Quality',    confidence.get('data_quality', 0)),
            ('Consistency',     confidence.get('consistency', 0)),
            ('Model Agreement', confidence.get('model_rule_agreement', 0)),
            ('Overall',         confidence.get('overall', 0)),
        ]
        labels_c  = [c[0] for c in conf_items]
        vals_c    = [c[1] for c in conf_items]
        bar_cols_c= [C_PASS if v >= 0.7 else (C_MILD if v >= 0.5 else C_FAIL) for v in vals_c]

        ax_conf.barh(labels_c, [1.0] * len(labels_c), color='#ECEFF1',
                     edgecolor='none', height=0.55)
        bars_c = ax_conf.barh(labels_c, vals_c, color=bar_cols_c,
                              alpha=0.90, edgecolor='none', height=0.55)
        ax_conf.axvline(x=0.7, color='#546E7A', linestyle='--', linewidth=1.2, alpha=0.6)
        ax_conf.axvline(x=0.5, color='#FF9800', linestyle='--', linewidth=1.0, alpha=0.5)
        for bar, val in zip(bars_c, vals_c):
            ax_conf.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                         f'{val:.0%}', va='center', fontsize=10, fontweight='bold',
                         color=C_PASS if val >= 0.7 else (C_MILD if val >= 0.5 else C_FAIL))
        ax_conf.set_xlim(0, 1.08)
        ax_conf.set_xlabel('Score', fontsize=9)
        ax_conf.set_title(f'Confidence Components\nOverall: {overall_conf:.0%}',
                          fontsize=11, fontweight='bold')
        ax_conf.spines['top'].set_visible(False)
        ax_conf.spines['right'].set_visible(False)
        ax_conf.invert_yaxis()

        ax_ind = fig.add_subplot(gs[1, :2])
        sev_rank = {'severe': 3, 'moderate': 2, 'mild': 1}
        sev_map  = {'mild': C_MILD, 'moderate': C_MOD, 'severe': C_SEV}

        if indications:
            sorted_inds = sorted(
                indications,
                key=lambda x: (sev_rank.get(x.get('severity','mild').lower(), 1),
                                x.get('confidence', 0)),
                reverse=True,
            )
            ind_labels = []
            ind_vals   = []
            ind_colors = []
            sev_labels = []
            for ind in sorted_inds:
                itype = ind.get('indication_type', 'unknown').replace('_', ' ').title()
                task  = (ind.get('task_name', '') or '').strip()
                if not task:
                    tg = str(ind.get('task_group', '') or '').strip()
                    if tg and tg not in ('0', 'nan', 'None'):
                        task = f'Group {tg}'
                task  = (task.split(': ', 1)[-1] if ': ' in task else task)[:22]
                sev   = ind.get('severity', 'mild').lower()
                conf  = ind.get('confidence', 0.0)
                label = f'{itype}  [{task}]' if task else itype
                ind_labels.append(label)
                ind_vals.append(conf)
                ind_colors.append(sev_map.get(sev, C_MILD))
                sev_labels.append(sev.capitalize())

            bars_i = ax_ind.barh(ind_labels, ind_vals, color=ind_colors,
                                 alpha=0.88, edgecolor='none', height=0.65)
            ax_ind.axvline(x=0.7, color='#546E7A', linestyle='--',
                           linewidth=1.2, alpha=0.65, label='High conf. (0.7)')
            ax_ind.axvline(x=0.5, color='#FF9800', linestyle='--',
                           linewidth=1.0, alpha=0.55, label='Medium conf. (0.5)')
            for bar, val, sev_txt in zip(bars_i, ind_vals, sev_labels):
                yc = bar.get_y() + bar.get_height() / 2
                ax_ind.text(val + 0.01, yc, f'{val:.0%}  ({sev_txt})',
                            va='center', fontsize=9, fontweight='bold')
            ax_ind.set_xlim(0, 1.15)
            ax_ind.set_xlabel('Confidence', fontsize=10)
            ax_ind.set_title(f'Screening Indications  ({n_ind} total, sorted by severity)',
                             fontsize=11, fontweight='bold')
            ax_ind.legend(fontsize=8, loc='lower right')
            for sev, col in [('Mild', C_MILD), ('Moderate', C_MOD), ('Severe', C_SEV)]:
                ax_ind.bar([], [], color=col, alpha=0.88, label=sev)
            ax_ind.legend(fontsize=8, loc='lower right')
        else:
            ax_ind.text(0.5, 0.5, 'No indications detected — within normal range',
                        ha='center', va='center', fontsize=13,
                        transform=ax_ind.transAxes,
                        color=C_PASS, fontweight='bold')
            ax_ind.set_title('Screening Indications', fontsize=11, fontweight='bold')
        ax_ind.spines['top'].set_visible(False)
        ax_ind.spines['right'].set_visible(False)

        ax_verd = fig.add_subplot(gs[1, 2])
        ax_verd.axis('off')
        n_anom_count = int(sum(anomaly_results.get('is_anomaly', []))) if anomaly_results else 0
        verdict_color = C_FAIL if n_anom_count > 0 or n_ind > 0 else C_PASS
        verdict_lines = (
            [f'ANOMALIES DETECTED', f'{n_anom_count} rep(s) flagged',
             f'{n_ind} indication(s)']
            if n_anom_count > 0 or n_ind > 0
            else ['WITHIN NORMAL', 'RANGE', 'No anomalies']
        )
        badge = mpatches.FancyBboxPatch(
            (0.05, 0.35), 0.90, 0.50,
            boxstyle='round,pad=0.04',
            transform=ax_verd.transAxes,
            facecolor=verdict_color, edgecolor='none', alpha=0.90, zorder=1,
        )
        ax_verd.add_patch(badge)
        for li, line in enumerate(verdict_lines):
            ax_verd.text(0.50, 0.82 - li * 0.14, line,
                         transform=ax_verd.transAxes,
                         ha='center', va='top', fontsize=11 if li == 0 else 9,
                         fontweight='bold' if li == 0 else 'normal',
                         color='white', zorder=2)
        ax_verd.text(0.50, 0.28, f'Overall confidence: {overall_conf:.0%}',
                     transform=ax_verd.transAxes, ha='center', va='top',
                     fontsize=10, color='#37474F', fontweight='bold')

        ax_det = fig.add_subplot(gs[2, :])
        ax_det.axis('off')

        if per_task_detail:
            ax_det.set_title('Anomaly Details per Task  (what went wrong)',
                             fontsize=11, fontweight='bold',
                             loc='left', pad=6)
            n_det = len(per_task_detail)
            row_h = max(0.060, min(0.155, 0.90 / (n_det + 1.5)))
            col_xs = [0.0, 0.26, 0.41, 0.55, 0.80, 0.96]
            hdr_y  = 0.98 - row_h * 0.15
            hdr_bg = mpatches.FancyBboxPatch(
                (0.0, hdr_y - row_h * 0.52), 1.0, row_h * 0.72,
                boxstyle='square,pad=0', transform=ax_det.transAxes,
                facecolor='#37474F', edgecolor='none', alpha=0.90, zorder=1,
            )
            ax_det.add_patch(hdr_bg)
            for hdr_txt, hdr_x in zip(
                ['Task (group)', 'Anomaly Type', 'Side', 'Top Contributing Features', 'Reps Flagged', 'Dev'],
                col_xs,
            ):
                ax_det.text(hdr_x + 0.005, hdr_y - row_h * 0.10, hdr_txt,
                            transform=ax_det.transAxes, va='center',
                            fontsize=8, fontweight='bold', color='white', zorder=2)

            for ri, det in enumerate(per_task_detail):
                y_centre = hdr_y - row_h * (ri + 1.05)
                bg_col   = '#F5F5F5' if ri % 2 == 0 else '#ECEFF1'
                row_patch = mpatches.FancyBboxPatch(
                    (0.0, y_centre - row_h * 0.44), 1.0, row_h * 0.86,
                    boxstyle='square,pad=0', transform=ax_det.transAxes,
                    facecolor=bg_col, edgecolor='none', alpha=0.95, zorder=1,
                )
                ax_det.add_patch(row_patch)

                sev_col = (C_FAIL if det['mean_dev'] > 0.65
                           else C_MOD if det['mean_dev'] > 0.40
                           else C_MILD)
                sev_stripe = mpatches.FancyBboxPatch(
                    (0.0, y_centre - row_h * 0.44), 0.004, row_h * 0.86,
                    boxstyle='square,pad=0', transform=ax_det.transAxes,
                    facecolor=sev_col, edgecolor='none', alpha=1.0, zorder=2,
                )
                ax_det.add_patch(sev_stripe)

                task_short = (det['name'].split(': ', 1)[-1]
                              if ': ' in det['name'] else det['name'])
                task_group_char = det['name'].split(':')[0].strip() if ':' in det['name'] else ''
                grp_col = {'A': '#0072B2', 'B': '#009E73', 'C': '#E69F00'}.get(
                    task_group_char, '#546E7A')
                ax_det.text(col_xs[0] + 0.006, y_centre,
                            task_group_char + ': ' if task_group_char else '',
                            transform=ax_det.transAxes, va='center',
                            fontsize=8, fontweight='bold', color=grp_col, zorder=3)
                ax_det.text(col_xs[0] + 0.006 + (0.018 if task_group_char else 0),
                            y_centre, task_short[:28],
                            transform=ax_det.transAxes, va='center',
                            fontsize=8, color='#212121', zorder=3)

                dom = det['dom_type']
                atype_lbl = _ATYPE_LABEL.get(dom, dom.replace('_', ' ').title())
                atype_col = _ATYPE_COLOR.get(dom, '#BDBDBD')
                badge_w   = min(0.14, 0.012 * len(atype_lbl) + 0.02)
                type_badge = mpatches.FancyBboxPatch(
                    (col_xs[1], y_centre - row_h * 0.35), badge_w, row_h * 0.70,
                    boxstyle='round,pad=0.005', transform=ax_det.transAxes,
                    facecolor=atype_col, edgecolor='none', alpha=0.88, zorder=3,
                )
                ax_det.add_patch(type_badge)
                ax_det.text(col_xs[1] + badge_w / 2, y_centre, atype_lbl,
                            transform=ax_det.transAxes, va='center', ha='center',
                            fontsize=7.5, fontweight='bold',
                            color='white' if atype_col != '#F0E442' else '#222222',
                            zorder=4)

                side = det['side']
                side_col = ('#C62828' if 'Right' in side
                            else '#1565C0' if 'Left' in side
                            else '#546E7A')
                if side:
                    ax_det.text(col_xs[2] + 0.005, y_centre, side.strip(),
                                transform=ax_det.transAxes, va='center',
                                fontsize=8, color=side_col, fontweight='bold', zorder=3)

                feat_text = '  •  '.join(_feat_label(f) for f in det['top_feats'])
                if not feat_text:
                    feat_text = '—'
                ax_det.text(col_xs[3] + 0.005, y_centre, feat_text[:52],
                            transform=ax_det.transAxes, va='center',
                            fontsize=7.5, color='#424242',
                            style='italic', zorder=3)

                frac = det['n_flag'] / det['n_tot']
                _bar_span = col_xs[5] - col_xs[4] - 0.04
                bar_w = _bar_span * frac
                ax_det.add_patch(mpatches.FancyBboxPatch(
                    (col_xs[4], y_centre - row_h * 0.18), _bar_span, row_h * 0.36,
                    boxstyle='square,pad=0', transform=ax_det.transAxes,
                    facecolor='#E0E0E0', edgecolor='none', alpha=0.9, zorder=3,
                ))
                ax_det.add_patch(mpatches.FancyBboxPatch(
                    (col_xs[4], y_centre - row_h * 0.18), bar_w, row_h * 0.36,
                    boxstyle='square,pad=0', transform=ax_det.transAxes,
                    facecolor=sev_col, edgecolor='none', alpha=0.85, zorder=4,
                ))
                ax_det.text(col_xs[4] + _bar_span + 0.005, y_centre,
                            f'{det["n_flag"]}/{det["n_tot"]}',
                            transform=ax_det.transAxes, va='center',
                            fontsize=7.5, color='#212121', fontweight='bold', zorder=3)

                ax_det.text(col_xs[5], y_centre, f'{det["mean_dev"]:.2f}',
                            transform=ax_det.transAxes, va='center', ha='right',
                            fontsize=8.5, fontweight='bold', color=sev_col, zorder=3)

            legend_y = hdr_y - row_h * (n_det + 0.9)
            shown_types = list(dict.fromkeys(d['dom_type'] for d in per_task_detail))
            lx = 0.0
            for atype in shown_types[:8]:
                lbl  = _ATYPE_ABBR.get(atype, '?')
                col  = _ATYPE_COLOR.get(atype, '#BDBDBD')
                full = _ATYPE_LABEL.get(atype, atype)
                ax_det.add_patch(mpatches.FancyBboxPatch(
                    (lx, legend_y - 0.012), 0.012, 0.022,
                    boxstyle='round,pad=0.002', transform=ax_det.transAxes,
                    facecolor=col, edgecolor='none', zorder=3,
                ))
                ax_det.text(lx + 0.015, legend_y, f'{lbl} = {full}',
                            transform=ax_det.transAxes, va='center',
                            fontsize=7, color='#424242', zorder=3)
                lx += 0.13
        else:
            ax_det.text(0.5, 0.5,
                        'No anomalous repetitions detected',
                        ha='center', va='center', fontsize=12,
                        transform=ax_det.transAxes, color=C_PASS, fontweight='bold')

        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
        output_config = self.config.get('output', {})
        fig.savefig(
            output_path.with_suffix('.png'),
            dpi=output_config.get('save_dpi', 150),
            bbox_inches='tight',
        )
        plt.close(fig)

    def plot_anomaly_results(
        self,
        anomaly_results: Dict[str, Any],
        output_path: Path,
        title: str = "Anomaly Detection Results",
        baseline_stats: Optional[Dict] = None,
    ) -> None:
        """Multi-page PDF: adaptive anomaly report that scales cleanly from
        single-task to full-session data.

        Page 1 — Session Overview Heatmap (task × repetition) with anomaly
                 flags, per-task anomaly rate bar, and summary panel.
        Page 2 — Score distribution: individual bars when n < 20, else a
                 per-task violin/box overview.
        Page 3 — Feature deviation: top deviant features for worst reps.
        Page 4 — Method consensus heatmap (rows = methods, cols = tasks).
        Page 5+ — Radar, waterfall, and PCA analytical pages.
        """
        from matplotlib.backends.backend_pdf import PdfPages
        from matplotlib.gridspec import GridSpec
        import matplotlib.colors as mcolors

        if not anomaly_results or 'anomaly_scores' not in anomaly_results:
            return

        scores          = anomaly_results.get('anomaly_scores', [])
        is_anomaly      = anomaly_results.get('is_anomaly', [False] * len(scores))
        dev_scores      = anomaly_results.get('deviation_score', [0.0] * len(scores))
        conf_scores     = anomaly_results.get('score_confidence', [0.0] * len(scores))
        mahal_scores    = anomaly_results.get('mahalanobis_score', [0.0] * len(scores))
        centroid_scores = anomaly_results.get('centroid_score', [0.0] * len(scores))
        within_scores   = anomaly_results.get('within_session_score', [0.0] * len(scores))
        method_votes    = anomaly_results.get('method_votes', [[] for _ in scores])
        n               = len(scores)
        n_ref           = anomaly_results.get('n_reference', 0)
        n_pca           = anomaly_results.get('n_pca_components', 0)
        pca_var         = anomaly_results.get('pca_explained_variance', [])

        rep_ids         = anomaly_results.get('repetitions', list(range(1, n + 1)))
        task_names_list = anomaly_results.get('task_names', [])
        task_groups_list= anomaly_results.get('task_groups', [])

        has_multi_task = (
            len(set(task_names_list)) > 1 or len(set(task_groups_list)) > 1
        )

        if has_multi_task and len(task_names_list) == n:
            short_tasks = []
            for tn in task_names_list:
                parts = str(tn).split(': ', 1)
                short_tasks.append(parts[-1][:12] if parts else str(tn)[:12])
            rep_labels = [f'{short_tasks[i]} R{int(rep_ids[i])}' for i in range(n)]
        else:
            rep_labels = [f'R{int(r)}' for r in rep_ids]

        method_sigmoid = anomaly_results.get('method_sigmoid_scores', [])
        if method_sigmoid and len(method_sigmoid) == n:
            model_norm    = [s[0] for s in method_sigmoid]
            mahal_norm    = [s[1] for s in method_sigmoid]
            centroid_norm = [s[2] for s in method_sigmoid]
            within_norm   = [s[3] for s in method_sigmoid]
        else:
            def _sigmoid_norm(s, center=0.0, steep=3.0):
                """Map an anomaly score to [0, 1] via a centred sigmoid."""
                return float(1.0 / (1.0 + np.exp(-steep * (s - center))))
            model_norm    = [_sigmoid_norm(s, 0.8, 2.5)  for s in scores]
            mahal_norm    = [_sigmoid_norm(s, 1.3, 2.0)  for s in mahal_scores]
            centroid_norm = [_sigmoid_norm(s, 1.3, 2.0)  for s in centroid_scores]
            within_norm   = [_sigmoid_norm(s, 1.5, 2.5)  for s in within_scores]

        C_PASS   = '#2E7D32'
        C_FAIL   = '#C62828'
        C_WARN   = '#E65100'

        feature_category_colors = {
            'asymmetry': '#0072B2',
            'amplitude': '#E69F00',
            'temporal':  '#009E73',
            'other':     '#999999',
        }

        def _feature_category(fname):
            """Classify a feature name into a broad category for colour coding."""
            fl = fname.lower()
            if 'asymmetry' in fl or 'ratio' in fl:
                return 'asymmetry'
            if 'time_to_peak' in fl or 'velocity' in fl or 'acceleration' in fl or 'duration' in fl:
                return 'temporal'
            if any(k in fl for k in ('mean', 'max', 'std', 'range', 'min', 'activation')):
                return 'amplitude'
            return 'other'

        unique_tasks = list(dict.fromkeys(task_names_list)) if task_names_list else ['Session']
        unique_reps  = sorted(set(int(r) for r in rep_ids))
        task_mat   = np.full((len(unique_tasks), len(unique_reps)), np.nan)
        anom_mat   = np.zeros((len(unique_tasks), len(unique_reps)), dtype=bool)
        votes_mat  = np.zeros((len(unique_tasks), len(unique_reps)), dtype=int)
        for i in range(n):
            tn  = task_names_list[i] if i < len(task_names_list) else 'Session'
            rid = int(rep_ids[i])
            ti  = unique_tasks.index(tn) if tn in unique_tasks else 0
            ri  = unique_reps.index(rid) if rid in unique_reps else i % len(unique_reps)
            if ti < task_mat.shape[0] and ri < task_mat.shape[1]:
                task_mat[ti, ri]  = dev_scores[i]
                anom_mat[ti, ri]  = is_anomaly[i]
                votes_mat[ti, ri] = sum(1 for v in (method_votes[i] if i < len(method_votes) else []) if v)

        n_anom = int(np.sum(is_anomaly))
        flagged_indices = [i for i, a in enumerate(is_anomaly) if a]
        if not flagged_indices:
            flagged_indices = [int(np.argmax(dev_scores))]

        pdf_path     = output_path.with_suffix('.pdf')
        output_config= self.config.get('output', {})

        cmap_traffic = mcolors.LinearSegmentedColormap.from_list(
            'traffic', [(0, C_PASS), (0.5, '#FDD835'), (1.0, C_FAIL)]
        )

        with PdfPages(pdf_path) as pdf:
            n_tasks_display = len(unique_tasks)
            n_reps_display  = len(unique_reps)
            fig_w = max(14, n_reps_display * 1.1 + 5)
            fig_h = max(8,  n_tasks_display * 0.65 + 5)

            fig = plt.figure(figsize=(fig_w, fig_h))
            gs  = GridSpec(
                1, 3, figure=fig,
                width_ratios=[max(3, n_reps_display * 0.8), 1.4, 1.2],
                wspace=0.05,
            )

            ax_hm = fig.add_subplot(gs[0, 0])
            masked_mat = np.ma.array(task_mat, mask=np.isnan(task_mat))
            im = ax_hm.imshow(
                masked_mat, aspect='auto', cmap=cmap_traffic,
                vmin=0, vmax=1, interpolation='nearest',
            )
            cbar = plt.colorbar(im, ax=ax_hm, shrink=0.7, pad=0.01)
            cbar.set_label('Deviation Score', fontsize=9)
            cbar.ax.tick_params(labelsize=8)

            for ti in range(n_tasks_display):
                for ri in range(n_reps_display):
                    val = task_mat[ti, ri]
                    if np.isnan(val):
                        continue
                    txt_col = 'white' if (val > 0.55 or (val < 0.2 and val > 0)) else '#222222'
                    cell_text = f'{val:.2f}'
                    if anom_mat[ti, ri]:
                        cell_text += '\n★'
                    ax_hm.text(ri, ti, cell_text, ha='center', va='center',
                               fontsize=max(6, min(9, 80 // max(n_reps_display, 1))),
                               color=txt_col, fontweight='bold' if anom_mat[ti, ri] else 'normal')

            ax_hm.set_xticks(range(n_reps_display))
            ax_hm.set_xticklabels([f'R{r}' for r in unique_reps],
                                   fontsize=max(7, min(10, 80 // max(n_reps_display, 1))))
            ax_hm.set_yticks(range(n_tasks_display))
            short_task_labels = [
                (t.split(': ', 1)[-1] if ': ' in t else t)[:22]
                for t in unique_tasks
            ]
            ax_hm.set_yticklabels(short_task_labels,
                                   fontsize=max(7, min(10, 200 // max(n_tasks_display, 1))))
            ax_hm.set_xlabel('Repetition', fontsize=10)
            n_task_rel = anomaly_results.get('ml_metadata', {}).get('n_task_relevant', n_pca)
            ax_hm.set_title(
                f'Deviation Score  (★ = anomaly flagged)\n'
                f'n_ref={n_ref}  ·  {n_task_rel} task-relevant features',
                fontsize=11, fontweight='bold',
            )

            ax_rate = fig.add_subplot(gs[0, 1], sharey=ax_hm)
            task_anom_rates = []
            task_mean_scores = []
            for ti in range(n_tasks_display):
                valid = ~np.isnan(task_mat[ti])
                if valid.any():
                    task_anom_rates.append(anom_mat[ti, valid].mean())
                    task_mean_scores.append(task_mat[ti, valid].mean())
                else:
                    task_anom_rates.append(0.0)
                    task_mean_scores.append(0.0)

            rate_colors = [C_FAIL if r > 0.5 else (C_WARN if r > 0 else C_PASS)
                           for r in task_anom_rates]
            ax_rate.barh(range(n_tasks_display), task_anom_rates, color=rate_colors,
                         alpha=0.85, edgecolor='none', height=0.6)
            ax_rate.axvline(x=0.5, color='#999999', linestyle='--', linewidth=1)
            ax_rate.set_xlim(0, 1.08)
            ax_rate.set_xlabel('Anomaly Rate', fontsize=9)
            ax_rate.set_title('Task\nAnomaly Rate', fontsize=9, fontweight='bold')
            ax_rate.set_yticks([])
            ax_rate.tick_params(axis='x', labelsize=8)
            ax_rate.spines['top'].set_visible(False)
            ax_rate.spines['right'].set_visible(False)
            for ti, r in enumerate(task_anom_rates):
                ax_rate.text(r + 0.02, ti, f'{r:.0%}', va='center', fontsize=8,
                             color=C_FAIL if r > 0.5 else '#333333')

            ax_sum = fig.add_subplot(gs[0, 2])
            ax_sum.axis('off')
            pca_var_str = '  '.join(f'PC{i+1}:{v:.0%}' for i, v in enumerate(pca_var[:3]))
            summary_lines = [
                ('Session Summary', True),
                (f'Total reps:  {n}', False),
                (f'Ref sessions:{n_ref}', False),
                (f'Flagged:     {n_anom}/{n}  ({n_anom/max(n,1):.0%})', False),
                ('', False),
                ('Detection', True),
                (f'Features:    {n_task_rel}', False),
                (f'PCA dims:    {n_pca}D', False),
                (pca_var_str, False),
                ('', False),
                ('Threshold:  0.45', False),
                ('Rule: t-intervals', False),
                ('+ composite > 0.45', False),
            ]
            verdict_color = C_FAIL if n_anom > 0 else C_PASS
            verdict_text  = f'ANOMALIES\nDETECTED ({n_anom})' if n_anom > 0 else 'NO\nANOMALIES'
            badge = mpatches.FancyBboxPatch(
                (0.05, 0.72), 0.90, 0.22,
                boxstyle='round,pad=0.03',
                transform=ax_sum.transAxes,
                facecolor=verdict_color, edgecolor='none', alpha=0.9, zorder=1,
            )
            ax_sum.add_patch(badge)
            ax_sum.text(0.50, 0.83, verdict_text, transform=ax_sum.transAxes,
                        ha='center', va='center', fontsize=12, fontweight='bold',
                        color='white', zorder=2, linespacing=1.4)

            y_cur = 0.68
            for line, bold in summary_lines:
                if not line:
                    y_cur -= 0.025
                    continue
                if bold:
                    ax_sum.text(0.05, y_cur, line, transform=ax_sum.transAxes,
                                fontsize=9, fontweight='bold', va='top', color='#37474F')
                    y_cur -= 0.035
                    ax_sum.plot([0.04, 0.96], [y_cur + 0.015, y_cur + 0.015],
                                color='#BDBDBD', linewidth=0.7,
                                transform=ax_sum.transAxes, clip_on=False)
                else:
                    ax_sum.text(0.08, y_cur, line, transform=ax_sum.transAxes,
                                fontsize=8, va='top', color='#444444', family='monospace')
                y_cur -= 0.048

            border = mpatches.FancyBboxPatch(
                (0.02, 0.02), 0.96, 0.96,
                boxstyle='round,pad=0.02',
                transform=ax_sum.transAxes,
                linewidth=1.5, edgecolor='#90A4AE',
                facecolor='#F5F7FA', zorder=0,
            )
            ax_sum.add_patch(border)

            fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
            pdf.savefig(fig, dpi=output_config.get('save_dpi', 150), bbox_inches='tight')
            plt.close(fig)

            self._plot_anomaly_score_distribution_page(
                pdf, anomaly_results, rep_labels, unique_tasks, task_mat, anom_mat,
                n, C_PASS, C_FAIL, C_WARN, output_config
            )

            self._plot_anomaly_feature_deviation_page(
                pdf, anomaly_results, flagged_indices, rep_labels,
                feature_category_colors, output_config
            )

            self._plot_method_consensus_page(
                pdf, anomaly_results, unique_tasks, task_names_list,
                model_norm, mahal_norm, centroid_norm, within_norm,
                method_votes, C_PASS, C_FAIL, output_config
            )

            self._plot_method_radar_page(pdf, anomaly_results, rep_labels, output_config)
            self._plot_waterfall_page(pdf, anomaly_results, rep_labels, output_config)
            self._plot_pca_biplot_page(pdf, anomaly_results, output_config)

    def _plot_anomaly_score_distribution_page(
        self,
        pdf,
        anomaly_results: Dict[str, Any],
        rep_labels: List[str],
        unique_tasks: List[str],
        task_mat: np.ndarray,
        anom_mat: np.ndarray,
        n: int,
        C_PASS: str,
        C_FAIL: str,
        C_WARN: str,
        output_config: Dict,
    ) -> None:
        """Page 2: score distribution.  Bar chart for ≤18 reps, per-task box
        overview for larger sessions.  Confidence intervals overlaid when available.
        """
        import matplotlib.colors as mcolors
        dev_scores = anomaly_results.get('deviation_score', [0.0] * n)
        is_anomaly = anomaly_results.get('is_anomaly', [False] * n)
        conf_scores= anomaly_results.get('score_confidence', [0.0] * n)
        ci_lower   = anomaly_results.get('deviation_ci_lower', [])
        ci_upper   = anomaly_results.get('deviation_ci_upper', [])
        method_votes = anomaly_results.get('method_votes', [[] for _ in range(n)])
        has_ci = len(ci_lower) == n and len(ci_upper) == n

        if n <= 18:
            fig_h = max(5, n * 0.45 + 2)
            fig, ax = plt.subplots(figsize=(12, fig_h))
            bar_colors = [C_FAIL if a else C_PASS for a in is_anomaly]
            bars = ax.barh(rep_labels, dev_scores,
                           color=bar_colors, alpha=0.87, edgecolor='white',
                           linewidth=0.6, height=0.65)
            if has_ci:
                xerr_l = [max(0, dev_scores[i] - ci_lower[i]) for i in range(n)]
                xerr_h = [max(0, ci_upper[i] - dev_scores[i]) for i in range(n)]
                ax.errorbar(dev_scores, rep_labels,
                            xerr=[xerr_l, xerr_h],
                            fmt='none', ecolor='#444444', elinewidth=1.5,
                            capsize=3, capthick=1.0, zorder=5)
            ax.axvline(x=0.45, color=C_FAIL, linestyle='--',
                       linewidth=1.5, alpha=0.7, label='Threshold (0.45)')
            for i, (bar, score, conf, is_anom) in enumerate(
                zip(bars, dev_scores, conf_scores, is_anomaly)
            ):
                w  = bar.get_width()
                yc = bar.get_y() + bar.get_height() / 2
                ax.text(w + 0.01, yc, f'{score:.2f}',
                        va='center', fontsize=9, fontweight='bold' if is_anom else 'normal',
                        color=C_FAIL if is_anom else '#333333')
                n_yes = sum(1 for v in (method_votes[i] if i < len(method_votes) else []) if v)
                ax.text(max(0.01, min(w - 0.01, 0.01)), yc,
                        f'{n_yes}/4', va='center', ha='left', fontsize=7.5,
                        color='white' if w > 0.08 else '#333333')
            ax.set_xlim(-0.02, 1.05)
            ax.set_xlabel('Composite Deviation Score', fontsize=10)
            ax.legend(fontsize=9, loc='lower right')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.invert_yaxis()
            fig.suptitle('Repetition Deviation Scores', fontsize=13, fontweight='bold')
        else:
            n_tasks = task_mat.shape[0]
            fig, axes = plt.subplots(1, 2, figsize=(14, max(5, n_tasks * 0.55 + 2)),
                                     gridspec_kw={'width_ratios': [2, 1]})
            ax_box, ax_rate = axes

            positions = np.arange(n_tasks)
            for ti in range(n_tasks):
                row_vals = task_mat[ti][~np.isnan(task_mat[ti])]
                if len(row_vals) == 0:
                    continue
                bp = ax_box.boxplot([row_vals], positions=[ti], vert=False,
                                    widths=0.5, patch_artist=True,
                                    boxprops=dict(facecolor='#CFD8DC', alpha=0.8),
                                    medianprops=dict(color='#333333', linewidth=2),
                                    whiskerprops=dict(linewidth=1.2),
                                    capprops=dict(linewidth=1.2),
                                    flierprops=dict(marker='o', markersize=4, alpha=0.6))
                anom_vals = task_mat[ti][anom_mat[ti] & ~np.isnan(task_mat[ti])]
                if len(anom_vals):
                    ax_box.scatter(anom_vals, [ti] * len(anom_vals),
                                   color=C_FAIL, zorder=5, s=40, marker='*')

            ax_box.axvline(x=0.45, color=C_FAIL, linestyle='--', linewidth=1.5, alpha=0.7,
                           label='Threshold (0.45)')
            ax_box.set_yticks(positions)
            short_labels = [(t.split(': ', 1)[-1] if ': ' in t else t)[:22]
                            for t in unique_tasks]
            ax_box.set_yticklabels(short_labels, fontsize=9)
            ax_box.set_xlabel('Deviation Score', fontsize=10)
            ax_box.set_title('Score Distribution by Task  (★ = anomaly)', fontsize=11, fontweight='bold')
            ax_box.legend(fontsize=8, loc='lower right')
            ax_box.spines['top'].set_visible(False)
            ax_box.spines['right'].set_visible(False)
            ax_box.grid(True, alpha=0.25, axis='x')
            ax_box.invert_yaxis()

            task_rates = [anom_mat[ti][~np.isnan(task_mat[ti])].mean()
                          if (~np.isnan(task_mat[ti])).any() else 0.0
                          for ti in range(n_tasks)]
            rate_cols = [C_FAIL if r > 0.5 else (C_WARN if r > 0 else C_PASS)
                         for r in task_rates]
            ax_rate.barh(positions, task_rates, color=rate_cols, alpha=0.85,
                         edgecolor='none', height=0.55)
            ax_rate.axvline(x=0.5, color='#999999', linestyle='--', linewidth=1)
            ax_rate.set_xlim(0, 1.05)
            ax_rate.set_xlabel('Anomaly Rate', fontsize=9)
            ax_rate.set_title('Anomaly\nRate', fontsize=9, fontweight='bold')
            ax_rate.set_yticks([])
            ax_rate.spines['top'].set_visible(False)
            ax_rate.spines['right'].set_visible(False)
            ax_rate.invert_yaxis()
            for ti, r in enumerate(task_rates):
                ax_rate.text(r + 0.02, ti, f'{r:.0%}', va='center', fontsize=8,
                             color=C_FAIL if r > 0.5 else '#444444')

            fig.suptitle('Score Distribution (full session)', fontsize=13, fontweight='bold')

        plt.tight_layout()
        pdf.savefig(fig, dpi=output_config.get('save_dpi', 150), bbox_inches='tight')
        plt.close(fig)

    def _plot_anomaly_feature_deviation_page(
        self,
        pdf,
        anomaly_results: Dict[str, Any],
        flagged_indices: List[int],
        rep_labels: List[str],
        cat_colors: Dict[str, str],
        output_config: Dict,
    ) -> None:
        """Page 3: top deviant features visualised as grouped horizontal bars.

        Shows the maximum deviation across flagged reps (bar fill) alongside
        grey dots for non-flagged reps, making it easy to see whether a feature
        is globally or selectively deviant.
        """
        feature_devs = anomaly_results.get('feature_deviations', {})
        deviations   = anomaly_results.get('deviations', [])
        n_total      = len(anomaly_results.get('anomaly_scores', []))

        if not feature_devs or not flagged_indices:
            return

        feat_scores: Dict[str, float] = {}
        for fname in feature_devs:
            vals = []
            for fi in flagged_indices:
                if fi < len(deviations) and fname in deviations[fi]:
                    vals.append(deviations[fi][fname].get('range_dev', 0.0))
            if vals:
                feat_scores[fname] = max(vals)

        top_feats = sorted(feat_scores, key=feat_scores.get, reverse=True)[:20]
        if not top_feats:
            return

        def _cat(fname):
            """Classify a feature name into an anomaly evidence category."""
            fl = fname.lower()
            if 'asymmetry' in fl or 'ratio' in fl:
                return 'asymmetry'
            if 'time_to_peak' in fl or 'velocity' in fl or 'acceleration' in fl or 'duration' in fl:
                return 'temporal'
            if any(k in fl for k in ('mean', 'max', 'std', 'range', 'min', 'activation')):
                return 'amplitude'
            return 'other'

        n_feats   = len(top_feats)
        fig_h     = max(6, n_feats * 0.55 + 3)
        fig, axes = plt.subplots(1, 2, figsize=(16, fig_h),
                                 gridspec_kw={'width_ratios': [3, 1]})
        ax_bars, ax_cat = axes

        y_pos = np.arange(n_feats)
        flagged_label = rep_labels[flagged_indices[0]] if flagged_indices else '?'

        for yi, fname in enumerate(reversed(top_feats)):
            cat     = _cat(fname)
            col     = cat_colors.get(cat, '#999999')
            dev_val = feat_scores.get(fname, 0)
            ax_bars.barh(yi, dev_val, color=col, alpha=0.82,
                         edgecolor='none', height=0.65)
            for ri in range(n_total):
                if ri not in flagged_indices and ri < len(deviations):
                    if fname in deviations[ri]:
                        ref_v = deviations[ri][fname].get('range_dev', 0.0)
                        ax_bars.plot(ref_v, yi, 'o', color='#606060',
                                     markersize=5, alpha=0.55, zorder=5)
            ax_bars.text(dev_val + 0.15, yi, f'{dev_val:.1f}×',
                         va='center', fontsize=7.5, color='#333333')

        ax_bars.axvline(x=1.5, color='#D55E00', linestyle='--',
                        linewidth=1.5, alpha=0.75, label='Deviant threshold (1.5×)')
        ax_bars.axvline(x=5.0, color='#C62828', linestyle=':', linewidth=1,
                        alpha=0.6, label='Severe (5×)')
        ax_bars.set_yticks(y_pos)
        ax_bars.set_yticklabels(
            [f.replace('_', ' ')[:40] for f in reversed(top_feats)], fontsize=8
        )
        ax_bars.set_xlabel('Range Deviation  (×  reference spread)', fontsize=10)
        ax_bars.set_title(
            f'Top Deviant Features — flagged reps (■) vs ● non-flagged reps\n'
            f'Most-deviant flagged: {flagged_label}',
            fontsize=11, fontweight='bold',
        )
        for cat, col in cat_colors.items():
            ax_bars.bar([], [], color=col, alpha=0.85, label=cat.title())
        ax_bars.legend(fontsize=8, loc='lower right')
        ax_bars.spines['top'].set_visible(False)
        ax_bars.spines['right'].set_visible(False)
        ax_bars.grid(True, alpha=0.25, axis='x')

        cat_counts: Dict[str, int] = {}
        for fname in top_feats:
            c = _cat(fname)
            cat_counts[c] = cat_counts.get(c, 0) + 1
        if cat_counts:
            labels_d = list(cat_counts.keys())
            sizes_d  = [cat_counts[l] for l in labels_d]
            colors_d = [cat_colors.get(l, '#999999') for l in labels_d]
            wedges, texts, autotexts = ax_cat.pie(
                sizes_d, labels=labels_d, colors=colors_d,
                autopct='%1.0f%%', startangle=90,
                pctdistance=0.75,
                wedgeprops=dict(width=0.55, edgecolor='white', linewidth=1.5),
            )
            for at in autotexts:
                at.set_fontsize(9)
                at.set_fontweight('bold')
            for t in texts:
                t.set_fontsize(9)
            ax_cat.set_title('Feature\nCategory Mix', fontsize=9, fontweight='bold')
        else:
            ax_cat.axis('off')

        plt.tight_layout()
        pdf.savefig(fig, dpi=output_config.get('save_dpi', 150), bbox_inches='tight')
        plt.close(fig)

    def _plot_method_consensus_page(
        self,
        pdf,
        anomaly_results: Dict[str, Any],
        unique_tasks: List[str],
        task_names_list: List[str],
        model_norm: List[float],
        mahal_norm: List[float],
        centroid_norm: List[float],
        within_norm: List[float],
        method_votes: List[List],
        C_PASS: str,
        C_FAIL: str,
        output_config: Dict,
    ) -> None:
        """Page 4: method-consensus heatmap aggregated by task.

        Each cell shows the mean normalised method score across all repetitions
        of that task, giving a clean (n_methods × n_tasks) summary.  A separate
        strip below each task column shows the fraction of reps voted YES.
        """
        import matplotlib.colors as mcolors

        n      = len(model_norm)
        n_tasks = len(unique_tasks)
        method_names = ['ML (n≥10)', 'Mahalanobis', 'Nearest Centroid', 'Within-Session']
        n_methods = len(method_names)

        task_method_scores = np.zeros((n_methods, n_tasks))
        task_vote_rate     = np.zeros((n_methods, n_tasks))
        task_counts        = np.zeros(n_tasks, dtype=int)

        all_norm = [model_norm, mahal_norm, centroid_norm, within_norm]

        for i in range(n):
            tn = task_names_list[i] if i < len(task_names_list) else unique_tasks[0]
            ti = unique_tasks.index(tn) if tn in unique_tasks else 0
            task_counts[ti] += 1
            for mi, norm_list in enumerate(all_norm):
                task_method_scores[mi, ti] += norm_list[i] if i < len(norm_list) else 0.0
                voted = (method_votes[i][mi]
                         if i < len(method_votes) and mi < len(method_votes[i]) else False)
                task_vote_rate[mi, ti] += int(voted)

        for ti in range(n_tasks):
            if task_counts[ti] > 0:
                task_method_scores[:, ti] /= task_counts[ti]
                task_vote_rate[:, ti] /= task_counts[ti]

        cmap_rg = mcolors.LinearSegmentedColormap.from_list(
            'rg', [(0, C_PASS), (0.55, '#FDD835'), (1.0, C_FAIL)]
        )

        fig_w = max(10, n_tasks * 1.2 + 3)
        fig_h = max(5,  n_methods * 0.9 + 3)
        fig, axes = plt.subplots(2, 1, figsize=(fig_w, fig_h),
                                 gridspec_kw={'height_ratios': [3, 1]},
                                 sharex=True)
        ax_score, ax_vote = axes

        im = ax_score.imshow(task_method_scores, aspect='auto', cmap=cmap_rg,
                             vmin=0, vmax=1, interpolation='nearest')
        plt.colorbar(im, ax=ax_score, label='Mean Normalised Score', shrink=0.7)

        for mi in range(n_methods):
            for ti in range(n_tasks):
                val = task_method_scores[mi, ti]
                txt_c = 'white' if val > 0.55 else '#222222'
                ax_score.text(ti, mi, f'{val:.2f}', ha='center', va='center',
                              fontsize=max(7, min(10, 80 // max(n_tasks, 1))),
                              color=txt_c, fontweight='bold')

        ax_score.set_yticks(range(n_methods))
        ax_score.set_yticklabels(method_names, fontsize=9)
        ax_score.set_title(
            'Method Consensus by Task  (mean normalised score per task)',
            fontsize=11, fontweight='bold',
        )

        cmap_vote = mcolors.LinearSegmentedColormap.from_list(
            'vote', [(0, C_PASS), (0.5, '#FDD835'), (1.0, C_FAIL)]
        )
        im_v = ax_vote.imshow(task_vote_rate, aspect='auto', cmap=cmap_vote,
                              vmin=0, vmax=1, interpolation='nearest')
        for mi in range(n_methods):
            for ti in range(n_tasks):
                r = task_vote_rate[mi, ti]
                txt_c = 'white' if r > 0.55 else '#222222'
                ax_vote.text(ti, mi, f'{r:.0%}', ha='center', va='center',
                             fontsize=max(6, min(9, 70 // max(n_tasks, 1))),
                             color=txt_c)
        ax_vote.set_yticks(range(n_methods))
        ax_vote.set_yticklabels(method_names, fontsize=8)
        ax_vote.set_xticks(range(n_tasks))
        short_task_labels = [
            (t.split(': ', 1)[-1] if ': ' in t else t)[:18]
            for t in unique_tasks
        ]
        ax_vote.set_xticklabels(short_task_labels,
                                rotation=30, ha='right',
                                fontsize=max(7, min(9, 160 // max(n_tasks, 1))))
        ax_vote.set_xlabel('Task', fontsize=10)
        ax_vote.set_title('Vote Rate  (fraction of reps voted anomaly)',
                          fontsize=9, fontweight='bold')

        plt.tight_layout()
        pdf.savefig(fig, dpi=output_config.get('save_dpi', 150), bbox_inches='tight')
        plt.close(fig)

    def _plot_method_radar_page(
        self,
        pdf,
        anomaly_results: Dict[str, Any],
        rep_labels: List[str],
        output_config: Dict,
    ) -> None:
        """Page 3: radar/spider chart of method sigmoid scores per repetition."""
        method_sigmoid = anomaly_results.get('method_sigmoid_scores', [])
        is_anomaly = anomaly_results.get('is_anomaly', [])
        if not method_sigmoid:
            return

        n = len(method_sigmoid)
        categories = ['ML\n(n\u226510)', 'Mahalanobis', 'Nearest\nCentroid', 'Within-\nSession']
        n_cats = len(categories)
        angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
        angles += angles[:1]

        cols = min(n, 4)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(
            rows, cols, figsize=(4.5 * cols, 4.5 * rows),
            subplot_kw={'projection': 'polar'},
        )
        if n == 1:
            axes = np.array([[axes]])
        axes = np.atleast_2d(axes)

        C_PASS = '#2E7D32'
        C_FAIL = '#C62828'

        for idx in range(n):
            r, c = divmod(idx, cols)
            ax = axes[r][c]
            values = list(method_sigmoid[idx]) + [method_sigmoid[idx][0]]
            is_anom = is_anomaly[idx] if idx < len(is_anomaly) else False
            fill_color = C_FAIL if is_anom else C_PASS

            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)

            ax.plot(angles, values, 'o-', linewidth=2, color=fill_color,
                    markersize=6, alpha=0.9)
            ax.fill(angles, values, alpha=0.2, color=fill_color)

            thresh_vals = [0.50] * (n_cats + 1)
            ax.plot(angles, thresh_vals, '--', linewidth=1, color='#999999', alpha=0.7)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=8)
            ax.set_ylim(0, 1.0)
            ax.set_yticks([0.25, 0.50, 0.75, 1.0])
            ax.set_yticklabels(['0.25', '0.50', '0.75', '1.0'], fontsize=7, color='#666666')

            label = rep_labels[idx] if idx < len(rep_labels) else f'R{idx+1}'
            status = 'ANOMALY' if is_anom else 'NORMAL'
            ax.set_title(
                f'{label}  ({status})',
                fontsize=10, fontweight='bold',
                color=C_FAIL if is_anom else '#333333',
                pad=12,
            )

        for idx in range(n, rows * cols):
            r, c = divmod(idx, cols)
            axes[r][c].set_visible(False)

        fig.suptitle(
            'Method Scores Radar  (dashed ring = 0.50 threshold)',
            fontsize=13, fontweight='bold', y=1.02,
        )
        fig.tight_layout()
        pdf.savefig(fig, dpi=output_config.get('save_dpi', 150), bbox_inches='tight')
        plt.close(fig)

    def _plot_waterfall_page(
        self,
        pdf,
        anomaly_results: Dict[str, Any],
        rep_labels: List[str],
        output_config: Dict,
    ) -> None:
        """Page 4: waterfall chart showing per-method weighted contribution to composite."""
        components = anomaly_results.get('method_weighted_components', [])
        is_anomaly = anomaly_results.get('is_anomaly', [])
        if not components:
            return

        n = len(components)
        method_names = list(components[0].keys())
        n_methods = len(method_names)
        C_PASS = '#2E7D32'
        C_FAIL = '#C62828'

        method_colors = {
            'ML Model':         '#0077BB',
            'Mahalanobis':      '#E69F00',
            'Nearest Centroid': '#56B4E9',
            'Feature Dev':      '#009E73',
            'Within-Session':   '#A29BFE',
        }

        cols = min(n, 4)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
        if n == 1:
            axes = np.array([[axes]])
        axes = np.atleast_2d(axes)

        for idx in range(n):
            r, c = divmod(idx, cols)
            ax = axes[r][c]
            comp = components[idx]
            values = [comp[m] for m in method_names]
            cumulative = 0.0
            for mi, (mname, val) in enumerate(zip(method_names, values)):
                color = method_colors.get(mname, '#888888')
                ax.barh(mi, val, left=cumulative, color=color,
                        edgecolor='white', linewidth=0.8, height=0.6)
                if val > 0.02:
                    ax.text(cumulative + val / 2, mi, f'{val:.3f}',
                            ha='center', va='center', fontsize=7.5,
                            color='white', fontweight='bold')
                cumulative += val

            composite = anomaly_results.get('deviation_score', [0])[idx]
            ax.axvline(x=composite, color='#333333', linewidth=2, linestyle='-',
                       zorder=5)
            ax.axvline(x=0.45, color=C_FAIL, linewidth=1, linestyle='--',
                       alpha=0.6)

            ax.set_yticks(range(n_methods))
            ax.set_yticklabels(method_names, fontsize=8)
            ax.set_xlim(0, max(0.6, composite * 1.3))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            label = rep_labels[idx] if idx < len(rep_labels) else f'R{idx+1}'
            is_anom = is_anomaly[idx] if idx < len(is_anomaly) else False
            status = 'ANOMALY' if is_anom else 'NORMAL'
            ax.set_title(
                f'{label}  composite={composite:.3f}  ({status})',
                fontsize=9, fontweight='bold',
                color=C_FAIL if is_anom else '#333333',
            )
            ax.set_xlabel('Weighted Contribution', fontsize=8)

        for idx in range(n, rows * cols):
            r, c = divmod(idx, cols)
            axes[r][c].set_visible(False)

        fig.suptitle(
            'Score Waterfall  — per-method weighted contribution to composite',
            fontsize=13, fontweight='bold', y=1.02,
        )
        fig.tight_layout()
        pdf.savefig(fig, dpi=output_config.get('save_dpi', 150), bbox_inches='tight')
        plt.close(fig)

    def _plot_pca_biplot_page(
        self,
        pdf,
        anomaly_results: Dict[str, Any],
        output_config: Dict,
    ) -> None:
        """Page 5: PCA loading biplot (PC1 vs PC2) showing original feature contributions."""
        ml_meta = anomaly_results.get('ml_metadata', {})
        pca_var = ml_meta.get('pca_explained_variance', [])
        if len(pca_var) < 2:
            return

        loadings = anomaly_results.get('pca_loadings', None)
        feature_names = anomaly_results.get('pca_feature_names', None)
        test_pca = anomaly_results.get('pca_projected', None)
        is_anomaly = anomaly_results.get('is_anomaly', [])

        if loadings is None or feature_names is None:
            return

        loadings = np.array(loadings)
        if loadings.shape[0] < 2:
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        C_PASS = '#2E7D32'
        C_FAIL = '#C62828'

        if test_pca is not None and len(test_pca) > 0:
            pca_arr = np.array(test_pca)
            for i in range(len(pca_arr)):
                color = C_FAIL if (i < len(is_anomaly) and is_anomaly[i]) else C_PASS
                ax.scatter(pca_arr[i, 0], pca_arr[i, 1],
                           c=color, s=100, zorder=5, edgecolors='black',
                           linewidth=0.8, alpha=0.85)
                label = f'R{i+1}'
                ax.annotate(label, (pca_arr[i, 0], pca_arr[i, 1]),
                            textcoords='offset points', xytext=(6, 6),
                            fontsize=8, color=color, fontweight='bold')

        pc1 = loadings[0]
        pc2 = loadings[1]
        max_load = max(np.max(np.abs(pc1)), np.max(np.abs(pc2)), 1e-6)
        if test_pca is not None and len(test_pca) > 0:
            pca_arr = np.array(test_pca)
            data_range = max(np.ptp(pca_arr[:, 0]), np.ptp(pca_arr[:, 1]), 1e-6)
            scale = data_range * 0.4 / max_load
        else:
            scale = 1.0 / max_load

        n_feat = len(feature_names)
        top_k = min(n_feat, 10)
        magnitudes = np.sqrt(pc1 ** 2 + pc2 ** 2)
        top_indices = np.argsort(magnitudes)[-top_k:]

        arrow_colors = plt.cm.Set2(np.linspace(0, 1, top_k))
        for rank, fi in enumerate(top_indices):
            dx = pc1[fi] * scale
            dy = pc2[fi] * scale
            ax.annotate(
                '', xy=(dx, dy), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=arrow_colors[rank],
                                lw=1.8, alpha=0.85),
            )
            fname = feature_names[fi].replace('_', ' ')[:30]
            ax.text(dx * 1.08, dy * 1.08, fname,
                    fontsize=7, color='#333333', alpha=0.9,
                    ha='center', va='center')

        ax.axhline(0, color='#999999', linewidth=0.5, alpha=0.5)
        ax.axvline(0, color='#999999', linewidth=0.5, alpha=0.5)
        ax.set_xlabel(f'PC1 ({pca_var[0]:.1%} variance)', fontsize=10)
        ax.set_ylabel(f'PC2 ({pca_var[1]:.1%} variance)', fontsize=10)
        ax.set_title(
            'PCA Loading Biplot  (top-10 feature contributions to PC1/PC2)',
            fontsize=12, fontweight='bold',
        )
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.2)

        fig.tight_layout()
        pdf.savefig(fig, dpi=output_config.get('save_dpi', 150), bbox_inches='tight')
        plt.close(fig)

    def plot_deviations_summary(
        self,
        anomaly_results: Dict[str, Any],
        screening_results: Optional[Dict[str, Any]],
        output_path: Path,
        task_name: str = "",
    ) -> None:
        """Single-page figure: feature deviation heatmap + clinical result box."""
        import matplotlib.colors as mcolors

        feature_devs = anomaly_results.get('feature_deviations', {})
        deviations   = anomaly_results.get('deviations', [])
        is_anomaly   = anomaly_results.get('is_anomaly', [])
        rep_ids      = anomaly_results.get('repetitions', list(range(1, len(is_anomaly) + 1)))
        rep_labels   = [f'{task_name}\nR{int(r)}' for r in rep_ids]

        feat_max = {
            fname: fstats.get('max_range_dev', 0)
            for fname, fstats in feature_devs.items()
        }
        top_feats = sorted(feat_max, key=feat_max.get, reverse=True)[:20]

        if not top_feats:
            return

        fig, (ax_heat, ax_bar) = plt.subplots(
            1, 2, figsize=(14, max(7, len(top_feats) * 0.45 + 2)),
            gridspec_kw={'width_ratios': [3, 1]}
        )

        n_reps = len(rep_ids)
        mat = np.zeros((len(top_feats), n_reps))
        for fi, fname in enumerate(top_feats):
            for ri in range(n_reps):
                if ri < len(deviations) and fname in deviations[ri]:
                    mat[fi, ri] = deviations[ri][fname].get('range_dev', 0.0)

        cmap = mcolors.LinearSegmentedColormap.from_list(
            'dev', ['#FFFDE7', '#FF6F00', '#B71C1C']
        )
        im = ax_heat.imshow(
            mat, aspect='auto', cmap=cmap, vmin=0,
            vmax=min(max(feat_max.values(), default=5), 20),
            interpolation='nearest'
        )
        for fi in range(len(top_feats)):
            for ri in range(n_reps):
                val = mat[fi, ri]
                if val > 1.5:
                    ax_heat.text(ri, fi, f'{val:.1f}',
                                 ha='center', va='center',
                                 fontsize=8, fontweight='bold', color='white')
                elif val >= 1.0:
                    ax_heat.text(ri, fi, f'{val:.1f}',
                                 ha='center', va='center',
                                 fontsize=7, color='#5D4037', alpha=0.8)
        for ri, is_anom in enumerate(is_anomaly):
            if is_anom:
                ax_heat.text(ri, -0.7, '★', ha='center', fontsize=10,
                             color='#C62828', fontweight='bold')

        ax_heat.set_xticks(range(n_reps))
        ax_heat.set_xticklabels(rep_labels, fontsize=8)
        ax_heat.set_yticks(range(len(top_feats)))
        ax_heat.set_yticklabels(
            [f[:35].replace('_', ' ') for f in top_feats], fontsize=8
        )
        plt.colorbar(im, ax=ax_heat, label='Range Deviation (reference spreads)',
                     shrink=0.8)
        ax_heat.set_title(
            f'Feature Deviations by Task × Repetition\n(★ = anomaly detected)',
            fontsize=11
        )

        C_PASS = '#2E7D32'
        C_FAIL = '#C62828'
        dev_scores = anomaly_results.get('deviation_score', [0.0] * n_reps)
        bar_colors = [C_FAIL if a else C_PASS for a in is_anomaly]
        ax_bar.bar(range(n_reps), dev_scores, color=bar_colors,
                   alpha=0.85, edgecolor='black', linewidth=0.8)
        ax_bar.axhline(y=0.45, color=C_FAIL, linestyle='--',
                       linewidth=1.5, alpha=0.7)
        for ri, (s, a) in enumerate(zip(dev_scores, is_anomaly)):
            if a:
                ax_bar.text(ri, s + 0.02, '★', ha='center',
                            fontsize=10, color=C_FAIL)
        ax_bar.set_xticks(range(n_reps))
        ax_bar.set_xticklabels(
            [f'R{int(r)}' for r in rep_ids], fontsize=9
        )
        ax_bar.set_ylim(0, 1.05)
        ax_bar.set_ylabel('Anomaly Score', fontsize=9)
        ax_bar.set_title('Anomaly Score\nby Repetition', fontsize=11)
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)

        if screening_results:
            indications = screening_results.get('indications', [])
            if indications:
                ind = indications[0]
                result_text = (
                    f"⚑ {ind.get('indication_type', '').replace('_', ' ').title()}\n"
                    f"Task: {task_name or ind.get('task_name', '')}\n"
                    f"Severity: {ind.get('severity', '').upper()}\n"
                    f"Confidence: {screening_results.get('confidence', {}).get('overall', 0):.0%}"
                )
                ax_bar.text(
                    0.5, 0.05, result_text,
                    transform=ax_bar.transAxes,
                    fontsize=9, va='bottom', ha='center',
                    bbox=dict(boxstyle='round', facecolor='#FFF8E1',
                              edgecolor='#F9A825', alpha=0.9)
                )

        plt.tight_layout()
        output_config = self.config.get('output', {})
        fig.savefig(
            output_path.with_suffix('.pdf'),
            dpi=output_config.get('save_dpi', 150),
            bbox_inches='tight'
        )
        plt.close(fig)

    def plot_session_overview(self, session_metrics: Dict[str, Any],
                             screening_results: Dict[str, Any],
                             output_path: Path) -> None:
        """Bar chart of the four session-level confidence components.

        Draws data quality, consistency, model-rule agreement, and overall
        confidence as colour-coded bars (green >= 0.7, amber >= 0.5, red < 0.5)
        with the numeric score annotated above each bar.  Saved as a PNG.

        Parameters
        ----------
        session_metrics:
            Session-level metrics dict (currently unused by this method but kept
            for API consistency with other overview methods).
        screening_results:
            Screening result dict containing a ``confidence`` sub-dict with keys
            ``data_quality``, ``consistency``, ``model_rule_agreement``, ``overall``.
        output_path:
            Destination path; the suffix is replaced with ``.png``.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        confidence = screening_results.get('confidence', {})
        conf_components = ['data_quality', 'consistency', 'model_rule_agreement', 'overall']
        conf_labels = ['Data Quality', 'Consistency', 'Model Agreement', 'Overall']
        conf_values = [confidence.get(c, 0.0) for c in conf_components]

        conf_colors = self.colors.get('confidence', {})
        bar_colors = []
        for v in conf_values:
            if v >= 0.7:
                bar_colors.append(conf_colors.get('high', '#81C784'))
            elif v >= 0.5:
                bar_colors.append(conf_colors.get('medium', '#FFD54F'))
            else:
                bar_colors.append(conf_colors.get('low', '#E57373'))

        bars = ax.bar(conf_labels, conf_values, color=bar_colors, alpha=0.9, edgecolor='black')
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Confidence Score')
        ax.set_title('Confidence Summary (Components)')

        for i, v in enumerate(conf_values):
            ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=10)

        plt.tight_layout()
        self._save_figure(fig, output_path)
        plt.close(fig)

    def plot_heatmap(self, features_df: pd.DataFrame, output_path: Path,
                     feature_pattern: str = None, title: str = "Feature Activation Heatmap") -> None:
        """Multi-page PDF: per-task blendshape activation heatmap, one page per task.

        Each page shows a 2-D grid of feature activations over time (frames on the
        x-axis, features on the y-axis) using a diverging colour map.  At most
        50 feature columns are shown.  Metadata columns and asymmetry columns are
        excluded automatically.

        Parameters
        ----------
        features_df:
            Frame-level features DataFrame.
        output_path:
            Destination path; the suffix is replaced with ``.pdf``.
        feature_pattern:
            Optional substring filter to restrict which columns are plotted
            (e.g. ``"mouth"`` for mouth-related blendshapes only).
        title:
            Page-level super-title.
        """
        from matplotlib.backends.backend_pdf import PdfPages

        if features_df is None or len(features_df) == 0:
            return

        exclude = {'frame_index', 'timestamp_abs', 'segment', 'repetition',
                   'detection_success', 'detection_confidence', 'time_rel_sec',
                   'task_group', 'task_id', 'task_name', 'occluded', 'brightness',
                   'inter_ocular_distance'}

        if feature_pattern:
            feature_cols = [c for c in features_df.columns if feature_pattern in c]
        else:
            feature_cols = [c for c in features_df.columns
                            if c not in exclude and not c.startswith('asymmetry')
                            and not c.startswith('activation_')
                            and features_df[c].dtype in [np.float64, np.float32,
                                                         np.int64, np.int32]]

        if not feature_cols:
            return

        feature_cols = feature_cols[:50] if len(feature_cols) > 50 else feature_cols

        task_pages = self._build_task_pages(features_df)
        pdf_path = output_path.with_suffix('.pdf')
        output_config = self.config.get('output', {})
        heatmap_config = self.config.get('plot_types', {}).get('heatmap', {})
        cmap = heatmap_config.get('colormap', 'RdYlBu_r')

        with PdfPages(pdf_path) as pdf:
            total_pages = len(task_pages)
            for page_idx, (task_label, task_key, task_df) in enumerate(task_pages):
                available_cols = [c for c in feature_cols if c in task_df.columns]
                if not available_cols:
                    continue

                data = task_df[available_cols].values.T
                if data.shape[1] > 200:
                    step = data.shape[1] // 200
                    data = data[:, ::step]

                fig, ax = plt.subplots(figsize=(14, max(6, len(available_cols) * 0.3)))

                im = ax.imshow(data, aspect='auto', cmap=cmap, interpolation='nearest')
                cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label('Activation Level')

                ax.set_yticks(range(len(available_cols)))
                ax.set_yticklabels([c.replace('_', ' ')[:20] for c in available_cols], fontsize=8)
                ax.set_xlabel('Time (frames)')

                short_label = task_label.split(': ', 1)[-1] if ': ' in task_label else task_label
                page_num = page_idx + 1
                ax.set_title(f'{short_label} — Page {page_num}/{total_pages}',
                             fontsize=12, fontweight='bold')

                plt.tight_layout()
                pdf.savefig(fig, dpi=output_config.get('save_dpi', 300), bbox_inches='tight')
                plt.close(fig)

    def plot_muscle_group_activation_heatmap(
        self,
        features_df: pd.DataFrame,
        output_path: Path,
        task_name: str = "",
    ) -> None:
        """Multi-page PDF: muscle group × repetition heatmap, one page per task.

        Design matches the temporal heatmap (RdYlBu_r palette, grid lines,
        consistent font sizes).  Colour range is normalised to vmax of each
        page so low-activation sessions still show meaningful contrast.  Cell
        text is suppressed for very small values to reduce clutter.
        """
        from .anatomy import get_muscle_group_summary, MUSCLE_GROUP_MAP
        from matplotlib.backends.backend_pdf import PdfPages

        task_pages    = self._build_task_pages(features_df)
        output_config = self.config.get('output', {})
        pdf_path      = output_path.with_suffix('.pdf')
        heatmap_config= self.config.get('plot_types', {}).get('heatmap', {})
        cmap          = heatmap_config.get('colormap', 'RdYlBu_r')
        total_pages   = len(task_pages)

        with PdfPages(pdf_path) as pdf:
            for page_idx, (task_label, task_key, task_df) in enumerate(task_pages):
                summary_df = get_muscle_group_summary(task_df, by_repetition=True)
                if len(summary_df) == 0:
                    continue

                muscle_cols  = [c for c in summary_df.columns if c.endswith('_mean')]
                muscle_groups= [c.replace('muscle_', '').replace('_mean', '') for c in muscle_cols]

                repetitions = summary_df['repetition'].values if 'repetition' in summary_df.columns else np.array([0])
                n_reps   = len(repetitions)
                n_groups = len(muscle_groups)
                if n_groups == 0 or n_reps == 0:
                    continue

                mat = np.zeros((n_groups, n_reps))
                for i, group in enumerate(muscle_groups):
                    col_name = f'muscle_{group}_mean'
                    if col_name in summary_df.columns:
                        mat[i, :] = summary_df[col_name].values

                fig_w = max(10, n_reps * 1.5 + 3)
                fig_h = max(5, n_groups * 0.6 + 2)
                fig, ax = plt.subplots(figsize=(fig_w, fig_h))

                vmax = mat.max() if mat.max() > 0 else 1.0
                im = ax.imshow(
                    mat, aspect='auto', cmap=cmap,
                    interpolation='nearest', vmin=0, vmax=vmax,
                )

                thresh_ann = vmax * 0.05
                font_sz = max(7, min(11, 90 // max(n_reps, 1)))
                for i in range(n_groups):
                    for j in range(n_reps):
                        val = mat[i, j]
                        if val > thresh_ann:
                            rel = val / vmax
                            text_color = '#222222' if 0.25 < rel < 0.75 else 'white'
                            ax.text(j, i, f'{val:.2f}',
                                    ha='center', va='center',
                                    fontsize=font_sz, color=text_color, fontweight='bold')

                ax.set_xticks(range(n_reps))
                ax.set_xticklabels([f'R{int(r)}' for r in repetitions],
                                   fontsize=max(8, min(11, 80 // max(n_reps, 1))))
                ax.set_yticks(range(n_groups))
                group_labels = []
                for group in muscle_groups:
                    info = MUSCLE_GROUP_MAP.get(group, {})
                    desc = info.get('description', group)
                    group_labels.append(desc[:52] + '...' if len(desc) > 52 else desc)
                ax.set_yticklabels(group_labels, fontsize=10)

                cbar = plt.colorbar(im, ax=ax, label='Mean Activation Intensity',
                                    shrink=0.75, pad=0.02)
                cbar.ax.tick_params(labelsize=9)

                short_label = task_label.split(': ', 1)[-1] if ': ' in task_label else task_label
                ax.set_title(
                    f'Muscle Group Activation — {short_label}'
                    f'  (Page {page_idx + 1}/{total_pages})',
                    fontsize=12, fontweight='bold', pad=12,
                )
                ax.set_xlabel('Repetition', fontsize=11, fontweight='bold')
                ax.set_ylabel('Anatomical Muscle Group', fontsize=11, fontweight='bold')

                ax.set_xticks(np.arange(n_reps) - 0.5, minor=True)
                ax.set_yticks(np.arange(n_groups) - 0.5, minor=True)
                ax.grid(which='minor', color='#CCCCCC', linestyle='-', linewidth=0.3)
                ax.tick_params(which='minor', size=0)

                pdf.savefig(fig, dpi=output_config.get('save_dpi', 300), bbox_inches='tight')
                plt.close(fig)

        logger.info("Saved muscle group activation heatmap: %s", pdf_path)

    def plot_muscle_group_temporal_heatmap(
        self,
        features_df: pd.DataFrame,
        output_path: Path,
    ) -> None:
        """Multi-page PDF showing anatomical muscle group activation over time (frames).

        Each page corresponds to one task.  Y-axis lists muscle groups by
        anatomical name, X-axis is time in frames, and colour intensity
        (blue-to-red) represents mean activation magnitude.
        """
        from .anatomy import aggregate_activations_by_muscle_group, MUSCLE_GROUP_MAP
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.colors as mcolors

        if features_df is None or len(features_df) == 0:
            return

        muscle_df = aggregate_activations_by_muscle_group(features_df)
        muscle_cols = [c for c in muscle_df.columns if c.startswith('muscle_')]

        if not muscle_cols:
            logger.warning("No muscle group columns available for temporal heatmap")
            return

        group_names = [c.replace('muscle_', '') for c in muscle_cols]

        group_labels = []
        for name in group_names:
            info = MUSCLE_GROUP_MAP.get(name, {})
            desc = info.get('description', name)
            if len(desc) > 55:
                desc = desc[:52] + '...'
            group_labels.append(desc)

        task_pages = self._build_task_pages(features_df)
        heatmap_config = self.config.get('plot_types', {}).get('heatmap', {})
        cmap = heatmap_config.get('colormap', 'RdYlBu_r')
        output_config = self.config.get('output', {})
        pdf_path = output_path.with_suffix('.pdf')

        with PdfPages(pdf_path) as pdf:
            total_pages = len(task_pages)

            for page_idx, (task_label, task_key, task_df) in enumerate(task_pages):
                task_muscle = aggregate_activations_by_muscle_group(task_df)
                present_cols = [c for c in muscle_cols if c in task_muscle.columns]
                if not present_cols:
                    continue

                data = task_muscle[present_cols].values.T
                n_groups, n_frames = data.shape

                if n_frames == 0:
                    continue

                max_display_frames = 300
                if n_frames > max_display_frames:
                    step = n_frames // max_display_frames
                    data = data[:, ::step]
                    n_frames = data.shape[1]

                fig, ax = plt.subplots(
                    figsize=(max(12, n_frames * 0.04), max(5, n_groups * 0.6))
                )

                vmax = float(np.percentile(data[data > 0], 98)) if (data > 0).any() else 1.0
                vmax = max(vmax, 1e-6)
                im = ax.imshow(
                    data,
                    aspect='auto',
                    cmap=cmap,
                    interpolation='nearest',
                    vmin=0,
                    vmax=vmax,
                )

                cbar = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
                cbar.set_label('Mean Activation Intensity', fontsize=10)
                cbar.ax.tick_params(labelsize=9)

                ax.set_yticks(range(n_groups))
                ax.set_yticklabels(group_labels, fontsize=10)
                ax.set_xlabel('Time (frames)', fontsize=11, fontweight='bold')
                ax.set_ylabel('Anatomical Muscle Group', fontsize=11, fontweight='bold')

                n_ticks = min(10, n_frames)
                tick_positions = np.linspace(0, n_frames - 1, n_ticks, dtype=int)
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_positions, fontsize=9)

                short_label = (
                    task_label.split(': ', 1)[-1]
                    if ': ' in task_label
                    else task_label
                )
                page_num = page_idx + 1
                ax.set_title(
                    f'Muscle Group Activation Over Time — {short_label}'
                    f' (Page {page_num}/{total_pages})',
                    fontsize=12,
                    fontweight='bold',
                    pad=12,
                )

                ax.set_xticks(np.arange(n_frames) - 0.5, minor=True)
                ax.set_yticks(np.arange(n_groups) - 0.5, minor=True)
                ax.grid(which='minor', color='#CCCCCC', linestyle='-', linewidth=0.3)
                ax.tick_params(which='minor', size=0)

                pdf.savefig(fig, dpi=output_config.get('save_dpi', 300), bbox_inches='tight')
                plt.close(fig)

        logger.info("Saved muscle group temporal heatmap: %s", pdf_path)

    def plot_activation_overlay_by_metric(self, features_df: pd.DataFrame, metrics: List[str],
                                          output_path: Path, title: str = "Activation Overlay by Metric",
                                          baseline_stats: Optional[Dict] = None,
                                          reference_baseline_stats: Optional[Dict] = None,
                                          task_profile_ref: Optional[Dict] = None,
                                          all_task_profiles: Optional[Dict] = None) -> None:
        """Multi-page PDF: up to 4 tasks per page with one column per metric.

        Each cell in the grid shows all repetitions of that metric for that task
        overlaid on the same axes.  Requested metrics absent from *features_df*
        are skipped silently.

        Parameters
        ----------
        features_df:
            Frame-level features DataFrame.
        metrics:
            List of column names to plot as separate columns (e.g.
            ``["jawOpen", "mouthSmileLeft", "asymmetry_ratio_mouth"]``).
        output_path:
            Destination path; the suffix is replaced with ``.pdf``.
        baseline_stats, reference_baseline_stats, task_profile_ref, all_task_profiles:
            Forwarded to overlay helpers for profile bands and reference lines.
        """
        from matplotlib.backends.backend_pdf import PdfPages

        if features_df is None or len(features_df) == 0:
            return

        metrics = [m for m in metrics if m in features_df.columns]
        if not metrics:
            return

        task_pages = self._build_task_pages_by_task(features_df)
        n_metrics = len(metrics)
        tasks_per_page = 4
        pdf_path = output_path.with_suffix('.pdf')
        output_config = self.config.get('output', {})

        with PdfPages(pdf_path) as pdf:
            for batch_start in range(0, len(task_pages), tasks_per_page):
                batch = task_pages[batch_start:batch_start + tasks_per_page]
                n_tasks = len(batch)

                fig, axes = plt.subplots(n_tasks, n_metrics,
                                         figsize=(5 * n_metrics, 3.5 * n_tasks),
                                         squeeze=False, sharex=False)

                for row, (task_label, task_key, task_df) in enumerate(batch):
                    repetitions = sorted([r for r in task_df['repetition'].unique() if r != 0])

                    for col, metric in enumerate(metrics):
                        ax = axes[row][col]
                        all_data = []
                        max_duration = 0

                        for i, rep in enumerate(repetitions):
                            rep_df = task_df[task_df['repetition'] == rep]
                            if len(rep_df) == 0 or metric not in rep_df.columns:
                                continue
                            start_time = rep_df['timestamp_abs'].min()
                            time_rel = (rep_df['timestamp_abs'] - start_time).values
                            vals = rep_df[metric].values
                            max_duration = max(max_duration, time_rel.max() if len(time_rel) else 0)
                            color = REPETITION_COLORS[i % len(REPETITION_COLORS)]
                            ax.plot(time_rel, vals, color=color, alpha=0.75, linewidth=2, label=f'Rep {int(rep)}')
                            all_data.append((time_rel, vals))

                        if len(all_data) > 1:
                            min_len = min(len(d[0]) for d in all_data)
                            if min_len > 0:
                                aligned = np.array([d[1][:min_len] for d in all_data])
                                mean_curve = np.mean(aligned, axis=0)
                                time_axis = all_data[0][0][:min_len]
                                ax.plot(time_axis, mean_curve, color='black', linewidth=3,
                                        linestyle='--', label='Mean', zorder=10)

                        if task_profile_ref is not None or all_task_profiles is not None:
                            self._overlay_task_profiles(ax, metric, task_df, repetitions,
                                                        all_task_profiles=all_task_profiles,
                                                        task_profile_ref=task_profile_ref,
                                                        max_duration=max_duration,
                                                        standardization_stats=baseline_stats)

                        baseline_vals_for_ylim = []
                        ref_val = self._get_derived_baseline_value(reference_baseline_stats, metric,
                                                                    standardization_stats=baseline_stats)
                        if ref_val is not None and abs(ref_val) > 1e-6:
                            ref_label = f'{ref_val:.3f}' if abs(ref_val) >= 0.001 else f'{ref_val:.1e}'
                            ax.axhline(y=ref_val, color=COLORBLIND_SAFE_PALETTE['coral'],
                                       linestyle='-', linewidth=2, alpha=0.8,
                                       label=f'Ref Baseline ({ref_label})')
                            baseline_vals_for_ylim.append(ref_val)

                        if all_data:
                            all_vals = np.concatenate([d[1] for d in all_data])
                            if baseline_vals_for_ylim:
                                all_vals = np.concatenate([all_vals, baseline_vals_for_ylim])
                            finite_vals = all_vals[np.isfinite(all_vals)]
                            if len(finite_vals) > 0:
                                ymin, ymax = np.min(finite_vals), np.max(finite_vals)
                                margin = (ymax - ymin) * 0.1 if ymax != ymin else 0.1
                                ax.set_ylim(ymin - margin, ymax + margin)

                        ax.set_xlim(0, max_duration * 1.02 if max_duration else 1)
                        ax.grid(True, alpha=0.3)

                        if row == 0:
                            ax.set_title(metric.replace('_', ' ').title(), fontsize=10, fontweight='bold')
                        if row == n_tasks - 1:
                            ax.set_xlabel('Time (s)')
                        if col == 0:
                            short_label = task_label.split(': ', 1)[-1] if ': ' in task_label else task_label
                            ax.set_ylabel(short_label, fontsize=9, fontweight='bold')
                        if row == 0 and col == n_metrics - 1:
                            ax.legend(
                                loc='upper left', fontsize=7,
                                framealpha=0.85,
                                bbox_to_anchor=(1.01, 1.0),
                                borderaxespad=0,
                            )

                page_num = batch_start // tasks_per_page + 1
                total_pages = (len(task_pages) + tasks_per_page - 1) // tasks_per_page
                fig.suptitle(f"{title} (Page {page_num}/{total_pages})",
                             fontsize=14, fontweight='bold')
                fig.subplots_adjust(right=0.88)
                plt.tight_layout()
                pdf.savefig(fig, dpi=output_config.get('save_dpi', 300), bbox_inches='tight')
                plt.close(fig)

    def plot_feature_for_task(self,
                              features_df: pd.DataFrame,
                              feature: str,
                              task_key: str,
                              output_path: Path,
                              title: Optional[str] = None,
                              baseline_stats: Optional[Dict] = None,
                              reference_baseline_stats: Optional[Dict] = None,
                              task_profile_ref: Optional[Dict] = None,
                              all_task_profiles: Optional[Dict] = None) -> Optional[Path]:
        """On-demand PNG: all repetitions of *feature* for one task overlaid.

        Parameters
        ----------
        features_df:
            DataFrame with columns ``task_group``, ``task_id``, ``repetition``,
            ``timestamp_abs``, and the requested *feature*.
        feature:
            Blendshape or kinematic column name to plot.
        task_key:
            ``"TG_TID"`` string matching :meth:`_build_task_pages_by_task`
            (e.g. ``"A_1"`` or ``"B_3"``).
        output_path:
            Destination PNG path; the suffix is replaced with ``.png`` if necessary.
        title:
            Optional axes title; defaults to ``"<feature> — <task_key>"``.
        baseline_stats, reference_baseline_stats, task_profile_ref, all_task_profiles:
            Forwarded to :meth:`_overlay_task_profiles` and
            :meth:`_get_derived_baseline_value`.

        Returns
        -------
        Path to the saved PNG, or *None* if nothing could be plotted.
        """
        if features_df is None or len(features_df) == 0:
            return None
        if feature not in features_df.columns:
            logger.warning("plot_feature_for_task: column %r not found", feature)
            return None

        parts = task_key.split("_", 1)
        if len(parts) != 2:
            logger.warning("plot_feature_for_task: invalid task_key %r", task_key)
            return None
        task_group, task_id_str = parts[0], parts[1]
        try:
            task_id = int(task_id_str)
        except ValueError:
            logger.warning("plot_feature_for_task: non-integer task_id in key %r", task_key)
            return None

        task_df = features_df[
            (features_df['task_group'] == task_group) &
            (features_df['task_id'] == task_id)
        ].copy()

        if task_df.empty:
            logger.warning("plot_feature_for_task: no rows for task_key %r", task_key)
            return None

        repetitions = sorted([r for r in task_df['repetition'].unique() if r != 0]) \
            if 'repetition' in task_df.columns else []
        if not repetitions:
            return None

        output_config = self.config.get('output', {})
        png_path = output_path.with_suffix('.png')

        fig, ax = plt.subplots(figsize=(10, 4))
        all_data: list = []
        max_duration = 0.0

        for i, rep in enumerate(repetitions):
            rep_df = task_df[task_df['repetition'] == rep]
            if len(rep_df) == 0:
                continue
            if 'timestamp_abs' not in rep_df.columns:
                continue
            start_time = rep_df['timestamp_abs'].min()
            time_rel = (rep_df['timestamp_abs'] - start_time).values
            vals = rep_df[feature].values
            max_duration = max(max_duration, time_rel[-1] if len(time_rel) else 0)
            color = REPETITION_COLORS[i % len(REPETITION_COLORS)]
            ax.plot(time_rel, vals, color=color, alpha=0.75, linewidth=2,
                    label=f'Rep {int(rep)}')
            all_data.append((time_rel, vals))

        if len(all_data) > 1:
            min_len = min(len(d[0]) for d in all_data)
            if min_len > 0:
                aligned = np.array([d[1][:min_len] for d in all_data])
                mean_curve = np.mean(aligned, axis=0)
                time_axis = all_data[0][0][:min_len]
                ax.plot(time_axis, mean_curve, color='black', linewidth=3,
                        linestyle='--', label='Mean', zorder=10)

        if task_profile_ref is not None or all_task_profiles is not None:
            self._overlay_task_profiles(ax, feature, task_df, repetitions,
                                        all_task_profiles=all_task_profiles,
                                        task_profile_ref=task_profile_ref,
                                        max_duration=max_duration,
                                        standardization_stats=baseline_stats)

        ref_val = self._get_derived_baseline_value(reference_baseline_stats, feature,
                                                    standardization_stats=baseline_stats)
        if ref_val is not None and abs(ref_val) > 1e-6:
            ref_label = f'{ref_val:.3f}' if abs(ref_val) >= 0.001 else f'{ref_val:.1e}'
            ax.axhline(y=ref_val, color=COLORBLIND_SAFE_PALETTE['coral'],
                       linestyle='-', linewidth=2, alpha=0.8,
                       label=f'Ref Baseline ({ref_label})')

        if all_data:
            all_vals = np.concatenate([d[1] for d in all_data])
            if ref_val is not None and abs(ref_val) > 1e-6:
                all_vals = np.concatenate([all_vals, [ref_val]])
            finite_vals = all_vals[np.isfinite(all_vals)]
            if len(finite_vals) > 0:
                ymin, ymax = np.min(finite_vals), np.max(finite_vals)
                margin = (ymax - ymin) * 0.1 if ymax != ymin else 0.1
                ax.set_ylim(ymin - margin, ymax + margin)

        ax.set_xlim(0, max_duration * 1.02 if max_duration else 1)
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel(feature.replace('_', ' ').title(), fontsize=11)
        ax.set_title(
            title or f"{feature.replace('_', ' ').title()} \u2014 {task_key}",
            fontsize=12, fontweight='bold',
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9, framealpha=0.85)

        plt.tight_layout()
        fig.savefig(png_path, dpi=output_config.get('save_dpi', 150), bbox_inches='tight')
        plt.close(fig)

        logger.info("Saved feature-for-task PNG: %s", png_path)
        return png_path

    def plot_activation_overlay_by_feature_pdf(self, features_df: pd.DataFrame,
                                               output_path: Path,
                                               title: str = "Activation Overlay by Feature",
                                               baseline_stats: Optional[Dict] = None,
                                               reference_baseline_stats: Optional[Dict] = None,
                                               task_profile_ref: Optional[Dict] = None,
                                               all_task_profiles: Optional[Dict] = None) -> None:
        """Multi-page PDF organised by task, with up to 4 feature rows per page.

        A title page separates each task.  Each feature row shows overlaid
        repetitions, a black dashed mean curve, task-profile bands, and a coral
        reference baseline line.  Metadata and internal columns are excluded.

        Parameters
        ----------
        features_df:
            Frame-level features DataFrame for the session.
        output_path:
            Destination path; the suffix is replaced with ``.pdf``.
        baseline_stats:
            Session neutral-baseline statistics dict.
        reference_baseline_stats:
            External reference session baseline dict.
        task_profile_ref:
            Single-task kinematic reference profile dict.
        all_task_profiles:
            Full cross-task profile dict keyed by ``"TG_TID"``.
        """
        from matplotlib.backends.backend_pdf import PdfPages

        if features_df is None or len(features_df) == 0:
            return

        task_pages = self._build_task_pages_by_task(features_df)
        if not task_pages:
            return

        exclude = {'frame_index', 'timestamp_abs', 'segment', 'repetition',
                   'detection_success', 'detection_confidence', 'time_rel_sec',
                   'task_group', 'task_id', 'task_name', 'occluded', 'brightness',
                   'inter_ocular_distance'}
        feat_cols = [c for c in features_df.columns
                     if c not in exclude and not c.startswith('_')
                     and features_df[c].dtype in [np.float64, np.float32,
                                                   np.int64, np.int32]]

        def _categorise(cols):
            """Categorize columns into facial region groups."""
            groups = [
                ('Derived Metrics', [c for c in cols if c in
                    ('mean_activation', 'max_activation', 'activation_range',
                     'activation_velocity', 'activation_acceleration')]),
                ('Brow Region', [c for c in cols if c.startswith('brow')]),
                ('Eye Region', [c for c in cols if c.startswith('eye')]),
                ('Cheek Region', [c for c in cols if c.startswith('cheek')]),
                ('Nose Region', [c for c in cols if c.startswith('nose')]),
                ('Jaw Region', [c for c in cols if c.startswith('jaw')]),
                ('Mouth Region', [c for c in cols if c.startswith('mouth')]),
                ('Tongue Region', [c for c in cols if c.startswith('tongue')]),
                ('Asymmetry Ratios', [c for c in cols
                                       if c.startswith('asymmetry_ratio')]),
                ('Asymmetry (raw)', [c for c in cols
                                      if c.startswith('asymmetry_')
                                      and not c.startswith('asymmetry_ratio')]),
            ]
            assigned = set()
            for _, cs in groups:
                assigned.update(cs)
            rest = [c for c in cols if c not in assigned]
            if rest:
                groups.append(('Other', rest))
            return [(g, cs) for g, cs in groups if cs]

        feature_groups = _categorise(feat_cols)
        all_features: List[Tuple[str, str]] = []
        for group_name, group_feats in feature_groups:
            for f in group_feats:
                all_features.append((group_name, f))

        if not all_features:
            return

        features_per_page = 8
        fig_width = 14.0
        row_height = 2.0

        PROFILE_COLOR = '#7E57C2'
        PROFILE_BAND  = '#D1C4E9'
        MEAN_COLOR    = '#212121'
        REF_BL_COLOR  = COLORBLIND_SAFE_PALETTE.get('coral', '#FF6B6B')

        pdf_path = output_path.with_suffix('.pdf')
        output_config = self.config.get('output', {})
        save_dpi = output_config.get('save_dpi', 300)

        with PdfPages(pdf_path) as pdf:
            for task_label, task_key, task_df in task_pages:
                short_label = (
                    task_label.split(': ', 1)[-1]
                    if ': ' in task_label else task_label
                )
                task_features = [
                    (gn, f) for gn, f in all_features if f in task_df.columns
                ]
                if not task_features:
                    continue

                total_feat_pages = (len(task_features) + features_per_page - 1) // features_per_page
                prev_group = None

                for fp_idx in range(total_feat_pages):
                    start = fp_idx * features_per_page
                    batch = task_features[start:start + features_per_page]
                    n_feat = len(batch)

                    fig_height = row_height * n_feat + 1.8
                    fig, axes = plt.subplots(
                        n_feat, 1, figsize=(fig_width, fig_height),
                        squeeze=False,
                    )

                    for feat_i, (group_name, feature) in enumerate(batch):
                        ax = axes[feat_i, 0]
                        display_name = feature.replace('_', ' ').title()
                        if group_name != prev_group:
                            subplot_title = f'{group_name}  ›  {display_name}'
                            prev_group = group_name
                        else:
                            subplot_title = display_name

                        repetitions = sorted(
                            [r for r in task_df['repetition'].unique() if r != 0])
                        if not repetitions:
                            repetitions = sorted(task_df['repetition'].unique())

                        all_data: List[Tuple[np.ndarray, np.ndarray]] = []
                        max_duration = 0.0

                        for i, rep in enumerate(repetitions):
                            rep_df = task_df[task_df['repetition'] == rep]
                            if len(rep_df) == 0:
                                continue
                            start_t = rep_df['timestamp_abs'].min()
                            t_rel = (rep_df['timestamp_abs'] - start_t).values
                            vals = rep_df[feature].values
                            dur = t_rel.max() if len(t_rel) > 0 else 0
                            max_duration = max(max_duration, dur)
                            color = REPETITION_COLORS[i % len(REPETITION_COLORS)]
                            ax.plot(t_rel, vals, color=color, alpha=0.55,
                                    linewidth=1.3, label=f'Rep {int(rep)}')
                            all_data.append((t_rel, vals))

                        if len(all_data) > 1:
                            n_interp = 100
                            common_t = np.linspace(0, max_duration, n_interp)
                            interp_vals = []
                            for t_rel_arr, vals_arr in all_data:
                                if len(t_rel_arr) < 2:
                                    continue
                                interp_vals.append(np.interp(common_t, t_rel_arr, vals_arr))
                            if len(interp_vals) > 1:
                                arr = np.array(interp_vals)
                                if arr.size > 0 and np.any(np.isfinite(arr)):
                                    with np.errstate(all='ignore'):
                                        mean_curve = np.nanmean(arr, axis=0)
                                    if np.any(np.isfinite(mean_curve)):
                                        ax.plot(common_t, mean_curve, color=MEAN_COLOR,
                                                linewidth=2.5, linestyle='--',
                                                label='Mean', zorder=8)

                        profile_plotted = False
                        if all_task_profiles and task_key:
                            task_ref = all_task_profiles.get(task_key)
                            if task_ref is not None:
                                pattern = task_ref.get(
                                    'activation_pattern', {}).get(feature)
                                if (pattern is not None
                                        and 'mean_pattern' in pattern
                                        and max_duration > 0):
                                    ref_median = np.array(
                                        pattern['mean_pattern'], dtype=float)
                                    if 'mad_pattern' in pattern:
                                        robust_1s = self._robust_sigma(
                                            pattern['mad_pattern'])
                                    else:
                                        robust_1s = np.array(
                                            pattern.get('std_pattern',
                                                         np.zeros_like(ref_median)),
                                            dtype=float)
                                    n_curves = pattern.get('n_curves', 1)
                                    if n_curves < 30 and n_curves >= 2:
                                        from scipy.stats import t as t_dist
                                        scale = float(
                                            t_dist.ppf(0.84, max(n_curves - 1, 1))
                                            * np.sqrt(1.0 + 1.0 / n_curves)
                                        )
                                    else:
                                        scale = 1.0
                                    band_inner = robust_1s * scale
                                    band_outer = robust_1s * scale * 2.0
                                    t_prof = np.linspace(
                                        0, max_duration, len(ref_median))
                                    inner_label = (
                                        f'Profile ±1σ MAD (n={n_curves})'
                                        if n_curves < 30
                                        else 'Profile ±1σ (MAD)'
                                    )
                                    ax.fill_between(
                                        t_prof,
                                        ref_median - band_outer,
                                        ref_median + band_outer,
                                        color='#E8D5F5', alpha=0.18,
                                        label='Profile ±2σ', zorder=0)
                                    ax.fill_between(
                                        t_prof,
                                        ref_median - band_inner,
                                        ref_median + band_inner,
                                        color=PROFILE_BAND, alpha=0.35,
                                        label=inner_label, zorder=1)
                                    ax.plot(
                                        t_prof, ref_median,
                                        color=PROFILE_COLOR, linewidth=2,
                                        linestyle='-', alpha=0.85,
                                        label='Task Profile', zorder=2)
                                    profile_plotted = True

                        ref_val = self._get_derived_baseline_value(
                            reference_baseline_stats, feature,
                            standardization_stats=baseline_stats)
                        if ref_val is not None and abs(ref_val) > 1e-6:
                            ref_lbl = (f'{ref_val:.2f}'
                                       if abs(ref_val) >= 0.01
                                       else f'{ref_val:.1e}')
                            ax.axhline(y=ref_val, color=REF_BL_COLOR,
                                       linestyle='-', linewidth=1.8,
                                       alpha=0.7,
                                       label=f'Ref Baseline ({ref_lbl})',
                                       zorder=3)

                        ylim_vals = []
                        for td, vd in all_data:
                            ylim_vals.extend(vd[np.isfinite(vd)].tolist())
                        if ref_val is not None and abs(ref_val) > 1e-6:
                            ylim_vals.append(ref_val)
                        if profile_plotted and task_ref is not None:
                            pat = task_ref.get(
                                'activation_pattern', {}).get(feature)
                            if pat and 'mean_pattern' in pat:
                                rm = np.array(pat['mean_pattern'], dtype=float)
                                if 'mad_pattern' in pat:
                                    r1s = self._robust_sigma(pat['mad_pattern'])
                                else:
                                    r1s = np.array(pat.get('std_pattern',
                                                  np.zeros_like(rm)), dtype=float)
                                r2s = r1s * 2.0
                                profile_bounds = np.concatenate([rm - r2s, rm + r2s])
                                ylim_vals.extend(
                                    profile_bounds[np.isfinite(profile_bounds)].tolist())

                        finite_ylim = [v for v in ylim_vals if np.isfinite(v)]
                        if finite_ylim:
                            ymin = min(finite_ylim)
                            ymax = max(finite_ylim)
                            margin = (ymax - ymin) * 0.08 if ymax != ymin else 0.1
                            ax.set_ylim(ymin - margin, ymax + margin)

                        ax.set_xlim(0, max_duration * 1.02 if max_duration > 0 else 1)
                        ax.set_title(subplot_title, fontsize=9, fontweight='bold',
                                     pad=3)
                        is_bottom = (feat_i == n_feat - 1)
                        if is_bottom:
                            ax.set_xlabel('Time from task onset (s)', fontsize=7)
                        else:
                            ax.set_xlabel('')
                            ax.tick_params(axis='x', labelbottom=False)
                        ax.set_ylabel(display_name[:22], fontsize=7)
                        ax.tick_params(axis='both', labelsize=6)
                        ax.grid(True, alpha=0.25)

                        handles, labels = ax.get_legend_handles_labels()
                        if handles:
                            seen = set()
                            deduped_h, deduped_l = [], []
                            for h, l in zip(handles, labels):
                                if l not in seen:
                                    seen.add(l)
                                    deduped_h.append(h)
                                    deduped_l.append(l)
                            ax.legend(
                                deduped_h, deduped_l,
                                fontsize=6.5, framealpha=0.85,
                                loc='upper left',
                                bbox_to_anchor=(1.01, 1.0),
                                borderaxespad=0,
                            )

                    fig.suptitle(
                        f'{short_label}  —  {title}'
                        f'  (Page {fp_idx + 1}/{total_feat_pages})',
                        fontsize=12, fontweight='bold', y=0.998)
                    fig.subplots_adjust(top=0.95, bottom=0.04, right=0.88,
                                        hspace=0.35)
                    pdf.savefig(fig, dpi=save_dpi, bbox_inches='tight')
                    plt.close(fig)

    def plot_fatigue_analysis(self, continuous_metrics: Dict[str, Any], output_path: Path,
                              title: str = "Fatigue Analysis") -> None:
        """Four-panel figure of fatigue indicators from a continuous recording.

        Panels show activation over time with a fitted linear trend, repeated
        task amplitude decay, asymmetry drift, and response-time proxy.
        Saved as a PNG.

        Parameters
        ----------
        continuous_metrics:
            Dict returned by the continuous metrics computer containing a
            ``fatigue`` sub-dict with ``window_metrics`` list.
        output_path:
            Destination path; the suffix is replaced with ``.png``.
        title:
            Figure super-title.
        """
        if not continuous_metrics:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax1 = axes[0, 0]
        if 'fatigue' in continuous_metrics:
            fatigue = continuous_metrics['fatigue']
            window_data = fatigue.get('window_metrics', [])
            if window_data:
                times = [w['window_start'] / 60 for w in window_data]
                activations = [w['mean_activation'] for w in window_data]

                ax1.plot(times, activations, 'o-', color=COLORBLIND_SAFE_PALETTE['blue'],
                        linewidth=2, markersize=6)
                ax1.fill_between(times, activations, alpha=0.3, color=COLORBLIND_SAFE_PALETTE['cyan'])

                if len(times) > 2:
                    z = np.polyfit(times, activations, 1)
                    p = np.poly1d(z)
                    ax1.plot(times, p(times), '--', color=COLORBLIND_SAFE_PALETTE['red'],
                            linewidth=2, label=f'Trend (slope: {z[0]:.4f})')
                    ax1.legend()

                ax1.set_xlabel('Time (minutes)')
                ax1.set_ylabel('Mean Activation')
                ax1.set_title(f"Activation Over Time\n(Decay: {fatigue.get('decay_percent', 0):.1f}%)")
                ax1.grid(True, alpha=0.3)

        ax2 = axes[0, 1]
        if 'asymmetry_trend' in continuous_metrics:
            trend = continuous_metrics['asymmetry_trend']
            window_data = trend.get('window_data', [])
            if window_data:
                times = [w['window_start'] / 60 for w in window_data]
                asymmetries = [w['mean_asymmetry'] for w in window_data]

                color = COLORBLIND_SAFE_PALETTE['orange']
                ax2.plot(times, asymmetries, 's-', color=color, linewidth=2, markersize=6)
                ax2.fill_between(times, asymmetries, alpha=0.3, color=COLORBLIND_SAFE_PALETTE['peach'])

                ax2.axhline(y=0.15, color=COLORBLIND_SAFE_PALETTE['yellow'],
                           linestyle='--', alpha=0.7, label='Mild threshold')
                ax2.axhline(y=0.25, color=COLORBLIND_SAFE_PALETTE['red'],
                           linestyle='--', alpha=0.7, label='Moderate threshold')

                is_increasing = trend.get('is_increasing', False)
                trend_text = "INCREASING ↑" if is_increasing else "STABLE"
                ax2.set_title(f"Asymmetry Trend: {trend_text}\n(Slope: {trend.get('slope_per_minute', 0):.4f}/min)")
                ax2.set_xlabel('Time (minutes)')
                ax2.set_ylabel('Mean Asymmetry')
                ax2.legend(loc='upper left')
                ax2.grid(True, alpha=0.3)

        ax3 = axes[1, 0]
        if 'fatigue' in continuous_metrics:
            fatigue = continuous_metrics['fatigue']
            categories = ['Early Session', 'Late Session']
            activations = [fatigue.get('early_activation_mean', 0),
                          fatigue.get('late_activation_mean', 0)]

            colors = [COLORBLIND_SAFE_PALETTE['green'], COLORBLIND_SAFE_PALETTE['coral']]
            bars = ax3.bar(categories, activations, color=colors, alpha=0.8, edgecolor='black')

            for bar, val in zip(bars, activations):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', fontsize=10)

            ax3.set_ylabel('Mean Activation')
            ax3.set_title('Early vs Late Session Comparison')
            ax3.set_ylim(0, max(activations) * 1.2 if activations else 1)

        ax4 = axes[1, 1]
        if 'response_latency' in continuous_metrics:
            latency = continuous_metrics['response_latency']
            ax4.text(0.5, 0.5,
                    f"Response Latency Analysis\n\n"
                    f"Mean: {latency.get('mean_latency', 0):.2f}s\n"
                    f"Std: {latency.get('std_latency', 0):.2f}s\n"
                    f"Trend: {latency.get('trend', 0):.4f}s/response",
                    ha='center', va='center', fontsize=12,
                    transform=ax4.transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax4.text(0.5, 0.5, "No response latency data available",
                    ha='center', va='center', transform=ax4.transAxes)
        ax4.axis('off')
        ax4.set_title('Response Latency')

        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        self._save_figure(fig, output_path)
        plt.close(fig)

    def plot_statistical_comparison(self, ref_metrics_df: pd.DataFrame,
                                    test_metrics_df: pd.DataFrame,
                                    output_path: Path,
                                    title: str = "Statistical Comparison") -> None:
        """Bar chart comparing reference and test session metrics with t-test results.

        Computes independent-samples t-tests and Cohen's d effect sizes for up to
        10 numeric columns that are shared between the two DataFrames.  Bars show
        the mean for each condition, with the p-value and effect size annotated.
        Saved as a PNG.

        Parameters
        ----------
        ref_metrics_df:
            Repetition metrics DataFrame from the reference (baseline) session.
        test_metrics_df:
            Repetition metrics DataFrame from the test session.
        output_path:
            Destination path; the suffix is replaced with ``.png``.
        title:
            Figure super-title.
        """
        if ref_metrics_df is None or test_metrics_df is None:
            return
        if len(ref_metrics_df) == 0 or len(test_metrics_df) == 0:
            return

        ref_cols = set(ref_metrics_df.select_dtypes(include=[np.number]).columns)
        test_cols = set(test_metrics_df.select_dtypes(include=[np.number]).columns)
        common_cols = list(ref_cols & test_cols)

        exclude = {'repetition', 'n_frames', 'frame_index'}
        common_cols = [c for c in common_cols if c not in exclude][:10]

        if not common_cols:
            return

        results = []
        for col in common_cols:
            ref_vals = ref_metrics_df[col].dropna().values
            test_vals = test_metrics_df[col].dropna().values

            if len(ref_vals) > 1 and len(test_vals) > 1:
                t_stat, p_value = stats.ttest_ind(ref_vals, test_vals)

                pooled_std = np.sqrt((np.var(ref_vals) + np.var(test_vals)) / 2)
                cohens_d = (np.mean(test_vals) - np.mean(ref_vals)) / pooled_std if pooled_std > 0 else 0

                results.append({
                    'feature': col,
                    'ref_mean': np.mean(ref_vals),
                    'test_mean': np.mean(test_vals),
                    'ref_std': np.std(ref_vals),
                    'test_std': np.std(test_vals),
                    'difference': np.mean(test_vals) - np.mean(ref_vals),
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'significant': p_value < 0.05
                })

        if not results:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax1 = axes[0]
        features = [r['feature'][:15] for r in results]
        ref_means = [r['ref_mean'] for r in results]
        test_means = [r['test_mean'] for r in results]
        ref_stds = [r['ref_std'] for r in results]
        test_stds = [r['test_std'] for r in results]

        x = np.arange(len(features))
        width = 0.35

        bars1 = ax1.bar(x - width/2, ref_means, width, yerr=ref_stds,
                       label='Reference', color=COLORBLIND_SAFE_PALETTE['blue'],
                       alpha=0.8, capsize=3)
        bars2 = ax1.bar(x + width/2, test_means, width, yerr=test_stds,
                       label='Test', color=COLORBLIND_SAFE_PALETTE['orange'],
                       alpha=0.8, capsize=3)

        for i, r in enumerate(results):
            if r['significant']:
                max_val = max(ref_means[i] + ref_stds[i], test_means[i] + test_stds[i])
                ax1.text(i, max_val + 0.05, '*', ha='center', fontsize=14, fontweight='bold')

        ax1.set_ylabel('Value')
        ax1.set_title('Mean Comparison (Reference vs Test)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(features, rotation=45, ha='right', fontsize=13)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        ax2 = axes[1]
        cohens_d = [r['cohens_d'] for r in results]
        colors = [COLORBLIND_SAFE_PALETTE['green'] if abs(d) < 0.5 else
                 COLORBLIND_SAFE_PALETTE['orange'] if abs(d) < 0.8 else
                 COLORBLIND_SAFE_PALETTE['red'] for d in cohens_d]

        bars = ax2.barh(features, cohens_d, color=colors, alpha=0.8, edgecolor='black')

        ax2.axvline(x=0, color='black', linewidth=1)
        ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        ax2.axvline(x=-0.5, color='gray', linestyle='--', alpha=0.5)
        ax2.axvline(x=0.8, color='gray', linestyle=':', alpha=0.5)
        ax2.axvline(x=-0.8, color='gray', linestyle=':', alpha=0.5)

        ax2.set_xlabel("Cohen's d (Effect Size)")
        ax2.set_title("Effect Sizes\n(Small<0.5, Medium<0.8, Large≥0.8)")
        ax2.grid(True, alpha=0.3, axis='x')

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_figure(fig, output_path)
        plt.close(fig)

    def plot_clinical_comparison(self, comparison_data: Dict[str, Any], output_path: Path,
                                 title: str = "Clinical Notes vs ML Predictions") -> None:
        """Two-panel figure comparing clinical observations against pipeline predictions.

        Left panel: bar chart of match, clinical-only, and ML-only counts with
        the overall agreement rate in the title.  Right panel: indication-level
        breakdown.  Saved as a PNG.

        Parameters
        ----------
        comparison_data:
            Dict from the clinical-notes comparison step, containing an
            ``agreement`` sub-dict with keys ``matches``, ``clinical_only``,
            ``ml_only``, and ``agreement_rate``.
        output_path:
            Destination path; the suffix is replaced with ``.png``.
        title:
            Figure super-title.
        """
        if not comparison_data:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax1 = axes[0]
        agreement = comparison_data.get('agreement', {})

        matches = len(agreement.get('matches', []))
        clinical_only = len(agreement.get('clinical_only', []))
        ml_only = len(agreement.get('ml_only', []))

        categories = ['Matches', 'Clinical Only\n(Missed by ML)', 'ML Only\n(Not in Clinical)']
        values = [matches, clinical_only, ml_only]
        colors = [COLORBLIND_SAFE_PALETTE['green'], COLORBLIND_SAFE_PALETTE['coral'],
                 COLORBLIND_SAFE_PALETTE['cyan']]

        bars = ax1.bar(categories, values, color=colors, alpha=0.8, edgecolor='black')

        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(val), ha='center', fontsize=12, fontweight='bold')

        ax1.set_ylabel('Count')
        agreement_rate = agreement.get('agreement_rate', 0) * 100
        ax1.set_title(f'Agreement Summary\n(Agreement Rate: {agreement_rate:.1f}%)')

        ax2 = axes[1]
        discrepancies = comparison_data.get('discrepancies', [])

        if discrepancies:
            text = "Discrepancies:\n\n"
            for i, d in enumerate(discrepancies[:8]):
                text += f"• {d.get('note', '')[:50]}...\n"
        else:
            text = "No discrepancies found!\n\nML predictions match clinical observations."

        ax2.text(0.1, 0.5, text, ha='left', va='center', fontsize=10,
                transform=ax2.transAxes, family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax2.axis('off')
        ax2.set_title('Discrepancy Details')

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_figure(fig, output_path)
        plt.close(fig)

    def create_screening_table(self, screening_results: Dict[str, Any],
                               output_path: Path) -> None:
        """PDF table of screening indications with severity, confidence, and source columns.

        Row 1: indication-level table (indication type, severity, confidence,
        source node, description).  Row 2: overall confidence summary bar.
        Saved as a PDF.

        Parameters
        ----------
        screening_results:
            Screening result dict with ``indications`` list and ``confidence``
            sub-dict.
        output_path:
            Destination path; the suffix is replaced with ``.pdf``.
        """
        indications = screening_results.get('indications', [])
        confidence = screening_results.get('confidence', {})

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])

        ax1 = axes[0]
        ax1.axis('off')

        if indications:
            table_data = []
            for ind in indications:
                row = [
                    ind.get('indication_type', '').replace('_', ' ').title(),
                    ind.get('severity', '').upper(),
                    f"{ind.get('confidence', 0):.1%}",
                    ind.get('source_node', '').replace('_', ' '),
                    ind.get('description', '')[:50] + '...' if len(ind.get('description', '')) > 50 else ind.get('description', '')
                ]
                table_data.append(row)

            columns = ['Indication', 'Severity', 'Confidence', 'Source', 'Description']

            table = ax1.table(cellText=table_data, colLabels=columns,
                             loc='center', cellLoc='left',
                             colWidths=[0.15, 0.1, 0.1, 0.2, 0.45])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.8)

            for i, ind in enumerate(indications):
                severity = ind.get('severity', '')
                if severity == 'severe':
                    table[(i+1, 1)].set_facecolor('#ffcccc')
                elif severity == 'moderate':
                    table[(i+1, 1)].set_facecolor('#ffe6cc')
                else:
                    table[(i+1, 1)].set_facecolor('#ccffcc')

            for j in range(len(columns)):
                table[(0, j)].set_facecolor('#37474F')
                table[(0, j)].set_text_props(color='white', fontweight='bold')
        else:
            ax1.text(0.5, 0.5, "No screening indications detected\n\nAll parameters within normal range",
                    ha='center', va='center', fontsize=14,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

        ax1.set_title('Screening Indications Table', fontsize=12, fontweight='bold', pad=20)

        ax2 = axes[1]
        ax2.axis('off')

        conf_data = [[
            f"{confidence.get('data_quality', 0):.1%}",
            f"{confidence.get('consistency', 0):.1%}",
            f"{confidence.get('model_rule_agreement', 0):.1%}",
            f"{confidence.get('overall', 0):.1%}"
        ]]
        conf_columns = ['Data Quality', 'Consistency', 'Model Agreement', 'Overall']

        conf_table = ax2.table(cellText=conf_data, colLabels=conf_columns,
                              loc='center', cellLoc='center',
                              colWidths=[0.25, 0.25, 0.25, 0.25])
        conf_table.auto_set_font_size(False)
        conf_table.set_fontsize(11)
        conf_table.scale(1.2, 2)

        for j, key in enumerate(['data_quality', 'consistency', 'model_rule_agreement', 'overall']):
            val = confidence.get(key, 0)
            if val >= 0.7:
                conf_table[(1, j)].set_facecolor('#ccffcc')
            elif val >= 0.5:
                conf_table[(1, j)].set_facecolor('#ffe6cc')
            else:
                conf_table[(1, j)].set_facecolor('#ffcccc')

        for j in range(len(conf_columns)):
            conf_table[(0, j)].set_facecolor('#37474F')
            conf_table[(0, j)].set_text_props(color='white', fontweight='bold')

        ax2.set_title('Confidence Metrics', fontsize=12, fontweight='bold', pad=20)

        plt.tight_layout()
        self._save_figure(fig, output_path, is_table=True)
        plt.close(fig)

    def create_anomaly_table(self, anomaly_results: Dict[str, Any],
                             output_path: Path) -> None:
        """PDF table of anomaly detection summary statistics and per-repetition scores.

        Top section: session-level summary (n_samples, n_anomalies, anomaly rate,
        mean deviation score, model type, PCA dimensionality, features with
        deviations).  Bottom section: per-repetition anomaly score table.
        Saved as a PDF.

        Parameters
        ----------
        anomaly_results:
            Dict returned by AnomalyDetector.detect_anomalies().
        output_path:
            Destination path; the suffix is replaced with ``.pdf``.
        """
        fig, axes = plt.subplots(2, 1, figsize=(16, 11), height_ratios=[1, 2])

        ax1 = axes[0]
        ax1.axis('off')

        summary = anomaly_results.get('summary', {})
        ml_meta = anomaly_results.get('ml_metadata', {})
        n_pca = summary.get('n_pca_components', ml_meta.get('n_pca_components', 0))
        summary_data = [[
            str(summary.get('n_samples', 0)),
            str(summary.get('n_anomalies', 0)),
            f"{summary.get('anomaly_rate', 0):.1%}",
            f"{summary.get('mean_deviation_score', 0):.3f}",
            f"{summary.get('mean_score_confidence', 0):.2f}",
            summary.get('model_type', 'unknown'),
            str(n_pca),
            str(summary.get('n_features_with_deviations', 0))
        ]]
        summary_columns = ['Samples', 'Anomalies', 'Rate', 'Dev. Score',
                           'Confidence', 'Model', 'PCA Dim', 'Dev. Feat.']

        summary_table = ax1.table(cellText=summary_data, colLabels=summary_columns,
                                 loc='center', cellLoc='center',
                                 colWidths=[0.11, 0.11, 0.09, 0.11, 0.11, 0.14, 0.1, 0.11])
        summary_table.auto_set_font_size(False)
        summary_table.set_fontsize(10)
        summary_table.scale(1.2, 2)

        for j in range(len(summary_columns)):
            summary_table[(0, j)].set_facecolor('#37474F')
            summary_table[(0, j)].set_text_props(color='white', fontweight='bold')

        ax1.set_title('Anomaly Detection Summary (PCA + Confidence-Weighted Consensus)',
                       fontsize=12, fontweight='bold', pad=20)

        ax2 = axes[1]
        ax2.axis('off')

        feature_devs = anomaly_results.get('feature_deviations', {})

        if feature_devs:
            sorted_feats = sorted(feature_devs.items(),
                                 key=lambda x: x[1].get('max_range_dev',
                                                         x[1].get('max_abs_modified_z',
                                                                   x[1].get('max_abs_z_score', 0))),
                                 reverse=True)[:15]

            table_data = []
            for feat, stats in sorted_feats:
                mean_rd = stats.get('mean_range_dev', 0)
                max_rd = stats.get('max_range_dev', 0)
                mz = stats.get('mean_modified_z', stats.get('mean_z_score', 0))
                w = stats.get('weight', 1.0)
                row = [
                    feat.replace('_', ' ')[:25],
                    f"{mean_rd:.2f}",
                    f"{max_rd:.2f}",
                    f"{mz:.2f}",
                    f"{w:.1f}",
                    str(stats.get('n_deviant', 0)),
                    '\u2605' if max_rd > 1.5 else ''
                ]
                table_data.append(row)

            columns = ['Feature', 'Mean Rng', 'Max Rng', 'Mean Mod-Z',
                       'Weight', 'N Dev', 'Signif.']

            table = ax2.table(cellText=table_data, colLabels=columns,
                             loc='center', cellLoc='center',
                             colWidths=[0.26, 0.11, 0.11, 0.12, 0.1, 0.1, 0.08])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)

            for i, (feat, stats) in enumerate(sorted_feats):
                max_rd = stats.get('max_range_dev', 0)
                if max_rd > 1.5:
                    for j in range(len(columns)):
                        table[(i+1, j)].set_facecolor('#ffcccc')
                elif max_rd > 0.75:
                    for j in range(len(columns)):
                        table[(i+1, j)].set_facecolor('#ffe6cc')

            for j in range(len(columns)):
                table[(0, j)].set_facecolor('#37474F')
                table[(0, j)].set_text_props(color='white', fontweight='bold')

        ax2.set_title('Feature Deviation Details (\u2605 = range dev > 1.5)',
                       fontsize=12, fontweight='bold', pad=20)

        plt.tight_layout()
        self._save_figure(fig, output_path, is_table=True)
        plt.close(fig)

    def create_heatmap_table(self, features_df: pd.DataFrame,
                             output_path: Path) -> None:
        """PDF table of mean, std, min, max, and range for up to 20 blendshape columns.

        Excludes metadata and asymmetry columns.  Saved as a PDF.

        Parameters
        ----------
        features_df:
            Frame-level features DataFrame for the session.
        output_path:
            Destination path; the suffix is replaced with ``.pdf``.
        """
        exclude = {'frame_index', 'timestamp_abs', 'segment', 'repetition',
                  'detection_success', 'detection_confidence', 'time_rel_sec',
                  'task_group', 'task_id', 'task_name'}

        feature_cols = [c for c in features_df.columns
                       if c not in exclude and not c.startswith('asymmetry')
                       and not c.startswith('activation_')][:20]

        if not feature_cols:
            return

        fig, ax = plt.subplots(figsize=(14, max(6, len(feature_cols) * 0.4)))
        ax.axis('off')

        table_data = []
        for col in feature_cols:
            values = features_df[col].dropna().values
            if len(values) > 0:
                row = [
                    col.replace('_', ' ')[:25],
                    f"{np.mean(values):.3f}",
                    f"{np.std(values):.3f}",
                    f"{np.min(values):.3f}",
                    f"{np.max(values):.3f}",
                    f"{np.max(values) - np.min(values):.3f}"
                ]
                table_data.append(row)

        columns = ['Feature', 'Mean', 'Std Dev', 'Min', 'Max', 'Range']

        table = ax.table(cellText=table_data, colLabels=columns,
                        loc='center', cellLoc='center',
                        colWidths=[0.3, 0.14, 0.14, 0.14, 0.14, 0.14])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.4)

        for j in range(len(columns)):
            table[(0, j)].set_facecolor('#37474F')
            table[(0, j)].set_text_props(color='white', fontweight='bold')

        ax.set_title('Activation Statistics by Feature', fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        self._save_figure(fig, output_path, is_table=True)
        plt.close(fig)

    def plot_confusion_matrix(self, anomaly_results: Dict[str, Any],
                              output_path: Path,
                              title: str = "Detection Agreement Matrix",
                              clinical_comparison: Optional[Dict[str, Any]] = None) -> None:
        """Method agreement matrix with optional clinical comparison panel.

        Left panel (when anomaly_results has data): inter-method agreement matrix
        showing how often each pair of detection methods (OC-SVM/IF, Mahalanobis,
        centroid, within-session) agreed on each repetition.  Right panel (when
        clinical_comparison has data): clinical vs pipeline agreement.  Saved as
        a PNG.

        Parameters
        ----------
        anomaly_results:
            Dict returned by AnomalyDetector.detect_anomalies().
        output_path:
            Destination path; the suffix is replaced with ``.png``.
        title:
            Figure title.
        clinical_comparison:
            Optional clinical agreement dict with an ``agreement`` sub-dict.
        """
        has_anomaly = (anomaly_results is not None and
                       len(anomaly_results.get('is_anomaly', [])) > 0)
        has_clinical = (clinical_comparison is not None and
                        'agreement' in clinical_comparison)
        if not has_anomaly and not has_clinical:
            return

        n_panels = (1 if has_anomaly else 0) + (1 if has_clinical else 0)
        fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
        if n_panels == 1:
            axes = [axes]
        panel_idx = 0

        if has_anomaly:
            ax = axes[panel_idx]
            panel_idx += 1

            is_anom = np.array(anomaly_results['is_anomaly'])
            method_votes = anomaly_results.get('method_votes', [])

            if method_votes and len(method_votes) == len(is_anom):
                votes_arr = np.array(method_votes)
                method_names = ['OC-SVM/IF', 'Mahalanobis', 'Centroid', 'Within-Session']
                n_methods = 4
                agree_matrix = np.zeros((n_methods, n_methods), dtype=int)
                for m1 in range(n_methods):
                    for m2 in range(n_methods):
                        agree_matrix[m1, m2] = int(np.sum(
                            votes_arr[:, m1].astype(bool) == votes_arr[:, m2].astype(bool)
                        ))

                im = ax.imshow(agree_matrix, cmap='Blues', aspect='equal',
                               vmin=0, vmax=max(agree_matrix.max(), 1))
                for mi in range(n_methods):
                    for mj in range(n_methods):
                        color = 'white' if agree_matrix[mi, mj] > agree_matrix.max() / 2 else 'black'
                        ax.text(mj, mi, str(agree_matrix[mi, mj]), ha='center', va='center',
                                fontsize=14, fontweight='bold', color=color)

                ax.set_xticks(range(n_methods))
                ax.set_xticklabels(method_names, fontsize=10)
                ax.set_yticks(range(n_methods))
                ax.set_yticklabels(method_names, fontsize=10)
                ax.set_xlabel('Method', fontsize=11)
                ax.set_ylabel('Method', fontsize=11)

                n_unanimous = int(np.sum(votes_arr.sum(axis=1) == n_methods)
                                  + np.sum(votes_arr.sum(axis=1) == 0))
                pct_agree = n_unanimous / len(is_anom) * 100 if len(is_anom) > 0 else 0
                ax.set_title(f'Method Pairwise Agreement\n(Unanimous: {pct_agree:.0f}%)', fontsize=11)
                plt.colorbar(im, ax=ax, shrink=0.8, label='N agree')
            else:
                deviations = anomaly_results.get('deviations', [])
                has_deviant = np.array([
                    any(d.get(f, {}).get('is_deviant', False) for f in d)
                    for d in deviations
                ]) if deviations else np.zeros(len(is_anom), dtype=bool)

                tp = int(np.sum(is_anom & has_deviant))
                fp = int(np.sum(is_anom & ~has_deviant))
                fn = int(np.sum(~is_anom & has_deviant))
                tn = int(np.sum(~is_anom & ~has_deviant))
                cm = np.array([[tn, fp], [fn, tp]])

                im = ax.imshow(cm, cmap='Blues', aspect='equal',
                               vmin=0, vmax=max(cm.max(), 1))
                for i in range(2):
                    for j in range(2):
                        color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
                        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                                fontsize=16, fontweight='bold', color=color)

                ax.set_xticks([0, 1])
                ax.set_xticklabels(['Normal', 'Anomaly'], fontsize=10)
                ax.set_yticks([0, 1])
                ax.set_yticklabels(['Normal', 'Deviant'], fontsize=10)
                model_label = anomaly_results.get('model_type', 'Consensus')
                ax.set_xlabel(model_label, fontsize=11)
                ax.set_ylabel('Range Deviation', fontsize=11)

                total = tp + fp + fn + tn
                agreement = (tp + tn) / total * 100 if total > 0 else 0
                ax.set_title(f'Consensus vs Range Dev\n(Agreement: {agreement:.0f}%)', fontsize=11)
                plt.colorbar(im, ax=ax, shrink=0.8)

        if has_clinical:
            ax = axes[panel_idx]
            agreement = clinical_comparison['agreement']

            matches = len(agreement.get('matches', []))
            clinical_only = len(agreement.get('clinical_only', []))
            ml_only = len(agreement.get('ml_only', []))

            cm = np.array([
                [matches, ml_only],
                [clinical_only, 0]
            ])

            im = ax.imshow(cm, cmap='Oranges', aspect='equal',
                          vmin=0, vmax=max(cm.max(), 1))
            for i in range(2):
                for j in range(2):
                    color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
                    label = str(cm[i, j]) if not (i == 1 and j == 1) else '\u2014'
                    ax.text(j, i, label, ha='center', va='center',
                           fontsize=16, fontweight='bold', color=color)

            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Predicted', 'ML Only'], fontsize=10)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Observed', 'Clinical Only'], fontsize=10)
            ax.set_xlabel('ML Predictions', fontsize=11)
            ax.set_ylabel('Clinical Notes', fontsize=11)

            rate = agreement.get('agreement_rate', 0) * 100
            ax.set_title(f'Clinical vs ML\n(Agreement: {rate:.0f}%)', fontsize=11)
            plt.colorbar(im, ax=ax, shrink=0.8)

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_figure(fig, output_path)
        plt.close(fig)

    def plot_cluster_embeddings(self, repetition_metrics_df: pd.DataFrame,
                                 anomaly_results: Optional[Dict[str, Any]],
                                 output_path: Path,
                                 title: str = "Repetition Cluster Embeddings",
                                 task_profile_ref: Optional[Dict] = None) -> None:
        """PCA and optional t-SNE scatter of repetition metrics coloured by task group and anomaly status.

        Numeric metric columns are standardised before dimensionality reduction.
        t-SNE is only attempted when the dataset has at least 5 samples.  Points
        are sized and coloured by anomaly status when *anomaly_results* is
        provided.  Saved as a PNG.

        Parameters
        ----------
        repetition_metrics_df:
            One row per (task, repetition) as produced by the metrics computer.
        anomaly_results:
            Optional anomaly results dict for colour coding.
        output_path:
            Destination path; the suffix is replaced with ``.png``.
        title:
            Figure super-title.
        task_profile_ref:
            Optional reference profile dict (reserved for future overlay).
        """
        if repetition_metrics_df is None or len(repetition_metrics_df) < 3:
            return

        exclude = {'repetition', 'n_frames', 'task_group', 'task_id', 'task_name'}
        numeric_cols = [c for c in repetition_metrics_df.select_dtypes(include=[np.number]).columns
                        if c not in exclude and not c.startswith('_')]
        if len(numeric_cols) < 2:
            return

        X = repetition_metrics_df[numeric_cols].fillna(0).values
        scaler = SkScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=min(2, X_scaled.shape[1]))
        X_pca = pca.fit_transform(X_scaled)

        do_tsne = len(X_scaled) >= 5 and X_scaled.shape[1] >= 2
        if do_tsne:
            perplexity = min(30, max(2, len(X_scaled) - 1))
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                        max_iter=500)
            X_tsne = tsne.fit_transform(X_scaled)

        n_cols = 2 if do_tsne else 1
        fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 6))
        if n_cols == 1:
            axes = [axes]

        is_anom = np.array(anomaly_results.get('is_anomaly', [False] * len(X)))  \
            if anomaly_results else np.zeros(len(X), dtype=bool)

        has_task = ('task_group' in repetition_metrics_df.columns and
                    repetition_metrics_df['task_group'].notna().any() and
                    (repetition_metrics_df['task_group'].astype(str) != '0').any())

        reps = (repetition_metrics_df['repetition'].values
                if 'repetition' in repetition_metrics_df.columns
                else np.arange(1, len(X) + 1))

        for ax, X_emb, method in [(axes[0], X_pca, 'PCA')] + \
                ([(axes[1], X_tsne, 't-SNE')] if do_tsne else []):

            if has_task:
                task_groups = repetition_metrics_df['task_group'].astype(str).values
                tg_colors = self.colors.get('task_groups', {})
                point_colors = [tg_colors.get(tg, COLORBLIND_SAFE_PALETTE['gray'])
                                for tg in task_groups]
            else:
                point_colors = [COLORBLIND_SAFE_PALETTE['blue']] * len(X_emb)

            for i in range(len(X_emb)):
                marker = 'X' if is_anom[i] else 'o'
                edge = COLORBLIND_SAFE_PALETTE['red'] if is_anom[i] else 'black'
                lw = 1.8 if is_anom[i] else 0.6
                ax.scatter(X_emb[i, 0], X_emb[i, 1], c=[point_colors[i]],
                          marker=marker, s=120, edgecolors=edge, linewidths=lw,
                          zorder=3)
                ax.annotate(f'R{int(reps[i])}', (X_emb[i, 0], X_emb[i, 1]),
                           textcoords='offset points', xytext=(5, 5),
                           fontsize=8, alpha=0.8)

            if method == 'PCA':
                ev = pca.explained_variance_ratio_
                ax.set_xlabel(f'PC1 ({ev[0]:.0%} var)')
                ax.set_ylabel(f'PC2 ({ev[1]:.0%} var)' if len(ev) > 1 else 'PC2')
            else:
                ax.set_xlabel('t-SNE 1')
                ax.set_ylabel('t-SNE 2')

            ax.set_title(f'{method} Embedding', fontsize=11)
            ax.grid(True, alpha=0.3)

            legend_handles = []
            if has_task:
                tg_colors_map = self.colors.get('task_groups', {})
                for tg in sorted(set(task_groups)):
                    if tg != '0':
                        legend_handles.append(
                            mpatches.Patch(color=tg_colors_map.get(tg, 'gray'),
                                          label=f'Group {tg}'))
            legend_handles.append(
                plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=COLORBLIND_SAFE_PALETTE['blue'],
                          markeredgecolor='black', markersize=8, label='Normal'))
            legend_handles.append(
                plt.Line2D([0], [0], marker='X', color='w',
                          markerfacecolor=COLORBLIND_SAFE_PALETTE['red'],
                          markeredgecolor=COLORBLIND_SAFE_PALETTE['red'],
                          markersize=8, label='Anomaly'))
            ax.legend(handles=legend_handles, fontsize=8, loc='best', framealpha=0.9)

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_figure(fig, output_path)
        plt.close(fig)

    def plot_trend_analysis(self, trend_data: Dict[str, Any], output_path: Path,
                            title: str = "Longitudinal Trend Analysis") -> None:
        """Visualize longitudinal trends across multiple sessions.

        Creates a 2x2 panel showing Mann-Kendall tau values, per-feature change
        percentages, a composite progression gauge, and a text summary.
        Requires at least three sessions of data.
        """
        if not trend_data or not trend_data.get("trends"):
            return

        trends = trend_data["trends"]
        n_sessions = trend_data.get("n_sessions", 0)
        progression = trend_data.get("progression_score", 0.0)
        overall = trend_data.get("overall_direction", "stable")

        sorted_feats = sorted(
            trends.items(),
            key=lambda x: abs(x[1]["mann_kendall_tau"]),
            reverse=True,
        )
        top_feats = sorted_feats[:12]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax1 = axes[0, 0]
        names = [f[:22] for f, _ in top_feats]
        taus = [t["mann_kendall_tau"] for _, t in top_feats]
        bar_colors = []
        for _, t in top_feats:
            if not t["is_significant"]:
                bar_colors.append(COLORBLIND_SAFE_PALETTE['gray'])
            elif t["mann_kendall_tau"] > 0:
                bar_colors.append(COLORBLIND_SAFE_PALETTE['red'])
            else:
                bar_colors.append(COLORBLIND_SAFE_PALETTE['blue'])

        y_pos = np.arange(len(names))
        ax1.barh(y_pos, taus, color=bar_colors, alpha=0.85,
                 edgecolor='black', linewidth=0.5)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(names, fontsize=8)
        ax1.set_xlabel("Mann-Kendall \u03c4")
        ax1.set_title(f"Trend Strength (top {len(top_feats)} features)")
        ax1.axvline(x=0, color='black', linewidth=0.8)
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.invert_yaxis()

        sig_up = mpatches.Patch(color=COLORBLIND_SAFE_PALETTE['red'],
                                label='Significant \u2191')
        sig_dn = mpatches.Patch(color=COLORBLIND_SAFE_PALETTE['blue'],
                                label='Significant \u2193')
        ns_patch = mpatches.Patch(color=COLORBLIND_SAFE_PALETTE['gray'],
                                  label='Not significant')
        ax1.legend(handles=[sig_up, sig_dn, ns_patch], fontsize=8,
                   loc='lower right')

        ax2 = axes[0, 1]
        change_feats = sorted(
            trends.items(),
            key=lambda x: abs(x[1]["change_pct"]),
            reverse=True,
        )[:12]
        c_names = [f[:22] for f, _ in change_feats]
        c_vals = [t["change_pct"] for _, t in change_feats]
        c_colors = [
            COLORBLIND_SAFE_PALETTE['coral'] if v > 0
            else COLORBLIND_SAFE_PALETTE['cyan']
            for v in c_vals
        ]

        y_pos2 = np.arange(len(c_names))
        ax2.barh(y_pos2, c_vals, color=c_colors, alpha=0.85,
                 edgecolor='black', linewidth=0.5)
        ax2.set_yticks(y_pos2)
        ax2.set_yticklabels(c_names, fontsize=8)
        ax2.set_xlabel("Change (%)")
        ax2.set_title("Overall Change (First \u2192 Last Session)")
        ax2.axvline(x=0, color='black', linewidth=0.8)
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.invert_yaxis()

        ax3 = axes[1, 0]
        theta_arc = np.linspace(np.pi, 0, 200)
        r_outer = 1.0
        r_inner = 0.6
        ax3.plot(r_outer * np.cos(theta_arc), r_outer * np.sin(theta_arc),
                 color='#444444', linewidth=2)
        ax3.plot(r_inner * np.cos(theta_arc), r_inner * np.sin(theta_arc),
                 color='#444444', linewidth=1)
        ax3.plot([-r_outer, -r_inner], [0, 0], color='#444444', linewidth=1)
        ax3.plot([r_inner, r_outer], [0, 0], color='#444444', linewidth=1)

        n_seg = 100
        for i in range(n_seg):
            t1 = np.pi - i * np.pi / n_seg
            t2 = np.pi - (i + 1) * np.pi / n_seg
            frac = i / n_seg
            if frac < 0.33:
                seg_color = COLORBLIND_SAFE_PALETTE['green']
            elif frac < 0.66:
                seg_color = COLORBLIND_SAFE_PALETTE['yellow']
            else:
                seg_color = COLORBLIND_SAFE_PALETTE['red']
            seg_t = np.linspace(t1, t2, 10)
            xs = np.concatenate([
                r_inner * np.cos(seg_t),
                r_outer * np.cos(seg_t[::-1]),
            ])
            ys = np.concatenate([
                r_inner * np.sin(seg_t),
                r_outer * np.sin(seg_t[::-1]),
            ])
            ax3.fill(xs, ys, alpha=0.4, color=seg_color, linewidth=0)

        needle_angle = np.pi * (1 - np.clip(progression, 0, 1))
        ax3.plot(
            [0, 0.85 * np.cos(needle_angle)],
            [0, 0.85 * np.sin(needle_angle)],
            color='#333333', linewidth=3, solid_capstyle='round',
        )
        ax3.plot(0, 0, 'o', color='#333333', markersize=8)
        ax3.text(0, -0.2, f"{progression:.0%}", ha='center', va='center',
                 fontsize=18, fontweight='bold')
        ax3.text(-1.05, -0.08, "Stable", ha='center', fontsize=9,
                 color=COLORBLIND_SAFE_PALETTE['green'])
        ax3.text(1.05, -0.08, "Progressing", ha='center', fontsize=9,
                 color=COLORBLIND_SAFE_PALETTE['red'])
        ax3.set_xlim(-1.35, 1.35)
        ax3.set_ylim(-0.4, 1.15)
        ax3.set_aspect('equal')
        ax3.axis('off')
        ax3.set_title("Progression Score")

        ax4 = axes[1, 1]
        n_analyzed = trend_data.get("n_features_analyzed", 0)
        n_sig = trend_data.get("n_significant_trends", 0)
        direction_map = {
            "worsening": "WORSENING \u2191",
            "improving": "IMPROVING \u2193",
            "stable": "STABLE \u2192",
            "insufficient_data": "INSUFFICIENT DATA",
        }
        direction_label = direction_map.get(overall, overall.upper())
        direction_color = {
            "worsening": COLORBLIND_SAFE_PALETTE['red'],
            "improving": COLORBLIND_SAFE_PALETTE['green'],
            "stable": COLORBLIND_SAFE_PALETTE['blue'],
        }.get(overall, COLORBLIND_SAFE_PALETTE['gray'])

        summary_text = (
            f"Sessions analyzed:   {n_sessions}\n"
            f"Features tracked:    {n_analyzed}\n"
            f"Significant trends:  {n_sig} / {n_analyzed}\n\n"
            f"Progression score:   {progression:.1%}\n"
        )
        ax4.text(0.5, 0.65, summary_text, ha='center', va='center',
                 fontsize=12, transform=ax4.transAxes, family='monospace',
                 bbox=dict(boxstyle='round,pad=0.8', facecolor='#F5F5F5',
                           edgecolor='#CCCCCC'))
        ax4.text(0.5, 0.18, direction_label, ha='center', va='center',
                 fontsize=16, fontweight='bold', color=direction_color,
                 transform=ax4.transAxes)
        ax4.axis('off')
        ax4.set_title("Summary")

        fig.suptitle(f"{title}  ({n_sessions} sessions)",
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        self._save_figure(fig, output_path)
        plt.close(fig)

    def plot_anatomical_report(self, report: Dict[str, Any], output_path: Path,
                               title: str = "Anatomical Muscle Group Analysis") -> None:
        """Visualize anatomical muscle group deviations on a schematic face diagram.

        Left panel draws a stylized face with colored zones at each muscle group
        position, sized and coloured by deviation severity.  Right panel shows a
        horizontal bar chart of mean deviations with cranial nerve annotations.
        """
        if not report or not report.get("muscle_groups"):
            return

        groups = report["muscle_groups"]
        laterality = report.get("laterality_hint", "")

        ZONE_POS = {
            "frontalis":         {"xy": (5.0, 11.0), "r": 0.55},
            "orbicularis_oculi": {"xy": (3.4, 9.2),  "r": 0.50, "mirror": (6.6, 9.2)},
            "nasal":             {"xy": (5.0, 7.3),  "r": 0.40},
            "zygomaticus":       {"xy": (3.2, 6.4),  "r": 0.50, "mirror": (6.8, 6.4)},
            "orbicularis_oris":  {"xy": (5.0, 5.0),  "r": 0.45},
            "buccinator":        {"xy": (2.4, 5.3),  "r": 0.40, "mirror": (7.6, 5.3)},
            "depressor":         {"xy": (4.0, 3.9),  "r": 0.35, "mirror": (6.0, 3.9)},
            "jaw":               {"xy": (5.0, 2.6),  "r": 0.50},
            "tongue":            {"xy": (5.0, 4.4),  "r": 0.30},
        }

        LABEL_SIDE = {
            "frontalis": "right",
            "orbicularis_oculi": "left",
            "nasal": "right",
            "zygomaticus": "right",
            "orbicularis_oris": "left",
            "buccinator": "left",
            "depressor": "right",
            "jaw": "left",
            "tongue": "right",
        }

        all_devs = [g.get("mean_deviation", 0) for g in groups.values()]
        dev_max = max(all_devs) if all_devs else 1.0
        dev_max = max(dev_max, 0.5)

        def _severity_color(dev: float) -> str:
            """Map a deviation score to a colour from green (low) to red (high)."""
            ratio = min(dev / dev_max, 1.0)
            if ratio < 0.25:
                return COLORBLIND_SAFE_PALETTE['green']
            if ratio < 0.50:
                return COLORBLIND_SAFE_PALETTE['yellow']
            if ratio < 0.75:
                return COLORBLIND_SAFE_PALETTE['orange']
            return COLORBLIND_SAFE_PALETTE['red']

        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(1, 2, width_ratios=[1, 1.1], wspace=0.35)

        ax_face = fig.add_subplot(gs[0])
        ax_face.set_xlim(-1.5, 11.5)
        ax_face.set_ylim(0.5, 13.5)
        ax_face.set_aspect('equal')
        ax_face.axis('off')

        head = mpatches.Ellipse((5, 7), 8.5, 11.5, fill=False,
                                edgecolor='#555555', linewidth=2.5)
        ax_face.add_patch(head)

        for ex in (3.4, 6.6):
            eye = mpatches.Ellipse((ex, 9.2), 1.4, 0.55, fill=False,
                                   edgecolor='#555555', linewidth=1.5)
            ax_face.add_patch(eye)

        brow_xs_l = np.linspace(2.5, 4.3, 30)
        brow_ys_l = 10.05 + 0.22 * np.sin(np.linspace(0, np.pi, 30))
        brow_xs_r = np.linspace(5.7, 7.5, 30)
        brow_ys_r = 10.05 + 0.22 * np.sin(np.linspace(0, np.pi, 30))
        ax_face.plot(brow_xs_l, brow_ys_l, color='#555555', linewidth=1.5)
        ax_face.plot(brow_xs_r, brow_ys_r, color='#555555', linewidth=1.5)

        ax_face.plot([5.0, 4.7], [8.3, 7.0], color='#555555', linewidth=1.2)
        ax_face.plot([5.0, 5.3], [8.3, 7.0], color='#555555', linewidth=1.2)
        ax_face.plot([4.7, 5.3], [7.0, 7.0], color='#555555', linewidth=1.2)

        mouth_xs = np.linspace(3.8, 6.2, 50)
        mouth_ys = 5.0 - 0.28 * np.sin(np.linspace(0, np.pi, 50))
        ax_face.plot(mouth_xs, mouth_ys, color='#555555', linewidth=1.5)

        label_y_left = 12.0
        label_y_right = 12.0

        for group_name, zone in ZONE_POS.items():
            if group_name not in groups:
                continue
            info = groups[group_name]
            dev = info.get("mean_deviation", 0)
            color = _severity_color(dev)
            cx, cy = zone["xy"]

            circle = mpatches.Circle((cx, cy), zone["r"], alpha=0.50,
                                     facecolor=color, edgecolor=color,
                                     linewidth=1.5)
            ax_face.add_patch(circle)

            if "mirror" in zone:
                mx, my = zone["mirror"]
                mirror_c = mpatches.Circle((mx, my), zone["r"], alpha=0.50,
                                           facecolor=color, edgecolor=color,
                                           linewidth=1.5)
                ax_face.add_patch(mirror_c)

            side = LABEL_SIDE.get(group_name, "right")
            label = group_name.replace("_", " ").title()

            if side == "left":
                lx = -1.2
                ly = label_y_left
                label_y_left -= 1.4
                ha = "left"
            else:
                lx = 11.2
                ly = label_y_right
                label_y_right -= 1.4
                ha = "right"

            ax_face.annotate(
                f"{label}\n({dev:.2f})",
                xy=(cx, cy), xytext=(lx, ly),
                fontsize=7.5, ha=ha, va='center', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#888888',
                                linewidth=0.8, connectionstyle='arc3,rad=0.1'),
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          edgecolor=color, alpha=0.92, linewidth=1.2),
            )

        legend_items = [
            mpatches.Patch(color=COLORBLIND_SAFE_PALETTE['green'], alpha=0.6,
                           label='Minimal'),
            mpatches.Patch(color=COLORBLIND_SAFE_PALETTE['yellow'], alpha=0.6,
                           label='Mild'),
            mpatches.Patch(color=COLORBLIND_SAFE_PALETTE['orange'], alpha=0.6,
                           label='Moderate'),
            mpatches.Patch(color=COLORBLIND_SAFE_PALETTE['red'], alpha=0.6,
                           label='Marked'),
        ]
        ax_face.legend(handles=legend_items, loc='upper center',
                       bbox_to_anchor=(0.5, -0.03), fontsize=8,
                       ncol=4, framealpha=0.9, title="Deviation Severity",
                       title_fontsize=8)
        ax_face.set_title("Face Muscle Group Map", fontsize=13,
                          fontweight='bold', pad=15)

        ax_bar = fig.add_subplot(gs[1])
        sorted_groups = sorted(
            groups.items(),
            key=lambda x: x[1]["mean_deviation"],
            reverse=True,
        )
        g_names = [g.replace("_", " ").title() for g, _ in sorted_groups]
        g_devs = [info["mean_deviation"] for _, info in sorted_groups]
        g_nerves = [info.get("cranial_nerve", "") for _, info in sorted_groups]
        g_colors = [_severity_color(d) for d in g_devs]

        y_pos = np.arange(len(g_names))
        ax_bar.barh(y_pos, g_devs, color=g_colors, alpha=0.85,
                    edgecolor='black', linewidth=0.5, height=0.65)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(g_names, fontsize=9)
        ax_bar.set_xlabel("Mean Deviation (z-score)", fontsize=10)
        ax_bar.set_title("Deviation by Muscle Group", fontsize=13,
                         fontweight='bold')
        ax_bar.axvline(x=1.0, color=COLORBLIND_SAFE_PALETTE['orange'],
                       linestyle='--', alpha=0.7, label='Clinical threshold')
        ax_bar.grid(True, alpha=0.3, axis='x')
        ax_bar.invert_yaxis()
        ax_bar.legend(fontsize=8)

        x_max = max(g_devs) * 1.15 if g_devs else 2.0
        ax_bar.set_xlim(right=x_max * 1.05)
        ax2 = ax_bar.twinx()
        ax2.set_ylim(ax_bar.get_ylim())
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(g_nerves, fontsize=7.5, color='#666666', style='italic')
        ax2.invert_yaxis()
        ax2.tick_params(axis='y', length=0)

        if laterality:
            fig.text(0.5, 0.02, laterality, ha='center', fontsize=10,
                     style='italic', color='#444444',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF9C4',
                               edgecolor='#FFD54F', alpha=0.9))

        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.06, 1, 0.95])
        self._save_figure(fig, output_path)
        plt.close(fig)

    def plot_articulation_profile(
        self,
        articulation_scores: Dict[str, Any],
        output_path: Path,
        title: str = "Speech Scores",
        reference_scores: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Speech scores figure — old-style unified layout with new B/C clinical content.

        Layout (2 rows × 3 cols):
          Row 0: Per-Task Scores (all tasks) | Component heatmap | Δ waterfall (or comp bars)
          Row 1: Component radar             | Clinical summary   | Component Δ bars (or legend)
        """
        from matplotlib.gridspec import GridSpec
        import matplotlib.colors as mcolors
        import matplotlib.patches as mpatches

        per_task = articulation_scores.get("per_task_scores", {})
        if not per_task:
            return

        ref_per_task = reference_scores.get("per_task_scores", {}) if reference_scores else {}
        has_ref      = bool(ref_per_task)

        C_GOOD    = "#2166AC"
        C_ACCEPT  = "#92C5DE"
        C_POOR    = "#FC8D59"
        C_SEVERE  = "#D7301F"
        C_IMPROVE = "#08519C"
        C_STABLE  = "#969696"
        C_B_BLUE  = "#1A237E"
        C_C_PURP  = "#4A148C"

        def _delta_color(d):
            """Return a colour for a delta score: positive=green, near-zero=grey, negative=yellow/red."""
            if d is None:  return C_STABLE
            if d >  0.05:  return C_IMPROVE
            if d >= -0.05: return C_STABLE
            if d >= -0.10: return C_POOR
            if d >= -0.20: return "#E34A33"
            return C_SEVERE

        def _score_color(s):
            """Return a colour for an articulation score: >= 0.80 green, >= 0.60 amber, else red."""
            if s >= 0.80: return C_GOOD
            if s >= 0.60: return C_ACCEPT
            if s >= 0.40: return C_POOR
            return C_SEVERE

        def _status_label(d):
            """Return a (text, colour) tuple describing the direction of a delta score."""
            if d is None:  return ("—",           C_STABLE)
            if d >  0.05:  return ("Improved",     C_IMPROVE)
            if d >= -0.05: return ("Stable",        C_GOOD)
            if d >= -0.10: return ("Mild concern",  C_POOR)
            if d >= -0.20: return ("Moderate ↓",    "#E34A33")
            return             ("Significant ↓",  C_SEVERE)

        def _task_label(tk):
            """Return a short human-readable task label from a task key string."""
            raw   = per_task[tk].get("task_name", tk)
            short = raw.split(": ", 1)[-1] if ": " in raw else raw
            return short.replace("_", " ").title() if short else tk

        def _clean_ax(ax):
            """Remove top/right spines and lighten the remaining axes for publication style."""
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color("#cccccc")
                ax.spines[sp].set_linewidth(0.8)
            ax.tick_params(colors="#333333", labelsize=9)

        components = ["timing", "smoothness", "amplitude"]
        b_keys = sorted(k for k in per_task if k.startswith("B_"))
        c_keys = sorted(k for k in per_task if k.startswith("C_"))
        all_keys   = b_keys + c_keys
        b_labels   = [_task_label(k) for k in b_keys]
        c_labels   = [_task_label(k) for k in c_keys]
        all_labels = b_labels + c_labels
        n_b, n_c   = len(b_keys), len(c_keys)
        n_all      = len(all_keys)

        all_scores  = [per_task[k]["score"] for k in all_keys]
        b_scores    = all_scores[:n_b]
        c_scores    = all_scores[n_b:]

        ref_all_sc  = ([ref_per_task.get(k, {}).get("score") for k in all_keys]
                       if has_ref else [None] * n_all)
        all_deltas  = [(all_scores[i] - ref_all_sc[i]
                        if ref_all_sc[i] is not None else None)
                       for i in range(n_all)]
        b_deltas    = all_deltas[:n_b]
        c_deltas    = all_deltas[n_b:]

        fig_w = max(15, n_all * 0.65 + 8)
        fig_h = max(10, n_all * 0.55 + 5)
        fig   = plt.figure(figsize=(fig_w, fig_h))
        fig.patch.set_facecolor("white")
        gs    = GridSpec(2, 3, figure=fig,
                         width_ratios=[1.4, 1.2, 0.9],
                         height_ratios=[1.4, 1.0],
                         hspace=0.48, wspace=0.38,
                         left=0.08, right=0.97, top=0.91, bottom=0.06)

        ax_scores = fig.add_subplot(gs[0, 0])
        ax_heat   = fig.add_subplot(gs[0, 1])
        ax_right0 = fig.add_subplot(gs[0, 2])
        ax_grp    = fig.add_subplot(gs[1, 0])
        ax_sum    = fig.add_subplot(gs[1, 1])
        ax_right1 = fig.add_subplot(gs[1, 2])

        yp       = np.arange(n_all)
        bar_cols = [(_delta_color(d) if has_ref else _score_color(s))
                    for d, s in zip(all_deltas, all_scores)]
        bw = 0.36 if has_ref else 0.55
        if has_ref:
            rv = [r if r is not None else 0 for r in ref_all_sc]
            ax_scores.barh(yp - bw / 2, all_scores, bw,
                           color=bar_cols, alpha=0.88, edgecolor="white", linewidth=0.5)
            ax_scores.barh(yp + bw / 2, rv, bw,
                           color=[_score_color(r) if r is not None else "#BDBDBD"
                                  for r in ref_all_sc],
                           alpha=0.28, edgecolor="gray", linewidth=0.4, hatch="//")
            for i, (sc, rf, d) in enumerate(zip(all_scores, ref_all_sc, all_deltas)):
                ann = f"{sc:.2f}" + (f" ({d:+.2f})" if d is not None else "")
                ax_scores.text(sc + 0.012, i - bw / 2, ann,
                               va="center", fontsize=8.5, fontweight="bold")
                if rf is not None:
                    ax_scores.text(rf + 0.012, i + bw / 2, f"{rf:.2f}",
                                   va="center", fontsize=7.5, color="#888", fontstyle="italic")
                if d is not None and d <= -0.20:
                    ax_scores.text(-0.03, i - bw / 2, "▼", va="center", fontsize=9,
                                   color=C_SEVERE, ha="right",
                                   transform=ax_scores.get_yaxis_transform())
                elif d is not None and d <= -0.10:
                    ax_scores.text(-0.03, i - bw / 2, "▽", va="center", fontsize=9,
                                   color=C_POOR, ha="right",
                                   transform=ax_scores.get_yaxis_transform())
        else:
            ax_scores.barh(yp, all_scores, bw, color=bar_cols,
                           alpha=0.88, edgecolor="white", linewidth=0.5)
            for i, (sc, yy) in enumerate(zip(all_scores, yp)):
                ax_scores.text(sc + 0.012, yy, f"{sc:.2f}",
                               va="center", fontsize=9, fontweight="bold")

        ax_scores.set_yticks(yp)
        ax_scores.set_yticklabels(all_labels, fontsize=9)
        for idx, lbl in enumerate(ax_scores.get_yticklabels()):
            lbl.set_color(C_B_BLUE if idx < n_b else C_C_PURP)
            lbl.set_fontweight("bold")
        if n_b:
            ax_scores.axhspan(-0.5, n_b - 0.5, facecolor=C_B_BLUE, alpha=0.035, zorder=0)
        if n_c:
            ax_scores.axhspan(n_b - 0.5, n_all - 0.5, facecolor=C_C_PURP, alpha=0.035, zorder=0)
        if n_b and n_c:
            ax_scores.axhline(y=n_b - 0.5, color="#BDBDBD", linewidth=1.2, linestyle="--")
        for thresh, col in [(0.80, C_GOOD), (0.60, C_ACCEPT), (0.40, C_POOR)]:
            ax_scores.axvline(x=thresh, ymax=0.97, color=col, linestyle="--", alpha=0.45, linewidth=1.2)
        ax_scores.set_xlim(0, 1.14)
        ax_scores.set_xlabel("Articulation Score", fontsize=9)
        subtitle_bar = ("color = deviation from baseline" if has_ref
                        else "color = score tier")
        ax_scores.set_title(f"Per-Task Scores\n({subtitle_bar})",
                            fontsize=10, fontweight="bold", color="#222222", pad=4)
        ax_scores.invert_yaxis()
        _clean_ax(ax_scores)

        comp_mat = np.array([[per_task[tk].get(c, 0) for c in components]
                             for tk in all_keys])
        if has_ref:
            ref_comp  = np.array([[ref_per_task.get(tk, {}).get(c, 0) for c in components]
                                  for tk in all_keys])
            heat_data = comp_mat - ref_comp
            cmap_heat = plt.cm.RdBu
            vmin, vmax = -0.50, 0.50
            cbar_lbl   = "Δ (test − baseline)"
        else:
            heat_data = comp_mat
            cmap_heat = mcolors.LinearSegmentedColormap.from_list(
                "comp", [(0, C_SEVERE), (0.4, C_POOR), (0.6, C_ACCEPT), (1, C_GOOD)])
            vmin, vmax = 0, 1
            cbar_lbl   = "Score"

        im = ax_heat.imshow(heat_data, aspect="auto", cmap=cmap_heat,
                            vmin=vmin, vmax=vmax, interpolation="nearest")
        for ri in range(n_all):
            for ci in range(len(components)):
                val   = heat_data[ri, ci]
                abs_v = comp_mat[ri, ci]
                tc    = "white" if abs(val) > 0.25 else "#222222"
                if has_ref:
                    ax_heat.text(ci, ri, f"{val:+.2f}\n({abs_v:.2f})",
                                 ha="center", va="center", fontsize=7.5,
                                 color=tc, fontweight="bold", linespacing=1.3)
                else:
                    ax_heat.text(ci, ri, f"{abs_v:.2f}",
                                 ha="center", va="center", fontsize=9,
                                 color=tc, fontweight="bold")
        if n_b and n_c:
            ax_heat.axhline(y=n_b - 0.5, color="white", linewidth=2.0)
            ax_heat.axhline(y=n_b - 0.5, color="#555", linewidth=0.9,
                            linestyle="--", alpha=0.6)
        plt.colorbar(im, ax=ax_heat, label=cbar_lbl, shrink=0.70, pad=0.02)
        ax_heat.set_xticks(range(len(components)))
        ax_heat.set_xticklabels([c.title() for c in components],
                                fontsize=10, fontweight="bold")
        ax_heat.set_yticks(range(n_all))
        ax_heat.set_yticklabels(all_labels, fontsize=9)
        for idx, lbl in enumerate(ax_heat.get_yticklabels()):
            lbl.set_color(C_B_BLUE if idx < n_b else C_C_PURP)
            lbl.set_fontweight("bold")
        ax_heat.tick_params(left=False, bottom=False)
        ax_heat.set_title("Component Δ from Baseline\n(Timing  ·  Smoothness  ·  Amplitude)",
                          fontsize=10, fontweight="bold", color="#222222", pad=4)

        if has_ref:
            fall_cols = [_delta_color(d) for d in all_deltas]
            yf = np.arange(n_all)
            ax_right0.barh(yf, [d if d is not None else 0 for d in all_deltas],
                           color=fall_cols, alpha=0.85, edgecolor="white",
                           linewidth=0.4, height=0.60)
            ax_right0.axvline(x=0, color="#333", linewidth=1.3)
            ax_right0.axvline(x=-0.10, color=C_POOR,   linestyle=":", alpha=0.5, linewidth=1.1)
            ax_right0.axvline(x=-0.20, color=C_SEVERE,  linestyle=":", alpha=0.5, linewidth=1.1)
            if all_deltas:
                ymax_f = n_all - 0.7
                ax_right0.text(-0.10, ymax_f, "−10%", fontsize=7.5, color=C_POOR,   ha="center")
                ax_right0.text(-0.20, ymax_f, "−20%", fontsize=7.5, color=C_SEVERE, ha="center")
            if n_b and n_c:
                ax_right0.axhline(y=n_b - 0.5, color="#BDBDBD", linewidth=1.0, linestyle="--")
            for i, (bar, d) in enumerate(zip(ax_right0.patches, all_deltas)):
                if d is not None and abs(d) > 0.005:
                    ax_right0.text(d + (0.005 if d >= 0 else -0.005),
                                   bar.get_y() + bar.get_height() / 2,
                                   f"{d:+.2f}", va="center", fontsize=8.5,
                                   ha="left" if d >= 0 else "right", fontweight="bold")
            ax_right0.set_yticks(range(n_all))
            ax_right0.set_yticklabels(all_labels, fontsize=9)
            for idx, lbl in enumerate(ax_right0.get_yticklabels()):
                lbl.set_color(C_B_BLUE if idx < n_b else C_C_PURP)
                lbl.set_fontweight("bold")
            ax_right0.set_xlabel("Δ Score (test − baseline)", fontsize=9)
            ax_right0.set_title("Change from Baseline", fontsize=10,
                                fontweight="bold", color="#222222", pad=4)
            ax_right0.invert_yaxis()
            _clean_ax(ax_right0)
        else:
            comp_means = [float(np.mean([per_task[tk].get(c, 0) for tk in all_keys]))
                          for c in components]
            comp_cols  = [_score_color(v) for v in comp_means]
            ax_right0.barh(components, comp_means, color=comp_cols,
                           alpha=0.88, edgecolor="white", linewidth=0.5, height=0.55)
            for i, (c, v) in enumerate(zip(components, comp_means)):
                ax_right0.text(v + 0.01, i, f"{v:.2f}", va="center",
                               fontsize=10, fontweight="bold")
            for thresh, col in [(0.80, C_GOOD), (0.60, C_ACCEPT)]:
                ax_right0.axvline(x=thresh, ymax=0.95, color=col, linestyle="--", alpha=0.4, linewidth=1.2)
            ax_right0.set_xlim(0, 1.12)
            ax_right0.set_xlabel("Mean Score", fontsize=9)
            ax_right0.set_title("Component Scores\n(mean across all tasks)",
                                fontsize=10, fontweight="bold", color="#222222", pad=4)
            _clean_ax(ax_right0)

        b_grp_sc  = articulation_scores.get("group_b_articulation_score") or (
                        float(np.mean(b_scores)) if b_scores else None)
        c_grp_sc  = (articulation_scores.get("group_c_articulation_score")
                     or articulation_scores.get("word_production_quality"))
        ref_b_grp = (reference_scores.get("group_b_articulation_score") if has_ref else None)
        ref_c_grp = ((reference_scores.get("group_c_articulation_score")
                      or reference_scores.get("word_production_quality"))
                     if has_ref else None)

        grp_rows = []
        if b_grp_sc is not None:
            db = (b_grp_sc - ref_b_grp) if ref_b_grp is not None else None
            grp_rows.append(("Group B\n(Simple Syllables)", b_grp_sc,
                             ref_b_grp, db, C_B_BLUE, "→ Dysarthria / Apraxia"))
        if c_grp_sc is not None:
            dc = (c_grp_sc - ref_c_grp) if ref_c_grp is not None else None
            grp_rows.append(("Group C\n(WPQ)", c_grp_sc,
                             ref_c_grp, dc, C_C_PURP, "→ Phonological Disorder"))

        yg   = np.arange(len(grp_rows))
        bw_g = 0.38 if has_ref else 0.55
        bar_g_cols = [(_delta_color(r[3]) if has_ref else _score_color(r[1]))
                      for r in grp_rows]

        ax_grp.barh(yg - bw_g/2 if has_ref else yg,
                    [r[1] for r in grp_rows], bw_g,
                    color=bar_g_cols, alpha=0.88, edgecolor="white", linewidth=0.5)
        if has_ref:
            ax_grp.barh(yg + bw_g/2,
                        [r[2] if r[2] is not None else 0 for r in grp_rows], bw_g,
                        color=[_score_color(r[2]) if r[2] is not None else "#BDBDBD"
                               for r in grp_rows],
                        alpha=0.28, edgecolor="gray", linewidth=0.4, hatch="//")

        for i, r in enumerate(grp_rows):
            score, d, accent = r[1], r[3], r[4]
            y_bar = yg[i] - bw_g/2 if has_ref else yg[i]
            ann = f"{score:.3f}" + (f"  (Δ = {d:+.3f})" if d is not None else "")
            ax_grp.text(score + 0.012, y_bar, ann,
                        va="center", fontsize=9, fontweight="bold")

        for thresh, col, lbl in [(0.80, C_GOOD, "0.80"), (0.60, C_ACCEPT, "0.60")]:
            ax_grp.axvline(x=thresh, ymin=0.04, ymax=0.94,
                           color=col, linestyle="--", alpha=0.55, linewidth=1.2)
            ax_grp.text(thresh, -0.14, lbl,
                        transform=ax_grp.get_xaxis_transform(),
                        ha="center", va="top", fontsize=7.5, color=col,
                        clip_on=False)

        ytick_labels = [
            f"{r[0]}\n{r[5]}" for r in grp_rows
        ]
        ax_grp.set_yticks(yg)
        ax_grp.set_yticklabels(ytick_labels, fontsize=8.5, fontweight="bold")
        for idx, lbl in enumerate(ax_grp.get_yticklabels()):
            lbl.set_color(grp_rows[idx][4])
        ax_grp.set_xlim(0, 1.18)
        ax_grp.set_xlabel("Score", fontsize=9)
        subtitle_grp = "test vs baseline" if has_ref else "absolute score"
        ax_grp.set_title(f"Group-Level Overview\n({subtitle_grp})",
                         fontsize=10, fontweight="bold", color="#222222", pad=4)
        ax_grp.invert_yaxis()
        _clean_ax(ax_grp)

        ax_sum.axis("off")
        ax_sum.set_xlim(0, 1); ax_sum.set_ylim(0, 1)

        mean_sc   = articulation_scores.get("mean_articulation_score", 0) or 0
        b_score   = articulation_scores.get("group_b_articulation_score")
        wpq       = (articulation_scores.get("group_c_articulation_score")
                     or articulation_scores.get("word_production_quality"))
        b_dev     = articulation_scores.get("group_b_score_deviation")
        c_dev     = (articulation_scores.get("group_c_score_deviation")
                     or articulation_scores.get("delta_word_production_quality"))
        pa        = articulation_scores.get("articulation_score_pa")
        ta        = articulation_scores.get("articulation_score_ta")
        ka        = articulation_scores.get("articulation_score_ka")
        n_extreme = int(articulation_scores.get("n_c_complex_extreme_amp_drop", 0) or 0)
        cw_cons   = articulation_scores.get("cross_word_consistency", 0) or 0

        if has_ref:
            ref_mean = reference_scores.get("mean_articulation_score", 0) or 0
            d_mean   = mean_sc - ref_mean
        else:
            d_mean = None

        ax_sum.set_title("Clinical Summary", fontsize=10, fontweight="bold",
                         color="#222222", pad=4)

        badge_col  = _delta_color(d_mean) if has_ref else _score_color(mean_sc)
        badge_line = f"Mean:  {mean_sc:.3f}"
        if d_mean is not None:
            st_txt, _ = _status_label(d_mean)
            badge_line += f"   ·   {st_txt}"
        ax_sum.add_patch(mpatches.FancyBboxPatch(
            (0.03, 0.87), 0.94, 0.11,
            boxstyle="round,pad=0.02", transform=ax_sum.transAxes,
            facecolor=badge_col, edgecolor="none", alpha=0.88))
        ax_sum.text(0.50, 0.928, badge_line, transform=ax_sum.transAxes,
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    color="white")

        def _sum_card(y_top, card_h, bg, border_col, label, score_str, detail=None):
            """Draw a single rounded summary card with label and score text on the summary axes."""
            ax_sum.add_patch(mpatches.FancyBboxPatch(
                (0.03, y_top - card_h), 0.94, card_h,
                boxstyle="round,pad=0.01", transform=ax_sum.transAxes,
                facecolor=bg, edgecolor=border_col, linewidth=1.2, alpha=0.92))
            ax_sum.add_patch(mpatches.FancyBboxPatch(
                (0.03, y_top - card_h), 0.04, card_h,
                boxstyle="square,pad=0.0", transform=ax_sum.transAxes,
                facecolor=border_col, edgecolor="none", alpha=0.88))
            ax_sum.text(0.12, y_top - 0.026, label, transform=ax_sum.transAxes,
                        fontsize=8.5, color=border_col, fontweight="bold", va="top")
            ax_sum.text(0.12, y_top - 0.076, score_str, transform=ax_sum.transAxes,
                        fontsize=8.5, color="#222222", va="top", fontweight="bold")
            if detail:
                ax_sum.text(0.12, y_top - 0.126, detail, transform=ax_sum.transAxes,
                            fontsize=7.5, color="#555555", va="top")

        y_card = 0.83
        b_has_det = pa is not None
        c_has_det = n_extreme > 0 or bool(cw_cons)

        if b_score is not None:
            b_card_h = 0.175 if b_has_det else 0.125
            b_str    = f"{b_score:.3f}"
            if b_dev is not None: b_str += f"   Δ = {b_dev:+.3f}"
            b_det = (f"pa {pa:.2f}  ·  ta {ta:.2f}  ·  ka {ka:.2f}"
                     if b_has_det else None)
            _sum_card(y_card, b_card_h, "#F0F3FF", C_B_BLUE,
                      "Group B  ·  Syllables", b_str, b_det)
            y_card -= b_card_h + 0.055

        if wpq is not None:
            c_card_h = 0.175 if c_has_det else 0.125
            c_str    = f"{wpq:.3f}"
            if c_dev is not None: c_str += f"   Δ = {c_dev:+.3f}"
            c_parts  = []
            if n_extreme > 0: c_parts.append(f"amp drops >50%: {n_extreme}")
            if cw_cons:       c_parts.append(f"consistency: {cw_cons:.2f}")
            c_det = "  ·  ".join(c_parts) if c_parts else None
            _sum_card(y_card, c_card_h, "#F5F0FF", C_C_PURP,
                      "Group C  ·  Words / WPQ", c_str, c_det)

        if has_ref:
            ref_comp_means = [
                float(np.mean([ref_per_task.get(tk, {}).get(c, 0)
                               for tk in all_keys if ref_per_task.get(tk)]))
                for c in components
            ]
            comp_deltas = [comp_means_all[i] - ref_comp_means[i]
                           for i in range(len(components))]
            d_cols = [_delta_color(d) for d in comp_deltas]
            ax_right1.barh(components, comp_deltas, color=d_cols,
                           alpha=0.85, edgecolor="white", linewidth=0.4, height=0.5)
            ax_right1.axvline(x=0, color="#333", linewidth=1.2)
            for i, (c, d) in enumerate(zip(components, comp_deltas)):
                ax_right1.text(d + (0.004 if d >= 0 else -0.004), i, f"{d:+.2f}",
                               va="center", fontsize=9.5,
                               ha="left" if d >= 0 else "right", fontweight="bold")
            ax_right1.set_xlabel("Δ Component (mean)", fontsize=9)
            ax_right1.set_title("Component Δ from Baseline",
                                fontsize=10, fontweight="bold", color="#222222", pad=4)
            _clean_ax(ax_right1)
        else:
            ax_right1.axis("off")
            ax_right1.set_title("Color Key", fontsize=10, fontweight="bold", pad=4)
            swatch_h = 0.065
            step     = 0.150
            y_l2     = 0.90
            for col, lbl in [
                (C_GOOD,   "Good  (≥ 0.80)"),
                (C_ACCEPT, "Acceptable  (0.60–0.80)"),
                (C_POOR,   "Poor  (0.40–0.60)"),
                (C_SEVERE, "Severe  (< 0.40)"),
            ]:
                ax_right1.add_patch(mpatches.FancyBboxPatch(
                    (0.05, y_l2 - swatch_h), 0.17, swatch_h,
                    boxstyle="square,pad=0.0", transform=ax_right1.transAxes,
                    facecolor=col, edgecolor="none", alpha=0.88))
                ax_right1.text(0.28, y_l2 - swatch_h / 2, lbl,
                               transform=ax_right1.transAxes,
                               fontsize=8.5, va="center", color="#333")
                y_l2 -= step

            y_g = y_l2 - 0.04
            ax_right1.add_patch(mpatches.FancyBboxPatch(
                (0.03, y_g - 0.22), 0.94, 0.24,
                boxstyle="round,pad=0.02", transform=ax_right1.transAxes,
                facecolor="#F5F5F5", edgecolor="#BDBDBD", linewidth=1.0, alpha=0.9))
            for gy, gfc, gtxt in [
                (y_g - 0.02,  C_B_BLUE, "Group B  →  Dysarthria / Apraxia"),
                (y_g - 0.12, C_C_PURP, "Group C  →  Phonological"),
            ]:
                ax_right1.add_patch(mpatches.FancyBboxPatch(
                    (0.07, gy - 0.025), 0.12, 0.055,
                    boxstyle="square,pad=0.0", transform=ax_right1.transAxes,
                    facecolor=gfc, edgecolor="none", alpha=0.85))
                ax_right1.text(0.23, gy, gtxt,
                               transform=ax_right1.transAxes, fontsize=8,
                               va="center", color=gfc, fontweight="bold")

        fig.suptitle(title, fontsize=13, fontweight="bold", y=0.975, color="#111111")
        self._save_figure(fig, output_path)
        plt.close(fig)

    def plot_anomaly_indication_flow(
        self,
        anomaly_results: Dict[str, Any],
        screening_results: Dict[str, Any],
        output_path: Path,
        title: str = "Anomaly → Indication Flow",
    ) -> None:
        """Three-panel figure showing the path from detected anomalies to disorder indications.

        Panel 1 (left):  Task × anomaly-type heatmap — deviation score per cell.
        Panel 2 (centre): Anomaly-type × indication co-occurrence matrix —
                          how many tasks with each anomaly type contributed to each indication.
        Panel 3 (right): Indication evidence bar — confidence + supporting-feature count
                         for each triggered indication.
        """
        from matplotlib.gridspec import GridSpec

        _raw_per_task = anomaly_results.get("per_task_results", {})
        if isinstance(_raw_per_task, list):
            per_task_anomaly: Dict[str, Any] = {
                f"{r.get('task_group', '?')}_{r.get('task_id', i)}": r
                for i, r in enumerate(_raw_per_task)
            }
        elif isinstance(_raw_per_task, dict):
            per_task_anomaly = _raw_per_task
        else:
            per_task_anomaly = {}
        if not per_task_anomaly:
            per_task_anomaly = {
                k: v for k, v in anomaly_results.items()
                if isinstance(v, dict) and "deviation_score" in v
            }
        if not per_task_anomaly and "deviation_score" in anomaly_results:
            tg = anomaly_results.get("task_group", "?")
            tid = anomaly_results.get("task_id", 0)
            per_task_anomaly = {f"{tg}_{tid}": anomaly_results}
        indications: List[Dict[str, Any]] = screening_results.get("indications", [])

        if not per_task_anomaly:
            return

        KNOWN_ANOM_TYPES = [
            "timing_drop", "amplitude_reduction", "side_amplitude",
            "irregular_onset", "coordination_break", "phonological_substitution",
            "rate_deviation", "smoothness_drop",
        ]
        present_types: List[str] = []
        for atype in KNOWN_ANOM_TYPES:
            if any(
                atype in str(v.get("anomaly_type", "")) or
                atype == str(v.get("dominant_anomaly_type", ""))
                for v in per_task_anomaly.values()
            ):
                present_types.append(atype)
        if not present_types:
            present_types = list({
                str(v.get("dominant_anomaly_type", "unknown"))
                for v in per_task_anomaly.values()
                if v.get("dominant_anomaly_type")
            })
        if not present_types:
            present_types = ["unknown"]

        task_keys = sorted(per_task_anomaly.keys())
        task_labels = [
            per_task_anomaly[tk].get("task_name", tk).split(": ", 1)[-1]
            for tk in task_keys
        ]

        def _scalar_dev(v: Any) -> float:
            """Extract a scalar deviation score from a value that may be a list."""
            if v is None:
                return 0.0
            if isinstance(v, (list, tuple)):
                clean = [x for x in v if x is not None]
                return float(np.mean(clean)) if clean else 0.0
            try:
                return float(v)
            except (TypeError, ValueError):
                return 0.0

        dev_mat = np.zeros((len(task_keys), len(present_types)))
        for ri, tk in enumerate(task_keys):
            td = per_task_anomaly[tk]
            base_dev = _scalar_dev(td.get("deviation_score", 0))
            dom_type = str(td.get("dominant_anomaly_type", "") or "")
            all_type = str(td.get("anomaly_type", "") or "")
            for ci, atype in enumerate(present_types):
                if atype == dom_type:
                    dev_mat[ri, ci] = base_dev
                elif atype in all_type:
                    dev_mat[ri, ci] = base_dev * 0.65
                else:
                    _is_anom = td.get("is_anomaly")
                    _flagged = (
                        any(_is_anom) if isinstance(_is_anom, (list, tuple))
                        else bool(_is_anom)
                    )
                    if _flagged:
                        dev_mat[ri, ci] = base_dev * 0.20

        indication_names = [ind.get("indication", "?") for ind in indications]
        if not indication_names:
            indication_names = ["(no indications)"]
        co_mat = np.zeros((len(present_types), len(indication_names)))
        for ci_a, atype in enumerate(present_types):
            atype_task_count = sum(
                1 for tk in task_keys
                if atype in str(per_task_anomaly[tk].get("anomaly_type", "")) or
                   atype == str(per_task_anomaly[tk].get("dominant_anomaly_type", ""))
            )
            for ci_i, ind in enumerate(indications):
                sf = " ".join(str(f) for f in ind.get("supporting_features", []))
                if atype in sf or any(
                    atype in str(per_task_anomaly[tk].get("anomaly_type", ""))
                    for tk in task_keys
                ):
                    co_mat[ci_a, ci_i] = atype_task_count

        fig_w = max(16, len(task_keys) * 0.5 + len(present_types) * 0.8 + 6)
        fig_h = max(8, max(len(task_keys), len(present_types)) * 0.45 + 4)
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = GridSpec(1, 3, figure=fig,
                      width_ratios=[1.6, 1.2, 1.0],
                      wspace=0.55)

        ax_task  = fig.add_subplot(gs[0, 0])
        ax_co    = fig.add_subplot(gs[0, 1])
        ax_ind   = fig.add_subplot(gs[0, 2])

        cmap_dev = plt.cm.YlOrRd
        im_task = ax_task.imshow(dev_mat, aspect="auto", cmap=cmap_dev,
                                  vmin=0, vmax=1, interpolation="nearest")
        for ri in range(len(task_keys)):
            for ci in range(len(present_types)):
                val = dev_mat[ri, ci]
                if val > 0.05:
                    txt_c = "white" if val > 0.65 else "#333333"
                    ax_task.text(ci, ri, f"{val:.2f}", ha="center", va="center",
                                 fontsize=7, color=txt_c)
                    _ia = per_task_anomaly[task_keys[ri]].get("is_anomaly")
                    _flagged_ri = (
                        any(_ia) if isinstance(_ia, (list, tuple)) else bool(_ia)
                    )
                    if _flagged_ri:
                        ax_task.add_patch(plt.Rectangle(
                            (ci - 0.5, ri - 0.5), 1, 1,
                            linewidth=1.5, edgecolor="#C62828", facecolor="none"
                        ))
        atype_short = [a.replace("_", "\n") for a in present_types]
        ax_task.set_xticks(range(len(present_types)))
        ax_task.set_xticklabels(atype_short, fontsize=8, rotation=30, ha="right")
        ax_task.set_yticks(range(len(task_keys)))
        ax_task.set_yticklabels(task_labels, fontsize=8)
        plt.colorbar(im_task, ax=ax_task, label="Deviation score", shrink=0.6, pad=0.02)
        ax_task.set_title("Task × Anomaly Type\n(red border = flagged anomaly)",
                           fontsize=10, fontweight="bold")

        if co_mat.max() > 0:
            cmap_co = plt.cm.Blues
            im_co = ax_co.imshow(co_mat, aspect="auto", cmap=cmap_co,
                                  vmin=0, vmax=co_mat.max(), interpolation="nearest")
            for ri in range(len(present_types)):
                for ci in range(len(indication_names)):
                    val = co_mat[ri, ci]
                    if val > 0:
                        txt_c = "white" if val > co_mat.max() * 0.6 else "#333333"
                        ax_co.text(ci, ri, f"{int(val)}", ha="center", va="center",
                                   fontsize=8, color=txt_c, fontweight="bold")
            plt.colorbar(im_co, ax=ax_co, label="# contributing tasks", shrink=0.6, pad=0.02)
        else:
            ax_co.text(0.5, 0.5, "No evidence\nmapped",
                       transform=ax_co.transAxes, ha="center", va="center",
                       fontsize=11, color="#888888")
        ind_short = [n.replace("_", "\n") for n in indication_names]
        ax_co.set_xticks(range(len(indication_names)))
        ax_co.set_xticklabels(ind_short, fontsize=8, rotation=30, ha="right")
        ax_co.set_yticks(range(len(present_types)))
        ax_co.set_yticklabels(atype_short, fontsize=8)
        ax_co.set_title("Anomaly Type → Indication\n(# tasks contributing)",
                         fontsize=10, fontweight="bold")

        ax_ind.axis("on")
        if indications:
            conf_vals = [float(ind.get("confidence", 0) or 0) for ind in indications]
            n_feat    = [len(ind.get("supporting_features", [])) for ind in indications]
            y_pos     = np.arange(len(indications))
            bar_c = [
                "#C62828" if c >= 0.70 else ("#E65100" if c >= 0.45 else "#F9A825")
                for c in conf_vals
            ]
            bars = ax_ind.barh(y_pos, conf_vals, color=bar_c,
                               alpha=0.85, edgecolor="none", height=0.6)
            for i, (bar, c, nf) in enumerate(zip(bars, conf_vals, n_feat)):
                ax_ind.text(c + 0.01, i, f"{c:.2f}  ({nf} feat.)",
                            va="center", fontsize=8, fontweight="bold")
            ax_ind.axvline(x=0.45, color="#E65100", linestyle="--",
                           alpha=0.55, linewidth=1.2, label="Indication threshold (0.45)")
            ax_ind.axvline(x=0.70, color="#C62828", linestyle="--",
                           alpha=0.55, linewidth=1.2, label="High confidence (0.70)")
            ax_ind.set_yticks(y_pos)
            ax_ind.set_yticklabels(indication_names, fontsize=9)
            ax_ind.set_xlim(0, 1.15)
            ax_ind.legend(fontsize=7, loc="lower right")
            ax_ind.invert_yaxis()
        else:
            ax_ind.text(0.5, 0.5, "No indications\ntriggered",
                        transform=ax_ind.transAxes, ha="center", va="center",
                        fontsize=13, color="#4CAF50", fontweight="bold")
        ax_ind.set_xlabel("Confidence", fontsize=9)
        ax_ind.set_title("Disorder Indications\n(confidence + evidence count)",
                          fontsize=10, fontweight="bold")
        ax_ind.spines["top"].set_visible(False)
        ax_ind.spines["right"].set_visible(False)

        fig.suptitle(title, fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._save_figure(fig, output_path)
        plt.close(fig)

    def plot_word_production_profile(
        self,
        articulation_scores: Dict[str, Any],
        output_path: Path,
        title: str = "Word Production Profile",
        reference_scores: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Word production assessment with complexity gradient and baseline comparison.

        Panel 1: Per-word quality scores ordered by complexity, with baseline
        overlay when reference is available.
        Panel 2: Complexity gradient scatter with trend lines showing the
        relationship between word complexity and production quality.
        Panel 3 (reference only): Per-word delta bars showing change from
        baseline.
        """
        per_word = articulation_scores.get("per_word_scores", {})
        per_task = articulation_scores.get("per_task_scores", {})
        if not per_word:
            return

        ref_per_word = reference_scores.get("per_word_scores", {}) if reference_scores else {}
        has_ref = bool(ref_per_word)

        C_GOOD = "#2E7D32"
        C_ACCEPT = "#F9A825"
        C_POOR = "#E65100"
        C_SEVERE = "#C62828"
        C_IMPROVE = "#1565C0"
        C_DECLINE = "#C62828"
        C_STABLE = "#757575"
        C_REF_LINE = "#8E24AA"
        C_TEST_LINE = "#1565C0"

        sorted_keys = sorted(per_word.keys(), key=lambda k: per_word[k].get("complexity", 0))
        complexities = [per_word[k]["complexity"] for k in sorted_keys]
        scores = [per_word[k]["score"] for k in sorted_keys]

        labels = []
        for k in sorted_keys:
            task_info = per_task.get(k, {})
            name = task_info.get("task_name", "")
            if name and name != k and name != "(no task selected)":
                short = name.split(": ", 1)[-1] if ": " in name else name
                labels.append(f"C{per_word[k]['complexity']}: {short}")
            else:
                labels.append(f"Word (complexity {per_word[k]['complexity']})")

        n_panels = 3 if has_ref else 2
        width_ratios = [1, 1, 0.8] if has_ref else [1, 1]
        fig_height = max(5, len(sorted_keys) * 0.7 + 2.5)
        fig, axes = plt.subplots(
            1, n_panels,
            figsize=(6.5 * n_panels, fig_height),
            gridspec_kw={"width_ratios": width_ratios},
        )

        ax_scores = axes[0]
        ax_gradient = axes[1]
        ax_delta = axes[2] if has_ref else None

        def _score_color(s):
            """Return a colour for an articulation score in the speech scores figure."""
            if s >= 0.80:
                return C_GOOD
            if s >= 0.60:
                return C_ACCEPT
            if s >= 0.40:
                return C_POOR
            return C_SEVERE

        bar_colors = [_score_color(s) for s in scores]

        if has_ref:
            ref_scores_list = [ref_per_word.get(k, {}).get("score", None) for k in sorted_keys]
            bar_width = 0.35
            y_pos = np.arange(len(sorted_keys))
            ax_scores.barh(
                y_pos - bar_width / 2, scores, bar_width,
                color=bar_colors, alpha=0.90, edgecolor="black",
                linewidth=0.8, label="Test",
            )
            ref_vals = [r if r is not None else 0 for r in ref_scores_list]
            ref_colors = [_score_color(r) if r is not None else "#BDBDBD" for r in ref_scores_list]
            ax_scores.barh(
                y_pos + bar_width / 2, ref_vals, bar_width,
                color=ref_colors, alpha=0.40, edgecolor="gray",
                linewidth=0.6, label="Baseline", hatch="//",
            )
            ax_scores.set_yticks(y_pos)
            ax_scores.set_yticklabels(labels)
            for i, (tv, rv) in enumerate(zip(scores, ref_scores_list)):
                ax_scores.text(
                    tv + 0.01, i - bar_width / 2,
                    f"{tv:.2f}", va="center", fontsize=9, fontweight="bold",
                )
                if rv is not None:
                    ax_scores.text(
                        rv + 0.01, i + bar_width / 2,
                        f"{rv:.2f}", va="center", fontsize=8,
                        color="gray", fontstyle="italic",
                    )
        else:
            bars = ax_scores.barh(
                labels, scores, color=bar_colors,
                alpha=0.85, edgecolor="black", linewidth=0.8,
            )
            for bar, val in zip(bars, scores):
                ax_scores.text(
                    val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", fontsize=10, fontweight="bold",
                )

        ax_scores.axvline(x=0.80, color=C_GOOD, linestyle="--", alpha=0.6, label="Good")
        ax_scores.axvline(x=0.60, color=C_ACCEPT, linestyle="--", alpha=0.6, label="Acceptable")
        ax_scores.axvline(x=0.40, color=C_SEVERE, linestyle="--", alpha=0.6, label="Poor")
        ax_scores.set_xlim(0, 1.08)
        ax_scores.set_xlabel("Production Quality Score")
        ax_scores.set_title("Per-Word Production Quality", fontsize=12, fontweight="bold")
        ax_scores.legend(fontsize=7, loc="lower right")
        ax_scores.spines["top"].set_visible(False)
        ax_scores.spines["right"].set_visible(False)
        ax_scores.invert_yaxis()

        ax_gradient.scatter(
            complexities, scores, c=bar_colors,
            s=80, edgecolors="black", linewidth=0.8, zorder=3,
        )

        if len(complexities) >= 2:
            z = np.polyfit(complexities, scores, 1)
            x_line = np.linspace(min(complexities), max(complexities), 50)
            ax_gradient.plot(
                x_line, np.polyval(z, x_line), color=C_TEST_LINE,
                linewidth=2, label=f"Test (slope={z[0]:+.3f})", linestyle="-",
            )

        if has_ref:
            ref_complexities = []
            ref_scatter_scores = []
            for k in sorted_keys:
                rw = ref_per_word.get(k, {})
                if rw.get("score") is not None:
                    ref_complexities.append(rw.get("complexity", per_word[k]["complexity"]))
                    ref_scatter_scores.append(rw["score"])

            if ref_scatter_scores:
                ax_gradient.scatter(
                    ref_complexities, ref_scatter_scores,
                    c=[_score_color(s) for s in ref_scatter_scores],
                    s=60, edgecolors="gray", linewidth=0.6, alpha=0.5,
                    marker="D", zorder=2,
                )
                if len(ref_complexities) >= 2:
                    z_ref = np.polyfit(ref_complexities, ref_scatter_scores, 1)
                    ax_gradient.plot(
                        x_line, np.polyval(z_ref, x_line), color=C_REF_LINE,
                        linewidth=1.5, label=f"Baseline (slope={z_ref[0]:+.3f})",
                        linestyle="--", alpha=0.7,
                    )

        ax_gradient.set_xlabel("Word Complexity Level")
        ax_gradient.set_ylabel("Production Quality Score")
        ax_gradient.set_ylim(0, 1.08)
        ax_gradient.set_title("Complexity Gradient", fontsize=12, fontweight="bold")
        ax_gradient.legend(fontsize=8, loc="upper right")
        ax_gradient.axhline(y=0.60, color=C_ACCEPT, linestyle=":", alpha=0.5)
        ax_gradient.spines["top"].set_visible(False)
        ax_gradient.spines["right"].set_visible(False)

        if ax_delta is not None:
            deltas = []
            delta_labels = []
            delta_colors = []
            for i, k in enumerate(sorted_keys):
                ref_s = ref_per_word.get(k, {}).get("score", None)
                if ref_s is None:
                    continue
                d = scores[i] - ref_s
                deltas.append(d)
                delta_labels.append(labels[i])
                if d < -0.05:
                    delta_colors.append(C_DECLINE)
                elif d > 0.05:
                    delta_colors.append(C_IMPROVE)
                else:
                    delta_colors.append(C_STABLE)

            if deltas:
                y_pos_d = np.arange(len(deltas))
                bars_d = ax_delta.barh(
                    y_pos_d, deltas, color=delta_colors,
                    alpha=0.80, edgecolor="black", linewidth=0.6,
                )
                ax_delta.set_yticks(y_pos_d)
                ax_delta.set_yticklabels(delta_labels)
                ax_delta.axvline(x=0, color="black", linewidth=0.8)
                ax_delta.axvline(x=-0.10, color=C_DECLINE, linestyle=":", alpha=0.5, label="\u22120.10")
                ax_delta.axvline(x=-0.20, color=C_SEVERE, linestyle=":", alpha=0.5, label="\u22120.20")

                for bar, d in zip(bars_d, deltas):
                    ax_delta.text(
                        d + (0.005 if d >= 0 else -0.005),
                        bar.get_y() + bar.get_height() / 2,
                        f"{d:+.2f}", va="center", fontsize=9,
                        ha="left" if d >= 0 else "right", fontweight="bold",
                    )

                ax_delta.set_xlabel("\u0394 Score (test \u2013 baseline)")
                ax_delta.set_title("Change from Baseline", fontsize=12, fontweight="bold")
                ax_delta.legend(fontsize=7, loc="lower left")
                ax_delta.spines["top"].set_visible(False)
                ax_delta.spines["right"].set_visible(False)
                ax_delta.invert_yaxis()

        wpq_val = articulation_scores.get("word_production_quality", 0)
        gradient_val = articulation_scores.get("complexity_gradient", 0)
        consistency_val = articulation_scores.get("cross_word_consistency", 0)
        n_words = articulation_scores.get("n_words_scored", 0)

        if has_ref:
            ref_wpq = reference_scores.get("word_production_quality") if reference_scores else None
            if ref_wpq is not None:
                delta_wpq = wpq_val - ref_wpq
                wp_summary = (
                    f"Quality: {wpq_val:.2f} (ref: {ref_wpq:.2f}, "
                    f"\u0394={delta_wpq:+.2f})   |   "
                    f"Gradient: {gradient_val:+.2f}   |   "
                    f"Consistency: {consistency_val:.2f}   |   "
                    f"Words scored: {n_words}"
                )
            else:
                wp_summary = (
                    f"Quality: {wpq_val:.2f}   |   "
                    f"Gradient: {gradient_val:+.2f}   |   "
                    f"Consistency: {consistency_val:.2f}   |   "
                    f"Words scored: {n_words}"
                )
        else:
            wp_summary = (
                f"Quality: {wpq_val:.2f}   |   "
                f"Gradient: {gradient_val:+.2f}   |   "
                f"Consistency: {consistency_val:.2f}   |   "
                f"Words scored: {n_words}"
            )

        fig.text(
            0.5, -0.02, wp_summary, ha="center", fontsize=10, fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#E3F2FD",
                      edgecolor="#64B5F6", alpha=0.9),
        )

        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        self._save_figure(fig, output_path)
        plt.close(fig)

    def create_detection_quality_table(
        self,
        session_overview: List[Dict[str, Any]],
        output_path: Path,
        title: str = "Detection Quality per Session",
    ) -> None:
        """Render a PDF table of per-session detection rate, PSNR, and occlusion metrics (thesis Fig 1)."""
        if not session_overview:
            return

        headers = ["Session", "Type", "Frames", "Det. Rate", "Confidence", "Duration (s)"]
        rows = []
        for s in session_overview:
            rows.append([
                str(s.get("session_id", "")),
                str(s.get("session_type", "")),
                str(s.get("n_frames_analyzed", s.get("n_frames_captured", ""))),
                f"{s.get('overall_detection_rate', 0):.2%}" if s.get("overall_detection_rate") is not None else "—",
                f"{s.get('confidence_data_quality', 0):.2f}" if s.get("confidence_data_quality") is not None else "—",
                f"{s.get('total_duration_sec', 0):.1f}" if s.get("total_duration_sec") is not None else "—",
            ])

        fig, ax = plt.subplots(figsize=(10, max(2, 0.4 * len(rows) + 1.5)))
        ax.axis("off")
        table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.4)

        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_facecolor(COLORBLIND_SAFE_PALETTE["blue"])
                cell.set_text_props(color="white", fontweight="bold")
            else:
                cell.set_facecolor("#F8F9FA" if r % 2 == 0 else "white")
            cell.set_edgecolor("#DEE2E6")

        fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
        plt.tight_layout()
        self._save_figure(fig, output_path, is_table=True)
        plt.close(fig)

    def create_pilot_validation_table(
        self,
        validation_report: Dict[str, Any],
        output_path: Path,
        title: str = "Pilot Validation: Per-Indication Metrics",
    ) -> None:
        """Render a PDF table of per-indication precision, recall, and F1 (thesis Fig 2)."""
        metrics = validation_report.get("metrics", {})
        ind_metrics = metrics.get("indication_metrics", {})
        if not ind_metrics:
            return

        headers = ["Indication", "TP", "FP", "FN", "Precision", "Recall", "F1"]
        rows = []
        for name, m in sorted(ind_metrics.items()):
            rows.append([
                name.replace("_", " ").title(),
                str(m.get("true_positives", 0)),
                str(m.get("false_positives", 0)),
                str(m.get("false_negatives", 0)),
                f"{m.get('precision', 0):.2f}",
                f"{m.get('recall', 0):.2f}",
                f"{m.get('f1_score', 0):.2f}",
            ])

        overall = metrics.get("overall_metrics", {})
        if overall:
            rows.append([
                "Overall (macro avg)",
                "", "",  "",
                f"{overall.get('mean_precision', 0):.2f}",
                f"{overall.get('mean_recall', 0):.2f}",
                f"{overall.get('mean_f1', 0):.2f}",
            ])

        fig, ax = plt.subplots(figsize=(10, max(2, 0.4 * len(rows) + 1.5)))
        ax.axis("off")
        table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.4)

        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_facecolor(COLORBLIND_SAFE_PALETTE["green"])
                cell.set_text_props(color="white", fontweight="bold")
            elif r == len(rows):
                cell.set_facecolor("#E8F5E9")
                cell.set_text_props(fontweight="bold")
            else:
                cell.set_facecolor("#F8F9FA" if r % 2 == 0 else "white")
            cell.set_edgecolor("#DEE2E6")

        accuracy = validation_report.get("accuracy", 0)
        n_sess = validation_report.get("n_sessions", 0)
        fig.text(
            0.5, 0.02,
            f"Sessions evaluated: {n_sess}   |   Overall accuracy: {accuracy:.1%}",
            ha="center", fontsize=10, fontstyle="italic",
        )
        fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
        plt.tight_layout()
        self._save_figure(fig, output_path, is_table=True)
        plt.close(fig)

    def plot_baseline_stability(
        self,
        baseline_stats: Dict[str, Dict[str, float]],
        output_path: Path,
        title: str = "Baseline Blendshape Stability",
    ) -> None:
        """Two-panel stability display: left shows CV per feature (relative); right shows normalised IQR."""
        if not baseline_stats:
            return

        bs_cols = [k for k in baseline_stats if "Blendshape" in k or k.startswith("bs_")]
        if not bs_cols:
            bs_cols = sorted(baseline_stats.keys())[:30]

        active_cols = []
        for c in bs_cols:
            mean_val = abs(baseline_stats[c].get("mean", 0))
            if mean_val > 1e-4:
                active_cols.append(c)
        if not active_cols:
            active_cols = bs_cols[:20]

        short_names = [c.replace("_Blendshape", "").replace("bs_", "")[-25:] for c in active_cols]
        means = np.array([baseline_stats[c].get("mean", 0) for c in active_cols])
        stds = np.array([baseline_stats[c].get("std", 0) for c in active_cols])
        q25 = np.array([baseline_stats[c].get("q25", 0) for c in active_cols])
        q75 = np.array([baseline_stats[c].get("q75", 0) for c in active_cols])

        cv_vals = np.where(np.abs(means) > 1e-6, stds / np.abs(means), 0.0)
        iqr_norm = np.where(np.abs(means) > 1e-6, (q75 - q25) / np.abs(means), 0.0)

        sort_idx = np.argsort(cv_vals)[::-1]
        cv_vals = cv_vals[sort_idx]
        iqr_norm = iqr_norm[sort_idx]
        short_names = [short_names[i] for i in sort_idx]

        n = len(short_names)
        fig, (ax_cv, ax_iqr) = plt.subplots(1, 2, figsize=(max(12, n * 0.4), 7))

        x = np.arange(n)
        cv_colors = [
            COLORBLIND_SAFE_PALETTE['red'] if v > 0.5
            else (COLORBLIND_SAFE_PALETTE['orange'] if v > 0.25
                  else COLORBLIND_SAFE_PALETTE['cyan'])
            for v in cv_vals
        ]
        ax_cv.bar(x, cv_vals, width=0.6, color=cv_colors, alpha=0.85, edgecolor='black', linewidth=0.4)
        ax_cv.axhline(y=0.5, color=COLORBLIND_SAFE_PALETTE['red'], linestyle='--', linewidth=1.2, alpha=0.7, label='High CV (0.5)')
        ax_cv.axhline(y=0.25, color=COLORBLIND_SAFE_PALETTE['orange'], linestyle='--', linewidth=1.0, alpha=0.6, label='Moderate CV (0.25)')
        ax_cv.set_xticks(x)
        ax_cv.set_xticklabels(short_names, rotation=60, ha="right", fontsize=7)
        ax_cv.set_ylabel("Coefficient of Variation (std / |mean|)")
        ax_cv.set_title("Feature Stability — CV", fontsize=11, fontweight='bold')
        ax_cv.legend(fontsize=8, loc='upper right')
        ax_cv.spines['top'].set_visible(False)
        ax_cv.spines['right'].set_visible(False)

        iqr_colors = [
            COLORBLIND_SAFE_PALETTE['red'] if v > 1.0
            else (COLORBLIND_SAFE_PALETTE['orange'] if v > 0.5
                  else COLORBLIND_SAFE_PALETTE['blue'])
            for v in iqr_norm
        ]
        ax_iqr.bar(x, iqr_norm, width=0.6, color=iqr_colors, alpha=0.85, edgecolor='black', linewidth=0.4)
        ax_iqr.axhline(y=1.0, color=COLORBLIND_SAFE_PALETTE['red'], linestyle='--', linewidth=1.2, alpha=0.7, label='IQR = mean')
        ax_iqr.set_xticks(x)
        ax_iqr.set_xticklabels(short_names, rotation=60, ha="right", fontsize=7)
        ax_iqr.set_ylabel("Normalised IQR (IQR / |mean|)")
        ax_iqr.set_title("Feature Stability — normalised IQR", fontsize=11, fontweight='bold')
        ax_iqr.legend(fontsize=8, loc='upper right')
        ax_iqr.spines['top'].set_visible(False)
        ax_iqr.spines['right'].set_visible(False)

        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save_figure(fig, output_path)
        plt.close(fig)

    def create_deviation_score_table(
        self,
        screening_results: Dict[str, Any],
        output_path: Path,
        reference_screening: Optional[Dict[str, Any]] = None,
        title: str = "Deviation Score Summary",
    ) -> None:
        """PDF table of per-task deviation scores, optionally comparing pre vs post (thesis Fig 6)."""
        indications = screening_results.get("indications", [])
        if not indications:
            return

        ref_lookup = {}
        if reference_screening:
            for ind in reference_screening.get("indications", []):
                ref_lookup[ind.get("type", "")] = ind

        has_ref = bool(ref_lookup)
        headers = ["Indication", "Severity", "Confidence", "z-score"]
        if has_ref:
            headers.extend(["Ref Severity", "Ref Confidence", "Delta"])

        rows = []
        for ind in indications:
            itype = ind.get("type", "")
            row = [
                itype.replace("_", " ").title(),
                str(ind.get("severity", "")),
                f"{ind.get('confidence', 0):.2f}",
                f"{ind.get('z_score', 0):.2f}" if ind.get("z_score") is not None else "—",
            ]
            if has_ref:
                ref = ref_lookup.get(itype, {})
                ref_conf = ref.get("confidence")
                cur_conf = ind.get("confidence", 0)
                row.extend([
                    str(ref.get("severity", "—")),
                    f"{ref_conf:.2f}" if ref_conf is not None else "—",
                    f"{cur_conf - ref_conf:+.2f}" if ref_conf is not None else "—",
                ])
            rows.append(row)

        fig, ax = plt.subplots(figsize=(12, max(2, 0.4 * len(rows) + 1.5)))
        ax.axis("off")
        table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.4)

        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_facecolor(COLORBLIND_SAFE_PALETTE["orange"])
                cell.set_text_props(color="white", fontweight="bold")
            else:
                cell.set_facecolor("#FFF8E1" if r % 2 == 0 else "white")
            cell.set_edgecolor("#DEE2E6")

        fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
        plt.tight_layout()
        self._save_figure(fig, output_path, is_table=True)
        plt.close(fig)

    def plot_anatomical_comparison(
        self,
        current_report: Dict[str, Any],
        output_path: Path,
        reference_report: Optional[Dict[str, Any]] = None,
        title: str = "Anatomical Muscle Group Comparison",
    ) -> None:
        """Side-by-side horizontal bar chart of muscle group deviations, pre vs post (thesis Fig 7)."""
        groups = current_report.get("muscle_groups", {})
        if not groups:
            return

        names = list(groups.keys())
        cur_devs = [groups[n].get("mean_deviation", 0) for n in names]
        nerves = [groups[n].get("cranial_nerve", "") for n in names]

        has_ref = reference_report is not None
        ref_devs = []
        if has_ref:
            ref_groups = reference_report.get("muscle_groups", {})
            ref_devs = [ref_groups.get(n, {}).get("mean_deviation", 0) for n in names]

        display_names = [f"{n.replace('_', ' ').title()}\n({nerves[i]})" for i, n in enumerate(names)]
        n = len(names)
        fig, ax = plt.subplots(figsize=(10, max(4, n * 0.55)))
        y = np.arange(n)
        bar_h = 0.35 if has_ref else 0.6

        ax.barh(y + (bar_h / 2 if has_ref else 0), cur_devs, bar_h,
                color=COLORBLIND_SAFE_PALETTE["red"], alpha=0.85, label="Current")
        if has_ref:
            ax.barh(y - bar_h / 2, ref_devs, bar_h,
                    color=COLORBLIND_SAFE_PALETTE["cyan"], alpha=0.85, label="Reference")

        ax.set_yticks(y)
        ax.set_yticklabels(display_names, fontsize=8)
        ax.set_xlabel("Mean Deviation (z-score)")
        ax.axvline(1.0, color=COLORBLIND_SAFE_PALETTE["gray"], linestyle="--", linewidth=0.8)
        ax.legend(fontsize=9)
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        self._save_figure(fig, output_path)
        plt.close(fig)

    def plot_anatomical_comparison_per_task(
        self,
        per_task_reports: Dict[str, list],
        output_path: Path,
        reference_report: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Multi-page PDF of per-task anatomical muscle group comparison.

        One page per task.  Each page shows muscle groups on the Y axis and
        mean deviation (z-score) on the X axis, with a separate coloured
        marker/line for every repetition.  An optional reference session is
        drawn as a dashed grey baseline.
        """
        from .anatomy import MUSCLE_GROUP_MAP
        from matplotlib.backends.backend_pdf import PdfPages

        palette_list = [
            COLORBLIND_SAFE_PALETTE["blue"],
            COLORBLIND_SAFE_PALETTE["orange"],
            COLORBLIND_SAFE_PALETTE["green"],
            COLORBLIND_SAFE_PALETTE["pink"],
            COLORBLIND_SAFE_PALETTE["cyan"],
            COLORBLIND_SAFE_PALETTE["red"],
            COLORBLIND_SAFE_PALETTE["yellow"],
        ]
        marker_list = ["o", "s", "D", "^", "v", "P", "X"]

        ref_groups = (reference_report or {}).get("muscle_groups", {})
        output_config = self.config.get('output', {})
        pdf_path = output_path.with_suffix('.pdf')

        with PdfPages(pdf_path) as pdf:
            for task_key, rep_reports in per_task_reports.items():
                if not rep_reports:
                    continue

                all_group_names: list = []
                for rr in rep_reports:
                    for gn in rr.get("muscle_groups", {}):
                        if gn not in all_group_names:
                            all_group_names.append(gn)

                if not all_group_names:
                    continue

                display_names = []
                for gn in all_group_names:
                    info = MUSCLE_GROUP_MAP.get(gn, {})
                    nerve = info.get("cranial_nerve", "")
                    label = gn.replace("_", " ").title()
                    display_names.append(f"{label}\n({nerve})" if nerve else label)

                n_groups = len(all_group_names)
                fig, ax = plt.subplots(figsize=(10, max(4, n_groups * 0.6)))
                y = np.arange(n_groups)

                for r_idx, rr in enumerate(rep_reports):
                    rep_num = rr.get("repetition", r_idx)
                    groups = rr.get("muscle_groups", {})
                    devs = [groups.get(gn, {}).get("mean_deviation", 0.0) for gn in all_group_names]

                    color = palette_list[r_idx % len(palette_list)]
                    marker = marker_list[r_idx % len(marker_list)]

                    ax.plot(
                        devs, y,
                        marker=marker, linestyle="-", linewidth=1.2,
                        color=color, alpha=0.85, markersize=6,
                        label=f"Rep {rep_num}",
                    )

                if ref_groups:
                    ref_devs = [ref_groups.get(gn, {}).get("mean_deviation", 0.0) for gn in all_group_names]
                    ax.plot(
                        ref_devs, y,
                        marker="x", linestyle="--", linewidth=1.0,
                        color=COLORBLIND_SAFE_PALETTE["gray"], alpha=0.7,
                        markersize=7, label="Reference",
                    )

                ax.axvline(1.0, color=COLORBLIND_SAFE_PALETTE["gray"], linestyle="--", linewidth=0.8)

                ax.set_yticks(y)
                ax.set_yticklabels(display_names, fontsize=8)
                ax.set_xlabel("Mean Deviation (z-score)", fontsize=11)

                task_title = task_key.replace("_", " — Task ") if "_" in task_key else task_key
                ax.set_title(
                    f"Anatomical Muscle Group Comparison — {task_title}",
                    fontsize=13, fontweight="bold",
                )

                ax.legend(fontsize=8, loc="lower right")
                plt.tight_layout()
                pdf.savefig(fig, dpi=output_config.get('save_dpi', 300), bbox_inches='tight')
                plt.close(fig)

        logger.info("Saved per-task anatomical comparison: %s", pdf_path)

    def plot_patient_trajectory(
        self,
        session_overview: List[Dict[str, Any]],
        output_path: Path,
        title: str = "Patient Trajectory",
    ) -> None:
        """Time-series plot of asymmetry and anomaly rate across sessions (thesis Fig 10)."""
        if not session_overview or len(session_overview) < 2:
            return

        sorted_sessions = sorted(session_overview, key=lambda s: s.get("session_timestamp", ""))
        labels = [s.get("session_id", "")[-6:] for s in sorted_sessions]
        asymmetry = [s.get("overall_mean_asymmetry") for s in sorted_sessions]
        anomaly_rate = [s.get("anomaly_rate") for s in sorted_sessions]
        detection = [s.get("overall_detection_rate") for s in sorted_sessions]
        x = np.arange(len(labels))

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        palette = COLORBLIND_SAFE_PALETTE

        valid_asym = [(i, v) for i, v in enumerate(asymmetry) if v is not None]
        if valid_asym:
            xi, yi = zip(*valid_asym)
            axes[0].plot(xi, yi, "o-", color=palette["blue"], linewidth=1.5)
            axes[0].fill_between(xi, 0, yi, alpha=0.15, color=palette["blue"])
        axes[0].set_ylabel("Mean Asymmetry")
        axes[0].set_title("Asymmetry Over Time", fontsize=11)

        valid_anom = [(i, v) for i, v in enumerate(anomaly_rate) if v is not None]
        if valid_anom:
            xi, yi = zip(*valid_anom)
            axes[1].bar(xi, yi, color=palette["orange"], alpha=0.8, width=0.5)
        axes[1].set_ylabel("Anomaly Rate")
        axes[1].set_title("Anomaly Rate Over Time", fontsize=11)

        valid_det = [(i, v) for i, v in enumerate(detection) if v is not None]
        if valid_det:
            xi, yi = zip(*valid_det)
            axes[2].plot(xi, yi, "s-", color=palette["green"], linewidth=1.5)
            axes[2].axhline(0.95, color=palette["gray"], linestyle="--", linewidth=0.7)
        axes[2].set_ylabel("Detection Rate")
        axes[2].set_xlabel("Session")
        axes[2].set_title("Detection Rate Over Time", fontsize=11)

        for a in axes:
            a.set_xticks(x)
            a.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)

        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        self._save_figure(fig, output_path)
        plt.close(fig)

    def plot_intraop_timeline(
        self,
        continuous_metrics: Dict[str, Any],
        output_path: Path,
        clinical_events: Optional[List[Dict[str, Any]]] = None,
        title: str = "Intra-operative Timeline",
    ) -> None:
        """Combined timeline of activation, asymmetry, and clinical events for long recordings (thesis Fig 12)."""
        windows = continuous_metrics.get("fatigue_windows", [])
        if not windows:
            return

        times = [w["window_start"] for w in windows]
        activations = [w.get("mean_activation", 0) for w in windows]
        asym = [w.get("mean_asymmetry", 0) for w in windows if "mean_asymmetry" in w]

        palette = COLORBLIND_SAFE_PALETTE
        n_panels = 2 if asym else 1
        fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3.5 * n_panels), sharex=True)
        if n_panels == 1:
            axes = [axes]

        axes[0].plot(times, activations, color=palette["blue"], linewidth=1.2)
        axes[0].fill_between(times, activations, alpha=0.15, color=palette["blue"])
        axes[0].set_ylabel("Mean Activation")
        axes[0].set_title("Activation Over Time", fontsize=11)

        if asym and n_panels > 1:
            asym_times = times[:len(asym)]
            axes[1].plot(asym_times, asym, color=palette["orange"], linewidth=1.2)
            axes[1].fill_between(asym_times, asym, alpha=0.15, color=palette["orange"])
            axes[1].set_ylabel("Asymmetry")
            axes[1].set_xlabel("Time (s)")

        if clinical_events:
            for ev in clinical_events:
                t = ev.get("time", ev.get("timestamp", 0))
                label = ev.get("label", ev.get("type", ""))
                for a in axes:
                    a.axvline(t, color=palette["red"], linestyle="--", linewidth=1.0, alpha=0.7)
                axes[0].annotate(label, (t, axes[0].get_ylim()[1]),
                                 fontsize=7, rotation=90, va="top", ha="right",
                                 color=palette["red"])

        trend = continuous_metrics.get("activation_trend", {})
        if trend.get("slope") is not None:
            slope_text = f"Slope: {trend['slope']:.4f}/s   p={trend.get('p_value', 1):.3f}"
            axes[0].text(0.02, 0.95, slope_text, transform=axes[0].transAxes,
                         fontsize=8, verticalalignment="top",
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        self._save_figure(fig, output_path)
        plt.close(fig)

    def plot_continuous_anomaly_timeline(
        self,
        continuous_anomaly_report: Dict[str, Any],
        output_path: Path,
        session_label: str = "",
    ) -> None:
        """Plot a timeline of continuous anomaly periods and change-point density.

        Four panels:
        - Top:    composite score trace over time (rolling window scores), with anomalous
                  periods highlighted as shaded bands coloured by anomaly_type
        - Middle: anomalous period bars coloured by anomaly_type, height = composite score
        - CP:     change-point density histogram
        - Bottom: CUSUM feature-flag density histogram
        """
        periods = continuous_anomaly_report.get("anomalous_periods", [])
        change_points = continuous_anomaly_report.get("change_points", [])
        summary = continuous_anomaly_report.get("summary", {})
        baseline_quality = continuous_anomaly_report.get("baseline_quality", "ok")
        session_dur = float(summary.get("session_duration_s", 0))
        if session_dur <= 0 and periods:
            session_dur = max(p["end_s"] for p in periods)
        if session_dur <= 0:
            return

        palette = COLORBLIND_SAFE_PALETTE
        _TYPE_COLORS = {
            "transient_spike":       palette.get("red",     "#D55E00"),
            "sustained_elevation":   palette.get("orange",  "#E69F00"),
            "drift":                 palette.get("blue",    "#0072B2"),
            "kinematic_deviation":   palette.get("green",   "#009E73"),
            "pattern_shift":         palette.get("pink",    "#CC79A7"),
            "sustained_depression":  palette.get("cyan",    "#56B4E9"),
        }
        _DEFAULT_COLOR = "#999999"

        fig, axes = plt.subplots(
            4, 1, figsize=(14, 9), sharex=True,
            gridspec_kw={"height_ratios": [2.5, 1.5, 1, 1], "hspace": 0.10},
        )
        ax_score, ax_periods, ax_cp, ax_feat = axes

        ax_score.set_facecolor("#fafafa")
        ax_score.axhline(0.50, color="#cccccc", linewidth=0.8, linestyle="--",
                         label="Threshold (0.50)")
        ax_score.set_ylim(0, 1.05)

        for period in periods:
            t0 = float(period.get("start_s", 0))
            t1 = float(period.get("end_s", t0 + 1))
            score = float(period.get("composite_score", 0.5))
            atype = period.get("anomaly_type", "")
            color = _TYPE_COLORS.get(atype, _DEFAULT_COLOR)
            ax_score.fill_between([t0, t1], 0, score, alpha=0.35,
                                   color=color, linewidth=0)
            ax_score.plot([t0, t1], [score, score], color=color,
                          linewidth=1.2, alpha=0.8)

        ax_score.set_ylabel("Composite\nscore", fontsize=8)
        ax_score.spines["top"].set_visible(False)
        ax_score.spines["right"].set_visible(False)
        ax_score.grid(True, axis="y", alpha=0.2)

        n_anom = summary.get("n_anomalous_periods", len(periods))
        frac = summary.get("anomaly_fraction", 0)
        title_str = (
            f"Continuous Anomaly Timeline — {session_label}   "
            f"({n_anom} periods · {frac:.0%} flagged)"
        )
        if baseline_quality == "contaminated_full_session":
            title_str += "\n⚠ Baseline: full session used (no neutral segment found)"
        elif baseline_quality == "first_N_seconds":
            title_str += "\n⚠ Baseline: first N seconds used (no neutral segment found)"
        ax_score.set_title(title_str, fontsize=11, fontweight="bold", pad=6)

        ax_periods.set_facecolor("#f8f8f8")
        ax_periods.set_ylim(0, 1.05)
        ax_periods.axhline(0.5, color="#dddddd", linewidth=0.8, linestyle="--")

        for period in periods:
            t0 = float(period.get("start_s", 0))
            t1 = float(period.get("end_s", t0 + 1))
            score = float(period.get("composite_score", 0.5))
            atype = period.get("anomaly_type", "")
            color = _TYPE_COLORS.get(atype, _DEFAULT_COLOR)
            ax_periods.barh(score / 2, t1 - t0, left=t0, height=score,
                            color=color, alpha=min(1.0, 0.4 + score * 0.55),
                            edgecolor="none")
            ctx = period.get("task_context", {})
            tg = ctx.get("task_group", "")
            tname = ctx.get("task_name", "") or ctx.get("segment", "")
            label_parts = [str(tg) if tg and str(tg) != "0" else "",
                           str(tname)[:12] if tname else ""]
            ctx_label = "  ".join(p for p in label_parts if p)
            bar_width_s = t1 - t0
            if ctx_label and bar_width_s > session_dur * 0.04:
                ax_periods.text(
                    t0 + bar_width_s / 2, score / 2,
                    ctx_label,
                    ha="center", va="center",
                    fontsize=6, color="white",
                    clip_on=True,
                )

        handles = [
            plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.75,
                           label=k.replace("_", " ").title())
            for k, c in _TYPE_COLORS.items()
        ]
        ax_periods.legend(handles=handles, fontsize=7, loc="upper right",
                          ncol=3, framealpha=0.88)
        ax_periods.set_ylabel("Anomaly\nperiods", fontsize=8)
        ax_periods.set_yticks([0, 0.5, 1.0])
        ax_periods.set_yticklabels(["0", "0.5", "1.0"], fontsize=7)
        ax_periods.set_xlim(0, session_dur)
        ax_periods.spines["top"].set_visible(False)
        ax_periods.spines["right"].set_visible(False)

        if change_points:
            cp_arr = np.array(change_points, dtype=float)
            cp_arr = cp_arr[(cp_arr >= 0) & (cp_arr <= session_dur)]
            if len(cp_arr) > 0:
                _bins = max(50, int(session_dur / 5))
                ax_cp.hist(cp_arr, bins=_bins, color=palette.get("blue", "#0072B2"),
                           alpha=0.65, edgecolor="none")
        ax_cp.set_ylabel("Change-point\ndensity", fontsize=8)
        ax_cp.spines["top"].set_visible(False)
        ax_cp.spines["right"].set_visible(False)

        per_feat = continuous_anomaly_report.get("per_feature_cusum_flags", {})
        if per_feat:
            all_timestamps: List[float] = []
            for ts_list in per_feat.values():
                if isinstance(ts_list, list):
                    all_timestamps.extend(
                        float(t) for t in ts_list if 0 <= float(t) <= session_dur
                    )
            if all_timestamps:
                _bins2 = max(50, int(session_dur / 5))
                ax_feat.hist(all_timestamps, bins=_bins2,
                             color=palette.get("orange", "#E69F00"),
                             alpha=0.65, edgecolor="none")
        ax_feat.set_ylabel("CUSUM flags\n(features)", fontsize=8)
        ax_feat.set_xlabel("Time (s)", fontsize=9)
        ax_feat.spines["top"].set_visible(False)
        ax_feat.spines["right"].set_visible(False)

        plt.tight_layout()
        self._save_figure(fig, output_path)
        plt.close(fig)

    def plot_des_event_comparison(
        self,
        features_df: pd.DataFrame,
        des_events: List[Dict[str, Any]],
        output_path: Path,
        title: str = "DES Event Comparison",
    ) -> None:
        """Compare feature distributions in pre, during, and post windows around DES events (thesis Fig 13)."""
        if features_df is None or len(features_df) == 0 or not des_events:
            return

        has_time = "timestamp_abs" in features_df.columns
        if not has_time:
            return

        metric_cols = [c for c in features_df.columns
                       if any(k in c for k in ("activation", "asymmetry")) and features_df[c].dtype != object][:4]
        if not metric_cols:
            return

        palette = COLORBLIND_SAFE_PALETTE
        window_sec = 10.0
        fig, axes = plt.subplots(1, len(metric_cols), figsize=(4 * len(metric_cols), 5), sharey=False)
        if len(metric_cols) == 1:
            axes = [axes]

        for idx, col in enumerate(metric_cols):
            pre_vals, during_vals, post_vals = [], [], []
            for ev in des_events:
                t = ev.get("time", ev.get("timestamp", 0))
                dur = ev.get("duration", window_sec)
                mask_pre = (features_df["timestamp_abs"] >= t - window_sec) & (features_df["timestamp_abs"] < t)
                mask_dur = (features_df["timestamp_abs"] >= t) & (features_df["timestamp_abs"] < t + dur)
                mask_post = (features_df["timestamp_abs"] >= t + dur) & (features_df["timestamp_abs"] < t + dur + window_sec)
                pre_vals.extend(features_df.loc[mask_pre, col].dropna().tolist())
                during_vals.extend(features_df.loc[mask_dur, col].dropna().tolist())
                post_vals.extend(features_df.loc[mask_post, col].dropna().tolist())

            data = [pre_vals, during_vals, post_vals]
            bp = axes[idx].boxplot(data, labels=["Pre", "During", "Post"], patch_artist=True, widths=0.5)
            colors_order = [palette["cyan"], palette["red"], palette["green"]]
            for patch, color in zip(bp["boxes"], colors_order):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            axes[idx].set_title(col.replace("_", " ").title(), fontsize=10)
            axes[idx].set_ylabel("Value")

        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        self._save_figure(fig, output_path)
        plt.close(fig)

    def create_trend_summary_table(
        self,
        trend_data: Dict[str, Any],
        output_path: Path,
        title: str = "Longitudinal Trend Summary",
    ) -> None:
        """PDF table summarising per-feature trend direction, slope, and significance (thesis Fig 14)."""
        trends = trend_data.get("trends", {})
        if not trends:
            return

        headers = ["Feature", "Direction", "Tau", "p-value", "Slope", "Change %", "Sig."]
        rows = []
        for fname, t in sorted(trends.items()):
            rows.append([
                fname.replace("_", " ").title()[:30],
                t.get("direction", "—"),
                f"{t.get('mann_kendall_tau', 0):.3f}",
                f"{t.get('p_value', 1):.4f}",
                f"{t.get('sens_slope', 0):.5f}",
                f"{t.get('change_pct', 0):.1f}%",
                "Yes" if t.get("is_significant") else "No",
            ])

        fig, ax = plt.subplots(figsize=(12, max(2, 0.35 * len(rows) + 1.5)))
        ax.axis("off")
        table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.35)

        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_facecolor(COLORBLIND_SAFE_PALETTE["purple"])
                cell.set_text_props(color="white", fontweight="bold")
            else:
                cell.set_facecolor("#F3E5F5" if r % 2 == 0 else "white")
                if c == 6 and r > 0:
                    txt = cell.get_text().get_text()
                    if txt == "Yes":
                        cell.set_facecolor("#FFCDD2")
            cell.set_edgecolor("#DEE2E6")

        prog = trend_data.get("progression_score", 0)
        direction = trend_data.get("overall_direction", "—")
        fig.text(
            0.5, 0.02,
            f"Progression score: {prog:.3f}   |   Overall direction: {direction}",
            ha="center", fontsize=10, fontstyle="italic",
        )
        fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
        plt.tight_layout()
        self._save_figure(fig, output_path, is_table=True)
        plt.close(fig)

    def plot_clinical_agreement_combined(
        self,
        clinical_comparison: Dict[str, Any],
        validation_report: Optional[Dict[str, Any]],
        output_path: Path,
        title: str = "Clinical Agreement & Validation",
    ) -> None:
        """Combined figure: clinical agreement matrix plus precision/recall bars (thesis Fig 15)."""
        palette = COLORBLIND_SAFE_PALETTE
        has_validation = validation_report is not None and validation_report.get("metrics", {}).get("indication_metrics")
        ncols = 2 if has_validation else 1
        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
        if ncols == 1:
            axes = [axes]

        notes = clinical_comparison.get("notes", clinical_comparison.get("comparisons", []))
        if isinstance(notes, list) and notes:
            categories = sorted({n.get("type", n.get("indication", "")) for n in notes if n.get("type") or n.get("indication")})
            agree_counts = []
            disagree_counts = []
            for cat in categories:
                matching = [n for n in notes if n.get("type", n.get("indication", "")) == cat]
                ag = sum(1 for n in matching if n.get("agreement", n.get("match", False)))
                agree_counts.append(ag)
                disagree_counts.append(len(matching) - ag)
            y = np.arange(len(categories))
            axes[0].barh(y, agree_counts, 0.4, label="Agree", color=palette["green"], alpha=0.8)
            axes[0].barh(y + 0.4, disagree_counts, 0.4, label="Disagree", color=palette["red"], alpha=0.8)
            axes[0].set_yticks(y + 0.2)
            axes[0].set_yticklabels([c.replace("_", " ").title() for c in categories], fontsize=8)
            axes[0].legend(fontsize=8)
            axes[0].set_xlabel("Count")
            axes[0].set_title("Clinical Agreement", fontsize=11)
        else:
            total = clinical_comparison.get("total_comparisons", 0)
            agreed = clinical_comparison.get("agreements", 0)
            axes[0].bar(["Agree", "Disagree"], [agreed, total - agreed],
                        color=[palette["green"], palette["red"]], alpha=0.8)
            axes[0].set_title("Clinical Agreement", fontsize=11)
            axes[0].set_ylabel("Count")

        if has_validation:
            ind_m = validation_report["metrics"]["indication_metrics"]
            names = sorted(ind_m.keys())
            prec = [ind_m[n].get("precision", 0) for n in names]
            rec = [ind_m[n].get("recall", 0) for n in names]
            x = np.arange(len(names))
            w = 0.35
            axes[1].bar(x - w / 2, prec, w, label="Precision", color=palette["blue"], alpha=0.8)
            axes[1].bar(x + w / 2, rec, w, label="Recall", color=palette["orange"], alpha=0.8)
            axes[1].set_xticks(x)
            axes[1].set_xticklabels([n.replace("_", " ").title() for n in names], rotation=30, ha="right", fontsize=8)
            axes[1].set_ylim(0, 1.05)
            axes[1].set_ylabel("Score")
            axes[1].legend(fontsize=8)
            axes[1].set_title("Per-Indication Precision & Recall", fontsize=11)

        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        self._save_figure(fig, output_path)
        plt.close(fig)

    def plot_cross_task_matching(
        self,
        cross_task_results: Dict[str, Any],
        output_path: Path,
        task_name_map: Optional[Dict[str, str]] = None,
        title: str = "Cross-Task Profile Matching",
    ) -> None:
        """Heatmap showing test repetition similarity to each reference task profile.

        Rows represent test repetitions (grouped by expected task), columns
        represent reference task profiles.  Cells where the best match
        differs from the expected task are outlined to highlight potential
        task substitution patterns associated with buccofacial apraxia.
        """
        import matplotlib.colors as mcolors

        if not cross_task_results:
            return

        all_ref_keys: List[str] = []
        row_labels: List[str] = []
        row_expected: List[str] = []
        similarity_rows: List[List[float]] = []

        for task_key in sorted(cross_task_results.keys()):
            task_data = cross_task_results[task_key]
            per_rep = task_data.get("per_repetition", [])
            for rep in per_rep:
                sims = rep.get("all_similarities", {})
                for rk in sims:
                    if rk not in all_ref_keys:
                        all_ref_keys.append(rk)

        all_ref_keys.sort()
        if not all_ref_keys:
            return

        for task_key in sorted(cross_task_results.keys()):
            task_data = cross_task_results[task_key]
            per_rep = task_data.get("per_repetition", [])
            for rep in per_rep:
                rep_id = rep.get("repetition", 0)
                sims = rep.get("all_similarities", {})
                if task_name_map and task_key in task_name_map:
                    label = f"{task_name_map[task_key]} R{rep_id}"
                else:
                    label = f"{task_key} R{rep_id}"
                row_labels.append(label)
                row_expected.append(task_key)
                similarity_rows.append([sims.get(rk, 0.0) for rk in all_ref_keys])

        if not similarity_rows:
            return

        mat = np.array(similarity_rows)
        n_rows, n_cols = mat.shape

        fig_h = max(4, 0.5 * n_rows + 1.5)
        fig_w = max(6, 0.8 * n_cols + 3)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        cmap = mcolors.LinearSegmentedColormap.from_list(
            "sim", ["#FFFFFF", "#56B4E9", "#0072B2"]
        )
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")

        CONFUSION_THRESHOLD = 0.45

        for ri in range(n_rows):
            expected_col = all_ref_keys.index(row_expected[ri]) if row_expected[ri] in all_ref_keys else -1
            best_col = int(np.argmax(mat[ri]))
            for ci in range(n_cols):
                val = mat[ri, ci]
                color = "white" if val > 0.55 else "#333333"
                ax.text(ci, ri, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

                is_substitution = (
                    expected_col >= 0
                    and ci != expected_col
                    and (val >= CONFUSION_THRESHOLD or (ci == best_col and best_col != expected_col))
                )
                if is_substitution:
                    rect = plt.Rectangle(
                        (ci - 0.5, ri - 0.5), 1, 1,
                        linewidth=2.5, edgecolor="#D55E00", facecolor="none"
                    )
                    ax.add_patch(rect)

        col_labels = [task_name_map.get(rk, rk) if task_name_map else rk for rk in all_ref_keys]
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(row_labels, fontsize=9)
        ax.set_xlabel("Reference Task Profile", fontsize=10)
        ax.set_ylabel("Test Repetition", fontsize=10)

        plt.colorbar(im, ax=ax, label="Similarity", shrink=0.8)

        sub_patch = mpatches.Patch(
            facecolor="none", edgecolor="#D55E00", linewidth=2,
            label="Potential substitution"
        )
        ax.legend(
            handles=[sub_patch],
            loc="lower right",
            bbox_to_anchor=(1.0, 1.01),
            bbox_transform=ax.transAxes,
            fontsize=8,
            borderaxespad=0,
        )

        ax.set_title(title, fontsize=12, fontweight="bold")
        plt.tight_layout()
        self._save_figure(fig, output_path)
        plt.close(fig)

    def plot_dtw_pattern_analysis(
        self,
        dtw_results: Dict[str, Any],
        output_path: Path,
        title: str = "DTW Temporal Pattern Analysis",
    ) -> None:
        """Multi-page PDF showing DTW shape-deviation results per task.

        Page 1 — Summary bar chart: mean DTW distance per task with shape-anomaly
                 counts annotated.  Provides a quick at-a-glance view of where
                 temporal misalignment is most pronounced.
        Page 2+ — Per-task detail: per-repetition DTW distance bar, coloured by
                  is_shape_anomaly flag.  Distinct visual separation between
                  normal-range repetitions (green) and shape anomalies (red).

        These figures directly correspond to the "same pattern 1–2 s later /
        same pattern but much slower" scenarios described in the study protocol.
        """
        from matplotlib.backends.backend_pdf import PdfPages

        if not dtw_results:
            return

        task_keys_sorted = sorted(dtw_results.keys())
        C_OK = "#2E7D32"
        C_ANOM = "#C62828"
        C_WARN = "#E65100"

        pdf_path = output_path.with_suffix(".pdf")
        output_config = self.config.get("output", {})

        with PdfPages(pdf_path) as pdf:
            n_tasks = len(task_keys_sorted)
            fig, axes = plt.subplots(1, 2, figsize=(14, max(4, n_tasks * 0.6 + 2)))

            mean_dtws = [dtw_results[k].get("mean_dtw_task", 0.0) for k in task_keys_sorted]
            n_anom_per = [dtw_results[k].get("n_shape_anomalies", 0) for k in task_keys_sorted]
            short_labels = [k.replace("_", " ") for k in task_keys_sorted]

            bar_colors = [
                C_ANOM if a > 0 else C_OK
                for a in n_anom_per
            ]
            ax_bar = axes[0]
            bars = ax_bar.barh(range(n_tasks), mean_dtws, color=bar_colors,
                               alpha=0.85, edgecolor="none", height=0.6)
            for i, (v, n) in enumerate(zip(mean_dtws, n_anom_per)):
                label = f"{v:.3f}"
                if n > 0:
                    label += f"  ★×{n}"
                x_offset = max(mean_dtws) * 0.02 if mean_dtws else 0.001
                ax_bar.text(v + x_offset, i, label, va="center", fontsize=8,
                            color=C_ANOM if n > 0 else "#333333")
            ax_bar.set_yticks(range(n_tasks))
            ax_bar.set_yticklabels(short_labels, fontsize=9)
            ax_bar.set_xlabel("Mean normalised DTW distance", fontsize=10)
            ax_bar.set_title("Per-task mean DTW\n(★ = shape anomaly repetitions)", fontsize=11, fontweight="bold")
            ax_bar.spines["top"].set_visible(False)
            ax_bar.spines["right"].set_visible(False)

            from matplotlib.patches import Patch
            legend_patches = [
                Patch(facecolor=C_OK, label="No shape anomaly"),
                Patch(facecolor=C_ANOM, label="≥1 shape anomaly"),
            ]
            ax_bar.legend(handles=legend_patches, fontsize=8, loc="lower right")

            ax_rate = axes[1]
            rates = []
            for k in task_keys_sorted:
                reps = dtw_results[k].get("repetitions", [])
                n_total = len(reps)
                n_sa = sum(1 for r in reps if r.get("is_shape_anomaly", False))
                rates.append(n_sa / n_total if n_total > 0 else 0.0)
            rate_colors = [C_ANOM if r > 0.5 else (C_WARN if r > 0 else C_OK) for r in rates]
            ax_rate.barh(range(n_tasks), rates, color=rate_colors, alpha=0.85,
                         edgecolor="none", height=0.6)
            ax_rate.axvline(x=0.5, color="#999999", linestyle="--", linewidth=1)
            ax_rate.set_xlim(0, 1.08)
            ax_rate.set_xlabel("Shape-anomaly rate", fontsize=10)
            ax_rate.set_title("Shape-anomaly\nrate (per task)", fontsize=11, fontweight="bold")
            ax_rate.set_yticks(range(n_tasks))
            ax_rate.set_yticklabels(short_labels, fontsize=9)
            for i, r in enumerate(rates):
                ax_rate.text(r + 0.02, i, f"{r:.0%}", va="center", fontsize=8,
                             color=C_ANOM if r > 0.5 else "#333333")
            ax_rate.spines["top"].set_visible(False)
            ax_rate.spines["right"].set_visible(False)

            c_task_keys = [k for k in task_keys_sorted if k.startswith("C_")]
            if c_task_keys:
                c_means = [dtw_results[k].get("mean_dtw_task", 0.0) for k in c_task_keys]
                c_n_high = sum(1 for m in c_means if m > 0.08)
                c_overall_mean = sum(c_means) / len(c_means) if c_means else 0.0
                gate_passed = c_n_high >= 2 and c_overall_mean > 0.04
                gate_color = "#2E7D32" if gate_passed else "#C62828"
                gate_label = "GATE PASSED" if gate_passed else "GATE BLOCKED"
                gate_text = (
                    f"C-task DTW gate: {gate_label}\n"
                    f"{c_n_high}/{len(c_task_keys)} tasks elevated (>0.08)  |  "
                    f"mean DTW = {c_overall_mean:.3f}"
                )
                fig.text(
                    0.5, -0.02, gate_text, ha="center", va="top",
                    fontsize=9, color=gate_color, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5",
                              edgecolor=gate_color, alpha=0.9),
                )

            fig.suptitle(title, fontsize=14, fontweight="bold")
            plt.tight_layout()
            pdf.savefig(fig, dpi=output_config.get("save_dpi", 300), bbox_inches="tight")
            plt.close(fig)

            tasks_per_page = 4
            for batch_start in range(0, n_tasks, tasks_per_page):
                batch_keys = task_keys_sorted[batch_start:batch_start + tasks_per_page]
                n_b = len(batch_keys)
                fig, axes = plt.subplots(1, n_b, figsize=(5 * n_b, 5), squeeze=False)
                axes_flat = axes.flatten()

                for j, tkey in enumerate(batch_keys):
                    ax = axes_flat[j]
                    reps_data = dtw_results[tkey].get("repetitions", [])
                    if not reps_data:
                        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                                transform=ax.transAxes)
                        ax.set_title(tkey.replace("_", " "), fontsize=10)
                        continue
                    rep_ids_t = [r.get("repetition", i + 1) for i, r in enumerate(reps_data)]
                    dtw_vals = [r.get("mean_dtw", 0.0) for r in reps_data]
                    is_sa = [r.get("is_shape_anomaly", False) for r in reps_data]
                    colors_t = [C_ANOM if s else C_OK for s in is_sa]
                    ax.bar(range(len(rep_ids_t)), dtw_vals, color=colors_t,
                           alpha=0.85, edgecolor="none")
                    ax.set_xticks(range(len(rep_ids_t)))
                    ax.set_xticklabels([f"R{r}" for r in rep_ids_t], fontsize=8)
                    ax.set_ylabel("Norm. DTW distance", fontsize=9)
                    ax.set_xlabel("Repetition", fontsize=9)
                    ax.set_title(
                        tkey.replace("_", " ") + f"\n(feature: {dtw_results[tkey].get('feature', '?')})",
                        fontsize=10, fontweight="bold",
                    )
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)

                for j in range(n_b, len(axes_flat)):
                    axes_flat[j].set_visible(False)

                page_num = batch_start // tasks_per_page + 2
                fig.suptitle(
                    f"{title} — Per-Task Detail (Page {page_num})",
                    fontsize=13, fontweight="bold",
                )
                plt.tight_layout()
                pdf.savefig(fig, dpi=output_config.get("save_dpi", 300), bbox_inches="tight")
                plt.close(fig)

    def plot_disorder_evidence(
        self,
        screening_results: Dict[str, Any],
        output_path: Path,
        title: str = "Disorder Evidence Profile",
    ) -> None:
        """Bar chart summarising evidence strength for each screened disorder.

        Groups evidence by disorder type and shows both the number of
        supporting evidence sources and the maximum evidence strength,
        making mixed profile patterns clearly visible.
        """
        indications = screening_results.get("indications", [])
        if not indications:
            return

        disorder_types = [
            "facial_paresis", "buccofacial_apraxia",
            "dysarthria", "speech_apraxia", "phonological_disorder",
        ]
        disorder_labels = [
            "Facial\nParesis", "Buccofacial\nApraxia",
            "Dysarthria", "Speech\nApraxia", "Phonological\nDisorder",
        ]

        confidences = {d: 0.0 for d in disorder_types}
        severities = {d: "none" for d in disorder_types}
        evidence_counts = {d: 0 for d in disorder_types}
        descriptions = {d: "" for d in disorder_types}

        for ind in indications:
            itype = ind.get("indication_type", "")
            if itype in confidences:
                conf = ind.get("confidence", 0.0)
                if conf > confidences[itype]:
                    confidences[itype] = conf
                    severities[itype] = ind.get("severity", "mild")
                    descriptions[itype] = ind.get("description", "")
                support = ind.get("supporting_features", {})
                evidence_counts[itype] += len(support)

        sev_colors = {
            "none": "#CCCCCC",
            "mild": "#F9A825",
            "moderate": "#E65100",
            "severe": "#B71C1C",
        }

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(disorder_types))
        w = 0.55

        conf_vals = [confidences[d] for d in disorder_types]
        bar_colors = [sev_colors.get(severities[d], "#CCCCCC") for d in disorder_types]

        bars = ax.bar(x, conf_vals, w, color=bar_colors, edgecolor="black", linewidth=0.8, alpha=0.85)

        for i, (bar, val) in enumerate(zip(bars, conf_vals)):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, val + 0.02,
                    f"{val:.0%}\n({severities[disorder_types[i]]})",
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                )

        ax.axhline(y=0.7, color="#37474F", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.axhline(y=0.5, color="#FF9800", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(disorder_labels, fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Confidence", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        n_flagged = sum(1 for v in conf_vals if v > 0)
        if n_flagged > 1:
            ax.text(
                0.98, 0.95, f"Mixed profile: {n_flagged} disorders flagged",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=10, fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3E0", edgecolor="#E65100"),
            )

        for sev_label, sev_col in [("Mild", "#F9A825"), ("Moderate", "#E65100"), ("Severe", "#B71C1C")]:
            ax.bar([], [], color=sev_col, alpha=0.85, label=sev_label)
        ax.legend(fontsize=8, loc="upper left")

        plt.tight_layout()
        self._save_figure(fig, output_path)
        plt.close(fig)

    def plot_deviation_scoring_summary(
        self,
        anomaly_results: Dict[str, Any],
        output_path: Path,
        title: str = "Deviation Scoring Summary",
        threshold: float = 0.45,
    ) -> None:
        """Single-page overview of composite deviation score statistics for a session.

        Shows the distribution of composite deviation scores across all scored
        repetitions with the detection threshold marked, a per-task bar chart of
        mean deviation, method-contribution proportions (how often each scoring
        method contributed a vote), and a summary table of key detection
        statistics.  This figure is suitable for inclusion in a thesis methods
        or results section.

        Parameters
        ----------
        anomaly_results:
            The dict returned by AnomalyDetector.detect_anomalies().
        output_path:
            Destination path (extension will be forced to .pdf).
        title:
            Figure super-title.
        threshold:
            Detection threshold shown as a vertical dashed line on the score
            distribution panel.
        """
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.gridspec as gridspec

        scores = anomaly_results.get("anomaly_scores", [])
        dev_scores = anomaly_results.get("deviation_score", scores)
        is_anomaly = anomaly_results.get("is_anomaly", [])
        task_names = anomaly_results.get("task_names", [])
        method_votes = anomaly_results.get("method_votes", [])

        if not scores:
            return

        scores_arr = np.array(scores, dtype=float)
        dev_arr = np.array(dev_scores, dtype=float)
        is_anom_arr = np.array(is_anomaly, dtype=bool)
        n = len(scores_arr)
        n_flagged = int(np.sum(is_anom_arr))

        output_config = self.config.get("output", {})
        C_PASS = "#2E7D32"
        C_FAIL = "#C62828"
        C_THOLD = "#E65100"

        with PdfPages(output_path.with_suffix(".pdf")) as pdf:
            fig = plt.figure(figsize=(16, 10))
            gs = gridspec.GridSpec(
                2, 3, figure=fig,
                hspace=0.45, wspace=0.38,
                left=0.07, right=0.97, top=0.88, bottom=0.10,
            )

            ax_hist = fig.add_subplot(gs[0, :2])
            colors_hist = [C_FAIL if a else C_PASS for a in is_anom_arr]
            ax_hist.bar(range(n), dev_arr, color=colors_hist, alpha=0.80, edgecolor="none", width=0.85)
            ax_hist.axhline(threshold, color=C_THOLD, linestyle="--", linewidth=1.8,
                            label=f"Threshold ({threshold})")
            ax_hist.set_xlabel("Repetition index", fontsize=10)
            ax_hist.set_ylabel("Composite deviation score", fontsize=10)
            ax_hist.set_xlim(-0.5, n - 0.5)
            ax_hist.set_ylim(0, max(float(dev_arr.max()) * 1.15 if n > 0 else 1.0, threshold * 1.3))
            ax_hist.legend(fontsize=9)
            ax_hist.set_title("Per-repetition composite deviation score", fontsize=11, fontweight="bold")
            ax_hist.spines["top"].set_visible(False)
            ax_hist.spines["right"].set_visible(False)

            from matplotlib.patches import Patch
            ax_hist.legend(
                handles=[
                    Patch(facecolor=C_PASS, label="Normal"),
                    Patch(facecolor=C_FAIL, label="Flagged"),
                    plt.Line2D([], [], color=C_THOLD, linestyle="--", linewidth=1.8,
                               label=f"Threshold = {threshold}"),
                ],
                fontsize=9, loc="upper right",
            )

            ax_dens = fig.add_subplot(gs[0, 2])
            bins = np.linspace(0, max(1.1, float(dev_arr.max()) * 1.05), 20)
            ax_dens.hist(dev_arr[~is_anom_arr], bins=bins, color=C_PASS, alpha=0.65,
                         label="Normal", edgecolor="none")
            ax_dens.hist(dev_arr[is_anom_arr], bins=bins, color=C_FAIL, alpha=0.65,
                         label="Flagged", edgecolor="none")
            ax_dens.axvline(threshold, color=C_THOLD, linestyle="--", linewidth=1.8)
            ax_dens.set_xlabel("Deviation score", fontsize=10)
            ax_dens.set_ylabel("Count", fontsize=10)
            ax_dens.set_title("Score distribution", fontsize=11, fontweight="bold")
            ax_dens.legend(fontsize=9)
            ax_dens.spines["top"].set_visible(False)
            ax_dens.spines["right"].set_visible(False)

            ax_task = fig.add_subplot(gs[1, :2])
            if task_names and len(task_names) == n:
                unique_tasks = list(dict.fromkeys(task_names))
                task_means = []
                task_rates = []
                for tn in unique_tasks:
                    idx = [i for i, t in enumerate(task_names) if t == tn]
                    task_means.append(float(np.mean(dev_arr[idx])))
                    task_rates.append(float(np.mean(is_anom_arr[idx])))
                bar_colors = [C_FAIL if r > 0 else C_PASS for r in task_rates]
                short_tasks = [(t.split(": ", 1)[-1] if ": " in t else t)[:20] for t in unique_tasks]
                ax_task.bar(range(len(unique_tasks)), task_means, color=bar_colors,
                            alpha=0.85, edgecolor="none")
                ax_task.axhline(threshold, color=C_THOLD, linestyle="--", linewidth=1.5)
                ax_task.set_xticks(range(len(unique_tasks)))
                ax_task.set_xticklabels(short_tasks, rotation=35, ha="right", fontsize=8)
                ax_task.set_ylabel("Mean deviation score", fontsize=10)
                ax_task.set_title("Mean deviation score per task", fontsize=11, fontweight="bold")
                ax_task.spines["top"].set_visible(False)
                ax_task.spines["right"].set_visible(False)
            else:
                ax_task.axis("off")

            ax_stats = fig.add_subplot(gs[1, 2])
            ax_stats.axis("off")
            mean_sc = float(np.mean(dev_arr)) if n > 0 else 0.0
            sd_sc = float(np.std(dev_arr)) if n > 0 else 0.0
            median_sc = float(np.median(dev_arr)) if n > 0 else 0.0
            detect_rate = n_flagged / n if n > 0 else 0.0
            snip_preserved = anomaly_results.get("snippet_function_preserved", None)
            snip_row = []
            if snip_preserved is not None:
                snip_label = "Yes — performed correctly ≥1 snippet" if snip_preserved \
                    else "No — impaired across all snippets"
                snip_row = [["Function preserved (snippet)", snip_label]]
            summary_rows = [
                ["Metric", "Value"],
                ["Repetitions scored", str(n)],
                ["Flagged (deviation > threshold)", f"{n_flagged} ({detect_rate:.0%})"],
                ["Mean score", f"{mean_sc:.3f}"],
                ["Median score", f"{median_sc:.3f}"],
                ["SD of scores", f"{sd_sc:.3f}"],
                ["Detection threshold", f"{threshold:.2f}"],
            ] + snip_row
            tbl = ax_stats.table(
                cellText=summary_rows[1:],
                colLabels=summary_rows[0],
                cellLoc="left",
                loc="center",
                bbox=[0, 0, 1, 1],
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9)
            n_data_rows = len(summary_rows) - 1
            for (r, c), cell in tbl.get_celld().items():
                if r == 0:
                    cell.set_facecolor("#EEEEEE")
                    cell.set_text_props(fontweight="bold")
                elif snip_preserved is not None and r == n_data_rows:
                    cell.set_facecolor("#C8E6C9" if snip_preserved else "#FFCDD2")
                    cell.set_text_props(fontweight="bold")
                cell.set_edgecolor("#CCCCCC")
            ax_stats.set_title("Session statistics", fontsize=11, fontweight="bold", pad=8)

            fig.suptitle(title, fontsize=14, fontweight="bold")
            pdf.savefig(fig, dpi=output_config.get("save_dpi", 300), bbox_inches="tight")
            plt.close(fig)

    def plot_group_a_kinematics(
        self,
        features_df: pd.DataFrame,
        output_path: Path,
        session_label: str = "",
        fps: float = 30.0,
        reference_profiles: Optional[Dict] = None,
        task_name_map: Optional[Dict[str, str]] = None,
        is_reference_session: bool = True,
        all_task_profiles: Optional[Dict] = None,
        task_profile_ref: Optional[Dict] = None,
    ) -> None:
        """Generate a multi-page PDF of Group A (facial expression) kinematic profiles.

        Formatting matches :meth:`plot_kinematic_spatiotemporal` used for Groups B/C:
        one page per task, showing time-normalised activation traces per repetition
        (tab10 colours), and — for reference sessions — ±1 SD / ±2 SD bands and
        95 % CI.  For test sessions (is_reference_session=False) those session-derived
        bands are suppressed so only the crimson reference overlay is shown.
        If *reference_profiles* contains a key ``"A_<task_id>"`` the reference mean
        is always overlaid as a crimson dashed line with ±1 SD band.

        A second axis shows signed left–right asymmetry over normalised time.
        Individual per-task PNGs are written alongside the PDF.
        """
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.cm as _cm
        from scipy.stats import t as _t_dist

        GROUP_A_TASK_LABELS = {
            1: "Lip Purse / Whistle",
            2: "Broad Smile",
            3: "Open-Mouth Smile",
            4: "Tongue Protrusion (midline)",
            5: "Tongue Lateral (right)",
            6: "Tongue Lateral (left)",
            7: "Frown / Sad Face",
            8: "Cheek Puff",
            9: "Brow Raise",
        }

        if "task_group" not in features_df.columns:
            logger.warning("plot_group_a_kinematics: task_group column missing.")
            return

        a_mask = features_df["task_group"] == "A"
        if not a_mask.any():
            logger.info("plot_group_a_kinematics: no Group A frames found; skipping.")
            return

        kin_a_cols = [c for c in features_df.columns if c.startswith("kin_a_")]
        if not kin_a_cols:
            logger.warning("plot_group_a_kinematics: kin_a_ columns missing; run kinematic extraction first.")
            return

        a_df      = features_df[a_mask].copy()
        task_ids  = sorted(a_df["task_id"].dropna().unique())
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        N_INTERP = 500
        grid = np.linspace(0, 1, N_INTERP)

        all_figs = []

        for tid in task_ids:
            tid_int   = int(tid)
            task_key_a = f"A_{tid_int}"
            if task_name_map and task_key_a in task_name_map:
                task_label = task_name_map[task_key_a]
            else:
                task_label = GROUP_A_TASK_LABELS.get(tid_int, f"Task A-{tid_int}")
            t_df      = a_df[a_df["task_id"] == tid].copy()
            if t_df.empty:
                continue

            rep_col = t_df["repetition"] if "repetition" in t_df.columns else pd.Series(1, index=t_df.index)
            reps    = sorted(rep_col.dropna().unique())

            act_col  = "kin_a_mean_activation" if "kin_a_mean_activation" in t_df.columns else None
            asym_col = "kin_a_asymmetry"        if "kin_a_asymmetry"       in t_df.columns else None
            if act_col is None:
                continue

            traces_act, traces_asym, valid_rep_labels = [], [], []
            for rep_id in reps:
                seg = t_df[rep_col == rep_id]
                if len(seg) < 3:
                    continue
                x       = np.linspace(0, 1, len(seg))
                act_raw = seg[act_col].fillna(0.0).to_numpy()
                traces_act.append(np.interp(grid, x, act_raw))
                valid_rep_labels.append(f"Rep {int(rep_id)}")
                if asym_col:
                    asym_raw = seg[asym_col].fillna(0.0).to_numpy()
                    traces_asym.append(np.interp(grid, x, asym_raw))

            if not traces_act:
                continue

            arr   = np.stack(traces_act)
            n     = len(arr)
            mean  = arr.mean(axis=0)
            std   = arr.std(axis=0)

            if n > 1:
                from scipy.stats import t as _t_dist2
                se    = std / np.sqrt(n)
                t_c   = _t_dist2.ppf(0.975, df=n - 1)
                ci_lo = mean - t_c * se
                ci_hi = mean + t_c * se
            else:
                ci_lo = ci_hi = mean

            ref_profile = None
            if reference_profiles:
                ref_key = f"A_{tid_int}"
                if ref_key not in reference_profiles:
                    try:
                        from .task_profile import _DISORDER_TASK_CANONICAL_MAP
                        _canon = _DISORDER_TASK_CANONICAL_MAP.get(("A", tid_int))
                        if _canon is not None:
                            ref_key = f"{_canon[0]}_{_canon[1]}"
                    except Exception:
                        pass
                ref_entry = reference_profiles.get(ref_key, {})
                if isinstance(ref_entry, dict):
                    ref_profile = ref_entry.get(act_col)
                    if ref_profile is None and "mean" in ref_entry:
                        ref_profile = ref_entry

            n_rows = 3 if traces_asym else 2
            hr     = [3, 1.5, 2] if n_rows == 3 else [3, 2]
            fig, axes = plt.subplots(n_rows, 1, figsize=(9, 4.5 + 2.0 * n_rows),
                                     gridspec_kw={"height_ratios": hr, "hspace": 0.45})
            if n_rows == 2:
                ax_act, ax_bar = axes
                ax_asym = None
            else:
                ax_act, ax_asym, ax_bar = axes

            colours = _cm.get_cmap("tab10", max(n, 1))

            for i, (trace, lbl) in enumerate(zip(traces_act, valid_rep_labels)):
                ax_act.plot(grid, trace, color=colours(i), alpha=0.75,
                            linewidth=1.6, label=lbl, zorder=8)

            if is_reference_session:
                ax_act.fill_between(grid, mean - 2 * std, mean + 2 * std,
                                    alpha=0.10, color="steelblue", zorder=4, label="±2 SD")
                ax_act.fill_between(grid, mean - std, mean + std,
                                    alpha=0.22, color="steelblue", zorder=5, label="±1 SD")
                ax_act.fill_between(grid, ci_lo, ci_hi, alpha=0.08, color="#1f3a5f", zorder=6, label="95% CI")
            ax_act.plot(grid, mean, color="black", linewidth=2.0, label=f"Mean (n={n})", zorder=10)

            _tp_entry = None
            if all_task_profiles:
                _tp_entry = all_task_profiles.get(f"A_{tid_int}")
                if _tp_entry is None:
                    try:
                        from .task_profile import _DISORDER_TASK_CANONICAL_MAP
                        _canon = _DISORDER_TASK_CANONICAL_MAP.get(("A", tid_int))
                        if _canon is not None:
                            _tp_entry = all_task_profiles.get(f"{_canon[0]}_{_canon[1]}")
                    except Exception:
                        pass
            if _tp_entry is None and task_profile_ref is not None:
                _tp_entry = task_profile_ref

            if _tp_entry is not None:
                _act_pattern = _tp_entry.get("activation_pattern", {}).get(act_col)
                if _act_pattern and "mean_pattern" in _act_pattern:
                    _tp_mean = np.array(_act_pattern["mean_pattern"], dtype=float)
                    if "mad_pattern" in _act_pattern:
                        _tp_1s = self._robust_sigma(_act_pattern["mad_pattern"])
                    else:
                        _tp_1s = np.array(
                            _act_pattern.get("std_pattern", np.zeros_like(_tp_mean)),
                            dtype=float,
                        )
                    _tp_2s = _tp_1s * 2.0
                    _tp_grid = np.linspace(0, 1, len(_tp_mean))
                    _tp_mean_i = np.interp(grid, _tp_grid, _tp_mean)
                    _tp_1s_i   = np.interp(grid, _tp_grid, _tp_1s)
                    _tp_2s_i   = np.interp(grid, _tp_grid, _tp_2s)
                    PROF_OUTER = '#E8D5F5'
                    PROF_INNER = COLORBLIND_SAFE_PALETTE.get('lavender', '#9b59b6')
                    PROF_CI    = '#7c4fc7'
                    ax_act.fill_between(grid, _tp_mean_i - _tp_2s_i, _tp_mean_i + _tp_2s_i,
                                        color=PROF_OUTER, alpha=0.13, zorder=0,
                                        label="Profile ±2σ")
                    ax_act.fill_between(grid, _tp_mean_i - _tp_1s_i, _tp_mean_i + _tp_1s_i,
                                        color=PROF_INNER, alpha=0.28, zorder=1,
                                        label="Profile ±1σ (MAD)")
                    _tp_n = _act_pattern.get("n_curves", _tp_entry.get("n_repetitions_total", 0))
                    if _tp_n and _tp_n > 1:
                        from scipy.stats import t as _t_prof
                        _tp_se = _tp_1s_i / np.sqrt(_tp_n)
                        _tp_tc = _t_prof.ppf(0.975, df=_tp_n - 1)
                        ax_act.fill_between(grid,
                                            _tp_mean_i - _tp_tc * _tp_se,
                                            _tp_mean_i + _tp_tc * _tp_se,
                                            color=PROF_CI, alpha=0.14, zorder=2,
                                            label="Profile 95% CI")
                    ax_act.plot(grid, _tp_mean_i, color=PROF_INNER, lw=1.8,
                                linestyle="-", alpha=0.9, zorder=3,
                                label="Task Profile Median")

            if ref_profile and "mean" in ref_profile:
                ref_m = np.asarray(ref_profile["mean"])
                ref_s = np.asarray(ref_profile.get("std", np.zeros_like(ref_m)))
                ref_grid = np.linspace(0, 1, len(ref_m))
                ref_m_i = np.interp(grid, ref_grid, ref_m)
                ref_s_i = np.interp(grid, ref_grid, ref_s)
                ax_act.fill_between(grid, ref_m_i - 2 * ref_s_i, ref_m_i + 2 * ref_s_i,
                                    alpha=0.08, color="crimson", zorder=1)
                ax_act.fill_between(grid, ref_m_i - ref_s_i, ref_m_i + ref_s_i,
                                    alpha=0.16, color="crimson", zorder=2,
                                    label="Ref ±1 SD / ±2 SD")
                ref_n = ref_profile.get("n", 0)
                if ref_n > 1:
                    from scipy.stats import t as _t2
                    ref_se = ref_s_i / np.sqrt(ref_n)
                    ref_tc = _t2.ppf(0.975, df=ref_n - 1)
                    ax_act.fill_between(grid, ref_m_i - ref_tc * ref_se, ref_m_i + ref_tc * ref_se,
                                        alpha=0.14, color="crimson", zorder=3,
                                        label="Ref 95% CI")
                ax_act.plot(grid, ref_m_i, color="crimson", lw=1.4,
                            linestyle="--", label="Ref mean", zorder=7)

            _onset_positions = []
            for _trace in traces_act:
                _pk = float(np.max(_trace))
                if _pk > 1e-6:
                    _hits = np.where(_trace >= 0.25 * _pk)[0]
                    if len(_hits) > 0:
                        _onset_positions.append(_hits[0] / len(_trace))
            if _onset_positions:
                _mean_onset = float(np.mean(_onset_positions))
                ax_act.axvline(_mean_onset, color="#F57C00", linewidth=1.4,
                               linestyle=":", alpha=0.85,
                               label=f"Onset (25% peak, μ={_mean_onset:.2f})")

            ax_act.axhline(0, color="#AAAAAA", lw=0.6, linestyle="--")
            ax_act.set_ylabel("Activation (a.u.)")
            ax_act.set_title(f"{task_label}  —  Group A Kinematics",
                             fontsize=10, fontweight="bold", pad=6)
            n_ref_items = (2 if ref_profile and "mean" in ref_profile else 0) + (1 if ref_profile and ref_profile.get("n", 0) > 1 else 0)
            session_band_items = 3 if is_reference_session else 0
            n_leg_act = n + 1 + session_band_items + n_ref_items
            if n_leg_act > 8:
                ax_act.legend(fontsize=6.5, loc='upper left',
                              bbox_to_anchor=(1.02, 1.0), borderaxespad=0,
                              ncol=1, framealpha=0.85)
                fig.subplots_adjust(right=0.80)
            else:
                ax_act.legend(fontsize=7, loc='upper right',
                              ncol=min(n_leg_act, 3), framealpha=0.85)
            ax_act.set_xlim(0, 1)
            ax_act.grid(True, alpha=0.25)

            if ax_asym is not None and traces_asym:
                arr_asym  = np.stack(traces_asym)
                mean_asym = arr_asym.mean(axis=0)
                std_asym  = arr_asym.std(axis=0)
                for i, trace in enumerate(traces_asym):
                    ax_asym.plot(grid, trace, color=colours(i), lw=1.0, alpha=0.50)
                ax_asym.fill_between(grid, mean_asym - std_asym, mean_asym + std_asym,
                                     alpha=0.15, color="steelblue")
                ax_asym.plot(grid, mean_asym, color="black", lw=1.8)
                ax_asym.axhline(0, color="#D55E00", lw=1.0, linestyle="--", label="Symmetric")
                ax_asym.set_ylim(-1.05, 1.05)
                ax_asym.set_ylabel("L−R Asymmetry")
                ax_asym.legend(fontsize=7, loc="upper right")
                ax_asym.set_xlim(0, 1)
                ax_asym.grid(True, alpha=0.25)

            peak_vals = [float(np.max(trace)) for trace in traces_act]
            n_reps = len(peak_vals)
            x_pos  = np.arange(n_reps)
            bar_cs = [colours(i) for i in range(n_reps)]
            ax_bar.bar(x_pos, peak_vals, color=bar_cs, edgecolor="black",
                       linewidth=0.7, alpha=0.85)
            mean_peak = float(np.mean(peak_vals))
            std_peak  = float(np.std(peak_vals)) if n_reps > 1 else 0.0
            ax_bar.axhline(mean_peak, color="black", lw=1.6,
                           label=f"Mean ± SD  ({mean_peak:.3f} ± {std_peak:.3f})")
            ax_bar.axhspan(mean_peak - std_peak, mean_peak + std_peak,
                           alpha=0.12, color="steelblue")
            ax_bar.set_xticks(x_pos)
            ax_bar.set_xticklabels(valid_rep_labels, fontsize=8)
            ax_bar.set_ylabel("Peak Amplitude (a.u.)")
            ax_bar.set_xlabel("Repetition")
            ax_bar.legend(fontsize=7)
            ax_bar.grid(True, alpha=0.25, axis='y')

            for ax in axes:
                ax.tick_params(labelsize=8)

            png_path = output_path.parent / (
                f"group_a_kin_task{tid_int:02d}_{task_label.replace(' ','_').replace('/','')}.png"
            )
            try:
                fig.savefig(png_path, dpi=150, bbox_inches="tight")
            except Exception:
                pass

            all_figs.append((task_label, fig))

        if not all_figs:
            logger.warning("plot_group_a_kinematics: no figures generated.")
            return

        pdf_path = output_path.with_suffix(".pdf")
        try:
            with PdfPages(pdf_path) as pdf:
                fig_cover, ax_c = plt.subplots(figsize=(7.5, 5))
                ax_c.axis("off")
                ax_c.text(0.5, 0.62, "Group A — Facial Expression Kinematics",
                          ha="center", va="center", fontsize=15, fontweight="bold",
                          transform=ax_c.transAxes)
                ax_c.text(0.5, 0.46,
                          f"{len(all_figs)} task(s) · blendshape-derived kinematic profiles",
                          ha="center", va="center", fontsize=9, color="#777777",
                          transform=ax_c.transAxes)
                try:
                    pdf.savefig(fig_cover, bbox_inches="tight")
                finally:
                    plt.close(fig_cover)

                for _task_label, fig in all_figs:
                    try:
                        pdf.savefig(fig, bbox_inches="tight")
                    finally:
                        plt.close(fig)

            logger.info("Saved Group A kinematic PDF: %s", pdf_path)
        except Exception as exc:
            logger.error("Failed to write Group A kinematic PDF: %s", exc)
            for _, fig in all_figs:
                try:
                    plt.close(fig)
                except Exception:
                    pass

    def plot_group_a_landmark_kinematics(
        self,
        kin_df: pd.DataFrame,
        features_df: pd.DataFrame,
        output_dir: Path,
        session_label: str = "",
        reference_profiles: Optional[Dict] = None,
        fps: float = 30.0,
        task_name_map: Optional[Dict[str, str]] = None,
        is_reference_session: bool = True,
    ) -> List[Path]:
        """Generate one PDF per Group A task using 3D-landmark kinematic columns.

        Each PDF has one page per measurement column (same style as B/C spatiotemporal
        plots): all reps overlaid in colour, ±1/2 SD bands and 95 % CI only for
        reference sessions (is_reference_session=True), thick mean, and crimson
        reference profile overlay when available.

        Returns list of PDF paths created.
        """
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pdf_paths: List[Path] = []

        a_mask = (
            features_df["task_group"] == "A"
            if "task_group" in features_df.columns
            else pd.Series(False, index=features_df.index)
        )
        if not a_mask.any() or "task_id" not in features_df.columns:
            return pdf_paths

        a_feat = features_df[a_mask]
        a_kin  = kin_df.reindex(a_feat.index)

        GROUP_A_TASK_LABELS = {
            1: "Lip Purse / Whistle",
            2: "Broad Smile",
            3: "Open-Mouth Smile",
            4: "Tongue Protrusion (midline)",
            5: "Tongue Lateral (right)",
            6: "Tongue Lateral (left)",
            7: "Frown / Sad Face",
            8: "Cheek Puff",
            9: "Brow Raise",
        }

        for tid in sorted(a_feat["task_id"].dropna().unique()):
            tid_int  = int(tid)
            task_key = f"A_{tid_int}"
            if task_name_map and task_key in task_name_map:
                task_name = task_name_map[task_key]
            else:
                task_name = GROUP_A_TASK_LABELS.get(tid_int, f"Task A-{tid_int}")

            task_mask = a_feat["task_id"] == tid
            task_kin  = a_kin[task_mask]
            task_feat = a_feat[task_mask]

            if task_kin.empty:
                continue

            available_cols = [
                c for c in self._GROUP_A_LANDMARK_COLS
                if c in task_kin.columns and task_kin[c].notna().any()
            ]
            if not available_cols:
                continue

            safe_name = task_name.replace("/", "-").replace(" ", "_")[:30]
            pdf_path  = output_dir / f"kinematic_A_{tid_int:02d}_{safe_name}_{session_label}.pdf"

            all_figs = []
            for meas_col in available_cols:
                ref_profile = None
                _ref_key = task_key
                if reference_profiles and _ref_key not in reference_profiles:
                    try:
                        from .task_profile import _DISORDER_TASK_CANONICAL_MAP
                        _canon = _DISORDER_TASK_CANONICAL_MAP.get(("A", tid_int))
                        if _canon is not None:
                            _ref_key = f"{_canon[0]}_{_canon[1]}"
                    except Exception:
                        pass
                if reference_profiles and _ref_key in reference_profiles:
                    ref_profile = reference_profiles[_ref_key].get(meas_col)

                fig = self.plot_kinematic_spatiotemporal(
                    kin_df=task_kin,
                    features_df=task_feat,
                    measurement_col=meas_col,
                    task_label=f"A / {task_name}",
                    reference_profile=ref_profile,
                    output_path=None,
                    fps=fps,
                    is_reference_session=is_reference_session,
                )
                if fig is not None:
                    all_figs.append(fig)

            if all_figs:
                try:
                    with PdfPages(str(pdf_path)) as pdf_file:
                        for fig in all_figs:
                            pdf_file.savefig(fig, bbox_inches="tight")
                            plt.close(fig)
                    pdf_paths.append(pdf_path)
                    logger.info("Saved Group A kinematic PDF (task %d): %s", tid_int, pdf_path)
                except Exception as _pdf_exc:
                    logger.error("Failed to write Group A task %d PDF: %s", tid_int, _pdf_exc)
                    for fig in all_figs:
                        try:
                            plt.close(fig)
                        except Exception:
                            pass

        return pdf_paths

    def plot_kinematic_spatiotemporal(
        self,
        kin_df: pd.DataFrame,
        features_df: pd.DataFrame,
        measurement_col: str,
        task_label: str,
        reference_profile: Optional[Dict] = None,
        output_path: Optional[Path] = None,
        fps: float = 30.0,
        is_reference_session: bool = True,
    ) -> Optional["plt.Figure"]:
        """
        Spatiotemporal kinematic profile for ONE measurement on ONE task.

        One faint coloured line per repetition (raw, time-normalised).
        For reference sessions (is_reference_session=True): draws ±1 SD / ±2 SD
        (steelblue fills) and 95 % CI (dark-blue strip) derived from this session.
        For test sessions (is_reference_session=False): skips session-derived SD/CI
        bands so only the reference profile bands (crimson) serve as the envelope.
        If reference_profile supplied: crimson dashed mean + light red ±1 SD / ±2 SD band.
        Returns the Figure object (caller decides save vs show).
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from scipy.stats import t as t_dist

        N_INTERP = 1000
        grid = np.linspace(0, 1, N_INTERP)

        traces = []
        rep_labels = []

        merged = kin_df.copy()
        if "repetition" not in merged.columns and "repetition" in features_df.columns:
            merged["repetition"] = features_df["repetition"]
        if "task_group" not in merged.columns and "task_group" in features_df.columns:
            merged["task_group"] = features_df["task_group"]
        if "task_id" not in merged.columns and "task_id" in features_df.columns:
            merged["task_id"] = features_df["task_id"]

        if measurement_col not in merged.columns:
            return None

        rep_col_vals = merged["repetition"] if "repetition" in merged.columns else pd.Series(1, index=merged.index)
        for rep_id in sorted(rep_col_vals.dropna().unique()):
            sub = merged[rep_col_vals == rep_id][measurement_col].dropna()
            if len(sub) < 2:
                continue
            x = np.linspace(0, 1, len(sub))
            traces.append(np.interp(grid, x, sub.to_numpy()))
            rep_labels.append(f"Rep {int(rep_id)}")

        if not traces:
            return None

        arr = np.stack(traces)
        n = len(traces)
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)

        if n > 1:
            se = std / np.sqrt(n)
            t_crit = t_dist.ppf(0.975, df=n - 1)
            ci_lo = mean - t_crit * se
            ci_hi = mean + t_crit * se
        else:
            ci_lo = ci_hi = mean

        fig, ax = plt.subplots(figsize=(9, 4), dpi=130)

        colours = cm.get_cmap("tab10", max(n, 1))
        for i, (trace, label) in enumerate(zip(traces, rep_labels)):
            ax.plot(grid, trace, color=colours(i), alpha=0.75, linewidth=1.6, label=label, zorder=8)

        if reference_profile and "mean" in reference_profile:
            ref_mean_arr = np.asarray(reference_profile["mean"], dtype=float)
            ref_std_arr  = np.asarray(reference_profile.get("std", np.zeros_like(ref_mean_arr)), dtype=float)
            if len(ref_mean_arr) != N_INTERP:
                ref_x = np.linspace(0, 1, len(ref_mean_arr))
                ref_mean_arr = np.interp(grid, ref_x, ref_mean_arr)
                if len(ref_std_arr) == len(ref_x):
                    ref_std_arr = np.interp(grid, ref_x, ref_std_arr)
                else:
                    ref_std_arr = np.full(N_INTERP, float(ref_std_arr.mean()) if len(ref_std_arr) else 0.0)
            ax.fill_between(grid,
                            ref_mean_arr - 2 * ref_std_arr,
                            ref_mean_arr + 2 * ref_std_arr,
                            alpha=0.08, color="crimson", zorder=1)
            ax.fill_between(grid,
                            ref_mean_arr - ref_std_arr,
                            ref_mean_arr + ref_std_arr,
                            alpha=0.16, color="crimson", zorder=2,
                            label="Ref ±1 SD / ±2 SD")
            ref_n = reference_profile.get("n", 0)
            if ref_n > 1:
                ref_se = ref_std_arr / np.sqrt(ref_n)
                ref_t  = t_dist.ppf(0.975, df=ref_n - 1)
                ax.fill_between(grid,
                                ref_mean_arr - ref_t * ref_se,
                                ref_mean_arr + ref_t * ref_se,
                                alpha=0.14, color="crimson", zorder=3,
                                label="Ref 95% CI")
            ax.plot(grid, ref_mean_arr,
                    color="crimson", linewidth=1.4, linestyle="--", label="Ref mean", zorder=7)

        if is_reference_session:
            ax.fill_between(grid, mean - 2 * std, mean + 2 * std,
                            alpha=0.10, color="steelblue", zorder=4, label="±2 SD")
            ax.fill_between(grid, mean - std, mean + std,
                            alpha=0.22, color="steelblue", zorder=5, label="±1 SD")
            ax.fill_between(grid, ci_lo, ci_hi, alpha=0.08, color="#1f3a5f", zorder=6, label="95% CI")

        ax.plot(grid, mean, color="black", linewidth=2.0, label=f"Mean (n={n})", zorder=10)

        col_label = measurement_col.replace("kin_", "").replace("_", " ").title()
        ax.set_xlabel("Normalised Duration")
        ax.set_ylabel(col_label)
        ax.set_title(f"{task_label}  ·  {col_label}  (N={n}; SD={std.mean():.3f})")
        has_ref = reference_profile and "mean" in reference_profile
        ref_items = (2 if has_ref else 0) + (1 if has_ref and reference_profile.get("n", 0) > 1 else 0)
        session_items = 3 if is_reference_session else 0
        n_leg = n + 1 + ref_items + session_items
        if n_leg > 8:
            ax.legend(loc='upper left', fontsize=7, ncol=1,
                      bbox_to_anchor=(1.02, 1), borderaxespad=0)
            fig.subplots_adjust(right=0.78)
        else:
            ax.legend(loc='upper right', fontsize=7, ncol=2, framealpha=0.85)
        ax.grid(True, alpha=0.25)
        plt.tight_layout(rect=[0, 0, 1 if n_leg <= 8 else 0.78, 1])

        if output_path:
            fig.savefig(output_path, dpi=130, bbox_inches="tight")
            plt.close(fig)
            return None
        return fig

    def plot_all_kinematic_tasks(
        self,
        kin_df: pd.DataFrame,
        features_df: pd.DataFrame,
        output_dir: Path,
        session_label: str,
        task_groups: Optional[List[str]] = None,
        reference_profiles: Optional[Dict] = None,
        fps: float = 30.0,
        task_name_map: Optional[Dict[str, str]] = None,
        is_reference_session: bool = True,
        ddk_summaries: Optional[Dict] = None,
    ) -> Optional[Path]:
        """
        Generate kinematic spatiotemporal profiles for all (task_group, task_id) combinations.

        For each (task_group, task_id) pair, creates PNG plots for each measurement in
        _KINEMATIC_PRIMARY_COLS, then collects all Figures into a multi-page PDF.
        is_reference_session controls whether session-derived SD/CI bands are drawn
        (True for reference/baseline sessions) or suppressed (False for test sessions).

        Returns path to the PDF, or None if no data.
        """
        from pathlib import Path
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        task_filter = set()
        if "task_group" in features_df.columns and "task_id" in features_df.columns:
            for tg, tid in zip(features_df["task_group"].dropna(), features_df["task_id"].dropna()):
                if task_groups is None or tg in task_groups:
                    task_filter.add((tg, tid))

        if not task_filter:
            logger.warning("No tasks found in features_df for kinematic plotting.")
            return None

        all_figures = []
        pdf_path = output_dir / f"kinematic_profiles_{session_label}.pdf"

        for task_group, task_id in sorted(task_filter):
            task_mask = (
                (features_df["task_group"] == task_group) &
                (features_df["task_id"] == task_id)
            )
            if not task_mask.any():
                continue

            task_kin_df = kin_df[task_mask].copy()
            task_features_df = features_df[task_mask].copy()

            if task_kin_df.empty:
                continue

            task_key = f"{task_group}_{int(task_id)}"
            if task_name_map and task_key in task_name_map:
                task_label = f"{task_group} / {task_name_map[task_key]}"
            else:
                task_label = f"{task_group} / Task {task_id}"

            for meas_col in self._KINEMATIC_PRIMARY_COLS:
                if meas_col not in task_kin_df.columns:
                    continue

                png_name = f"kinematic_{task_group}_{int(task_id)}_{meas_col}.png"
                png_path = output_dir / png_name

                ref_profile = None
                ref_key_all = f"{task_group}_{int(task_id)}"
                if reference_profiles and ref_key_all not in reference_profiles:
                    try:
                        from .task_profile import _DISORDER_TASK_CANONICAL_MAP
                        _canon = _DISORDER_TASK_CANONICAL_MAP.get((task_group, int(task_id)))
                        if _canon is not None:
                            ref_key_all = f"{_canon[0]}_{_canon[1]}"
                    except Exception:
                        pass
                if reference_profiles and ref_key_all in reference_profiles:
                    ref_profile = reference_profiles[ref_key_all].get(meas_col)

                fig = self.plot_kinematic_spatiotemporal(
                    kin_df=task_kin_df,
                    features_df=task_features_df,
                    measurement_col=meas_col,
                    task_label=task_label,
                    reference_profile=ref_profile,
                    output_path=None,
                    fps=fps,
                    is_reference_session=is_reference_session,
                )

                if fig is not None:
                    kin_fmt = self.config.get("output", {}).get(
                        "kinematic_format", "pdf_only"
                    )
                    if kin_fmt in ("png_only", "both"):
                        try:
                            fig.savefig(png_path, dpi=130, bbox_inches="tight")
                        except Exception:
                            pass
                    if kin_fmt != "png_only":
                        all_figures.append((task_label, meas_col, fig))
                    else:
                        plt.close(fig)

        if all_figures:
            try:
                with PdfPages(pdf_path) as pdf:
                    fig_cover, ax = plt.subplots(figsize=(8.5, 11), dpi=130)
                    ax.axis("off")
                    ax.text(
                        0.5, 0.7,
                        "Kinematic Profiles",
                        ha="center", va="center", fontsize=18, weight="bold",
                    )
                    ax.text(
                        0.5, 0.5,
                        f"{len(all_figures)} measurements across {len(task_filter)} task(s)",
                        ha="center", va="center", fontsize=11, style="italic",
                    )
                    try:
                        pdf.savefig(fig_cover, bbox_inches="tight")
                    finally:
                        plt.close(fig_cover)

                    for task_label, meas_col, fig in all_figures:
                        try:
                            pdf.savefig(fig, bbox_inches="tight")
                        finally:
                            plt.close(fig)

                    if ddk_summaries and "B" in task_groups if task_groups else True:
                        _ddk = ddk_summaries if isinstance(ddk_summaries, dict) else {}
                        _ddk_metrics = {
                            "DDK rate (Hz)":        _ddk.get("ddk_rate_hz"),
                            "D_mean (lip excursion)":_ddk.get("ddk_D_mean"),
                            "D_max":                _ddk.get("ddk_D_max"),
                            "Tsd (temporal SD)":    _ddk.get("ddk_Tsd"),
                            "STI (spatiotempl idx)":_ddk.get("ddk_STI"),
                            "Duration (s)":         _ddk.get("ddk_Duration_s"),
                            "# Cycles":             _ddk.get("ddk_Num_Cycles"),
                            "Speed pct25":          _ddk.get("ddk_speed_pct25"),
                            "Speed pct50":          _ddk.get("ddk_speed_pct50"),
                            "Speed pct75":          _ddk.get("ddk_speed_pct75"),
                            "Speed pct95":          _ddk.get("ddk_speed_pct95"),
                        }
                        _valid = {k: v for k, v in _ddk_metrics.items()
                                  if v is not None and not (isinstance(v, float) and np.isnan(v))}
                        if _valid:
                            import matplotlib.gridspec as _gs_ddk
                            fig_ddk = plt.figure(figsize=(10, 7), dpi=130)
                            _gs = _gs_ddk.GridSpec(1, 2, figure=fig_ddk,
                                                   left=0.08, right=0.98,
                                                   wspace=0.40, bottom=0.12, top=0.84)
                            _bar_keys = ["DDK rate (Hz)", "Tsd (temporal SD)",
                                         "STI (spatiotempl idx)", "D_mean (lip excursion)"]
                            _bar_vals = [_valid.get(k, float("nan")) for k in _bar_keys]
                            _bar_valid = [(k, v) for k, v in zip(_bar_keys, _bar_vals)
                                          if not np.isnan(v)]
                            if _bar_valid:
                                ax_ddk = fig_ddk.add_subplot(_gs[0, 0])
                                _bk, _bv = zip(*_bar_valid)
                                _bx = np.arange(len(_bk))
                                ax_ddk.barh(_bx, _bv, color="#4C72B0", alpha=0.82,
                                            edgecolor="none", height=0.55)
                                ax_ddk.set_yticks(_bx)
                                ax_ddk.set_yticklabels(_bk, fontsize=9)
                                ax_ddk.set_xlabel("Value", fontsize=9)
                                ax_ddk.set_title("DDK key metrics", fontsize=10,
                                                 fontweight="bold")
                                ax_ddk.spines["top"].set_visible(False)
                                ax_ddk.spines["right"].set_visible(False)
                                ax_ddk.grid(True, axis="x", alpha=0.25)
                            _spd_keys = ["Speed pct25", "Speed pct50",
                                         "Speed pct75", "Speed pct95"]
                            _spd_vals = [_valid.get(k) for k in _spd_keys]
                            _spd_labels = ["25th", "50th", "75th", "95th"]
                            _spd_valid = [(lbl, v) for lbl, v in zip(_spd_labels, _spd_vals)
                                          if v is not None and not np.isnan(v)]
                            if _spd_valid:
                                ax_spd = fig_ddk.add_subplot(_gs[0, 1])
                                _slbl, _sv = zip(*_spd_valid)
                                _sx = np.arange(len(_slbl))
                                ax_spd.bar(_sx, _sv, color="#DD8452", alpha=0.82,
                                           edgecolor="none", width=0.55)
                                ax_spd.set_xticks(_sx)
                                ax_spd.set_xticklabels([f"Ls{l}" for l in _slbl],
                                                       fontsize=9)
                                ax_spd.set_ylabel("Speed (a.u.)", fontsize=9)
                                ax_spd.set_title("DDK speed percentiles",
                                                 fontsize=10, fontweight="bold")
                                ax_spd.spines["top"].set_visible(False)
                                ax_spd.spines["right"].set_visible(False)
                                ax_spd.grid(True, axis="y", alpha=0.25)
                            fig_ddk.suptitle(
                                "DDK Clinical Metrics — Group B\n"
                                "(Allison et al. 2022 · Simmatis et al. 2023 · Segal et al. 2022)",
                                fontsize=11, fontweight="bold",
                            )
                            try:
                                pdf.savefig(fig_ddk, bbox_inches="tight")
                            finally:
                                plt.close(fig_ddk)

                logger.info("Saved kinematic multi-page PDF: %s", pdf_path)
                return pdf_path
            except Exception as e:
                logger.error("Failed to create kinematic PDF: %s", e)
                return None
        else:
            logger.warning("No figures generated for kinematic PDF.")
            return None

    def plot_condition_comparison(
        self,
        session_overview_df: "pd.DataFrame",
        output_path: "Path",
        subject_id: str = "",
    ) -> None:
        """Compare sessions across any two or more distinct posture / condition labels.

        Three-panel figure:
         - Panel 1: mean asymmetry ratio per session, coloured by condition
         - Panel 2: overall detection rate per session, coloured by condition
         - Panel 3: anomaly rate per session, coloured by condition, with
           the session label on the x-axis so the viewer can identify
           the session type (baseline vs COMBINED, pre/intra/post-op).

        Triggers for any two or more distinct posture values — not only the
        canonical upright/supine pairing.  This makes it suitable for
        pre-op / intra-op / post-op comparisons and any other multi-condition
        study design.
        """
        import pandas as _pd

        df = session_overview_df.copy() if hasattr(session_overview_df, "copy") else session_overview_df
        if len(df) == 0:
            logger.warning("plot_condition_comparison: empty session overview DataFrame")
            return

        C = self._get_colors()

        posture_col = "posture" if "posture" in df.columns else None
        label_col   = "session_label" if "session_label" in df.columns else "session_id"

        def _posture(row):
            """Infer recording posture from row metadata: 'supine' or 'upright'."""
            if posture_col:
                return str(row[posture_col])
            token = str(row.get("session_id", "") + " " + row.get(label_col, "")).lower()
            if any(k in token for k in ("supine", "or_sim", "intra")):
                return "supine"
            return "upright"

        df["_posture"] = df.apply(_posture, axis=1)
        df["_label"]   = df[label_col].astype(str).str.replace(r"^[A-Z0-9]+_", "", regex=True)

        unique_conditions = sorted(df["_posture"].dropna().unique())
        condition_color_map = {cond: C[i % len(C)] for i, cond in enumerate(unique_conditions)}
        colors = df["_posture"].map(condition_color_map).fillna(C[4])
        x = range(len(df))
        x_labels = df["_label"].tolist()

        metrics = [
            ("overall_mean_asymmetry", "Mean asymmetry ratio", "Asymmetry ratio"),
            ("overall_detection_rate", "Face detection rate", "Detection rate"),
            ("anomaly_rate",           "Anomaly rate", "Anomaly rate"),
        ]
        metrics = [(k, t, y) for k, t, y in metrics if k in df.columns]
        if not metrics:
            logger.warning("plot_condition_comparison: none of the required metric columns present")
            return

        n_panels = len(metrics)
        fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.5))
        if n_panels == 1:
            axes = [axes]

        _line_styles = ["--", ":", "-.", (0, (3, 1, 1, 1))]
        for ax, (col, title, ylabel) in zip(axes, metrics):
            vals = df[col].fillna(0).values
            bars = ax.bar(list(x), vals, color=list(colors), width=0.65, edgecolor="#333333",
                          linewidth=0.6, zorder=2)
            for ci, cond in enumerate(unique_conditions):
                med_val = df.loc[df["_posture"] == cond, col].median()
                if not _pd.isna(med_val):
                    ls = _line_styles[ci % len(_line_styles)]
                    ax.axhline(med_val, color=condition_color_map[cond], linestyle=ls,
                               linewidth=1.0, alpha=0.75, label=f"{cond} median ({med_val:.2f})")
            ax.set_xticks(list(x))
            ax.set_xticklabels(x_labels, rotation=40, ha="right", fontsize=7)
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.yaxis.grid(True, alpha=0.3, linestyle=":")
            ax.set_axisbelow(True)
            ax.legend(fontsize=7, framealpha=0.85)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=6.5)

        legend_patches = [
            mpatches.Patch(facecolor=condition_color_map[cond], label=cond.title())
            for cond in unique_conditions
        ]
        fig.legend(handles=legend_patches, loc="lower center", ncol=min(4, len(unique_conditions)),
                   fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.06))

        cond_str = " vs ".join(c.title() for c in unique_conditions)
        title_str = f"Condition Comparison — {subject_id}: {cond_str}" if subject_id else f"Condition Comparison: {cond_str}"
        fig.suptitle(title_str, fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0.04, 1, 0.96])
        try:
            fig.savefig(str(output_path), dpi=self.save_dpi, bbox_inches="tight")
        finally:
            plt.close(fig)

    def plot_detection_quality_summary(
        self,
        session_overview_df: "pd.DataFrame",
        output_path: "Path",
        subject_id: str = "",
    ) -> None:
        """Four-panel dataset / recording quality overview figure.

        Intended as the thesis Dataset or Data Quality section figure;
        summarises the quality and completeness of recordings across all
        sessions for a single subject.

         - Panel 1: Per-session face detection rate (bar chart)
         - Panel 2: Distribution of detection rates across sessions (histogram)
         - Panel 3: Mean blendshape activation amplitude per session
         - Panel 4: Per-session total duration (bar chart)
        """
        import pandas as _pd

        df = session_overview_df.copy() if hasattr(session_overview_df, "copy") else session_overview_df
        if len(df) == 0:
            logger.warning("plot_detection_quality_summary: empty session overview DataFrame")
            return

        C = self._get_colors()
        label_col = "session_label" if "session_label" in df.columns else "session_id"
        df["_label"] = df[label_col].astype(str).str.replace(r"^[A-Z0-9]+_", "", regex=True)

        posture_col = "posture" if "posture" in df.columns else None
        colors = (
            df[posture_col].map({"upright": C[0], "supine": C[1]}).fillna(C[0]).tolist()
            if posture_col else [C[0]] * len(df)
        )

        fig = plt.figure(figsize=(13, 9))
        gs  = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.38)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        x = list(range(len(df)))

        if "overall_detection_rate" in df.columns:
            vals = df["overall_detection_rate"].fillna(0).values
            ax1.bar(x, vals, color=colors, edgecolor="#333333", linewidth=0.6, zorder=2)
            ax1.axhline(0.8, color="#CC3311", linestyle="--", linewidth=1.1,
                        label="Acceptable threshold (0.8)", alpha=0.85)
            ax1.set_xticks(x)
            ax1.set_xticklabels(df["_label"].tolist(), rotation=40, ha="right", fontsize=7)
            ax1.set_ylabel("Detection rate")
            ax1.set_ylim(0, 1.05)
            ax1.set_title("Face Detection Rate per Session", fontsize=10, fontweight="bold")
            ax1.yaxis.grid(True, alpha=0.3, linestyle=":")
            ax1.legend(fontsize=7, framealpha=0.85)

            det_vals = df["overall_detection_rate"].dropna().values
            if len(det_vals) >= 2:
                ax2.hist(det_vals, bins=min(8, len(det_vals)), color=C[0],
                         edgecolor="#333333", linewidth=0.7, alpha=0.85)
                ax2.axvline(float(_pd.Series(det_vals).median()), color="#CC3311",
                            linestyle="--", linewidth=1.2,
                            label=f"Median = {float(_pd.Series(det_vals).median()):.2f}")
                ax2.set_xlabel("Detection rate")
                ax2.set_ylabel("Count")
                ax2.set_title("Detection Rate Distribution", fontsize=10, fontweight="bold")
                ax2.legend(fontsize=7, framealpha=0.85)
                ax2.yaxis.grid(True, alpha=0.3, linestyle=":")
            else:
                ax2.text(0.5, 0.5, "Insufficient sessions\nfor distribution",
                         ha="center", va="center", fontsize=9, color="#666666",
                         transform=ax2.transAxes)
                ax2.set_title("Detection Rate Distribution", fontsize=10, fontweight="bold")
        else:
            for ax in (ax1, ax2):
                ax.text(0.5, 0.5, "Detection rate\nnot available",
                        ha="center", va="center", fontsize=9, color="#666666",
                        transform=ax.transAxes)

        act_col = "mean_activation_session_mean" if "mean_activation_session_mean" in df.columns else None
        if act_col:
            act_vals = df[act_col].fillna(0).values
            ax3.bar(x, act_vals, color=colors, edgecolor="#333333", linewidth=0.6, zorder=2)
            ax3.set_xticks(x)
            ax3.set_xticklabels(df["_label"].tolist(), rotation=40, ha="right", fontsize=7)
            ax3.set_ylabel("Mean blendshape activation")
            ax3.set_title("Mean Activation per Session", fontsize=10, fontweight="bold")
            ax3.yaxis.grid(True, alpha=0.3, linestyle=":")

        dur_col = "total_duration_sec" if "total_duration_sec" in df.columns else None
        if dur_col:
            dur_vals = df[dur_col].fillna(0).values
            ax4.bar(x, dur_vals, color=colors, edgecolor="#333333", linewidth=0.6, zorder=2)
            ax4.set_xticks(x)
            ax4.set_xticklabels(df["_label"].tolist(), rotation=40, ha="right", fontsize=7)
            ax4.set_ylabel("Duration (s)")
            ax4.set_title("Session Duration", fontsize=10, fontweight="bold")
            ax4.yaxis.grid(True, alpha=0.3, linestyle=":")

        legend_patches = [
            mpatches.Patch(facecolor=C[0], label="Upright"),
            mpatches.Patch(facecolor=C[1], label="Supine"),
        ]
        fig.legend(handles=legend_patches, loc="lower center", ncol=2,
                   fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.03))

        title_str = f"Recording Quality Overview — {subject_id}" if subject_id else "Recording Quality Overview"
        fig.suptitle(title_str, fontsize=13, fontweight="bold")
        try:
            fig.savefig(str(output_path), dpi=self.save_dpi, bbox_inches="tight")
        finally:
            plt.close(fig)

    def plot_brain_activation_map(
        self,
        screening_results: Dict[str, Any],
        output_path: Path,
        title: str = "Neural Substrates",
    ) -> None:
        """Render a nilearn glass-brain activation map showing implicated neural regions.

        Builds a NIfTI stat map by placing Gaussian blobs (sigma=3 voxels, 6 mm)
        at the MNI-152 coordinates in brain._MNI_COORDS, weighted by each
        region's activation score.  nilearn renders four orthographic glass-brain
        views (left lateral, coronal, right lateral, axial) with an amber-to-crimson
        colormap.  A legend panel on the right lists activated regions by structure
        type with horizontal activation bars and severity badges.

        Always produces a figure. When there are no findings the glass-brain is
        empty (all zeros) and the subtitle reads "No findings".  Requires the
        optional dependencies ``nilearn`` and ``nibabel``.

        Parameters
        ----------
        screening_results:
            Screening result dict from the decision support step.
        output_path:
            Destination path; the suffix is replaced with ``.png``.
        title:
            Figure super-title.
        """
        import nibabel as nib
        from scipy.ndimage import gaussian_filter
        from io import BytesIO
        import matplotlib.image as mpimg
        import nilearn.plotting as nplot
        from matplotlib.colors import LinearSegmentedColormap
        from .brain import generate_brain_report, _MNI_COORDS

        brain_report   = generate_brain_report(screening_results)
        activation_map = brain_report.get("activation_map", {})

        indication_list = sorted(set(
            ind.get("indication_type", "").replace("_", " ").title()
            for ind in screening_results.get("indications", [])
        ))
        subtitle = "Findings: " + ", ".join(indication_list) if indication_list else "No findings"

        def _act_color(act: float) -> str:
            """Return a hex colour interpolated across a warm activation gradient for the brain map."""
            stops = [
                (0.00, (0.92, 0.92, 0.93)),
                (0.30, (1.00, 0.93, 0.68)),
                (0.60, (0.94, 0.55, 0.15)),
                (0.85, (0.83, 0.16, 0.10)),
                (1.00, (0.52, 0.02, 0.08)),
            ]
            act = max(0.0, min(1.0, float(act)))
            for i in range(len(stops) - 1):
                t0, c0 = stops[i]
                t1, c1 = stops[i + 1]
                if t0 <= act <= t1:
                    f = (act - t0) / (t1 - t0) if t1 > t0 else 0.0
                    return "#{:02X}{:02X}{:02X}".format(
                        int((c0[0] + f * (c1[0] - c0[0])) * 255),
                        int((c0[1] + f * (c1[1] - c0[1])) * 255),
                        int((c0[2] + f * (c1[2] - c0[2])) * 255),
                    )
            return "#AAAAAA"

        _MNI_AFFINE = np.array([
            [2., 0., 0., -98.],
            [0., 2., 0., -134.],
            [0., 0., 2., -72.],
            [0., 0., 0.,   1.],
        ], dtype=float)
        _MNI_SHAPE = (99, 117, 95)
        inv_aff = np.linalg.inv(_MNI_AFFINE)

        data = np.zeros(_MNI_SHAPE, dtype=np.float32)
        for region_id, coord_list in _MNI_COORDS.items():
            rinfo = activation_map.get(region_id, {})
            act = float(rinfo.get("activation", 0.0))
            if act < 0.05:
                continue
            for coord in coord_list:
                v = np.round((inv_aff @ (*coord, 1.0))[:3]).astype(int)
                if all(0 <= v[i] < _MNI_SHAPE[i] for i in range(3)):
                    data[v[0], v[1], v[2]] = max(data[v[0], v[1], v[2]], act)

        cmap_brain = LinearSegmentedColormap.from_list(
            'brain_act', ['#FFE580', '#F5A020', '#E04010', '#830808'],
        )
        if data.max() > 0:
            peak = float(data.max())
            data = gaussian_filter(data, sigma=3)
            data = data * (peak / float(data.max()))
        stat_img = nib.Nifti1Image(data, _MNI_AFFINE)
        vmax   = float(data.max()) if data.max() > 0 else 1.0
        thresh = vmax * 0.08

        display = nplot.plot_glass_brain(
            stat_img,
            display_mode='lyrz',
            colorbar=False,
            black_bg=False,
            alpha=0.28,
            cmap=cmap_brain,
            vmin=thresh,
            vmax=vmax,
            annotate=True,
            threshold=thresh,
        )
        buf = BytesIO()
        display.savefig(buf, dpi=180)
        buf.seek(0)
        brain_img = mpimg.imread(buf)
        display.close()

        fig      = plt.figure(figsize=(22, 10), facecolor='white')
        gs       = GridSpec(1, 2, figure=fig, width_ratios=[1.85, 0.72], wspace=0.06,
                            left=0.01, right=0.97, top=0.88, bottom=0.06)
        ax_brain = fig.add_subplot(gs[0, 0])
        ax_leg   = fig.add_subplot(gs[0, 1])

        ax_brain.set_facecolor('white')
        ax_brain.imshow(brain_img)
        ax_brain.axis('off')

        for label, xpos in [
            ("Left lateral",  0.105),
            ("Coronal",       0.365),
            ("Right lateral", 0.625),
            ("Axial",         0.875),
        ]:
            ax_brain.text(
                xpos, -0.01, label,
                transform=ax_brain.transAxes,
                ha='center', va='top', fontsize=9, style='italic',
                color='#546E7A',
            )

        _draw_legend_panel(ax_leg, brain_report, _act_color)

        fig.suptitle(
            title + ("\n" + subtitle if subtitle else ""),
            fontsize=14, fontweight='bold', y=0.97, color='#222222',
        )
        self._save_figure(fig, output_path)
        plt.close(fig)


    def _save_figure(self, fig: plt.Figure, output_path: Path, is_table: bool = False) -> None:
        """Save *fig* as PDF (tables) or PNG (plots) with config-driven DPI and padding."""
        output_config = self.config.get('output', {})
        suffix = '.pdf' if is_table else '.png'
        fig.savefig(
            output_path.with_suffix(suffix),
            dpi=output_config.get('save_dpi', 300),
            bbox_inches=output_config.get('bbox_inches', 'tight'),
            pad_inches=output_config.get('pad_inches', 0.1),
            transparent=output_config.get('transparent_background', False),
        )
        try:
            plt.close(fig)
        except Exception:
            pass

    def plot_fatigue_drift_report(
        self,
        fatigue_report: Dict[str, Any],
        output_path: Path,
        title: str = "Fatigue & Motor Drift Analysis",
    ) -> None:
        """Visualise per-window fatigue drift analysis from FatigueDriftMonitor.

        Produces a 4-panel clinical-report figure:
          Panel A — Regional motor velocity normalised to session median (%)
                    with shaded background on velocity-decay windows.
          Panel B — Facial symmetry index (absolute L-R asymmetry ratio, 0-1)
                    per region, with mild (0.10) and notable (0.20) threshold lines
                    (Kong et al. 2021 PVT correlates).
          Panel C — Dynamic motor range as % of baseline range.
                    100% = same ROM as session start; declining = progressive fatigue.
                    (Adapted from Brach & VanSwearingen 1995.)
          Panel D — Fatigue flag heatmap (flag type x time), showing WHEN each
                    flag type occurred across the session.

        References
        ----------
        Kong Y et al. (2021) Atten Percept Psychophys 83:525.
          doi:10.3758/s13414-020-02199-5
        Di Stasi LL et al. (2014) Ann Surg 259:824.
          doi:10.1097/SLA.0000000000000260
        Brach JS, VanSwearingen J (1995) Arch Phys Med Rehabil 76:905. PMID:7668964
        """
        import numpy as _np
        from matplotlib.colors import LinearSegmentedColormap

        windows = fatigue_report.get("windows", [])
        if not windows:
            return

        summary = fatigue_report.get("summary", {})
        n_win  = summary.get("n_windows", len(windows))
        n_flag = summary.get("n_flagged", 0)

        C_EYE   = COLORBLIND_SAFE_PALETTE["blue"]
        C_BROW  = COLORBLIND_SAFE_PALETTE["orange"]
        C_MOUTH = COLORBLIND_SAFE_PALETTE["green"]
        C_ALERT = COLORBLIND_SAFE_PALETTE["red"]
        C_WARN  = COLORBLIND_SAFE_PALETTE["orange"]
        C_GRAY  = COLORBLIND_SAFE_PALETTE["gray"]
        region_colors = {"eye": C_EYE, "brow": C_BROW, "mouth": C_MOUTH}

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(title, fontsize=15, fontweight="bold", y=1.00)

        def _mid(w):
            """Return the midpoint time of a window dict in minutes."""
            return (w["start_s"] + w["end_s"]) / 2.0 / 60.0

        def _smooth(arr, k=3):
            """Apply a symmetric moving-average kernel of width k to an array."""
            if len(arr) < k:
                return arr
            out = _np.convolve(arr, _np.ones(k) / k, mode="same")
            out[:k // 2] = arr[:k // 2]
            out[-(k // 2):] = arr[-(k // 2):]
            return out

        ax_a = axes[0, 0]

        decay_spans = {
            (w["start_s"], w["end_s"])
            for w in windows
            for f in w.get("flags", [])
            if f.get("type") == "velocity_decay"
        }
        for s, e in decay_spans:
            ax_a.axvspan(s / 60.0, e / 60.0, color=C_ALERT, alpha=0.08, zorder=0)

        for region, color in region_colors.items():
            times, vels = [], []
            for w in windows:
                if region in w.get("regions", {}):
                    times.append(_mid(w))
                    vels.append(w["regions"][region]["velocity_mean"])
            if not times:
                continue
            times = _np.array(times)
            vels  = _np.array(vels, dtype=float)
            med = float(_np.median(vels))
            vels_pct = (vels / med * 100.0) if med > 1e-6 else vels
            smoothed = _smooth(vels_pct)
            ax_a.plot(times, smoothed, linewidth=2.0, color=color,
                      label=region.capitalize(), zorder=3)
            ax_a.fill_between(times, smoothed, 100.0,
                              where=(smoothed < 100.0),
                              alpha=0.10, color=color, zorder=2)

        ax_a.axhline(y=100.0, linestyle="--", linewidth=1.2, color=C_GRAY, alpha=0.6,
                     label="Median (100%)")
        ax_a.axhline(y=75.0,  linestyle=":",  linewidth=1.0, color=C_ALERT, alpha=0.5,
                     label="-25% threshold")
        _bot_a, _top_a = ax_a.get_ylim()
        ax_a.set_ylim(bottom=_bot_a, top=_top_a + (_top_a - _bot_a) * 0.30)
        ax_a.set_xlabel("Time (min)")
        ax_a.set_ylabel("Velocity  (% of session median)")
        ax_a.set_title(
            "Motor Velocity — Normalised\n"
            "(pale red background = velocity-decay flag windows)",
            fontsize=9
        )
        ax_a.legend(fontsize=8, loc="upper right")
        ax_a.grid(True, alpha=0.25, linestyle=":")

        ax_b = axes[0, 1]

        for region, color in region_colors.items():
            times, asym_vals = [], []
            for w in windows:
                if region in w.get("regions", {}):
                    rdata = w["regions"][region]
                    asym = rdata.get(
                        "asymmetry_index",
                        min(1.0, abs(rdata.get("asymmetry_pct_change", 0.0)) / 100.0),
                    )
                    times.append(_mid(w))
                    asym_vals.append(max(0.0, asym))
            if not times:
                continue
            smoothed = _smooth(_np.array(asym_vals, dtype=float))
            ax_b.plot(times, smoothed, linewidth=2.0, color=color,
                      label=region.capitalize(), zorder=3)

        ax_b.axhline(y=0.20, linestyle="--", linewidth=1.3, color=C_ALERT, alpha=0.8,
                     label="Notable (0.20, Kong 2021)")
        ax_b.axhline(y=0.10, linestyle=":",  linewidth=1.0, color=C_WARN,  alpha=0.7,
                     label="Mild (0.10)")
        ax_b.axhline(y=0.0,  linewidth=0.8,  color=C_GRAY, alpha=0.4)
        ax_b.set_ylim(bottom=0.0)
        ylim_top = max(0.35, ax_b.get_ylim()[1])
        ylim_top = ylim_top + ylim_top * 0.30
        ax_b.set_ylim(top=ylim_top)
        ax_b.axhspan(0.20, ylim_top, color=C_ALERT, alpha=0.05, zorder=0)
        ax_b.set_xlabel("Time (min)")
        ax_b.set_ylabel("Asymmetry index  (0=symmetric, 1=full)")
        ax_b.set_title(
            "Facial Symmetry Index\n"
            "(Kong et al. 2021 — Lid Tighten & Lip Corner correlates)",
            fontsize=9
        )
        ax_b.legend(fontsize=8, loc="upper right")
        ax_b.grid(True, alpha=0.25, linestyle=":")

        ax_c = axes[1, 0]

        times_dr, mean_dr = [], []
        for w in windows:
            fr = w.get("fatigue_risk", {})
            mdr = fr.get("mean_activation_range_pct",
                         fr.get("mean_percent_fatigue", None))
            if mdr is not None:
                times_dr.append(_mid(w))
                mean_dr.append(float(mdr))

        if times_dr:
            td  = _np.array(times_dr)
            mdr = _np.array(mean_dr, dtype=float)
            sdr = _smooth(mdr)
            ax_c.plot(td, sdr, linewidth=2.2,
                      color=COLORBLIND_SAFE_PALETTE["cyan"],
                      label="Mean ROM range", zorder=3)
            ax_c.fill_between(td, sdr, 100.0, where=(sdr < 100.0),
                              alpha=0.18,
                              color=COLORBLIND_SAFE_PALETTE["cyan"],
                              zorder=2, label="Below baseline ROM")
            if len(td) >= 4:
                try:
                    trend = _np.poly1d(_np.polyfit(td, mdr, 1))
                    ax_c.plot(td, trend(td), linestyle="--", linewidth=1.4,
                              color=C_ALERT, alpha=0.7, label="Trend")
                except Exception:
                    pass

        ax_c.axhline(y=100.0, linestyle="--", linewidth=1.2, color=C_GRAY,  alpha=0.6,
                     label="Baseline (100%)")
        ax_c.axhline(y=75.0,  linestyle=":",  linewidth=1.2, color=C_WARN,  alpha=0.8,
                     label="-25% ROM threshold (Brach 1995)")

        rom_trend = summary.get("rom_trend") or {}
        if rom_trend and times_dr:
            p         = rom_trend.get("p_value", 1.0)
            slope_min = rom_trend.get("slope_pct_per_min", 0.0)
            r2        = rom_trend.get("r_squared", 0.0)
            sig       = rom_trend.get("significant_decline", False)
            p_str     = ("p < 0.001" if p < 0.001
                         else f"p = {p:.3f}" if p < 0.01
                         else f"p = {p:.2f}")
            star      = "  *" if sig else ""
            ann_color = C_ALERT if sig else C_GRAY
            ann_text  = (
                f"Trend: {slope_min:+.1f} %/min\n"
                f"{p_str}{star},  R\u00b2 = {r2:.2f}"
            )
            ax_c.text(
                0.98, 0.97, ann_text,
                transform=ax_c.transAxes,
                fontsize=813, va="top", ha="right", color=ann_color,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor=ann_color, alpha=0.88),
            )

        _bot_c, _top_c = ax_c.get_ylim()
        ax_c.set_ylim(bottom=_bot_c, top=_top_c + (_top_c - _bot_c) * 0.30)
        ax_c.set_xlabel("Time (min)")
        ax_c.set_ylabel("Dynamic range  (% of baseline range)")
        ax_c.set_title(
            "Motor Range-of-Motion  vs  Baseline\n"
            "(100% = baseline ROM; declining = progressive fatigue)",
            fontsize=9
        )
        ax_c.legend(fontsize=7, loc="upper left")
        ax_c.grid(True, alpha=0.25, linestyle=":")

        ax_d = axes[1, 1]

        flag_types  = ["velocity_decay", "asymmetry_creep", "rom_tightening"]
        flag_labels = ["Velocity\nDecay", "Asymmetry\nCreep", "ROM\nTightening"]

        n_w  = len(windows)
        heat = _np.zeros((len(flag_types), n_w), dtype=float)
        t_mid_all = [_mid(w) for w in windows]

        for wi, w in enumerate(windows):
            for fi, ftype in enumerate(flag_types):
                heat[fi, wi] = sum(
                    1 for f in w.get("flags", [])
                    if f.get("type") == ftype
                )

        _cmap = LinearSegmentedColormap.from_list(
            "fatigue_heat",
            ["#F9FAFB", "#FDE68A", "#F59E0B", "#DC2626"],
            N=256,
        )
        t0, t1 = t_mid_all[0], t_mid_all[-1]
        im = ax_d.imshow(
            heat, aspect="auto", cmap=_cmap, vmin=0, vmax=3,
            interpolation="nearest",
            extent=[t0, t1, -0.5, len(flag_types) - 0.5],
        )
        ax_d.set_yticks(range(len(flag_types)))
        ax_d.set_yticklabels(flag_labels, fontsize=9)
        ax_d.set_xlabel("Time (min)")
        ax_d.set_title(
            f"Fatigue Flag Timeline\n"
            f"{n_flag}/{n_win} windows flagged"
            f"  ({summary.get('flag_fraction', 0) * 100:.0f}%)\n"
            f"Colour intensity = number of affected regions (0-3)",
            fontsize=9,
        )
        cbar = fig.colorbar(im, ax=ax_d, orientation="vertical",
                            fraction=0.04, pad=0.02)
        cbar.set_ticks([0, 1, 2, 3])
        cbar.set_ticklabels(["0", "1", "2", "3 regions"], fontsize=7)

        plt.tight_layout()
        self._save_figure(fig, output_path)
        plt.close(fig)


def _draw_legend_panel(
    ax,
    brain_report: Dict[str, Any],
    color_fn,
) -> None:
    """Draw the activation legend with colour bar, grouped region list, and localisation note.

    Renders a continuous colour-scale bar at the top, then a bulleted list of
    activated regions grouped by structure type (Cortical / Subcortical /
    Brainstem), each with a horizontal activation bar and severity badge.
    The clinical localisation summary is appended at the bottom.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    ax.add_patch(mpatches.FancyBboxPatch(
        (0.03, 0.01), 0.94, 0.97,
        boxstyle="round,pad=0.01",
        facecolor='#F8FAFB', edgecolor='#BDBDBD', linewidth=1.0,
        zorder=0,
    ))

    ax.text(0.50, 0.955, "Activated Regions",
            ha='center', va='top', fontsize=11, fontweight='bold',
            color='#222222')
    ax.text(0.50, 0.920, "Neural substrates by activation level",
            ha='center', va='top', fontsize=8, style='italic', color='#546E7A')

    ax.plot([0.08, 0.92], [0.903, 0.903], color='#BDBDBD', linewidth=0.8, zorder=2)

    ax.text(0.50, 0.886, "Activation scale",
            ha='center', va='top', fontsize=8, color='#546E7A')
    bar_top  = 0.820
    bar_h    = 0.034
    bar_xs   = np.linspace(0.08, 0.92, 120)
    for i, bx in enumerate(bar_xs[:-1]):
        ax.add_patch(mpatches.Rectangle(
            (bx, bar_top), bar_xs[1] - bar_xs[0], bar_h,
            facecolor=color_fn(i / (len(bar_xs) - 2)), edgecolor='none',
        ))
    ax.add_patch(mpatches.Rectangle(
        (0.08, bar_top), 0.84, bar_h,
        facecolor='none', edgecolor='#90A4AE', linewidth=0.8, zorder=3,
    ))
    ax.text(0.08, bar_top - 0.012, "Low",  ha='center', va='top', fontsize=7, color='#546E7A')
    ax.text(0.92, bar_top - 0.012, "High", ha='center', va='top', fontsize=7, color='#546E7A')

    ax.plot([0.08, 0.92], [0.783, 0.783], color='#CFD8DC', linewidth=0.7, zorder=2)

    activated_ids  = brain_report.get("regions_activated", [])
    activation_map = brain_report.get("activation_map", {})

    _SEV_COLORS = {
        "mild":     "#E69F00",
        "moderate": "#D55E00",
        "severe":   "#CC0000",
    }

    y_cur = 0.770
    row_h = 0.070

    _GROUPS = [
        ("Cortical",              {"cortical", "insular"}),
        ("Subcortical",           {"subcortical"}),
        ("Brainstem & Cerebellar",{"brainstem", "cerebellar"}),
    ]

    for grp_label, struct_types in _GROUPS:
        members = [
            (rid, activation_map[rid]) for rid in activated_ids
            if rid in activation_map
            and activation_map[rid].get("structure_type", "cortical") in struct_types
        ]
        if not members:
            continue
        if y_cur < 0.13:
            break

        ax.text(0.09, y_cur, grp_label.upper(),
                fontsize=7, fontweight='bold', color='#78909C', va='top')
        y_cur -= row_h * 0.50

        for rid, rinfo in members:
            if y_cur < 0.13:
                break
            act = float(rinfo.get("activation", 0))
            name = rinfo.get("name", rid)
            if len(name) > 27:
                name = name[:25] + "…"
            sev = rinfo.get("max_severity", "")

            dot_cy = y_cur - 0.012
            ax.add_patch(plt.Circle(
                (0.09, dot_cy), 0.013,
                facecolor=color_fn(act), edgecolor='#90A4AE',
                linewidth=0.6, zorder=3,
            ))
            ax.text(0.155, y_cur, name,
                    fontsize=7, color='#37474F', va='top')
            bar_bot = y_cur - row_h * 0.52
            bar_ht  = row_h * 0.26
            ax.add_patch(mpatches.Rectangle(
                (0.155, bar_bot), 0.755, bar_ht,
                facecolor='#ECEFF1', edgecolor='none',
            ))
            if act > 0:
                ax.add_patch(mpatches.Rectangle(
                    (0.155, bar_bot), 0.755 * act, bar_ht,
                    facecolor=color_fn(act), edgecolor='none', alpha=0.88,
                ))
            if sev and sev != "none":
                ax.text(0.935, y_cur, sev,
                        fontsize=6.5, color=_SEV_COLORS.get(sev, '#999999'),
                        va='top', ha='right', style='italic', fontweight='bold')
            y_cur -= row_h

        y_cur -= row_h * 0.20

    localisation = brain_report.get("clinical_localisation", "")
    if localisation and localisation != "No specific localisation pattern identified":
        y_loc = max(y_cur - 0.008, 0.055)
        ax.plot([0.08, 0.92], [y_loc + 0.006, y_loc + 0.006],
                color='#CFD8DC', linewidth=0.7, zorder=2)
        y_loc -= 0.016
        ax.text(0.09, y_loc, "Clinical Localisation",
                fontsize=8, fontweight='bold', color='#37474F', va='top')
        y_loc -= 0.038
        for part in localisation.split(" | ")[:3]:
            for line in _wrap_text(part, 40)[:2]:
                if y_loc < 0.025:
                    break
                ax.text(0.09, y_loc, line,
                        fontsize=7, color='#546E7A', va='top', style='italic')
                y_loc -= 0.038


def _wrap_text(text: str, max_chars: int = 50) -> List[str]:
    """Split *text* into lines no longer than *max_chars* characters."""
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 <= max_chars:
            current = (current + " " + word).lstrip()
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [text[:max_chars]]


def _feature_category_str(fname: str) -> str:
    """Map a feature name to a human-readable category label."""
    fl = fname.lower()
    if 'asymmetry' in fl or 'ratio' in fl:
        return 'Asymmetry'
    if 'time_to_peak' in fl or 'velocity' in fl or 'acceleration' in fl or 'duration' in fl:
        return 'Temporal'
    if any(k in fl for k in ('mean', 'max', 'std', 'range', 'min', 'activation')):
        return 'Amplitude'
    return 'Other'


def create_visualizer(plotting_config: Dict[str, Any]) -> Visualizer:
    """Factory: instantiate a :class:`Visualizer` from the plotting YAML config."""
    return Visualizer(plotting_config)
