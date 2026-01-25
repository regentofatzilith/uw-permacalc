
# --- Imports: all at the top, no duplicates ---
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, callback, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dataclasses import dataclass, field
from typing import Dict, Any


# --- Simulation/stat/figure logic (inlined from perma_calc_core) ---
WA_Card = {
	"None": 0.0,
	"1 Star": 0.30,
	"2 Star": 0.34,
	"3 Star": 0.38,
	"4 Star": 0.42,
	"5 Star": 0.46,
	"6 Star": 0.50,
	"7 Star": 0.54
}
Wave_Basetime = 26
Farming_Perks = {
	"Black_Hole_Duration": 12,
	"Death_Wave_Duration": 4,
	"Chrono_Field_Duration": 5,
}
GC_EFFECTS = {
	'None': 0.0,
	'Epic': 10.0,
	'Legendary': 13.0,
	'Mythic': 17.0,
	'Ancestral': 20.0,
}
MVN_EFFECTS = {
	'None': 0.0,
	'Epic': 20.0,
	'Legendary': 10.0,
	'Mythic': 1.0,
	'Ancestral': -10.0,
}
total_time = 3600

def is_package(time: int, rng: np.random.Generator, pkg_chance: float, pkg_reduction: float, wave_time: float, boss_every_x: int, boss_package_enabled: bool = True) -> tuple[float, bool]:
    package_reduction = 0.0
    is_boss_package = False
    boss_time = wave_time * boss_every_x
    is_boss_wave = time % boss_time < 1
    if is_boss_wave and boss_package_enabled:
        package_reduction = pkg_reduction
        is_boss_package = True
    elif time % wave_time < 1:
        package_reduction = pkg_reduction if rng.random() < pkg_chance else 0.0
        is_boss_package = False
    return package_reduction, is_boss_package

def simulate_package_reductions(total_time: int, pkg_chance: float, pkg_reduction: float, wave_time: float, boss_every_x: int, seed: int = 0, boss_package_enabled: bool = True) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    reductions = np.zeros(total_time, dtype=float)
    is_boss = np.zeros(total_time, dtype=bool)
    for t in range(total_time):
        reductions[t], is_boss[t] = is_package(t, rng, pkg_chance, pkg_reduction, wave_time, boss_every_x, boss_package_enabled)
    return reductions, is_boss

def calculate_uw_uptime(
    uw_cooldown: float,
    uw_duration: float,
    total_time: int,
    return_df: bool = False,
    package_reduction_series: np.ndarray | None = None,
    is_boss_package_series: np.ndarray | None = None,
    wave_time: float = 30.140,
    can_queue: bool = True,
) -> dict | pd.DataFrame:
    timeline = np.arange(0, total_time)
    if package_reduction_series is None:
        package_reduction_series, is_boss_package_series = simulate_package_reductions(total_time, 0.76, 13.0, wave_time, 10)
    package_reduction_series = np.asarray(package_reduction_series, dtype=float)
    if package_reduction_series.shape[0] != total_time:
        raise ValueError("package_reduction_series length must equal total_time")
    if is_boss_package_series is None:
        is_boss_package_series = np.zeros(total_time, dtype=bool)
    is_boss_package_series = np.asarray(is_boss_package_series, dtype=bool)
    package_reduction = package_reduction_series.copy()
    cooldown_remaining = np.zeros(total_time, dtype=float)
    active_remaining = np.zeros(total_time, dtype=float)
    remaining_active_time = np.zeros(total_time, dtype=float)
    is_active = np.zeros(total_time, dtype=bool)
    wave_number = np.zeros(total_time, dtype=int)
    cooldown_remaining_val = float(uw_cooldown)
    active_remaining_val = 0.0
    for t in timeline:
        pr = float(package_reduction_series[t])
        wave_number[t] = int(t / wave_time) + 1
        cooldown_remaining_val -= 1
        if pr > 0:
            cooldown_remaining_val -= pr
        activation_occurred = False
        if can_queue:
            while cooldown_remaining_val <= 0:
                activation_occurred = True
                active_remaining_val += uw_duration
                cooldown_remaining_val += uw_cooldown
        else:
            if cooldown_remaining_val <= 0:
                activation_occurred = True
                active_remaining_val = uw_duration
                cooldown_remaining_val += uw_cooldown
        if activation_occurred:
            cooldown_remaining[t] = 0.0
        else:
            cooldown_remaining[t] = max(0.0, cooldown_remaining_val)
        active_remaining[t] = active_remaining_val
        remaining_active_time[t] = active_remaining_val
        is_active[t] = active_remaining_val > 0
        if active_remaining_val > 0:
            active_remaining_val = max(0.0, active_remaining_val - 1)
    perma_result = bool(np.all(is_active))
    if return_df:
        return pd.DataFrame({
            "t": timeline,
            "wave_number": wave_number,
            "package_reduction": package_reduction,
            "is_boss_package": is_boss_package_series,
            "cooldown_remaining": cooldown_remaining,
            "active_remaining": active_remaining,
            "remaining_active_time": remaining_active_time,
            "is_active": is_active,
        })
    return {
        "perma_result": perma_result,
        "perma_timeline": timeline.tolist(),
        "perma_active": is_active.tolist(),
        "perma_remaining_active_time": remaining_active_time.tolist(),
    }

def _calculate_uptime_downtime_stats(df: pd.DataFrame) -> Dict[str, float | None]:
    def run_stats(mask: np.ndarray, value: bool) -> tuple[float, float, float]:
        # Returns (mean, min, max) for runs of value in mask
        changes = np.concatenate(([True], mask[1:] != mask[:-1], [True]))
        change_indices = np.where(changes)[0]
        run_lengths = np.diff(change_indices)
        run_values = mask[change_indices[:-1]]
        if len(run_lengths) > 0:
            run_lengths = run_lengths[:-1]
            run_values = run_values[:-1]
        runs = run_lengths[run_values == value] if len(run_values) > 0 else np.array([])
        if len(runs) > 0:
            return runs.mean(), runs.min(), runs.max()
        else:
            return 0, 0, 0

    active_mask = df["is_active"].to_numpy()
    total_seconds = len(df)
    uptime_seconds = active_mask.sum()
    downtime_seconds = total_seconds - uptime_seconds
    uptime_pct = (uptime_seconds / total_seconds * 100) if total_seconds > 0 else 0
    downtime_pct = (downtime_seconds / total_seconds * 100) if total_seconds > 0 else 0

    avg_uptime, min_uptime, max_uptime = run_stats(active_mask, True)
    avg_downtime, min_downtime, max_downtime = run_stats(active_mask, False)

    if min_downtime == max_downtime == avg_downtime and avg_downtime > 0:
        min_downtime = None
        max_downtime = None

    activation_starts = np.where((active_mask) & (np.concatenate(([True], ~active_mask[:-1]))))[0]
    avg_activation_interval = None
    if len(activation_starts) > 1:
        intervals = np.diff(activation_starts)
        avg_activation_interval = intervals.mean()

    return {
        "uptime_pct": uptime_pct,
        "downtime_pct": downtime_pct,
        "avg_uptime": avg_uptime,
        "max_downtime": max_downtime,
        "min_downtime": min_downtime,
        "avg_downtime": avg_downtime,
        "avg_activation_interval": avg_activation_interval,
    }

def _downsample_for_plot(df: pd.DataFrame, max_points: int = 600) -> pd.DataFrame:
    n = len(df)
    if n <= max_points:
        return df
    keep_mask = np.zeros(n, dtype=bool)
    keep_mask[0] = True
    keep_mask[-1] = True
    keep_mask |= (df["package_reduction"] > 0).to_numpy()
    wave_changes = df["wave_number"].diff().fillna(0) != 0
    keep_mask |= wave_changes.to_numpy()
    active_changes = df["is_active"].astype(int).diff().fillna(0) != 0
    keep_mask |= active_changes.to_numpy()
    kept_count = keep_mask.sum()
    if kept_count < max_points:
        step = max(1, n // (max_points - kept_count))
        keep_mask[::step] = True
    return df[keep_mask].copy()

def _make_uw_figure(df: pd.DataFrame, title: str, uw_color: str = "#00B0F6", show_cooldown: bool = True) -> go.Figure:
    df_plot = _downsample_for_plot(df, max_points=800)
    package_mask = df_plot["package_reduction"].to_numpy() > 0
    package_times = df_plot.loc[package_mask, "t"].to_numpy()
    package_values = df_plot.loc[package_mask, "package_reduction"].to_numpy()
    uptime_y = np.where(df_plot["is_active"], df_plot["remaining_active_time"], 0)
    downtime_y = -df_plot["cooldown_remaining"]
    stats = _calculate_uptime_downtime_stats(df)
    y_top = float(df_plot["remaining_active_time"].max())
    package_y = np.full_like(package_times, y_top, dtype=float)
    fig = go.Figure()
    fill_color = uw_color + "4D" if len(uw_color) == 7 else uw_color
    fig.add_trace(go.Scatter(x=df_plot["t"], y=uptime_y, mode="lines", name="Uptime", fill="tozeroy", line=dict(color=uw_color, width=1), fillcolor=fill_color, hovertemplate="t=%{x}s<br>time=%{y}s<extra></extra>"))
    if show_cooldown:
        fig.add_trace(go.Scatter(x=df_plot["t"], y=downtime_y, mode="lines", name="Downtime", line=dict(color="#888", width=1), hovertemplate="t=%{x}s<br>time=%{y}s<extra></extra>"))
    inactive_mask = ~df_plot["is_active"].to_numpy()
    if inactive_mask.any():
        t_vals = df_plot["t"].to_numpy()
        y_zeros = np.zeros(len(t_vals))
        inactive_y = np.where(inactive_mask, y_zeros, np.nan)
        fig.add_trace(go.Scatter(x=t_vals, y=inactive_y, mode="lines", name="No Activation", line=dict(color="#DC2626", width=4), hovertemplate="t=%{x}s<br>NO ACTIVATION<extra></extra>"))
    if show_cooldown:
        for i, pkg_time in enumerate(package_times):
            pkg_idx = np.where(df_plot["t"] == pkg_time)[0]
            if len(pkg_idx) > 0:
                idx = pkg_idx[0]
                pkg_reduction = package_values[i]
                cooldown_after = df_plot.iloc[idx]["cooldown_remaining"]
                cooldown_before = cooldown_after + pkg_reduction
                y_after = -cooldown_after
                y_before = -cooldown_before
                fig.add_trace(go.Scatter(x=[pkg_time, pkg_time], y=[y_before, y_after], mode="lines", line=dict(color="#90EE90", width=2), showlegend=False, hoverinfo="skip"))
    is_boss_at_packages = df_plot.loc[package_mask, "is_boss_package"].to_numpy()
    regular_mask = ~is_boss_at_packages
    if regular_mask.any():
        fig.add_trace(go.Scatter(x=package_times[regular_mask], y=package_y[regular_mask], mode="markers", name="Package", marker=dict(size=6, color="#6495ED"), customdata=package_values[regular_mask], hovertemplate="t=%{x}s<br>reduction=%{customdata}s<extra></extra>"))
    boss_mask = is_boss_at_packages
    if boss_mask.any():
        fig.add_trace(go.Scatter(x=package_times[boss_mask], y=package_y[boss_mask], mode="markers", name="Boss Package", marker=dict(size=10, color="#FFD700", symbol="star"), customdata=package_values[boss_mask], hovertemplate="t=%{x}s<br>Boss Package<br>reduction=%{customdata}s<extra></extra>"))
    fig.update_layout(template="plotly_dark", title=title, xaxis_title="t (seconds)", xaxis=dict(range=[0, df["t"].max()]), yaxis_title="Time (s, +active/-cooldown)", legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5), margin=dict(l=50, r=100, t=50, b=80))
    return fig

def _make_sync_figure(results: Dict[str, pd.DataFrame], selected_uw: str | None = None) -> go.Figure:
    fig = go.Figure()
    uw_positions = {uw_name: i for i, uw_name in enumerate(UW_ORDER)}
    bar_height = 0.8
    selected_active = None
    if selected_uw and selected_uw in results:
        selected_active = np.asarray(results[selected_uw]["is_active"].values, dtype=bool)
    for uw_name in UW_ORDER:
        if uw_name in results:
            df = results[uw_name]
            y_position = uw_positions[uw_name]
            is_active = np.asarray(df["is_active"].values, dtype=bool)
            t_values = np.asarray(df["t"].values, dtype=float).flatten()
            is_active_padded = np.concatenate([np.array([False]), is_active, np.array([False])])
            diff = np.diff(is_active_padded.astype(int))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            for start_idx, end_idx in zip(starts, ends):
                t_start = float(t_values[start_idx]) if hasattr(t_values[start_idx], 'item') else t_values[start_idx]
                t_end = float(t_values[end_idx - 1]) + 1 if hasattr(t_values[end_idx - 1], 'item') else t_values[end_idx - 1] + 1
                if selected_active is not None:
                    segment_active = np.asarray(is_active[start_idx:end_idx])
                    segment_selected = np.asarray(selected_active[start_idx:end_idx])
                    both_active = segment_active & segment_selected
                    only_this_active = segment_active & ~segment_selected
                    if both_active.any():
                        seg_starts = np.where(np.diff(np.concatenate(([0], both_active.astype(int)))) == 1)[0]
                        seg_ends = np.where(np.diff(np.concatenate((both_active.astype(int), [0]))) == -1)[0]
                        for seg_start, seg_end in zip(seg_starts, seg_ends):
                            seg_t_start = float(t_values[start_idx + seg_start]) if hasattr(t_values[start_idx + seg_start], 'item') else t_values[start_idx + seg_start]
                            seg_t_end = float(t_values[start_idx + seg_end - 1]) + 1 if hasattr(t_values[start_idx + seg_end - 1], 'item') else t_values[start_idx + seg_end - 1] + 1
                            fig.add_shape(type="rect", x0=seg_t_start, x1=seg_t_end, y0=y_position - bar_height/2, y1=y_position + bar_height/2, fillcolor="#00B0F6", opacity=1.0, line_width=0)
                    if only_this_active.any():
                        seg_starts = np.where(np.diff(np.concatenate(([0], only_this_active.astype(int)))) == 1)[0]
                        seg_ends = np.where(np.diff(np.concatenate((only_this_active.astype(int), [0]))) == -1)[0]
                        for seg_start, seg_end in zip(seg_starts, seg_ends):
                            seg_t_start = float(t_values[start_idx + seg_start]) if hasattr(t_values[start_idx + seg_start], 'item') else t_values[start_idx + seg_start]
                            seg_t_end = float(t_values[start_idx + seg_end - 1]) + 1 if hasattr(t_values[start_idx + seg_end - 1], 'item') else t_values[start_idx + seg_end - 1] + 1
                            fig.add_shape(type="rect", x0=seg_t_start, x1=seg_t_end, y0=y_position - bar_height/2, y1=y_position + bar_height/2, fillcolor="#00B0F6", opacity=0.3, line_width=0)
                else:
                    fig.add_shape(type="rect", x0=t_start, x1=t_end, y0=y_position - bar_height/2, y1=y_position + bar_height/2, fillcolor="#00B0F6", opacity=1.0, line_width=0)
    fig.update_layout(template="plotly_dark", title="Sync Chart: UWs sync with {}".format(selected_uw if selected_uw else "(None)"), xaxis_title="t (seconds)", yaxis=dict(tickvals=list(uw_positions.values()), ticktext=list(uw_positions.keys()), range=[-0.5, len(UW_ORDER)-0.5]), yaxis_title="Ultimate Weapon", showlegend=False, margin=dict(l=50, r=100, t=50, b=80))
    return fig

# --- Static UW config DataFrame (must be defined before any function uses it) ---
UW_CONFIG_DF = pd.DataFrame([
    {"name": "Chrono Field", "base_cooldown": 100, "base_duration": 29, "color_hex": "#00FFFF", "can_queue": False},
    {"name": "Black Hole", "base_cooldown": 46, "base_duration": 32, "color_hex": "#9933FF", "can_queue": True},
    {"name": "Golden Tower", "base_cooldown": 170, "base_duration": 45, "color_hex": "#FF6600", "can_queue": True},
    {"name": "Death Wave", "base_cooldown": 170, "base_duration": 20, "color_hex": "#FF0000", "can_queue": False},
    {"name": "Golden Bot", "base_cooldown": 100, "base_duration": 26, "color_hex": "#FFD700", "can_queue": False},
    {"name": "Summon Guardian", "base_cooldown": 100, "base_duration": 30, "color_hex": "#C532CD", "can_queue": False},
    {"name": "Smart Missiles", "base_cooldown": 120, "base_duration": 15, "color_hex": "#00FF00", "can_queue": False},
    {"name": "Inner Land Mines", "base_cooldown": 130, "base_duration": 25, "color_hex": "#DC143C", "can_queue": False},
    {"name": "Poison Swamp", "base_cooldown": 140, "base_duration": 30, "color_hex": "#32CD32", "can_queue": False},
])

# Now that UW_CONFIG_DF is defined, define UW_ORDER
UW_ORDER = list(UW_CONFIG_DF['name'])


dash.register_page(
    __name__,
    path="/perma-calc-hybrid",
    name="UW PermaCalc (Hybrid)",
    order=8,
)


def get_uw_param_ids():
    ids = []
    for _, row in UW_CONFIG_DF.iterrows():
        uw = row['name']
        uw_id = uw.lower().replace(' ', '-')
        ids.append(f"uw-{uw_id}-cooldown")
        ids.append(f"uw-{uw_id}-duration")
    return ids

# --- Simulation callback ---
@callback(
    Output("uw-hybrid-simulation-store", "data"),
    Input("uw-hybrid-play-mode", "value"),
    Input("uw-hybrid-wa-card", "value"),
    Input("uw-hybrid-boss-waves", "value"),
    Input("uw-hybrid-package-after-boss", "value"),
    Input("uw-hybrid-cooldown-bc", "value"),
    Input("uw-hybrid-package-chance", "value"),
    Input("uw-hybrid-galaxy-compressor", "value"),
    Input("uw-hybrid-mvn", "value"),
    Input("uw-hybrid-perk-status", "value"),
    Input("uw-hybrid-random-seed", "data"),
    *(Input(i, "value") for i in get_uw_param_ids()),
)

def hybrid_calculate_simulations(
    play_mode, wa_card_level, boss_waves, package_after_boss_state, cooldown_bc_state, pkg_chance_pct, gc_tier, mvn_state,
    perk_status_value, random_seed,
    *uw_param_values
):
    # Map UW param values to names
    uw_param_ids = get_uw_param_ids()
    uw_param_map = dict(zip(uw_param_ids, uw_param_values))

    # Get WA card reduction
    wa_reduction = WA_Card.get(wa_card_level, 0.54)
    if play_mode == "Tournament":
        wave_cooldown = 4.5 * (1 - wa_reduction)
    else:
        wave_cooldown = 9 * (1 - wa_reduction)
    wave_time = Wave_Basetime + wave_cooldown

    package_after_boss = package_after_boss_state == "Unlocked"
    bc_cooldown = 10 if cooldown_bc_state == "Active" else 0
    gc_bonus = GC_EFFECTS.get(gc_tier.split()[0], 0.0) if gc_tier else 0.0
    pkg_chance = (pkg_chance_pct or 76) / 100.0

    # Perks logic (now controlled by checklist)
    perks_on = "show" in (perk_status_value or [])
    bh_perk_duration = Farming_Perks["Black_Hole_Duration"] if perks_on else 0
    dw_perk_duration = Farming_Perks["Death_Wave_Duration"] if perks_on else 0
    cf_perk_duration = Farming_Perks["Chrono_Field_Duration"] if perks_on else 0

    # Multiverse Nexus
    mvn_offset = MVN_EFFECTS.get(mvn_state.split()[0], None) if mvn_state else None

    # Create UW objects
    uws = create_uw_objects()

    # Set effective cooldown/duration for each UW (fix: set per-UW, not via locals)
    for uw in uws:
        uw_id = uw.name.lower().replace(' ', '-')
        cooldown = uw_param_map.get(f"uw-{uw_id}-cooldown", uw.base_cooldown)
        duration = uw_param_map.get(f"uw-{uw_id}-duration", uw.base_duration)
        # Apply BC bonus
        cooldown = (cooldown or uw.base_cooldown) + bc_cooldown
        # Apply perks
        if uw.name == "Black Hole":
            duration = (duration or uw.base_duration) + bh_perk_duration
        elif uw.name == "Death Wave":
            duration = (duration or uw.base_duration) + dw_perk_duration
        elif uw.name == "Chrono Field":
            duration = (duration or uw.base_duration) + cf_perk_duration
        # Set effective values directly
        uw.effective_cooldown = cooldown
        uw.effective_duration = duration

    # Apply MVN sync (after all effective_cooldown are set)
    bh = next(u for u in uws if u.name == "Black Hole")
    gt = next(u for u in uws if u.name == "Golden Tower")
    dw = next(u for u in uws if u.name == "Death Wave")
    if mvn_offset is not None:
        avg_cooldown = (bh.effective_cooldown + gt.effective_cooldown + dw.effective_cooldown) / 3
        synced_cooldown = avg_cooldown + mvn_offset
        bh.effective_cooldown = synced_cooldown
        gt.effective_cooldown = synced_cooldown
        dw.effective_cooldown = synced_cooldown

    # Simulate package reductions (shared for all UWs)
    reductions, is_boss = simulate_package_reductions(
        total_time=total_time,
        pkg_chance=pkg_chance,
        pkg_reduction=gc_bonus,
        wave_time=wave_time,
        boss_every_x=boss_waves or 10,
        seed=random_seed or 0,
        boss_package_enabled=package_after_boss,
    )

    # Run simulation for each UW
    for uw in uws:
        # Golden Bot and Summon Guardian: no GC bonus
        pkg_red = 0 if uw.name in ["Golden Bot", "Summon Guardian"] else gc_bonus
        pr_series = reductions if pkg_red > 0 else np.zeros(total_time, dtype=float)
        is_boss_arr = np.asarray(is_boss, dtype=bool)
        uw_df = calculate_uw_uptime(
            uw.effective_cooldown,
            uw.effective_duration,
            total_time=total_time,
            return_df=True,
            package_reduction_series=pr_series,
            is_boss_package_series=is_boss_arr,
            wave_time=wave_time,
            can_queue=uw.can_queue,
        )
        # Ensure uw_df is a DataFrame (for Pylance type checking)
        if not isinstance(uw_df, pd.DataFrame):
            uw_df = pd.DataFrame(uw_df)
        uw.set_df(uw_df)
        uw.set_stats(_calculate_uptime_downtime_stats(uw_df))

    # Prepare serializable results for UI
    results = {}
    for uw in uws:
        results[uw.name] = {
            "params": {
                "base_cooldown": uw.base_cooldown,
                "effective_cooldown": uw.effective_cooldown,
                "base_duration": uw.base_duration,
                "effective_duration": uw.effective_duration,
            },
            "stats": uw.stats,
            "df": uw.df.to_dict('list'),
        }
    return results

# --- Stat card outputs (one Output per stat card) ---
def make_stat_card_outputs():
    outputs = []
    for _, row in UW_CONFIG_DF.iterrows():
        uw = row['name']
        uw_id = uw.lower().replace(' ', '-')
        outputs.extend([
            Output(f'stats-{uw_id}-params', 'children'),
            Output(f'stats-{uw_id}-uptime', 'children'),
            Output(f'stats-{uw_id}-downtime', 'children'),
            Output(f'stats-{uw_id}-interval', 'children'),
            Output(f'stats-{uw_id}-card', 'style'),
        ])
    return outputs

@callback(
    *make_stat_card_outputs(),
    Input("uw-hybrid-simulation-store", "data"),
)
def update_stat_cards(sim_data):
    if not sim_data:
        return ["", "", "", "", {}] * len(UW_CONFIG_DF)
    outputs = []
    for _, row in UW_CONFIG_DF.iterrows():
        uw = row['name']
        uw_id = uw.lower().replace(' ', '-')
        params = sim_data[uw]["params"]
        stats = sim_data[uw]["stats"]
        # Format params
        if params['base_cooldown'] == params['effective_cooldown']:
            cd_text = f"CD: {params['base_cooldown']}s"
        else:
            cd_text = f"CD: {params['base_cooldown']}s → {params['effective_cooldown']}s"
        if params['base_duration'] == params['effective_duration']:
            dur_text = f"Dur: {params['base_duration']}s"
        else:
            dur_text = f"Dur: {params['base_duration']}s → {params['effective_duration']}s"
        params_text = f"{cd_text} | {dur_text}"
        # Format stats
        uptime_text = f"Uptime: {stats['uptime_pct']:.1f}%" if stats['uptime_pct'] is not None else "Uptime: N/A"
        downtime_text = f"Downtime: Avg: {stats['avg_downtime']:.1f}s" if stats['avg_downtime'] is not None else "Downtime: N/A"
        interval_text = f"Activation Interval: {stats['avg_activation_interval']:.1f}s" if stats['avg_activation_interval'] is not None else "Activation Interval: N/A"
        card_style = {'borderLeft': f'4px solid {row["color_hex"]}'}
        outputs.extend([params_text, uptime_text, downtime_text, interval_text, card_style])
    return outputs


# --- Dynamic UW class ---
@dataclass
class UltimateWeapon:
    name: str
    # Only dynamic fields are stored; all static metadata is always looked up from UW_CONFIG_DF
    effective_cooldown: float = 0.0
    effective_duration: float = 0.0
    stats: Dict[str, Any] = field(default_factory=dict)
    df: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def base_cooldown(self) -> float:
        row = UW_CONFIG_DF[UW_CONFIG_DF['name'] == self.name]
        return float(row['base_cooldown'].values[0]) if not row.empty else 0.0

    @property
    def base_duration(self) -> float:
        row = UW_CONFIG_DF[UW_CONFIG_DF['name'] == self.name]
        return float(row['base_duration'].values[0]) if not row.empty else 0.0

    @property
    def color_hex(self) -> str:
        row = UW_CONFIG_DF[UW_CONFIG_DF['name'] == self.name]
        return str(row['color_hex'].values[0]) if not row.empty else "#00B0F6"

    @property
    def can_queue(self) -> bool:
        row = UW_CONFIG_DF[UW_CONFIG_DF['name'] == self.name]
        return bool(row['can_queue'].values[0]) if not row.empty else False

    def update_effective(self, cooldown, duration):
        self.effective_cooldown = cooldown
        self.effective_duration = duration

    def set_stats(self, stats: Dict[str, Any]):
        self.stats = stats

    def set_df(self, df: pd.DataFrame):
        self.df = df

# --- Utility to create UW objects from config ---
def create_uw_objects():
    return [UltimateWeapon(name=row['name']) for _, row in UW_CONFIG_DF.iterrows()]


# --- Static config and class from above ---
# (UW_CONFIG_DF, UltimateWeapon, create_uw_objects)

# --- Layout (dynamically generated from UW_CONFIG_DF) ---
def make_uw_param_controls():
    controls = []
    for _, row in UW_CONFIG_DF.iterrows():
        uw = row['name']
        controls.append(
            dbc.Col([
                dbc.Label(f"{uw} Cooldown (s)"),
                dbc.Input(id=f"uw-{uw.lower().replace(' ', '-')}-cooldown", type="number", value=row['base_cooldown'], step=1),
            ], width=2)
        )
        controls.append(
            dbc.Col([
                dbc.Label(f"{uw} Duration (s)"),
                dbc.Input(id=f"uw-{uw.lower().replace(' ', '-')}-duration", type="number", value=row['base_duration'], step=1),
            ], width=2)
        )
    return controls

def make_uw_stat_cards():
    cards = []
    for _, row in UW_CONFIG_DF.iterrows():
        uw = row['name']
        uw_id = uw.lower().replace(' ', '-')
        cards.append(
            dbc.Col([
                dbc.Card([dbc.CardBody([
                    html.H6(uw, className='card-title', style={'marginBottom': '0.5rem'}),
                    html.Div(id=f'stats-{uw_id}-params', style={'color': '#6c757d', 'fontSize': '0.8rem', 'marginBottom': '0.5rem'}),
                    html.Div(id=f'stats-{uw_id}-uptime', style={'fontSize': '1rem', 'marginBottom': '0.25rem'}),
                    html.Div(id=f'stats-{uw_id}-downtime', style={'color': '#999', 'fontSize': '0.85rem', 'marginBottom': '0.25rem'}),
                    html.Div(id=f'stats-{uw_id}-interval', style={'color': '#999', 'fontSize': '0.85rem'}),
                ])], id=f'stats-{uw_id}-card', className='mb-2'),
            ], width=12, md=6, lg=4)
        )
    return cards

layout = html.Div([
    html.H1("UW PermaCalc Hybrid", className="page-title"),
    dcc.Store(id="uw-hybrid-simulation-store"),
    dcc.Store(id="uw-hybrid-random-seed", data=0),

    dbc.Card(
        dbc.CardBody([
            html.H4("Global Settings", className="mb-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Play Mode"),
                    dcc.Dropdown(
                        id="uw-hybrid-play-mode",
                        options=[{"label": "Farming", "value": "Farming"}, {"label": "Tournament", "value": "Tournament"}],
                        value="Farming", clearable=False),
                ], width=2),
                dbc.Col([
                    dbc.Label("Wave Accelerator Card"),
                    dcc.Dropdown(
                        id="uw-hybrid-wa-card",
                        options=[{"label": s, "value": s} for s in ["None", "1 Star", "2 Star", "3 Star", "4 Star", "5 Star", "6 Star", "7 Star"]],
                        value="7 Star", clearable=False),
                ], width=2),
                dbc.Col([
                    dbc.Label("Boss Every X Waves"),
                    dcc.Dropdown(
                        id="uw-hybrid-boss-waves",
                        options=[{"label": str(i), "value": i} for i in range(5, 11)],
                        value=10, clearable=False),
                ], width=2),
                dbc.Col([
                    dbc.Label("Package After Boss"),
                    dcc.Dropdown(
                        id="uw-hybrid-package-after-boss",
                        options=[{"label": "None", "value": "None"}, {"label": "Unlocked", "value": "Unlocked"}],
                        value="Unlocked", clearable=False),
                ], width=2),
                dbc.Col([
                    dbc.Label("UW Cooldown BC"),
                    dcc.Dropdown(
                        id="uw-hybrid-cooldown-bc",
                        options=[{"label": "Inactive", "value": "Inactive"}, {"label": "Active (+10s)", "value": "Active"}],
                        value="Inactive", clearable=False),
                ], width=2),
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Package Chance (%)"),
                    dbc.Input(id="uw-hybrid-package-chance", type="number", value=76, min=0, max=100, step=1),
                ], width=2),
                dbc.Col([
                    dbc.Label("Galaxy Compressor"),
                    dcc.Dropdown(
                        id="uw-hybrid-galaxy-compressor",
                        options=[{"label": l, "value": l} for l in ["None", "Epic (+10s)", "Legendary (+13s)", "Mythic (+17s)", "Ancestral (+20s)"]],
                        value="Mythic (+17s)", clearable=False),
                ], width=2),
                dbc.Col([
                    dbc.Label("Multiverse Nexus"),
                    dcc.Dropdown(
                        id="uw-hybrid-mvn",
                        options=[{"label": l, "value": l} for l in ["None", "Epic (+20s)", "Legendary (+10s)", "Mythic (+1s)", "Ancestral (-10s)"]],
                        value="None", clearable=False),
                ], width=2),
            ], className="mt-3"),
        ]), className="mb-4"
    ),

    dbc.Card(
        dbc.CardBody([
            html.H4("UW Parameters", className="mb-3"),
            dbc.Row(make_uw_param_controls()),
        ]), className="mb-4"
    ),

    html.H4("UW Statistics", className="mb-3 mt-4"),
    dbc.Row(make_uw_stat_cards(), className="mb-4"),


    # Display Options, Perks Toggle, Re-randomize
    dbc.Row([
        dbc.Col([
            html.Label('Display Options:', className='mb-2'),
            dbc.Checklist(
                id='uw-hybrid-show-cooldown',
                options=[{'label': ' Show Cooldown', 'value': 'show'}],
                value=[],
                switch=True,
                className='mt-4'
            ),
        ], width=12, md=3, lg=2),
        dbc.Col([
            html.Label('Perks On/Off:', className='mb-2'),
            dbc.Checklist(
                id='uw-hybrid-perk-status',
                options=[{'label': ' Perks On/off', 'value': 'show'}],
                value=['show'],
                switch=True,
                className='mt-4'
            ),
        ], width=12, md=3, lg=2),
        dbc.Col([
            html.Label('Re-randomize Packages:', className='mb-2'),
            dbc.Button(
                "Re-randomize",
                id="uw-hybrid-rerandomize-btn",
                color="primary",
                size="sm",
                className="mt-4"
            ),
        ], width=12, md=2, lg=2),
    ], className="mb-2"),

    # Graphs
    dcc.Dropdown(
        id='uw-hybrid-detail-selector',
        options=[{'label': uw, 'value': uw} for uw in UW_CONFIG_DF['name']],
        value='Golden Bot',
        clearable=False,
        className='mb-3',
    ),
    dcc.Graph(id="uw-hybrid-detail-graph", config={"displayModeBar": False}, className="mb-4"),

    html.H2("Sync Chart"),
    html.P("E.g., [Golden Tower (Sync: 30%)] means Golden Tower is active 30% of the time when the selected UW is active."),
    dcc.Graph(id="uw-hybrid-graph-sync", config={"displayModeBar": False}),
])

# --- Re-randomize callback (for completeness, but not used in sim yet) ---
@callback(
    Output("uw-hybrid-random-seed", "data"),
    Input("uw-hybrid-rerandomize-btn", "n_clicks"),
    prevent_initial_call=True,
)
def update_random_seed(n_clicks):
    import time
    return int(time.time() * 1000000) % 1000000

# --- Graph and sync chart callbacks ---
@callback(
    Output("uw-hybrid-detail-graph", "figure"),
    Input("uw-hybrid-simulation-store", "data"),
    Input("uw-hybrid-detail-selector", "value"),
    Input("uw-hybrid-show-cooldown", "value"),
)
def update_detail_graph(sim_data, selected_uw, show_cooldown):
    if not sim_data or selected_uw not in sim_data:
        return go.Figure()
    df_dict = sim_data[selected_uw]["df"]
    df = pd.DataFrame(df_dict)
    # Get color_hex from UW_CONFIG_DF
    uw_row = UW_CONFIG_DF[UW_CONFIG_DF['name'] == selected_uw]
    uw_color = uw_row['color_hex'].values[0] if not uw_row.empty else "#00B0F6"
    show_cd = 'show' in show_cooldown if show_cooldown else False
    return _make_uw_figure(df, selected_uw, uw_color, show_cooldown=show_cd)

@callback(
    Output("uw-hybrid-graph-sync", "figure"),
    Input("uw-hybrid-simulation-store", "data"),
    Input("uw-hybrid-detail-selector", "value"),
)
def update_sync_chart(sim_data, selected_uw):
    if not sim_data:
        return go.Figure()
    # Reconstruct all UW DataFrames
    results = {uw: pd.DataFrame(sim_data[uw]["df"]) for uw in sim_data if "df" in sim_data[uw]}
    return _make_sync_figure(results, selected_uw=selected_uw)
