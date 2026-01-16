import numpy as np
import pandas as pd
from typing import Any, Dict, Literal, Union, overload

import dash
from dash import callback, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go


#definitions:
# perma_remaining_active_time = list of remaining active time given current time in timeline
# perma_active = list of booleans if remaining active time is larger than 0
# package_reduction = list of package reduction values based on package_chance and package_after_boss for each wave

# ============================================================================
# SECTION 1: VARIABLES
# ============================================================================

# Global simulation settings
package_after_boss = True  # whether package reduction applies after boss
total_time = 3600  # total simulation time in seconds
cooldown_bc = 10  # Battle condition increasing all UW cooldowns by 10 seconds

# Game mode settings as per https://mvn.thetower.tools/
WAVE_TIME_MODES = {
    "Tournament": 28.070,
    "Farming": 30.140
}
WA_Card ={
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
# Wave cooldown is calculated dynamically in the callback based on:
# Tournament: 4.5 * (1 - WA_reduction)
# Farming: 9 * (1 - WA_reduction)

# Farming perks (additional durations when in Farming mode)
Farming_Perks = {
    "Black_Hole_Duration": 12,
    "Death_Wave_Duration": 4,
    "Chrono_Field_Duration": 5,
}

# Galaxy Compressor effects (package reduction bonuses)
GC_EFFECTS = {
    'None': 0.0,
    'Epic': 10.0,
    'Legendary': 13.0,
    'Mythic': 17.0,
    'Ancestral': 20.0,
}

# Multiverse Nexus effects (cooldown synchronization adjustment)
MVN_EFFECTS = {
    'None': None,  # MVN not active
    'Epic': 20.0,
    'Legendary': 10.0,
    'Mythic': 1.0,
    'Ancestral': -10.0,
}

# Ultimate Weapon configuration
UW_CONFIG = {
    "Black Hole": {
        "cooldown": 46,
        "duration": 34,
        "color_hex": "#9933FF",
        "color_rgba": "rgba(153, 51, 255, 0.8)",
        "color_rgba_light": "rgba(153, 51, 255, 0.6)",
        "can_queue": True,
    },
    "Golden Tower": {
        "cooldown": 170,
        "duration": 45,
        "color_hex": "#FF6600",
        "color_rgba": "rgba(255, 102, 0, 0.8)",
        "color_rgba_light": "rgba(255, 102, 0, 0.6)",
        "can_queue": True,
    },
    "Death Wave": {
        "cooldown": 170,
        "duration": 20,
        "color_hex": "#FF0000",
        "color_rgba": "rgba(255, 0, 0, 0.8)",
        "color_rgba_light": "rgba(255, 0, 0, 0.6)",
        "can_queue": False,
    },
    "Chrono Field": {
        "cooldown": 180,
        "duration": 28,
        "color_hex": "#00FFFF",
        "color_rgba": "rgba(0, 255, 255, 0.8)",
        "color_rgba_light": "rgba(0, 255, 255, 0.6)",
        "can_queue": False,
    },
    "Golden Bot": {
        "cooldown": 100,
        "duration": 24.5,
        "color_hex": "#FFD700",
        "color_rgba": "rgba(255, 215, 0, 0.8)",
        "color_rgba_light": "rgba(255, 215, 0, 0.6)",
        "can_queue": False,
    },
    "Smart Missiles": {
        "cooldown": 120,
        "duration": 15,
        "color_hex": "#00FF00",
        "color_rgba": "rgba(0, 255, 0, 0.8)",
        "color_rgba_light": "rgba(0, 255, 0, 0.6)",
        "can_queue": False,
    },
    "Inner Land Mines": {
        "cooldown": 130,
        "duration": 25,
        "color_hex": "#DC143C",
        "color_rgba": "rgba(220, 20, 60, 0.8)",
        "color_rgba_light": "rgba(220, 20, 60, 0.6)",
        "can_queue": False,
    },
    "Poison Swamp": {
        "cooldown": 140,
        "duration": 30,
        "color_hex": "#32CD32",
        "color_rgba": "rgba(50, 205, 50, 0.8)",
        "color_rgba_light": "rgba(50, 205, 50, 0.6)",
        "can_queue": False,
    },
}

# UW order for consistent display
UW_ORDER = ["Black Hole", "Golden Tower", "Death Wave", "Chrono Field", "Golden Bot", "Smart Missiles", "Inner Land Mines", "Poison Swamp"]


# ============================================================================
# SECTION 2: FUNCTIONS
# ============================================================================


def is_package(
    time: int, 
    rng: np.random.Generator, 
    pkg_chance: float, 
    pkg_reduction: float,
    wave_time: float,
    boss_every_x: int,
    boss_package_enabled: bool = True
) -> tuple[float, bool]:
    """Return package reduction for a given second and whether it's a boss package.

    Notes:
      - Boss check is evaluated first so boss packages aren't shadowed by wave packages at t=0.
    
    Returns:
      - (package_reduction, is_boss_package)
    """
    package_reduction = 0.0
    is_boss_package = False
    boss_time = wave_time * boss_every_x  # Boss package appears every boss_every_x waves
    
    if time % boss_time < 1:
        package_reduction = pkg_reduction if boss_package_enabled else 0.0
        is_boss_package = package_reduction > 0
    elif time % wave_time < 1:
        package_reduction = pkg_reduction if rng.random() < pkg_chance else 0.0
    return package_reduction, is_boss_package


def simulate_package_reductions(
    total_time: int, 
    pkg_chance: float, 
    pkg_reduction: float,
    wave_time: float,
    boss_every_x: int,
    seed: int = 0,
    boss_package_enabled: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a shared package reduction series for all UWs.
    
    Returns:
      - (reductions, is_boss_package_array)
    """
    rng = np.random.default_rng(seed)
    reductions = np.zeros(total_time, dtype=float)
    is_boss = np.zeros(total_time, dtype=bool)
    for t in range(total_time):
        reductions[t], is_boss[t] = is_package(t, rng, pkg_chance, pkg_reduction, wave_time, boss_every_x, boss_package_enabled)
    return reductions, is_boss

@overload
def calculate_uw_uptime(
    uw_cooldown: float,
    uw_duration: float,
    total_time: int,
    return_df: Literal[True],
    package_reduction_series: np.ndarray | None = None,
    is_boss_package_series: np.ndarray | None = None,
    wave_time: float = 30.140,
) -> pd.DataFrame: ...


@overload
def calculate_uw_uptime(
    uw_cooldown: float,
    uw_duration: float,
    total_time: int,
    return_df: Literal[False] = False,
    package_reduction_series: np.ndarray | None = None,
    is_boss_package_series: np.ndarray | None = None,
    wave_time: float = 30.140,
) -> Dict[str, Any]: ...


def calculate_uw_uptime(
    uw_cooldown: float,
    uw_duration: float,
    total_time: int,
    return_df: bool = False,
    package_reduction_series: np.ndarray | None = None,
    is_boss_package_series: np.ndarray | None = None,
    wave_time: float = 30.140,
    can_queue: bool = True,
) -> Union[pd.DataFrame, Dict[str, Any]]:
    """Calculate UW uptime over total_time given cooldown and duration.

    Args:
        uw_cooldown: Base cooldown in seconds.
        uw_duration: Active duration in seconds.
        total_time: Simulation length (seconds).
        return_df: If True, return a pandas DataFrame with per-second details.
        package_reduction_series: Optional precomputed package reductions (shared across UWs).
        wave_time: Time per wave in seconds.
        can_queue: If True, activations stack duration (BH, GT). If False, only reset cooldown.

    Returns:
        If return_df is False: dict with summary + series (lists).
        If return_df is True: pandas DataFrame with per-second columns.
    """
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

    # Continuous cooldown + stacking activation model:
    # - Cooldown counts down every second, regardless of whether UW is active.
    # - A package reduces the *remaining* cooldown.
    # - When cooldown reaches <= 0, UW activates and *adds* uw_duration to remaining active.
    # - Cooldown carryover: if reduction overshoots (e.g. 5 remaining, pr=10),
    #   the leftover reduces the next cooldown: new cooldown becomes uw_cooldown - 5 (=25).
    #
    # Implementation detail: we record state at the start of each second, after applying
    # package reductions and processing any activations triggered at that boundary.
    # 
    # Order of operations (matching SimulationClass.py):
    # 1. Decrement timer
    # 2. Apply package reduction  
    # 3. Check activation
    # 4. Record state
    cooldown_remaining_val = float(uw_cooldown)
    active_remaining_val = 0.0  # UW starts inactive

    for t in timeline:
        pr = float(package_reduction_series[t])
        
        # Calculate wave number
        wave_number[t] = int(t / wave_time) + 1

        # Step 1: Decrement cooldown timer (normal time passage - runs always)
        cooldown_remaining_val -= 1

        # Step 2: Apply package reduction
        if pr > 0:
            cooldown_remaining_val -= pr

        # Step 3: Trigger activation(s) if cooldown hits or overshoots 0
        # The overshoot amount carries over to reduce the next cooldown
        if can_queue:
            # BH and GT: Queue activations (stack duration)
            while cooldown_remaining_val <= 0:
                active_remaining_val += uw_duration
                cooldown_remaining_val += uw_cooldown  # Add full cooldown duration back
        else:
            # All other UWs: Single activation, reset cooldown
            if cooldown_remaining_val <= 0:
                active_remaining_val = uw_duration  # Reset to full duration
                cooldown_remaining_val += uw_cooldown

        # Step 4: Record state at this second (post-decrement, post-package, post-activation)
        cooldown_remaining[t] = max(0.0, cooldown_remaining_val)
        active_remaining[t] = active_remaining_val
        remaining_active_time[t] = active_remaining_val
        is_active[t] = active_remaining_val > 0

        # Decrement active time if UW is active
        if active_remaining_val > 0:
            active_remaining_val = max(0.0, active_remaining_val - 1)

    perma_result = bool(np.all(is_active))

    if return_df:
        return pd.DataFrame(
            {
                "t": timeline,
                "wave_number": wave_number,
                "package_reduction": package_reduction,
                "is_boss_package": is_boss_package_series,
                "cooldown_remaining": cooldown_remaining,
                "active_remaining": active_remaining,
                "remaining_active_time": remaining_active_time,
                "is_active": is_active,
            }
        )

    return {
        "perma_result": perma_result,
        "perma_timeline": timeline.tolist(),
        "perma_active": is_active.tolist(),
        "perma_remaining_active_time": remaining_active_time.tolist(),
    }


# Register page for multi-page Dash app
dash.register_page(__name__, path="/perma-calc-new", name="UW PermaCalc (New)", order=6)


def _downsample_for_plot(df: pd.DataFrame, max_points: int = 600) -> pd.DataFrame:
    """Downsample DataFrame for faster plotting while preserving visual accuracy.
    
    Keeps all package events and important transitions, downsamples the rest.
    """
    n = len(df)
    if n <= max_points:
        return df
    
    # Always keep: first, last, package events, wave boundaries
    keep_mask = np.zeros(n, dtype=bool)
    keep_mask[0] = True
    keep_mask[-1] = True
    
    # Keep package events
    keep_mask |= (df["package_reduction"] > 0).to_numpy()
    
    # Keep wave boundaries
    wave_changes = df["wave_number"].diff().fillna(0) != 0
    keep_mask |= wave_changes.to_numpy()
    
    # Keep activation/deactivation transitions
    active_changes = df["is_active"].astype(int).diff().fillna(0) != 0
    keep_mask |= active_changes.to_numpy()
    
    # Add evenly spaced points to fill in gaps
    kept_count = keep_mask.sum()
    if kept_count < max_points:
        # Add more points evenly distributed
        step = max(1, n // (max_points - kept_count))
        keep_mask[::step] = True
    
    return df[keep_mask].copy()


def _calculate_uptime_downtime_stats(df: pd.DataFrame) -> Dict[str, Union[float, None]]:
    """Calculate uptime and downtime statistics from DataFrame based on boolean is_active column.
    
    Calculates statistics on consecutive runs of active/inactive periods.
    Only includes complete periods (excludes partial period at end of dataframe).
    
    Returns:
        Dictionary with uptime_pct, downtime_pct, min/max/avg for uptime/downtime period lengths,
        and avg_activation_interval (average time between activation starts).
    """
    active_mask = df["is_active"].to_numpy()
    total_seconds = len(df)
    uptime_seconds = active_mask.sum()
    downtime_seconds = total_seconds - uptime_seconds
    
    uptime_pct = (uptime_seconds / total_seconds * 100) if total_seconds > 0 else 0
    downtime_pct = (downtime_seconds / total_seconds * 100) if total_seconds > 0 else 0
    
    # Find consecutive runs of True (uptime) and False (downtime)
    # Detect changes in the boolean array
    changes = np.concatenate(([True], active_mask[1:] != active_mask[:-1], [True]))
    change_indices = np.where(changes)[0]
    
    # Calculate run lengths and their corresponding values
    run_lengths = np.diff(change_indices)
    run_values = active_mask[change_indices[:-1]]
    
    # Exclude the last run if it's incomplete (ends at the dataframe boundary)
    # The last change_index is always len(df), so the last run might be cut off
    if len(run_lengths) > 0:
        # Remove the last run as it may be incomplete
        run_lengths = run_lengths[:-1]
        run_values = run_values[:-1]
    
    # Separate into uptime and downtime runs
    uptime_runs = run_lengths[run_values] if len(run_values) > 0 else np.array([])
    downtime_runs = run_lengths[~run_values] if len(run_values) > 0 else np.array([])
    
    # Calculate statistics for uptime periods
    avg_uptime = uptime_runs.mean() if len(uptime_runs) > 0 else 0
    min_uptime = uptime_runs.min() if len(uptime_runs) > 0 else 0
    max_uptime = uptime_runs.max() if len(uptime_runs) > 0 else 0
    
    # Calculate statistics for downtime periods
    avg_downtime = downtime_runs.mean() if len(downtime_runs) > 0 else 0
    min_downtime = downtime_runs.min() if len(downtime_runs) > 0 else 0
    max_downtime = downtime_runs.max() if len(downtime_runs) > 0 else 0
    
    # If min/max/avg are the same for downtime, only keep avg
    if min_downtime == max_downtime == avg_downtime and avg_downtime > 0:
        min_downtime = None
        max_downtime = None
    
    # Calculate average time between activation starts
    # Find indices where activation starts (transition from False to True)
    activation_starts = []
    for i in range(len(active_mask)):
        if active_mask[i] and (i == 0 or not active_mask[i-1]):
            activation_starts.append(i)
    
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


def _make_uw_figure(df: pd.DataFrame, title: str, uw_color: str = "rgba(0, 176, 246, 0.8)", show_cooldown: bool = True) -> go.Figure:
    # Downsample for faster rendering while keeping important events
    df_plot = _downsample_for_plot(df, max_points=800)
    
    package_mask = df_plot["package_reduction"].to_numpy() > 0
    package_times = df_plot.loc[package_mask, "t"].to_numpy()
    package_values = df_plot.loc[package_mask, "package_reduction"].to_numpy()
    
    # Create uptime and downtime arrays directly from DataFrame columns
    # Uptime: use remaining_active_time when active, 0 otherwise
    uptime_y = np.where(
        df_plot["is_active"],
        df_plot["remaining_active_time"],
        0
    )
    
    # Cooldown: always show -cooldown_remaining (cooldown runs continuously)
    downtime_y = -df_plot["cooldown_remaining"]
    
    # Calculate uptime/downtime statistics using helper function (on full dataset)
    stats = _calculate_uptime_downtime_stats(df)
    uptime_pct = stats["uptime_pct"]
    avg_uptime = stats["avg_uptime"]
    max_downtime = stats["max_downtime"]
    min_downtime = stats["min_downtime"]
    avg_downtime = stats["avg_downtime"]

    # Put package markers at the top
    y_top = float(df_plot["remaining_active_time"].max())
    package_y = np.full_like(package_times, y_top, dtype=float)

    fig = go.Figure()
    
    # Uptime trace (area chart)
    # Extract alpha from uw_color and create fill color
    fill_color = uw_color.replace("0.8)", "0.3)")
    fig.add_trace(
        go.Scatter(
            x=df_plot["t"],
            y=uptime_y,
            mode="lines",
            name="Uptime",
            fill="tozeroy",
            line=dict(color=uw_color, width=1),
            fillcolor=fill_color,
            hovertemplate="t=%{x}s<br>time=%{y}s<extra></extra>",
        )
    )
    
    # Downtime trace (grey line chart) - only show if show_cooldown is True
    if show_cooldown:
        fig.add_trace(
            go.Scatter(
                x=df_plot["t"],
                y=downtime_y,
                mode="lines",
                name="Downtime",
                line=dict(color="rgba(128, 128, 128, 0.8)", width=1),
                hovertemplate="t=%{x}s<br>time=%{y}s<extra></extra>",
            )
        )
    
    # No activation periods (thick red line at y=0)
    # Find continuous segments where is_active is False
    inactive_mask = ~df_plot["is_active"].to_numpy()
    if inactive_mask.any():
        # Create segments for inactive periods
        t_vals = df_plot["t"].to_numpy()
        y_zeros = np.zeros(len(t_vals))
        
        # Mark inactive times with 0, active times with NaN to create gaps
        inactive_y = np.where(inactive_mask, y_zeros, np.nan)
        
        fig.add_trace(
            go.Scatter(
                x=t_vals,
                y=inactive_y,
                mode="lines",
                name="No Activation",
                line=dict(color="rgba(220, 38, 38, 1.0)", width=4),
                hovertemplate="t=%{x}s<br>NO ACTIVATION<extra></extra>",
            )
        )
    
    # Add vertical segments for package reductions on cooldown (light green) - drawn BEFORE markers
    # Only show if show_cooldown is True
    if show_cooldown:
        for i, pkg_time in enumerate(package_times):
            pkg_idx = np.where(df_plot["t"] == pkg_time)[0]
            if len(pkg_idx) > 0:
                idx = pkg_idx[0]
                # Get the actual cooldown_remaining value at this point
                pkg_reduction = package_values[i]
                cooldown_after = df_plot.iloc[idx]["cooldown_remaining"]
                cooldown_before = cooldown_after + pkg_reduction
                
                # Plot as negative values (cooldown portion)
                y_after = -cooldown_after
                y_before = -cooldown_before
                
                fig.add_trace(
                    go.Scatter(
                        x=[pkg_time, pkg_time],
                        y=[y_before, y_after],
                        mode="lines",
                        line=dict(color="rgba(144, 238, 144, 0.8)", width=2),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
    
    # Package markers - split into boss and regular packages (drawn AFTER green lines)
    is_boss_at_packages = df_plot.loc[package_mask, "is_boss_package"].to_numpy()
    
    # Regular packages (cornflower blue)
    regular_mask = ~is_boss_at_packages
    if regular_mask.any():
        fig.add_trace(
            go.Scatter(
                x=package_times[regular_mask],
                y=package_y[regular_mask],
                mode="markers",
                name="Package",
                marker=dict(size=6, color="rgba(100, 149, 237, 1.0)"),
                customdata=package_values[regular_mask],
                hovertemplate="t=%{x}s<br>reduction=%{customdata}s<extra></extra>",
            )
        )
    
    # Boss packages (yellow stars, larger)
    boss_mask = is_boss_at_packages
    if boss_mask.any():
        fig.add_trace(
            go.Scatter(
                x=package_times[boss_mask],
                y=package_y[boss_mask],
                mode="markers",
                name="Boss Package",
                marker=dict(size=10, color="rgba(255, 215, 0, 1.0)", symbol="star"),
                customdata=package_values[boss_mask],
                hovertemplate="t=%{x}s<br>Boss Package<br>reduction=%{customdata}s<extra></extra>",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        title=title,
        xaxis_title="t (seconds)",
        xaxis=dict(range=[0, df["t"].max()]),
        yaxis_title="Time (s, +active/-cooldown)",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=50, r=100, t=50, b=80),
    )
    return fig


def _make_sync_figure(results: Dict[str, pd.DataFrame], selected_uw: str | None = None) -> go.Figure:
    """Create a swim lane chart showing active UWs as horizontal bars.
    
    If selected_uw is provided, shows segments synced with the selected UW as solid color,
    and unsynced segments as 30% transparent.
    """
    # DEBUG: Uncomment to troubleshoot sync chart rendering
    # print(f"[DEBUG] _make_sync_figure called with selected_uw={selected_uw}")
    fig = go.Figure()
    
    # Assign each UW a vertical position (swim lane)
    uw_positions = {uw_name: i for i, uw_name in enumerate(UW_ORDER)}
    bar_height = 0.8  # Height of each swim lane bar
    
    # Get the selected UW's active status for sync checking
    selected_active = None
    if selected_uw and selected_uw in results:
        selected_active = np.asarray(results[selected_uw]["is_active"].values, dtype=bool)
        # DEBUG: Uncomment to see active period counts
        # print(f"[DEBUG] Selected UW '{selected_uw}' active periods: {np.sum(selected_active)} seconds out of {len(selected_active)}")
    
    shapes = []
    
    for uw_name in UW_ORDER:
        if uw_name in results:
            df = results[uw_name]
            y_position = uw_positions[uw_name]
            
            # Find continuous active segments
            is_active = np.asarray(df["is_active"].values, dtype=bool)
            t_values = df["t"].values
            
            # Find start and end of each active segment
            # Add padding to detect edges
            is_active_padded = np.concatenate([np.array([False]), is_active, np.array([False])])
            diff = np.diff(is_active_padded.astype(int))
            
            starts = np.where(diff == 1)[0]  # Transitions from False to True
            ends = np.where(diff == -1)[0]   # Transitions from True to False
            
            # Create shapes for each active segment
            for start_idx, end_idx in zip(starts, ends):
                # end_idx points to first False, so actual end is end_idx - 1
                t_start = t_values[start_idx]
                t_end = t_values[end_idx - 1] + 1  # +1 to extend to end of that second
                
                # Determine if this segment is synced with selected UW
                if selected_active is not None:
                    # Split segment into synced and unsynced parts
                    segment_active = np.asarray(is_active[start_idx:end_idx])
                    segment_selected = np.asarray(selected_active[start_idx:end_idx])
                    
                    # Find where both are active (synced) vs only this UW is active (unsynced)
                    both_active = segment_active & segment_selected
                    only_this_active = segment_active & ~segment_selected
                    
                    # Create synced segments (solid color)
                    both_padded = np.concatenate([np.array([False]), both_active, np.array([False])])
                    both_diff = np.diff(both_padded.astype(int))
                    both_starts = np.where(both_diff == 1)[0]
                    both_ends = np.where(both_diff == -1)[0]
                    
                    for sync_start, sync_end in zip(both_starts, both_ends):
                        abs_start = start_idx + sync_start
                        abs_end = start_idx + sync_end - 1
                        shapes.append(
                            dict(
                                type="rect",
                                x0=t_values[abs_start],
                                x1=t_values[abs_end] + 1,
                                y0=y_position - bar_height/2,
                                y1=y_position + bar_height/2,
                                fillcolor=UW_CONFIG[uw_name]["color_hex"],
                                line=dict(width=0),
                                layer="below",
                            )
                        )
                    
                    # Create unsynced segments (30% transparent)
                    only_padded = np.concatenate([np.array([False]), only_this_active, np.array([False])])
                    only_diff = np.diff(only_padded.astype(int))
                    only_starts = np.where(only_diff == 1)[0]
                    only_ends = np.where(only_diff == -1)[0]
                    
                    for unsync_start, unsync_end in zip(only_starts, only_ends):
                        abs_start = start_idx + unsync_start
                        abs_end = start_idx + unsync_end - 1
                        # Convert hex to rgba with 30% opacity
                        hex_color = UW_CONFIG[uw_name]["color_hex"]
                        r = int(hex_color[1:3], 16)
                        g = int(hex_color[3:5], 16)
                        b = int(hex_color[5:7], 16)
                        rgba_color = f"rgba({r}, {g}, {b}, 0.3)"
                        
                        shapes.append(
                            dict(
                                type="rect",
                                x0=t_values[abs_start],
                                x1=t_values[abs_end] + 1,
                                y0=y_position - bar_height/2,
                                y1=y_position + bar_height/2,
                                fillcolor=rgba_color,
                                line=dict(width=0),
                                layer="below",
                            )
                        )
                else:
                    # No selection - show all as solid
                    shapes.append(
                        dict(
                            type="rect",
                            x0=t_start,
                            x1=t_end,
                            y0=y_position - bar_height/2,
                            y1=y_position + bar_height/2,
                            fillcolor=UW_CONFIG[uw_name]["color_hex"],
                            line=dict(width=0),
                            layer="below",
                        )
                    )
            
            # Add invisible trace for legend
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=10, color=UW_CONFIG[uw_name]["color_hex"], symbol="square"),
                    name=uw_name,
                    showlegend=True,
                )
            )
    
    # Calculate sync percentages for y-axis labels
    y_axis_labels = {}
    chart_title = "Sync Chart: Active UWs"
    
    if selected_active is not None and selected_uw:
        chart_title = f"Sync Chart: UWs sync with {selected_uw}"
        
        # Calculate selected UW's total active time (denominator for all sync percentages)
        selected_uw_active_seconds = np.sum(selected_active)
        
        for uw_name in UW_ORDER:
            if uw_name in results:
                uw_active = np.asarray(results[uw_name]["is_active"].values, dtype=bool)
                # Calculate sync: when selected UW is active, what % of that time is this UW also active?
                both_active = uw_active & selected_active
                sync_seconds = np.sum(both_active)
                
                if selected_uw_active_seconds > 0:
                    sync_percentage = (sync_seconds / selected_uw_active_seconds) * 100
                    y_axis_labels[uw_name] = f"{uw_name} (Sync: {sync_percentage:.1f}%)"
                else:
                    y_axis_labels[uw_name] = f"{uw_name} (Sync: 0%)"
            else:
                y_axis_labels[uw_name] = uw_name
    else:
        # No selection - just use UW names
        for uw_name in UW_ORDER:
            y_axis_labels[uw_name] = uw_name
    
    fig.update_layout(
        template="plotly_dark",
        title=chart_title,
        xaxis_title="t (seconds)",
        yaxis=dict(
            tickmode='array',
            tickvals=list(uw_positions.values()),
            ticktext=[y_axis_labels[uw] for uw in UW_ORDER],
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=False,
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)',
        ),
        shapes=shapes,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=150, r=50, t=50, b=80),  # Wider left margin for UW names
        showlegend=True,
        hovermode='x unified',
        uirevision=selected_uw,  # Force re-render when selection changes
    )
    return fig


# ============================================================================
# SECTION 3: DASH LAYOUT
# ============================================================================

layout = html.Div(
    [
        html.H1("UW PermaCalc (New)", className="page-title"),
        
        # Hidden storage for simulation results (to avoid recalculating on every interaction)
        dcc.Store(id="uw-simulation-store"),
        dcc.Store(id="uw-random-seed", data=0),
        
        # Global Controls
        dbc.Card(
            dbc.CardBody(
                [
                    html.H4("Global Settings", className="mb-3"),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Play Mode"),
                                    dcc.Dropdown(
                                        id="uw-play-mode",
                                        options=[
                                            {"label": "Farming", "value": "Farming"},
                                            {"label": "Tournament", "value": "Tournament"},
                                        ],
                                        value="Farming",
                                        clearable=False,
                                    ),
                                ],
                                width=2,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Wave Accelerator Card"),
                                    dcc.Dropdown(
                                        id="uw-wa-card",
                                        options=[
                                            {"label": "None", "value": "None"},
                                            {"label": "1 Star", "value": "1 Star"},
                                            {"label": "2 Star", "value": "2 Star"},
                                            {"label": "3 Star", "value": "3 Star"},
                                            {"label": "4 Star", "value": "4 Star"},
                                            {"label": "5 Star", "value": "5 Star"},
                                            {"label": "6 Star", "value": "6 Star"},
                                            {"label": "7 Star", "value": "7 Star"},
                                        ],
                                        value="7 Star",
                                        clearable=False,
                                    ),
                                ],
                                width=2,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Boss Every X Waves"),
                                    dcc.Dropdown(
                                        id="uw-boss-waves",
                                        options=[
                                            {"label": "5", "value": 5},
                                            {"label": "6", "value": 6},
                                            {"label": "7", "value": 7},
                                            {"label": "8", "value": 8},
                                            {"label": "9", "value": 9},
                                            {"label": "10", "value": 10}                                            
                                        ],
                                        value=10,
                                        clearable=False,
                                    ),
                                ],
                                width=2,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Package After Boss"),
                                    dcc.Dropdown(
                                        id="uw-package-after-boss",
                                        options=[
                                            {"label": "None", "value": "None"},
                                            {"label": "Unlocked", "value": "Unlocked"},
                                        ],
                                        value="Unlocked",
                                        clearable=False,
                                    ),
                                ],
                                width=2,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("UW Cooldown BC"),
                                    dcc.Dropdown(
                                        id="uw-cooldown-bc",
                                        options=[
                                            {"label": "Inactive", "value": "Inactive"},
                                            {"label": "Active (+10s)", "value": "Active"},
                                        ],
                                        value="Inactive",
                                        clearable=False,
                                    ),
                                ],
                                width=2,
                            ),
                        ],
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Package Chance (%)"),
                                    dbc.Input(
                                        id="uw-package-chance",
                                        type="number",
                                        value=76,
                                        min=0,
                                        max=100,
                                        step=1,
                                    ),
                                ],
                                width=2,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Galaxy Compressor"),
                                    dcc.Dropdown(
                                        id="uw-galaxy-compressor",
                                        options=[
                                            {"label": "None", "value": "None"},
                                            {"label": "Epic (+10s)", "value": "Epic"},
                                            {"label": "Legendary (+13s)", "value": "Legendary"},
                                            {"label": "Mythic (+17s)", "value": "Mythic"},
                                            {"label": "Ancestral (+20s)", "value": "Ancestral"},
                                        ],
                                        value="Mythic",
                                        clearable=False,
                                    ),
                                ],
                                width=2,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Multiverse Nexus"),
                                    dcc.Dropdown(
                                        id="uw-mvn",
                                        options=[
                                            {"label": "None", "value": "None"},
                                            {"label": "Epic (+20s)", "value": "Epic"},
                                            {"label": "Legendary (+10s)", "value": "Legendary"},
                                            {"label": "Mythic (+1s)", "value": "Mythic"},
                                            {"label": "Ancestral (-10s)", "value": "Ancestral"},
                                        ],
                                        value="None",
                                        clearable=False,
                                    ),
                                ],
                                width=2,
                            ),
                        ],
                        className="mt-3",
                    ),
                ]
            ),
            className="mb-4",
        ),
        
        # UW Parameter Controls
        dbc.Card(
            dbc.CardBody(
                [
                    html.H4("UW Parameters", className="mb-3"),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Black Hole Cooldown (s)"),
                                    dbc.Input(id="uw-bh-cooldown", type="number", value=46, step=0.1),
                                ],
                                width=2,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Black Hole Duration (s)"),
                                    dbc.Input(id="uw-bh-duration", type="number", value=34, step=0.1),
                                ],
                                width=2,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Golden Tower Cooldown (s)"),
                                    dbc.Input(id="uw-gt-cooldown", type="number", value=170, step=0.1),
                                ],
                                width=2,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Golden Tower Duration (s)"),
                                    dbc.Input(id="uw-gt-duration", type="number", value=45, step=0.1),
                                ],
                                width=2,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Death Wave Cooldown (s)"),
                                    dbc.Input(id="uw-dw-cooldown", type="number", value=170, step=0.1),
                                ],
                                width=2,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Death Wave Duration (s)"),
                                    dbc.Input(id="uw-dw-duration", type="number", value=20, step=0.1),
                                ],
                                width=2,
                            ),
                        ],
                        className="mb-3",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Chrono Field Cooldown (s)"),
                                    dbc.Input(id="uw-cf-cooldown", type="number", value=180, step=0.1),
                                ],
                                width=2,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Chrono Field Duration (s)"),
                                    dbc.Input(id="uw-cf-duration", type="number", value=28, step=0.1),
                                ],
                                width=2,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Golden Bot Cooldown (s)"),
                                    dbc.Input(id="uw-gb-cooldown", type="number", value=100, step=0.1),
                                ],
                                width=2,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Golden Bot Duration (s)"),
                                    dbc.Input(id="uw-gb-duration", type="number", value=24.5, step=0.1),
                                ],
                                width=2,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Smart Missiles Cooldown (s)"),
                                    dbc.Input(id="uw-sm-cooldown", type="number", value=120, step=0.1),
                                ],
                                width=2,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Smart Missiles Duration (s)"),
                                    dbc.Input(id="uw-sm-duration", type="number", value=15, step=0.1),
                                ],
                                width=2,
                            ),
                        ],
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Inner Land Mines Cooldown (s)"),
                                    dbc.Input(id="uw-ilm-cooldown", type="number", value=130, step=0.1),
                                ],
                                width=2,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Inner Land Mines Duration (s)"),
                                    dbc.Input(id="uw-ilm-duration", type="number", value=25, step=0.1),
                                ],
                                width=2,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Poison Swamp Cooldown (s)"),
                                    dbc.Input(id="uw-ps-cooldown", type="number", value=140, step=0.1),
                                ],
                                width=2,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Poison Swamp Duration (s)"),
                                    dbc.Input(id="uw-ps-duration", type="number", value=30, step=0.1),
                                ],
                                width=2,
                            ),
                        ],
                        className="mt-3",
                    ),
                ]
            ),
            className="mb-4",
        ),
        
        # UW Stats Cards
        html.H4("UW Statistics", className="mb-3 mt-4"),
        dbc.Row(
            [
                dbc.Col([
                    dbc.Card([dbc.CardBody([
                        html.H6('Black Hole', className='card-title', style={'marginBottom': '0.5rem'}),
                        html.Div(id='stats-bh-params', style={'color': '#6c757d', 'fontSize': '0.8rem', 'marginBottom': '0.5rem'}),
                        html.Div(id='stats-bh-uptime', style={'fontSize': '1rem', 'marginBottom': '0.25rem'}),
                        html.Div(id='stats-bh-downtime', style={'color': '#999', 'fontSize': '0.85rem', 'marginBottom': '0.25rem'}),
                        html.Div(id='stats-bh-interval', style={'color': '#999', 'fontSize': '0.85rem'}),
                    ])], id='stats-bh-card', className='mb-2'),
                ], width=12, md=6, lg=4),
                dbc.Col([
                    dbc.Card([dbc.CardBody([
                        html.H6('Golden Tower', className='card-title', style={'marginBottom': '0.5rem'}),
                        html.Div(id='stats-gt-params', style={'color': '#6c757d', 'fontSize': '0.8rem', 'marginBottom': '0.5rem'}),
                        html.Div(id='stats-gt-uptime', style={'fontSize': '1rem', 'marginBottom': '0.25rem'}),
                        html.Div(id='stats-gt-downtime', style={'color': '#999', 'fontSize': '0.85rem', 'marginBottom': '0.25rem'}),
                        html.Div(id='stats-gt-interval', style={'color': '#999', 'fontSize': '0.85rem'}),
                    ])], id='stats-gt-card', className='mb-2'),
                ], width=12, md=6, lg=4),
                dbc.Col([
                    dbc.Card([dbc.CardBody([
                        html.H6('Death Wave', className='card-title', style={'marginBottom': '0.5rem'}),
                        html.Div(id='stats-dw-params', style={'color': '#6c757d', 'fontSize': '0.8rem', 'marginBottom': '0.5rem'}),
                        html.Div(id='stats-dw-uptime', style={'fontSize': '1rem', 'marginBottom': '0.25rem'}),
                        html.Div(id='stats-dw-downtime', style={'color': '#999', 'fontSize': '0.85rem', 'marginBottom': '0.25rem'}),
                        html.Div(id='stats-dw-interval', style={'color': '#999', 'fontSize': '0.85rem'}),
                    ])], id='stats-dw-card', className='mb-2'),
                ], width=12, md=6, lg=4),
                dbc.Col([
                    dbc.Card([dbc.CardBody([
                        html.H6('Chrono Field', className='card-title', style={'marginBottom': '0.5rem'}),
                        html.Div(id='stats-cf-params', style={'color': '#6c757d', 'fontSize': '0.8rem', 'marginBottom': '0.5rem'}),
                        html.Div(id='stats-cf-uptime', style={'fontSize': '1rem', 'marginBottom': '0.25rem'}),
                        html.Div(id='stats-cf-downtime', style={'color': '#999', 'fontSize': '0.85rem', 'marginBottom': '0.25rem'}),
                        html.Div(id='stats-cf-interval', style={'color': '#999', 'fontSize': '0.85rem'}),
                    ])], id='stats-cf-card', className='mb-2'),
                ], width=12, md=6, lg=4),
                dbc.Col([
                    dbc.Card([dbc.CardBody([
                        html.H6('Golden Bot', className='card-title', style={'marginBottom': '0.5rem'}),
                        html.Div(id='stats-gb-params', style={'color': '#6c757d', 'fontSize': '0.8rem', 'marginBottom': '0.5rem'}),
                        html.Div(id='stats-gb-uptime', style={'fontSize': '1rem', 'marginBottom': '0.25rem'}),
                        html.Div(id='stats-gb-downtime', style={'color': '#999', 'fontSize': '0.85rem', 'marginBottom': '0.25rem'}),
                        html.Div(id='stats-gb-interval', style={'color': '#999', 'fontSize': '0.85rem'}),
                    ])], id='stats-gb-card', className='mb-2'),
                ], width=12, md=6, lg=4),
                dbc.Col([
                    dbc.Card([dbc.CardBody([
                        html.H6('Smart Missiles', className='card-title', style={'marginBottom': '0.5rem'}),
                        html.Div(id='stats-sm-params', style={'color': '#6c757d', 'fontSize': '0.8rem', 'marginBottom': '0.5rem'}),
                        html.Div(id='stats-sm-uptime', style={'fontSize': '1rem', 'marginBottom': '0.25rem'}),
                        html.Div(id='stats-sm-downtime', style={'color': '#999', 'fontSize': '0.85rem', 'marginBottom': '0.25rem'}),
                        html.Div(id='stats-sm-interval', style={'color': '#999', 'fontSize': '0.85rem'}),
                    ])], id='stats-sm-card', className='mb-2'),
                ], width=12, md=6, lg=4),
                dbc.Col([
                    dbc.Card([dbc.CardBody([
                        html.H6('Inner Land Mines', className='card-title', style={'marginBottom': '0.5rem'}),
                        html.Div(id='stats-ilm-params', style={'color': '#6c757d', 'fontSize': '0.8rem', 'marginBottom': '0.5rem'}),
                        html.Div(id='stats-ilm-uptime', style={'fontSize': '1rem', 'marginBottom': '0.25rem'}),
                        html.Div(id='stats-ilm-downtime', style={'color': '#999', 'fontSize': '0.85rem', 'marginBottom': '0.25rem'}),
                        html.Div(id='stats-ilm-interval', style={'color': '#999', 'fontSize': '0.85rem'}),
                    ])], id='stats-ilm-card', className='mb-2'),
                ], width=12, md=6, lg=4),
                dbc.Col([
                    dbc.Card([dbc.CardBody([
                        html.H6('Poison Swamp', className='card-title', style={'marginBottom': '0.5rem'}),
                        html.Div(id='stats-ps-params', style={'color': '#6c757d', 'fontSize': '0.8rem', 'marginBottom': '0.5rem'}),
                        html.Div(id='stats-ps-uptime', style={'fontSize': '1rem', 'marginBottom': '0.25rem'}),
                        html.Div(id='stats-ps-downtime', style={'color': '#999', 'fontSize': '0.85rem', 'marginBottom': '0.25rem'}),
                        html.Div(id='stats-ps-interval', style={'color': '#999', 'fontSize': '0.85rem'}),
                    ])], id='stats-ps-card', className='mb-2'),
                ], width=12, md=6, lg=4),
                dbc.Col([
                    dbc.Card([dbc.CardBody([
                        html.H6('Perma Calc Details', className='card-title', style={'marginBottom': '0.5rem'}),
                        html.Div(id='sim-details-wave-time', style={'color': '#999', 'fontSize': '0.85rem', 'marginBottom': '0.25rem'}),
                        html.Div(id='sim-details-total-time', style={'color': '#999', 'fontSize': '0.85rem', 'marginBottom': '0.25rem'}),
                        html.Div(id='sim-details-pkg-chance', style={'color': '#999', 'fontSize': '0.85rem'}),
                    ])], className='mb-2', style={'borderLeft': '4px solid #6c757d'}),
                ], width=12, md=6, lg=4),
            ],
            className="mb-4",
        ),
        
        # Single UW Detail Chart with Selector
        html.H4("Detailed UW Analysis", className="mb-3"),
        dbc.Row([
            dbc.Col([
                html.Label('Select Ultimate Weapon:', className='mb-2'),
                dcc.Dropdown(
                    id='uw-detail-selector',
                    options=[
                        {'label': 'Black Hole', 'value': 'Black Hole'},
                        {'label': 'Golden Tower', 'value': 'Golden Tower'},
                        {'label': 'Death Wave', 'value': 'Death Wave'},
                        {'label': 'Chrono Field', 'value': 'Chrono Field'},
                        {'label': 'Golden Bot', 'value': 'Golden Bot'},
                        {'label': 'Smart Missiles', 'value': 'Smart Missiles'},
                        {'label': 'Inner Land Mines', 'value': 'Inner Land Mines'},
                        {'label': 'Poison Swamp', 'value': 'Poison Swamp'},
                    ],
                    value='Chrono Field',
                    clearable=False,
                    className='mb-3'
                ),
            ], width=12, md=4, lg=3),
            dbc.Col([
                html.Label('Display Options:', className='mb-2'),
                dbc.Checklist(
                    id='uw-show-cooldown',
                    options=[{'label': ' Show Cooldown', 'value': 'show'}],
                    value=['show'],
                    switch=True,
                    className='mt-4'
                ),
            ], width=12, md=3, lg=2),
            dbc.Col([
                html.Label('Re-randomize Packages:', className='mb-2'),
                dbc.Button(
                    "Re-randomize",
                    id="uw-rerandomize-btn",
                    color="primary",
                    size="sm",
                    className="mt-4"
                ),
            ], width=12, md=2, lg=2),
        ], className="mb-2"),
        dcc.Graph(id="uw-detail-graph", config={"displayModeBar": False}, className="mb-4"),
        
        html.H2("Sync Chart"),
        dcc.Graph(id="uw-graph-sync", config={"displayModeBar": False}),
    ]
)


@callback(
    Output("uw-random-seed", "data"),
    Input("uw-rerandomize-btn", "n_clicks"),
    prevent_initial_call=True,
)
def update_random_seed(n_clicks):
    """Generate a new random seed when re-randomize button is clicked."""
    import time
    return int(time.time() * 1000000) % 1000000


@callback(
    Output("uw-simulation-store", "data"),
    Input("uw-play-mode", "value"),
    Input("uw-wa-card", "value"),
    Input("uw-boss-waves", "value"),
    Input("uw-package-after-boss", "value"),
    Input("uw-cooldown-bc", "value"),
    Input("uw-package-chance", "value"),
    Input("uw-galaxy-compressor", "value"),
    Input("uw-mvn", "value"),
    Input("uw-bh-cooldown", "value"),
    Input("uw-bh-duration", "value"),
    Input("uw-gt-cooldown", "value"),
    Input("uw-gt-duration", "value"),
    Input("uw-dw-cooldown", "value"),
    Input("uw-dw-duration", "value"),
    Input("uw-cf-cooldown", "value"),
    Input("uw-cf-duration", "value"),
    Input("uw-gb-cooldown", "value"),
    Input("uw-gb-duration", "value"),
    Input("uw-sm-cooldown", "value"),
    Input("uw-sm-duration", "value"),
    Input("uw-ilm-cooldown", "value"),
    Input("uw-ilm-duration", "value"),
    Input("uw-ps-cooldown", "value"),
    Input("uw-ps-duration", "value"),
    Input("uw-random-seed", "data"),
)
def calculate_simulations(
    play_mode,
    wa_card_level,
    boss_waves,
    package_after_boss_state,
    cooldown_bc_state,
    pkg_chance_pct,
    gc_tier,
    mvn_state,
    bh_cooldown, bh_duration,
    gt_cooldown, gt_duration,
    dw_cooldown, dw_duration,
    cf_cooldown, cf_duration,
    gb_cooldown, gb_duration,
    sm_cooldown, sm_duration,
    ilm_cooldown, ilm_duration,
    ps_cooldown, ps_duration,
    random_seed,
):
    """Calculate simulation data for all UWs and return serializable results."""
    import time
    start_time = time.time()
    
    seed = random_seed if random_seed is not None else 0
    
    # Get WA card reduction
    wa_reduction = WA_Card.get(wa_card_level, 0.54)  # Default to 7 Star
    
    # Calculate wave time based on play mode and WA card
    if play_mode == "Tournament":
        wave_cooldown = 4.5 * (1 - wa_reduction)
    else:  # Farming
        wave_cooldown = 9 * (1 - wa_reduction)
    
    wave_time = Wave_Basetime + wave_cooldown
    
    # Package after boss setting
    package_after_boss = package_after_boss_state == "Unlocked"
    
    # Battle condition cooldown bonus
    bc_cooldown = cooldown_bc if cooldown_bc_state == "Active" else 0
    
    # Get galaxy compressor bonus
    gc_bonus = GC_EFFECTS.get(gc_tier, 0.0)
    
    # Convert package chance from percentage to probability
    pkg_chance = (pkg_chance_pct or 76) / 100.0
    
    # Base package reduction + galaxy compressor (except Golden Bot)
    base_pkg_reduction = gc_bonus
    
    # Apply farming perks to durations
    bh_perk_duration = Farming_Perks["Black_Hole_Duration"] if play_mode == "Farming" else 0
    dw_perk_duration = Farming_Perks["Death_Wave_Duration"] if play_mode == "Farming" else 0
    cf_perk_duration = Farming_Perks["Chrono_Field_Duration"] if play_mode == "Farming" else 0
    
    # Multiverse Nexus: Synchronize BH, GT, DW cooldowns to their average + tier offset
    mvn_offset = MVN_EFFECTS.get(mvn_state)
    if mvn_offset is not None:
        # Calculate average cooldown of BH, GT, DW (before BC bonus)
        bh_base = bh_cooldown or 46
        gt_base = gt_cooldown or 170
        dw_base = dw_cooldown or 170
        avg_cooldown = (bh_base + gt_base + dw_base) / 3
        
        # Override individual cooldowns with average + tier offset
        synced_cooldown = avg_cooldown + mvn_offset
        bh_cooldown = synced_cooldown
        gt_cooldown = synced_cooldown
        dw_cooldown = synced_cooldown
    
    results: Dict[str, pd.DataFrame] = {}
    
    # Generate package reductions for each UW
    uw_configs = [
        ("Black Hole", (bh_cooldown or 46) + bc_cooldown, (bh_duration or 34) + bh_perk_duration, base_pkg_reduction, True),
        ("Golden Tower", (gt_cooldown or 170) + bc_cooldown, gt_duration or 45, base_pkg_reduction, True),
        ("Death Wave", (dw_cooldown or 170) + bc_cooldown, (dw_duration or 20) + dw_perk_duration, base_pkg_reduction, False),
        ("Chrono Field", (cf_cooldown or 180) + bc_cooldown, (cf_duration or 28) + cf_perk_duration, base_pkg_reduction, False),
        ("Golden Bot", (gb_cooldown or 100) + bc_cooldown, gb_duration or 24.5, 0, False),  # No GC bonus
        ("Smart Missiles", (sm_cooldown or 120) + bc_cooldown, sm_duration or 15, base_pkg_reduction, False),
        ("Inner Land Mines", (ilm_cooldown or 130) + bc_cooldown, ilm_duration or 25, base_pkg_reduction, False),
        ("Poison Swamp", (ps_cooldown or 140) + bc_cooldown, ps_duration or 30, base_pkg_reduction, False),
    ]
    
    start_sim = time.time()
    package_count = 0  # Track total packages across simulation
    for uw_name, cooldown, duration, pkg_red, can_queue in uw_configs:
        package_reductions, is_boss_packages = simulate_package_reductions(
            total_time=total_time,
            pkg_chance=pkg_chance,
            pkg_reduction=pkg_red,
            wave_time=wave_time,
            boss_every_x=boss_waves or 10,
            seed=seed,
            boss_package_enabled=package_after_boss,
        )
        # Count packages from first UW (all UWs share same package series)
        if uw_name == "Black Hole":
            package_count = int(np.sum(package_reductions > 0))
        
        df = calculate_uw_uptime(
            cooldown,
            duration,
            total_time=total_time,
            return_df=True,
            package_reduction_series=package_reductions,
            is_boss_package_series=is_boss_packages,
            wave_time=wave_time,
            can_queue=can_queue,
        )  # type: ignore[call-overload]
        assert isinstance(df, pd.DataFrame)
        results[uw_name] = df
    
    sim_time = time.time() - start_sim
    print(f"[PERF] Simulation loop: {sim_time:.3f}s")

    # Convert DataFrames to serializable format and include stats
    start_serialize = time.time()
    serializable_results = {}
    
    # Store simulation parameters
    serializable_results["wave_time"] = wave_time
    serializable_results["total_time"] = total_time
    serializable_results["package_count"] = package_count
    
    # Store effective parameters for display in cards
    effective_params = {}
    for i, uw_name in enumerate(UW_ORDER):
        _, cooldown, duration, _, _ = uw_configs[i]
        base_cd = [
            bh_cooldown or 46, gt_cooldown or 170, dw_cooldown or 170, cf_cooldown or 180, 
            gb_cooldown or 100, sm_cooldown or 120, ilm_cooldown or 130, ps_cooldown or 140
        ][i]
        base_dur = [
            bh_duration or 34, gt_duration or 45, dw_duration or 20, cf_duration or 28, 
            gb_duration or 24.5, sm_duration or 15, ilm_duration or 25, ps_duration or 30
        ][i]
        perk_dur = [bh_perk_duration, 0, dw_perk_duration, cf_perk_duration, 0, 0, 0, 0][i]
        
        effective_params[uw_name] = {
            "base_cooldown": base_cd,
            "bc_cooldown": bc_cooldown,
            "effective_cooldown": cooldown,
            "base_duration": base_dur,
            "perk_duration": perk_dur,
            "effective_duration": duration,
        }
    
    serializable_results["effective_params"] = effective_params
    
    for uw_name in UW_ORDER:
        df = results[uw_name]
        stats = _calculate_uptime_downtime_stats(df)
        
        serializable_results[uw_name] = {
            "df": df.to_dict('list'),  # Serialize DataFrame
            "stats": stats,
        }
    
    serialize_time = time.time() - start_serialize
    elapsed_time = time.time() - start_time
    print(f"[PERF] Serialization: {serialize_time:.3f}s")
    print(f"[PERF] Total calculate_simulations: {elapsed_time:.3f}s")
    
    return serializable_results


@callback(
    Output("sim-details-wave-time", "children"),
    Output("sim-details-total-time", "children"),
    Output("sim-details-pkg-chance", "children"),
    Output("stats-bh-params", "children"),
    Output("stats-bh-uptime", "children"),
    Output("stats-bh-downtime", "children"),
    Output("stats-bh-interval", "children"),
    Output("stats-bh-card", "style"),
    Output("stats-gt-params", "children"),
    Output("stats-gt-uptime", "children"),
    Output("stats-gt-downtime", "children"),
    Output("stats-gt-interval", "children"),
    Output("stats-gt-card", "style"),
    Output("stats-dw-params", "children"),
    Output("stats-dw-uptime", "children"),
    Output("stats-dw-downtime", "children"),
    Output("stats-dw-interval", "children"),
    Output("stats-dw-card", "style"),
    Output("stats-cf-params", "children"),
    Output("stats-cf-uptime", "children"),
    Output("stats-cf-downtime", "children"),
    Output("stats-cf-interval", "children"),
    Output("stats-cf-card", "style"),
    Output("stats-gb-params", "children"),
    Output("stats-gb-uptime", "children"),
    Output("stats-gb-downtime", "children"),
    Output("stats-gb-interval", "children"),
    Output("stats-gb-card", "style"),
    Output("stats-sm-params", "children"),
    Output("stats-sm-uptime", "children"),
    Output("stats-sm-downtime", "children"),
    Output("stats-sm-interval", "children"),
    Output("stats-sm-card", "style"),
    Output("stats-ilm-params", "children"),
    Output("stats-ilm-uptime", "children"),
    Output("stats-ilm-downtime", "children"),
    Output("stats-ilm-interval", "children"),
    Output("stats-ilm-card", "style"),
    Output("stats-ps-params", "children"),
    Output("stats-ps-uptime", "children"),
    Output("stats-ps-downtime", "children"),
    Output("stats-ps-interval", "children"),
    Output("stats-ps-card", "style"),
    Output("uw-detail-graph", "figure"),
    Output("uw-graph-sync", "figure"),
    Input("uw-simulation-store", "data"),
    Input("uw-detail-selector", "value"),
    Input("uw-show-cooldown", "value"),
)
def render_uw_figures(simulation_data, selected_uw, show_cooldown):
    """Render UI components from pre-calculated simulation data."""
    import time
    start_time = time.time()
    
    if simulation_data is None:
        # Return empty defaults (3 sim details + 40 card outputs + 2 figures = 45)
        return ["", "", ""] + [""] * 40 + [go.Figure(), go.Figure()]
    
    # Extract simulation parameters
    wave_time = simulation_data.get("wave_time", 0)
    total_waves = int(total_time / wave_time) if wave_time > 0 else 0
    
    sim_wave_time_text = f"Total Wave Time: {wave_time:.2f}s"
    sim_total_time_text = f"Simulation Time: {total_time}s ({total_waves} waves)"
    
    # Calculate effective package chance from first UW's data
    pkg_count = simulation_data.get("package_count", 0)
    effective_pkg_chance = (pkg_count / total_waves * 100) if total_waves > 0 else 0
    sim_pkg_chance_text = f"Effective Package Chance: {effective_pkg_chance:.1f}%"
    
    # Reconstruct DataFrames from serialized data
    start_deserialize = time.time()
    results = {}
    for uw_name in UW_ORDER:
        df_dict = simulation_data[uw_name]["df"]
        results[uw_name] = pd.DataFrame(df_dict)
    deserialize_time = time.time() - start_deserialize
    print(f"[PERF] Deserialization: {deserialize_time:.3f}s")
    
    # Generate card outputs from stored stats
    start_cards = time.time()
    
    # Get effective parameters from simulation_data
    effective_params = simulation_data.get("effective_params", {})
    
    card_outputs = []
    for uw_name in UW_ORDER:
        stats = simulation_data[uw_name]["stats"]
        
        # Get effective parameters for this UW
        uw_params = effective_params.get(uw_name, {})
        base_cooldown = uw_params.get("base_cooldown", 0)
        bc_cooldown = uw_params.get("bc_cooldown", 0)
        effective_cooldown = uw_params.get("effective_cooldown", 0)
        base_duration = uw_params.get("base_duration", 0)
        perk_duration = uw_params.get("perk_duration", 0)
        effective_duration = uw_params.get("effective_duration", 0)
        
        # Format cooldown values: show 0.0s if not full second, otherwise full seconds
        def format_cd(value):
            if value == int(value):
                return f"{int(value)}s"
            else:
                return f"{value:.1f}s"
        
        base_cooldown_str = format_cd(base_cooldown)
        bc_cooldown_str = format_cd(bc_cooldown)
        effective_cooldown_str = format_cd(effective_cooldown)
        
        # Format parameter display
        if bc_cooldown > 0 and perk_duration > 0:
            params_text = f"CD: {base_cooldown_str} + {bc_cooldown_str} BC = {effective_cooldown_str} | Dur: {base_duration}s + {perk_duration}s Perk = {effective_duration}s"
        elif bc_cooldown > 0:
            params_text = f"CD: {base_cooldown_str} + {bc_cooldown_str} BC = {effective_cooldown_str} | Dur: {effective_duration}s"
        elif perk_duration > 0:
            params_text = f"CD: {effective_cooldown_str} | Dur: {base_duration}s + {perk_duration}s Perk = {effective_duration}s"
        else:
            params_text = f"CD: {effective_cooldown_str} | Dur: {effective_duration}s"
        
        stats = simulation_data[uw_name]["stats"]
        uptime_pct = stats["uptime_pct"]
        avg_downtime = stats["avg_downtime"]
        min_downtime = stats["min_downtime"]
        max_downtime = stats["max_downtime"]
        avg_activation_interval = stats.get("avg_activation_interval")
        
        # Format uptime display
        if uptime_pct is not None and uptime_pct >= 100:
            uptime_text = "Uptime: Permanent"
        else:
            uptime_text = f"Uptime: {uptime_pct:.1f}%" if uptime_pct is not None else "Uptime: N/A"
        
        # Format downtime display (without activation interval)
        if uptime_pct is not None and uptime_pct >= 100:
            downtime_text = "Downtime: No downtime"
        elif min_downtime is None or max_downtime is None:
            # All downtimes are the same
            downtime_text = f"Downtime: Avg: {avg_downtime:.1f}s" if avg_downtime is not None else "Downtime: N/A"
        else:
            # Variable downtimes
            downtime_text = f"Downtime: Min: {min_downtime:.1f}s | Avg: {avg_downtime:.1f}s | Max: {max_downtime:.1f}s"
        
        # Format activation interval as separate line
        if uptime_pct is not None and uptime_pct >= 100:
            interval_text = "Activation Interval: Permanent"
        elif avg_activation_interval is not None:
            interval_text = f"Activation Interval: {avg_activation_interval:.1f}s"
        else:
            interval_text = "Activation Interval: N/A"
        
        # Card style with color
        card_color = UW_CONFIG[uw_name]["color_hex"]
        card_style = {'borderLeft': f'4px solid {card_color}'}
        
        card_outputs.extend([params_text, uptime_text, downtime_text, interval_text, card_style])
    
    cards_time = time.time() - start_cards
    print(f"[PERF] Card generation: {cards_time:.3f}s")
    
    # Generate figures from stored DataFrames
    start_figs = time.time()
    # DEBUG: Uncomment to trace figure generation
    # print(f"[DEBUG] render_uw_figures: selected_uw={selected_uw}")
    selected_color = UW_CONFIG[selected_uw]["color_rgba"]
    show_cd = 'show' in show_cooldown if show_cooldown else False
    fig_detail = _make_uw_figure(results[selected_uw], selected_uw, selected_color, show_cooldown=show_cd)
    # print(f"[DEBUG] About to call _make_sync_figure with selected_uw={selected_uw}")
    fig_sync = _make_sync_figure(results, selected_uw=selected_uw)
    # print(f"[DEBUG] _make_sync_figure returned")
    figs_time = time.time() - start_figs
    elapsed_time = time.time() - start_time
    print(f"[PERF] Figure generation: {figs_time:.3f}s")
    print(f"[PERF] Total render_uw_figures: {elapsed_time:.3f}s")

    return (sim_wave_time_text, sim_total_time_text, sim_pkg_chance_text, *card_outputs, fig_detail, fig_sync)


if __name__ == "__main__":
    # Standalone mode: run as independent Dash app
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        suppress_callback_exceptions=True
    )
    
    app.layout = layout
    
    print("Starting UW PermaCalc in standalone mode...")
    print("Open your browser and navigate to: http://127.0.0.1:8050")
    print("Press Ctrl+C to stop the server.")
    
    app.run(debug=True, host='127.0.0.1', port=8050)
else:
    # Production mode for deployment (e.g., Render.com with gunicorn)
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        suppress_callback_exceptions=True
    )
    
    app.layout = layout
    server = app.server  # Expose server for gunicorn
