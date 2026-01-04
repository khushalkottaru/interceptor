"""
3D Visualization for missile engagement simulations.

Uses Matplotlib for static and animated 3D plots showing:
- Missile trajectories (probability cloud)
- Target path
- Flare positions
- Engagement envelope
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.animation as animation
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from core.config import Vector3, MissileConfig
from simulation.simulation import SimulationResults, EngagementOutcome


# Color scheme
COLORS = {
    'missile_hit': '#00ff00',      # Green for successful intercepts
    'missile_miss': '#ff0000',     # Red for misses
    'missile_decoyed': '#ffff00',  # Yellow for decoyed
    'missile_other': '#888888',    # Gray for other outcomes
    'target': '#0088ff',           # Blue for target
    'flare': '#ff8800',            # Orange for flares
    'seeker_cone': '#00ff0044',    # Transparent green for seeker FOV
    'envelope': '#ffffff22',       # Transparent white for envelope
}


def plot_single_engagement(
    missile_trajectory: List[Tuple[float, float, float]],
    target_trajectory: List[Tuple[float, float, float]],
    outcome: str = "unknown",
    title: str = "Missile-Target Engagement",
    show_axes_labels: bool = True,
    figsize: Tuple[int, int] = (10, 8)
) -> Figure:
    """
    Plot a single missile-target engagement in 3D.

    Args:
        missile_trajectory: List of (x, y, z) positions for missile
        target_trajectory: List of (x, y, z) positions for target
        outcome: Engagement outcome for coloring
        title: Plot title
        show_axes_labels: Whether to show axis labels
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Convert to numpy arrays
    missile_arr = np.array(missile_trajectory)
    target_arr = np.array(target_trajectory)

    # Choose missile color based on outcome
    outcome_colors = {
        'hit': COLORS['missile_hit'],
        'miss': COLORS['missile_miss'],
        'decoyed': COLORS['missile_decoyed'],
    }
    missile_color = outcome_colors.get(
        outcome.lower(), COLORS['missile_other'])

    # Plot trajectories
    if len(missile_arr) > 0:
        ax.plot(
            missile_arr[:, 0], missile_arr[:, 2], missile_arr[:, 1],
            color=missile_color,
            linewidth=2,
            label=f'Missile ({outcome})'
        )
        # Start and end markers
        ax.scatter(
            missile_arr[0, 0], missile_arr[0, 2], missile_arr[0, 1],
            color=missile_color, s=100, marker='o', label='Missile Start'
        )
        ax.scatter(
            missile_arr[-1, 0], missile_arr[-1, 2], missile_arr[-1, 1],
            color=missile_color, s=100, marker='x'
        )

    if len(target_arr) > 0:
        ax.plot(
            target_arr[:, 0], target_arr[:, 2], target_arr[:, 1],
            color=COLORS['target'],
            linewidth=2,
            linestyle='--',
            label='Target'
        )
        ax.scatter(
            target_arr[0, 0], target_arr[0, 2], target_arr[0, 1],
            color=COLORS['target'], s=100, marker='o'
        )
        ax.scatter(
            target_arr[-1, 0], target_arr[-1, 2], target_arr[-1, 1],
            color=COLORS['target'], s=100, marker='^'
        )

    # Labels and formatting
    if show_axes_labels:
        ax.set_xlabel('East (m)')
        ax.set_ylabel('North (m)')
        ax.set_zlabel('Altitude (m)')

    ax.set_title(title)
    ax.legend()

    # Equal aspect ratio (approximately)
    _set_axes_equal(ax)

    return fig


def plot_monte_carlo_results(
    results: SimulationResults,
    sample_trajectories: int = 50,
    title: str = None,
    figsize: Tuple[int, int] = (12, 10)
) -> Figure:
    """
    Plot Monte Carlo simulation results with trajectory probability cloud.

    Args:
        results: SimulationResults from simulation
        sample_trajectories: Maximum number of trajectories to plot
        title: Plot title (auto-generated if None)
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Sample trajectories if too many
    missile_trajs = results.missile_trajectories[:sample_trajectories]
    target_trajs = results.target_trajectories[:sample_trajectories]
    outcomes = [
        r.outcome for r in results.individual_results[:sample_trajectories]]

    # Plot missile trajectories with alpha based on density
    alpha = max(0.05, min(0.5, 1.0 / (len(missile_trajs) ** 0.5)))

    for traj, outcome in zip(missile_trajs, outcomes):
        if len(traj) == 0:
            continue
        arr = np.array(traj)

        if outcome == EngagementOutcome.HIT:
            color = COLORS['missile_hit']
        elif outcome == EngagementOutcome.DECOYED:
            color = COLORS['missile_decoyed']
        else:
            color = COLORS['missile_miss']

        ax.plot(
            arr[:, 0], arr[:, 2], arr[:, 1],
            color=color,
            alpha=alpha,
            linewidth=1
        )

    # Plot one target trajectory (they're all similar)
    if target_trajs and len(target_trajs[0]) > 0:
        target_arr = np.array(target_trajs[0])
        ax.plot(
            target_arr[:, 0], target_arr[:, 2], target_arr[:, 1],
            color=COLORS['target'],
            linewidth=3,
            label='Target Path'
        )

    # Add Pk annotation
    ax.text2D(
        0.05, 0.95,
        f"Pk = {results.pk:.1%}\n{results.num_runs} runs",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='black', alpha=0.8),
        color='white'
    )

    # Create legend with outcome counts
    handles = []
    labels = []
    for outcome_name, color in [('Hit', COLORS['missile_hit']),
                                ('Miss', COLORS['missile_miss']),
                                ('Decoyed', COLORS['missile_decoyed'])]:
        count = results.outcomes.get(outcome_name.lower(), 0)
        if count > 0:
            line, = ax.plot([], [], color=color, linewidth=2)
            handles.append(line)
            labels.append(f"{outcome_name}: {count}")

    ax.legend(handles, labels, loc='upper right')

    # Formatting
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_zlabel('Altitude (m)')

    if title is None:
        title = f"{results.missile_config} vs {results.target_config}"
        if results.evasion_mode != 'none':
            title += f" (Evasion: {results.evasion_mode})"
    ax.set_title(title)

    _set_axes_equal(ax)

    return fig


def plot_pk_vs_range(
    ranges_km: List[float],
    pk_values: List[float],
    missile_name: str = "Missile",
    figsize: Tuple[int, int] = (10, 6)
) -> Figure:
    """
    Plot probability of kill vs range.

    Args:
        ranges_km: List of engagement ranges [km]
        pk_values: Corresponding Pk values (0-1)
        missile_name: Missile name for label
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(ranges_km, [p * 100 for p in pk_values],
            'o-', linewidth=2, markersize=8)
    ax.fill_between(ranges_km, 0, [p * 100 for p in pk_values], alpha=0.3)

    ax.set_xlabel('Engagement Range (km)')
    ax.set_ylabel('Probability of Kill (%)')
    ax.set_title(f'{missile_name} - Pk vs Range')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    # Add annotations
    max_pk = max(pk_values)
    max_range = ranges_km[pk_values.index(max_pk)]
    ax.annotate(
        f'Max Pk: {max_pk:.1%} at {max_range:.1f} km',
        xy=(max_range, max_pk * 100),
        xytext=(max_range + 1, max_pk * 100 + 10),
        arrowprops=dict(arrowstyle='->', color='red'),
        fontsize=10
    )

    return fig


def plot_outcome_comparison(
    missile_names: List[str],
    pk_values: List[float],
    flare_counts: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> Figure:
    """
    Bar chart comparing Pk across different missiles or scenarios.

    Args:
        missile_names: Names/labels for each scenario
        pk_values: Pk values for each scenario
        flare_counts: Optional flare counts for annotations
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = range(len(missile_names))
    bars = ax.bar(x, [p * 100 for p in pk_values],
                  color='steelblue', edgecolor='navy')

    # Add value labels on bars
    for bar, pk in zip(bars, pk_values):
        height = bar.get_height()
        ax.annotate(
            f'{pk:.1%}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=10, fontweight='bold'
        )

    ax.set_xticks(x)
    ax.set_xticklabels(missile_names, rotation=45, ha='right')
    ax.set_ylabel('Probability of Kill (%)')
    ax.set_title('Missile Effectiveness Comparison')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    return fig


def create_animation(
    results: SimulationResults,
    run_index: int = 0,
    interval_ms: int = 50,
    figsize: Tuple[int, int] = (10, 8)
) -> animation.FuncAnimation:
    """
    Create animated replay of an engagement.

    Args:
        results: SimulationResults from simulation
        run_index: Which run to animate
        interval_ms: Milliseconds between frames
        figsize: Figure size

    Returns:
        Matplotlib animation object
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    result = results.individual_results[run_index]
    missile_traj = np.array(result.missile_trajectory)
    target_traj = np.array(result.target_trajectory)

    # Initialize empty plots
    missile_line, = ax.plot([], [], [], 'g-', linewidth=2, label='Missile')
    target_line, = ax.plot([], [], [], 'b--', linewidth=2, label='Target')
    missile_point, = ax.plot([], [], [], 'go', markersize=10)
    target_point, = ax.plot([], [], [], 'b^', markersize=10)

    # Set up axes limits
    all_points = np.vstack([missile_traj, target_traj])
    ax.set_xlim(all_points[:, 0].min() - 500, all_points[:, 0].max() + 500)
    ax.set_ylim(all_points[:, 2].min() - 500, all_points[:, 2].max() + 500)
    ax.set_zlim(all_points[:, 1].min() - 500, all_points[:, 1].max() + 500)

    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_zlabel('Altitude (m)')
    ax.legend()

    # Time text
    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

    # Determine frame count (use shorter trajectory length)
    n_frames = min(len(missile_traj), len(target_traj))

    # Subsample if too many frames
    if n_frames > 500:
        step = n_frames // 500
        indices = list(range(0, n_frames, step))
    else:
        indices = list(range(n_frames))

    def init():
        missile_line.set_data([], [])
        missile_line.set_3d_properties([])
        target_line.set_data([], [])
        target_line.set_3d_properties([])
        missile_point.set_data([], [])
        missile_point.set_3d_properties([])
        target_point.set_data([], [])
        target_point.set_3d_properties([])
        time_text.set_text('')
        return missile_line, target_line, missile_point, target_point, time_text

    def update(frame_idx):
        idx = indices[frame_idx]

        # Update trajectory lines
        missile_line.set_data(missile_traj[:idx, 0], missile_traj[:idx, 2])
        missile_line.set_3d_properties(missile_traj[:idx, 1])

        target_line.set_data(target_traj[:idx, 0], target_traj[:idx, 2])
        target_line.set_3d_properties(target_traj[:idx, 1])

        # Update current positions
        missile_point.set_data([missile_traj[idx, 0]], [missile_traj[idx, 2]])
        missile_point.set_3d_properties([missile_traj[idx, 1]])

        target_point.set_data([target_traj[idx, 0]], [target_traj[idx, 2]])
        target_point.set_3d_properties([target_traj[idx, 1]])

        # Update time
        time_text.set_text(f't = {idx * 0.01:.2f}s')

        return missile_line, target_line, missile_point, target_point, time_text

    anim = animation.FuncAnimation(
        fig, update, frames=len(indices),
        init_func=init, blit=False, interval=interval_ms
    )

    return anim


def _set_axes_equal(ax):
    """Set equal aspect ratio for 3D plot."""
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))

    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


def save_figure(fig: Figure, filename: str, dpi: int = 150):
    """Save figure to file."""
    fig.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Saved: {filename}")


def show():
    """Display all open figures."""
    plt.show()
