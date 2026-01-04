"""
Monte Carlo simulation engine for missile-target engagements.

Runs multiple parallel trajectory samples to generate probability distributions
for intercept outcomes.
"""

import random
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from enum import Enum
import copy
import multiprocessing as mp
from multiprocessing import Pool
import os

from core.config import (
    MissileConfig, TargetConfig, SimulationConfig, FlareConfig,
    InitialConditions, Vector3, Orientation,
    AIM_9L, MISSILE_PRESETS, FIGHTER_JET, TARGET_PRESETS,
    DEFAULT_SIMULATION
)
from entities.missile import Missile, MissileStatus
from entities.target import Target, EvasiveManeuver, ThreatAwareness
from core.physics import add_gaussian_noise, add_vector_noise


class EngagementOutcome(Enum):
    """Outcome of a single engagement."""
    HIT = "hit"
    MISS = "miss"
    DECOYED = "decoyed"
    FUEL_OUT = "fuel_out"
    LOST_TRACK = "lost_track"
    IN_PROGRESS = "in_progress"


@dataclass
class EngagementResult:
    """Result of a single engagement simulation."""
    outcome: EngagementOutcome
    time_of_flight: float
    final_range: float           # Final missile-target distance
    max_g_pulled: float          # Maximum G during flight
    flares_deployed: int         # Number of flares used
    missile_trajectory: List[Tuple[float, float, float]]
    target_trajectory: List[Tuple[float, float, float]]
    final_missile_pos: Tuple[float, float, float]
    final_target_pos: Tuple[float, float, float]


@dataclass
class SimulationResults:
    """Aggregate results from Monte Carlo simulation."""
    num_runs: int
    outcomes: Dict[str, int]     # Count of each outcome type
    pk: float                    # Probability of kill (hit rate)
    mean_intercept_time: float   # Mean time to intercept (for hits)
    std_intercept_time: float    # Std dev of intercept time
    missile_trajectories: List[List[Tuple[float, float, float]]]
    target_trajectories: List[List[Tuple[float, float, float]]]
    individual_results: List[EngagementResult]

    # Configuration used
    missile_config: str
    target_config: str
    evasion_mode: str
    initial_range: float


class Simulation:
    """
    Monte Carlo missile engagement simulation.

    Runs multiple trajectory samples with randomized noise to generate
    probability distributions for intercept outcomes.
    """

    def __init__(
        self,
        missile_config: MissileConfig = AIM_9L,
        target_config: TargetConfig = FIGHTER_JET,
        sim_config: SimulationConfig = None,
        flare_config: FlareConfig = None,
        initial_conditions: InitialConditions = None,
        evasion_mode: EvasiveManeuver = EvasiveManeuver.NONE
    ):
        """
        Initialize simulation.

        Args:
            missile_config: Missile configuration
            target_config: Target aircraft configuration
            sim_config: Simulation parameters
            flare_config: Flare configuration
            initial_conditions: Initial positions and velocities
            evasion_mode: Type of evasive maneuvers
        """
        self.missile_config = missile_config
        self.target_config = target_config
        self.sim_config = sim_config or SimulationConfig()
        self.flare_config = flare_config or FlareConfig()
        self.initial_conditions = initial_conditions or InitialConditions()
        self.evasion_mode = evasion_mode

        # Random number generator
        if self.sim_config.random_seed is not None:
            self.rng = random.Random(self.sim_config.random_seed)
        else:
            self.rng = random.Random()

        # Results storage
        self.results: List[EngagementResult] = []

    def _create_missile(self, run_seed: int) -> Missile:
        """Create missile instance with randomized initial conditions."""
        rng = random.Random(run_seed)

        # Add noise to initial conditions
        initial_pos = add_vector_noise(
            self.initial_conditions.missile_position,
            10.0,  # Position noise [m]
            rng
        )

        # Initial velocity from orientation and speed
        direction = self.initial_conditions.missile_orientation.to_direction_vector()
        base_speed = self.initial_conditions.missile_velocity_mps
        noisy_speed = add_gaussian_noise(
            base_speed,
            base_speed * self.sim_config.launch_speed_noise_pct,
            rng
        )
        initial_vel = direction * noisy_speed

        missile = Missile(
            config=self.missile_config,
            initial_position=initial_pos,
            initial_velocity=initial_vel,
            initial_orientation=self.initial_conditions.missile_orientation,
            rng=rng
        )

        return missile

    def _create_target(self, run_seed: int) -> Target:
        """Create target instance with randomized initial conditions."""
        rng = random.Random(run_seed + 1000000)  # Different seed from missile

        initial_pos = add_vector_noise(
            self.initial_conditions.target_position,
            10.0,
            rng
        )

        direction = self.initial_conditions.target_orientation.to_direction_vector()
        base_speed = self.initial_conditions.target_velocity_mps
        noisy_speed = add_gaussian_noise(base_speed, base_speed * 0.02, rng)
        initial_vel = direction * noisy_speed

        target = Target(
            config=self.target_config,
            initial_position=initial_pos,
            initial_velocity=initial_vel,
            initial_orientation=self.initial_conditions.target_orientation,
            evasion_mode=self.evasion_mode,
            flare_config=self.flare_config,
            rng=rng
        )

        return target

    def run_single(self, run_id: int = 0) -> EngagementResult:
        """
        Run a single engagement simulation.

        Args:
            run_id: Unique run identifier (used for seeding)

        Returns:
            EngagementResult for this run
        """
        # Create entities
        missile = self._create_missile(run_id)
        target = self._create_target(run_id)

        # Launch missile
        missile.launch()

        # Alert target (if evasion enabled)
        if self.evasion_mode != EvasiveManeuver.NONE:
            target.set_awareness(ThreatAwareness.TRACKING)
            target.alert_to_missile(missile.state.position)

        # Simulation loop
        dt = self.sim_config.dt
        time = 0.0
        max_g = 0.0
        initial_flares = target.flare_dispenser.flares_remaining

        # Track distance to detect when missile passes target (distance starts increasing)
        prev_distance = (target.state.position -
                         missile.state.position).magnitude()
        distance_increasing_count = 0  # Counter to avoid false positives from noise

        while time < self.sim_config.max_time_s:
            # Get flare info
            flare_positions = target.get_flare_positions()
            flare_intensities = target.get_flare_intensities()

            # Update missile
            status = missile.update(
                target.state.position,
                target.state.velocity,
                target.get_ir_signature(),
                flare_positions,
                flare_intensities,
                dt
            )

            # Track max G
            max_g = max(max_g, missile.state.current_g_load)

            # Update target
            target.update(
                dt,
                time,
                missile.state.position,
                auto_deploy_flares=True
            )

            # Check termination conditions
            if status != MissileStatus.FLYING:
                break

            # Check if missile has passed target (distance is increasing)
            current_distance = (target.state.position -
                                missile.state.position).magnitude()
            if current_distance > prev_distance:
                distance_increasing_count += 1
                # Require 3 consecutive frames of increasing distance to confirm miss
                # This avoids false positives from momentary zigzags or noise
                if distance_increasing_count >= 3:
                    # Missile has passed target - it's a miss, no point continuing
                    missile.state.status = MissileStatus.MISS
                    break
            else:
                distance_increasing_count = 0  # Reset counter if distance is decreasing

            prev_distance = current_distance
            time += dt

        # Determine outcome
        if missile.state.status == MissileStatus.HIT:
            outcome = EngagementOutcome.HIT
        elif missile.state.status == MissileStatus.DECOYED:
            outcome = EngagementOutcome.DECOYED
        elif missile.state.status == MissileStatus.FUEL_OUT:
            outcome = EngagementOutcome.FUEL_OUT
        elif missile.state.status == MissileStatus.LOST_TRACK:
            outcome = EngagementOutcome.LOST_TRACK
        else:
            outcome = EngagementOutcome.MISS

        # Compute final range
        final_range = (
            target.state.position - missile.state.position
        ).magnitude()

        # Count flares deployed
        flares_deployed = initial_flares - target.flare_dispenser.flares_remaining

        result = EngagementResult(
            outcome=outcome,
            time_of_flight=time,
            final_range=final_range,
            max_g_pulled=max_g,
            flares_deployed=flares_deployed,
            missile_trajectory=missile.get_trajectory(),
            target_trajectory=target.get_trajectory(),
            final_missile_pos=missile.state.position.to_tuple(),
            final_target_pos=target.state.position.to_tuple()
        )

        return result

    def _run_single_wrapper(self, run_id: int) -> EngagementResult:
        """Wrapper for run_single that can be called from multiprocessing."""
        return self.run_single(run_id)

    def run_monte_carlo(
        self,
        num_runs: Optional[int] = None,
        progress_callback=None,
        num_workers: Optional[int] = None,
        use_multiprocessing: bool = True
    ) -> SimulationResults:
        """
        Run Monte Carlo simulation with multiple trajectories.

        Uses multiprocessing to run simulations in parallel for faster results.

        Args:
            num_runs: Number of runs (overrides config if provided)
            progress_callback: Function(run_id, total_runs) called per run
                              NOTE: Only works in sequential mode (use_multiprocessing=False)
            num_workers: Number of parallel workers (default: CPU count)
            use_multiprocessing: Whether to use parallel processing (default: True)

        Returns:
            SimulationResults with aggregate statistics
        """
        n = num_runs or self.sim_config.num_runs

        # Determine number of workers
        if num_workers is None:
            num_workers = mp.cpu_count()

        # Use sequential processing if requested, for small runs, or if callback needed
        if not use_multiprocessing or n <= 1 or (progress_callback is not None):
            # Sequential execution (original behavior)
            results = []
            for i in range(n):
                result = self.run_single(i)
                results.append(result)

                if progress_callback:
                    progress_callback(i + 1, n)
        else:
            # Parallel execution using multiprocessing
            # Create run IDs for each simulation
            run_ids = list(range(n))

            # Use Pool.map for parallel execution
            # Note: We use 'spawn' context on macOS to avoid fork issues
            ctx = mp.get_context('spawn')
            with ctx.Pool(processes=num_workers) as pool:
                results = pool.map(self._run_single_wrapper, run_ids)

        # Aggregate results
        outcomes = {}
        hit_times = []

        for r in results:
            outcome_name = r.outcome.value
            outcomes[outcome_name] = outcomes.get(outcome_name, 0) + 1

            if r.outcome == EngagementOutcome.HIT:
                hit_times.append(r.time_of_flight)

        # Calculate Pk
        pk = outcomes.get('hit', 0) / n if n > 0 else 0

        # Statistics for hit times
        if hit_times:
            mean_time = sum(hit_times) / len(hit_times)
            variance = sum((t - mean_time) **
                           2 for t in hit_times) / len(hit_times)
            std_time = variance ** 0.5
        else:
            mean_time = 0
            std_time = 0

        # Calculate initial range
        initial_range = (
            self.initial_conditions.target_position -
            self.initial_conditions.missile_position
        ).magnitude()

        return SimulationResults(
            num_runs=n,
            outcomes=outcomes,
            pk=pk,
            mean_intercept_time=mean_time,
            std_intercept_time=std_time,
            missile_trajectories=[r.missile_trajectory for r in results],
            target_trajectories=[r.target_trajectory for r in results],
            individual_results=results,
            missile_config=self.missile_config.name,
            target_config=self.target_config.name,
            evasion_mode=self.evasion_mode.value,
            initial_range=initial_range
        )

    @staticmethod
    def create_engagement_scenario(
        missile_preset: str = "AIM-9L",
        target_preset: str = "fighter",
        evasion: str = "none",
        initial_range_km: float = 5.0,
        aspect_angle_deg: float = 0.0,  # 0=tail, 180=head-on
        altitude_m: float = 5000.0,
        flare_count: int = 60,
        num_runs: int = 100
    ) -> 'Simulation':
        """
        Create a simulation with common scenario parameters.

        Args:
            missile_preset: Name of missile preset (AIM-9B, AIM-9L, etc.)
            target_preset: Name of target preset (fighter, transport)
            evasion: Evasion mode (none, random, break_left, etc.)
            initial_range_km: Initial missile-target range [km]
            aspect_angle_deg: Aspect angle (0=tail, 180=head-on)
            altitude_m: Engagement altitude [m]
            flare_count: Number of flares on target
            num_runs: Number of Monte Carlo runs

        Returns:
            Configured Simulation instance
        """
        import math

        # Get presets
        missile_config = MISSILE_PRESETS.get(missile_preset, AIM_9L)
        target_config = TARGET_PRESETS.get(target_preset, FIGHTER_JET)

        # Override flare count
        target_config = copy.copy(target_config)
        target_config.flare_count = flare_count

        # Parse evasion mode
        evasion_map = {
            "none": EvasiveManeuver.NONE,
            "random": EvasiveManeuver.RANDOM,
            "break_left": EvasiveManeuver.BREAK_LEFT,
            "break_right": EvasiveManeuver.BREAK_RIGHT,
            "break_down": EvasiveManeuver.BREAK_DOWN,
            "break_up": EvasiveManeuver.BREAK_UP,
            "notch": EvasiveManeuver.NOTCH
        }
        evasion_mode = evasion_map.get(evasion.lower(), EvasiveManeuver.NONE)

        # Calculate initial positions based on aspect angle
        range_m = initial_range_km * 1000
        aspect_rad = math.radians(aspect_angle_deg)

        # Missile at origin (at altitude), pointing north (heading 0)
        missile_pos = Vector3(0, altitude_m, 0)
        missile_heading = 0  # Pointing north

        # Target position is always north of missile (along Z axis)
        target_pos = Vector3(0, altitude_m, range_m)

        # Target heading based on aspect angle:
        # Aspect 0° = tail chase (target flying away, heading 0° = north)
        # Aspect 180° = head-on (target flying toward, heading 180° = south)
        # Aspect 90° = beam (target flying east/west)
        target_heading = aspect_angle_deg  # 0=away, 180=toward

        # Set up initial conditions
        initial_conditions = InitialConditions(
            missile_position=missile_pos,
            missile_velocity_mps=300.0,  # Launch platform speed
            missile_orientation=Orientation(heading_deg=0, pitch_deg=0),
            target_position=target_pos,
            target_velocity_mps=250.0,   # Target speed
            target_orientation=Orientation(
                heading_deg=target_heading, pitch_deg=0)
        )

        sim_config = SimulationConfig(num_runs=num_runs)

        return Simulation(
            missile_config=missile_config,
            target_config=target_config,
            sim_config=sim_config,
            initial_conditions=initial_conditions,
            evasion_mode=evasion_mode
        )


def print_results(results: SimulationResults):
    """Print formatted simulation results."""
    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)
    print(f"Missile: {results.missile_config}")
    print(f"Target: {results.target_config}")
    print(f"Evasion: {results.evasion_mode}")
    print(f"Initial Range: {results.initial_range / 1000:.1f} km")
    print(f"Runs: {results.num_runs}")
    print("-" * 60)
    print("\nOUTCOMES:")
    for outcome, count in sorted(results.outcomes.items()):
        pct = count / results.num_runs * 100
        print(f"  {outcome:15s}: {count:4d} ({pct:5.1f}%)")
    print("-" * 60)
    print(f"\nPk (Probability of Kill): {results.pk:.1%}")
    if results.pk > 0:
        print(
            f"Mean intercept time: {results.mean_intercept_time:.2f} ± {results.std_intercept_time:.2f} s")
    print("=" * 60)
