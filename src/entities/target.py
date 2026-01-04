"""
Target aircraft simulation.

Models the evading target aircraft with:
- Evasive maneuvers
- IR signature
- Countermeasure deployment
"""

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Tuple

from core.config import Vector3, Orientation, TargetConfig, FIGHTER_JET
from core.physics import (
    EntityState, propagate_trajectory_rk4, velocity_to_orientation,
    compute_drag, compute_g_load, GRAVITY_M_S2, mps_to_mach, get_speed_of_sound,
    add_gaussian_noise
)
from entities.flare import FlareDispenser, FlareConfig


class EvasiveManeuver(Enum):
    """Types of evasive maneuvers."""
    NONE = "none"                # No evasion
    BREAK_LEFT = "break_left"    # Hard left turn
    BREAK_RIGHT = "break_right"  # Hard right turn
    BREAK_DOWN = "break_down"    # Dive
    BREAK_UP = "break_up"        # Climb
    NOTCH = "notch"              # Beam (perpendicular to threat)
    SPLIT_S = "split_s"          # Roll inverted and pull down
    BARREL_ROLL = "barrel_roll"  # Defensive barrel roll
    RANDOM = "random"            # Random maneuvers


class ThreatAwareness(Enum):
    """Target's awareness of incoming missile."""
    UNAWARE = "unaware"          # No knowledge of threat
    ALERTED = "alerted"          # RWR warning or visual
    TRACKING = "tracking"        # Actively evading


@dataclass
class TargetState:
    """Complete state of target aircraft."""
    # Kinematics
    position: Vector3 = field(default_factory=lambda: Vector3(0, 5000, 10000))
    velocity: Vector3 = field(default_factory=lambda: Vector3(0, 0, -250))
    acceleration: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    orientation: Orientation = field(
        default_factory=lambda: Orientation(heading_deg=180))

    # Status
    current_maneuver: EvasiveManeuver = EvasiveManeuver.NONE
    awareness: ThreatAwareness = ThreatAwareness.UNAWARE

    # IR signature
    ir_signature: float = 0.8
    afterburner_active: bool = False

    # Performance
    current_g_load: float = 1.0
    current_mach: float = 0.8


class Target:
    """
    Evading target aircraft.

    Simulates:
    - Flight path with evasive maneuvers
    - IR signature (affected by afterburner)
    - Countermeasure deployment
    """

    def __init__(
        self,
        config: TargetConfig = FIGHTER_JET,
        initial_position: Optional[Vector3] = None,
        initial_velocity: Optional[Vector3] = None,
        initial_orientation: Optional[Orientation] = None,
        evasion_mode: EvasiveManeuver = EvasiveManeuver.NONE,
        flare_config: Optional[FlareConfig] = None,
        rng: Optional[random.Random] = None
    ):
        """
        Initialize target.

        Args:
            config: Target configuration
            initial_position: Starting position [m]
            initial_velocity: Starting velocity [m/s]
            initial_orientation: Starting orientation
            evasion_mode: Type of evasive maneuvers to perform
            flare_config: Flare configuration
            rng: Random number generator
        """
        self.config = config
        self.evasion_mode = evasion_mode
        self.rng = rng or random.Random()

        self.state = TargetState()

        if initial_position:
            self.state.position = initial_position
        if initial_velocity:
            self.state.velocity = initial_velocity
        if initial_orientation:
            self.state.orientation = initial_orientation
        else:
            self.state.orientation = velocity_to_orientation(
                self.state.velocity)

        # Set base IR signature
        self.state.ir_signature = config.ir_signature

        # Initialize flare dispenser
        self.flare_dispenser = FlareDispenser(
            total_flares=config.flare_count,
            config=flare_config,
            rng=self.rng
        )

        # Maneuver timing
        self.maneuver_start_time = 0.0
        self.maneuver_duration = 3.0  # seconds per maneuver (for random mode)
        self.time_in_maneuver = 0.0

        # Break turn parameters
        self.break_turn_duration = 2.0  # seconds to complete break turn (~90°)
        self.break_complete = False     # True when break turn is finished

        # Flare deployment timing
        self.last_flare_time = -float('inf')
        self.flare_interval = 1.0  # seconds between flare bursts

        # Aircraft physical properties for thrust/drag model
        self.mass_kg = 10000.0           # Typical fighter mass [kg]
        self.reference_area = 25.0       # Wing reference area [m²]
        self.drag_coefficient = 0.025    # Clean configuration Cd
        # Military thrust [N] (non-afterburner)
        self.mil_thrust_n = 50000.0
        self.ab_thrust_n = 80000.0       # Afterburner thrust [N]

        # Trajectory history
        self.trajectory_history: List[Vector3] = []
        self.max_history_length = 10000

    def set_awareness(self, awareness: ThreatAwareness):
        """Set target's threat awareness level."""
        self.state.awareness = awareness

    def alert_to_missile(self, missile_position: Vector3):
        """Alert target to incoming missile."""
        self.state.awareness = ThreatAwareness.TRACKING

        # Start evasive maneuvers if configured
        if self.evasion_mode == EvasiveManeuver.RANDOM:
            self._select_random_maneuver()
        elif self.evasion_mode != EvasiveManeuver.NONE:
            self.state.current_maneuver = self.evasion_mode

        self.maneuver_start_time = 0.0
        self.time_in_maneuver = 0.0
        self.break_complete = False  # Reset break turn state

    def _select_random_maneuver(self):
        """Select a random evasive maneuver."""
        maneuvers = [
            EvasiveManeuver.BREAK_LEFT,
            EvasiveManeuver.BREAK_RIGHT,
            EvasiveManeuver.BREAK_DOWN,
            EvasiveManeuver.BREAK_UP,
            EvasiveManeuver.NOTCH
        ]
        self.state.current_maneuver = self.rng.choice(maneuvers)

    def _compute_maneuver_acceleration(
        self,
        missile_position: Optional[Vector3] = None
    ) -> Vector3:
        """
        Compute acceleration for current maneuver.

        Args:
            missile_position: Position of threat missile (for notch)

        Returns:
            Commanded acceleration [m/s²]
        """
        if self.state.current_maneuver == EvasiveManeuver.NONE:
            return Vector3(0, 0, 0)

        max_accel = self.config.max_g * GRAVITY_M_S2
        speed = self.state.velocity.magnitude()

        if speed < 1.0:
            return Vector3(0, 0, 0)

        velocity_dir = self.state.velocity.normalized()

        # Define maneuver accelerations (in aircraft body frame)
        if self.state.current_maneuver == EvasiveManeuver.BREAK_LEFT:
            # Turn left at max G, but only until break is complete
            if self.break_complete:
                # Break turn done - fly straight
                return Vector3(0, 0, 0)
            up = Vector3(0, 1, 0)
            left = up.cross(velocity_dir).normalized()
            return left * max_accel

        elif self.state.current_maneuver == EvasiveManeuver.BREAK_RIGHT:
            # Turn right at max G, but only until break is complete
            if self.break_complete:
                # Break turn done - fly straight
                return Vector3(0, 0, 0)
            up = Vector3(0, 1, 0)
            right = velocity_dir.cross(up).normalized()
            return right * max_accel

        elif self.state.current_maneuver == EvasiveManeuver.BREAK_DOWN:
            # Dive
            return Vector3(0, -max_accel, 0)

        elif self.state.current_maneuver == EvasiveManeuver.BREAK_UP:
            # Climb
            return Vector3(0, max_accel - GRAVITY_M_S2, 0)

        elif self.state.current_maneuver == EvasiveManeuver.NOTCH:
            # Turn perpendicular to missile (beam maneuver)
            if missile_position:
                to_missile = missile_position - self.state.position
                to_missile.y = 0  # Horizontal plane
                if to_missile.magnitude() > 1.0:
                    to_missile = to_missile.normalized()
                    # Turn perpendicular
                    beam_dir = Vector3(-to_missile.z, 0, to_missile.x)
                    desired_vel = beam_dir * speed
                    accel = (desired_vel - self.state.velocity) * 2.0
                    if accel.magnitude() > max_accel:
                        accel = accel.normalized() * max_accel
                    return accel
            # Fallback to level turn
            up = Vector3(0, 1, 0)
            left = up.cross(velocity_dir).normalized()
            return left * max_accel

        elif self.state.current_maneuver == EvasiveManeuver.SPLIT_S:
            # Roll inverted and pull (dive and turn)
            return Vector3(0, -max_accel, 0)

        return Vector3(0, 0, 0)

    def engage_afterburner(self, engage: bool = True):
        """Engage or disengage afterburner."""
        self.state.afterburner_active = engage
        if engage:
            self.state.ir_signature = self.config.ir_signature * \
                self.config.afterburner_multiplier
        else:
            self.state.ir_signature = self.config.ir_signature

    def deploy_flares(self, current_time: float, num_flares: int = 2) -> int:
        """
        Deploy flares if cooldown elapsed.

        Args:
            current_time: Current simulation time [s]
            num_flares: Number of flares to deploy

        Returns:
            Number of flares deployed
        """
        if current_time - self.last_flare_time < self.flare_interval:
            return 0

        deployed = self.flare_dispenser.deploy_burst(
            self.state.position,
            self.state.velocity,
            current_time,
            num_flares
        )

        if deployed > 0:
            self.last_flare_time = current_time

        return deployed

    def update(
        self,
        dt: float,
        current_time: float,
        missile_position: Optional[Vector3] = None,
        auto_deploy_flares: bool = True
    ):
        """
        Update target state for one timestep using RK4 integration.

        Uses a simplified thrust/drag model so target loses energy realistically
        during high-G maneuvers.

        Args:
            dt: Time step [s]
            current_time: Current simulation time [s]
            missile_position: Position of incoming missile (if known)
            auto_deploy_flares: Automatically deploy flares when evading
        """
        # Update maneuver timing
        if self.state.current_maneuver != EvasiveManeuver.NONE:
            self.time_in_maneuver += dt

            # Check if break turn is complete (for break_left/right)
            if self.state.current_maneuver in (EvasiveManeuver.BREAK_LEFT, EvasiveManeuver.BREAK_RIGHT):
                if self.time_in_maneuver >= self.break_turn_duration:
                    self.break_complete = True

            # Switch maneuver periodically for random mode
            if self.evasion_mode == EvasiveManeuver.RANDOM:
                if self.time_in_maneuver > self.maneuver_duration:
                    self._select_random_maneuver()
                    self.time_in_maneuver = 0.0
                    self.break_complete = False  # Reset for new maneuver

        # Compute maneuver acceleration command
        maneuver_accel = self._compute_maneuver_acceleration(missile_position)

        # Determine thrust level based on afterburner state
        thrust_n = self.ab_thrust_n if self.state.afterburner_active else self.mil_thrust_n

        # Create EntityState for RK4 integration
        entity_state = EntityState(
            position=self.state.position,
            velocity=self.state.velocity,
            acceleration=self.state.acceleration,
            orientation=self.state.orientation,
            time=current_time,
            mass=self.mass_kg
        )

        # Define acceleration function for RK4
        def compute_acceleration(state: EntityState) -> Vector3:
            altitude = state.position.y
            speed = state.velocity.magnitude()

            # Thrust force in direction of travel
            if speed > 1.0:
                thrust_direction = state.velocity.normalized()
            else:
                thrust_direction = state.orientation.to_direction_vector()
            thrust_force = thrust_direction * thrust_n

            # Drag force (opposes velocity) - increases significantly during maneuvers
            # Induced drag scales with G-load squared (lift² for induced drag)
            # At 9G, induced drag is roughly 81x baseline
            g_load = maneuver_accel.magnitude() / GRAVITY_M_S2
            # Significant drag increase
            induced_drag_factor = 1.0 + (g_load ** 2) * 0.5
            effective_cd = self.drag_coefficient * induced_drag_factor

            drag_force = compute_drag(
                state.velocity,
                altitude,
                self.reference_area,
                effective_cd
            )

            # Gravity force
            gravity_force = Vector3(0, -state.mass * GRAVITY_M_S2, 0)

            # Total aerodynamic and propulsion forces
            total_force = thrust_force + drag_force + gravity_force

            # Convert to acceleration (a = F/m)
            base_accel = total_force * (1.0 / state.mass)

            # Add maneuver acceleration (from pilot input)
            total_accel = base_accel + maneuver_accel

            # Limit to max G
            max_accel = self.config.max_g * GRAVITY_M_S2
            accel_magnitude = total_accel.magnitude()
            if accel_magnitude > max_accel:
                total_accel = total_accel.normalized() * max_accel

            return total_accel

        # Propagate using RK4 integration
        new_state = propagate_trajectory_rk4(
            entity_state,
            compute_acceleration,
            dt
        )

        # Update target state from RK4 result
        self.state.position = new_state.position
        self.state.velocity = new_state.velocity
        self.state.acceleration = new_state.acceleration
        self.state.orientation = new_state.orientation

        # Limit speed to max Mach
        speed = self.state.velocity.magnitude()
        altitude = self.state.position.y
        max_speed = self.config.max_speed_mach * get_speed_of_sound(altitude)
        if speed > max_speed:
            self.state.velocity = self.state.velocity.normalized() * max_speed

        # Update performance metrics
        self.state.current_g_load = compute_g_load(self.state.acceleration)
        self.state.current_mach = mps_to_mach(speed, altitude)

        # Update flare dispenser
        self.flare_dispenser.update(dt)

        # Auto-deploy flares when evading
        if auto_deploy_flares and self.state.awareness == ThreatAwareness.TRACKING:
            if self.state.current_maneuver != EvasiveManeuver.NONE:
                self.deploy_flares(current_time, num_flares=2)

        # Store trajectory
        if len(self.trajectory_history) < self.max_history_length:
            self.trajectory_history.append(
                Vector3(
                    self.state.position.x,
                    self.state.position.y,
                    self.state.position.z
                )
            )

        # Prevent going underground
        if self.state.position.y < 100:  # Minimum altitude
            self.state.position.y = 100
            if self.state.velocity.y < 0:
                self.state.velocity.y = 0

    def get_flare_positions(self) -> List[Vector3]:
        """Get positions of active flares."""
        return self.flare_dispenser.get_active_positions()

    def get_flare_intensities(self) -> List[float]:
        """Get IR intensities of active flares."""
        return self.flare_dispenser.get_active_intensities()

    def get_trajectory(self) -> List[Tuple[float, float, float]]:
        """Get trajectory as list of (x, y, z) tuples."""
        return [pos.to_tuple() for pos in self.trajectory_history]

    def get_ir_signature(self) -> float:
        """Get current IR signature."""
        return self.state.ir_signature

    def get_state_dict(self) -> dict:
        """Get current state as dictionary."""
        return {
            'position': self.state.position.to_tuple(),
            'velocity': self.state.velocity.to_tuple(),
            'mach': self.state.current_mach,
            'g_load': self.state.current_g_load,
            'maneuver': self.state.current_maneuver.value,
            'ir_signature': self.state.ir_signature,
            'flares_remaining': self.flare_dispenser.flares_remaining
        }
