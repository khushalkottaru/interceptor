"""
Missile entity class for IR-guided air-to-air missiles.

Models:
- Missile state (position, velocity, fuel)
- IR seeker behavior (FOV, tracking, lock)
- Guidance logic (proportional navigation)
"""

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Tuple

from core.config import (
    MissileConfig, Vector3, Orientation, SeekerType, IRCCMLevel,
    AIM_9L, MISSILE_PRESETS
)
from core.physics import (
    EntityState, compute_drag, compute_thrust, compute_proportional_navigation,
    compute_intercept_geometry, propagate_trajectory_euler, propagate_trajectory_rk4,
    velocity_to_orientation, compute_g_load, limit_g_load, add_gaussian_noise,
    add_vector_noise, get_air_density, mps_to_mach, GRAVITY_M_S2, compute_net_force
)


class MissileStatus(Enum):
    """Current status of the missile."""
    PRELAUNCH = "prelaunch"      # Not yet launched
    FLYING = "flying"           # In flight, tracking
    HIT = "hit"                 # Intercept achieved
    MISS = "miss"               # Missed target
    DECOYED = "decoyed"         # Distracted by flare
    FUEL_OUT = "fuel_out"       # Exceeded max flight time
    LOST_TRACK = "lost_track"   # Lost target lock


class SeekerStatus(Enum):
    """Current status of the IR seeker."""
    UNCAGED = "uncaged"         # Searching for target
    LOCKED = "locked"           # Tracking target
    TRACKING_FLARE = "tracking_flare"  # Distracted by countermeasure
    LOST = "lost"               # Lost lock


@dataclass
class SeekerState:
    """State of the IR seeker head."""
    status: SeekerStatus = SeekerStatus.UNCAGED
    boresight_angle_deg: float = 0.0    # Current gimbal angle off boresight
    target_bearing_deg: float = 0.0     # Bearing to tracked target
    target_elevation_deg: float = 0.0   # Elevation to tracked target
    signal_strength: float = 0.0        # IR signal strength (0-1)
    lock_time_s: float = 0.0            # Time locked on target


@dataclass
class MissileState:
    """Complete state of a missile."""
    # Core kinematics
    position: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    velocity: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    acceleration: Vector3 = field(default_factory=lambda: Vector3(0, 0, 0))
    orientation: Orientation = field(default_factory=Orientation)

    # Time
    time_since_launch: float = 0.0

    # Status
    status: MissileStatus = MissileStatus.PRELAUNCH
    seeker: SeekerState = field(default_factory=SeekerState)

    # Propulsion
    fuel_remaining: float = 1.0  # 0-1 fraction
    is_motor_burning: bool = False
    current_thrust: float = 0.0
    current_mass: float = 0.0    # Current mass in kg (decreases as fuel burns)

    # Performance
    current_g_load: float = 0.0
    current_mach: float = 0.0


class Missile:
    """
    IR-guided missile simulation.

    Models a Fox-2 style heat-seeking missile with:
    - IR seeker with field of view constraints
    - Proportional navigation guidance
    - Aerodynamic performance limits
    - Countermeasure susceptibility
    """

    def __init__(
        self,
        config: MissileConfig = AIM_9L,
        initial_position: Optional[Vector3] = None,
        initial_velocity: Optional[Vector3] = None,
        initial_orientation: Optional[Orientation] = None,
        rng: Optional[random.Random] = None
    ):
        """
        Initialize missile with configuration.

        Args:
            config: Missile configuration (preset or custom)
            initial_position: Launch position [m]
            initial_velocity: Launch velocity [m/s] (from aircraft)
            initial_orientation: Launch orientation
            rng: Random number generator for noise
        """
        self.config = config
        self.rng = rng or random.Random()

        # Initialize state
        self.state = MissileState()

        if initial_position:
            self.state.position = initial_position
        if initial_velocity:
            self.state.velocity = initial_velocity
        if initial_orientation:
            self.state.orientation = initial_orientation

        # Compute reference area from diameter
        self.reference_area = math.pi * (config.diameter_m / 2) ** 2

        # Initialize mass to launch mass
        self.state.current_mass = config.mass_kg

        # Track history for visualization
        self.trajectory_history: List[Vector3] = []
        self.max_history_length = 10000

        # Navigation constant for proportional navigation
        self.nav_gain = 4.0

    def launch(self):
        """Launch the missile (start tracking and motor ignition)."""
        self.state.status = MissileStatus.FLYING
        self.state.seeker.status = SeekerStatus.LOCKED
        self.state.time_since_launch = 0.0
        self.state.is_motor_burning = True

    def can_track_target(
        self,
        target_position: Vector3,
        target_ir_signature: float = 1.0
    ) -> Tuple[bool, float]:
        """
        Check if seeker can track the target.

        Args:
            target_position: Target position [m]
            target_ir_signature: Target IR signature strength (0-1)

        Returns:
            (can_track, signal_strength)
        """
        # Line of sight to target
        los = target_position - self.state.position
        range_m = los.magnitude()

        if range_m < 1.0:
            return (True, 1.0)  # Very close

        los_unit = los.normalized()

        # Get missile boresight direction
        boresight = self.state.orientation.to_direction_vector()

        # Angle off boresight
        cos_angle = boresight.dot(los_unit)
        angle_off_boresight = math.degrees(
            math.acos(max(-1, min(1, cos_angle))))

        # Check gimbal limits
        if angle_off_boresight > self.config.seeker_gimbal_deg:
            return (False, 0.0)

        # Check aspect angle for rear-aspect only seekers
        if self.config.seeker_type == SeekerType.REAR_ASPECT:
            # Would need target velocity to compute aspect
            # For now, assume we have it if within gimbal
            pass

        # Signal strength decreases with range and angle
        # Use missile's max range as reference (signal ~0.5 at max range)
        max_range_m = self.config.max_range_km * 1000
        range_factor = 1.0 / (1.0 + (range_m / max_range_m) ** 2)
        angle_factor = math.cos(math.radians(angle_off_boresight))
        signal_strength = target_ir_signature * range_factor * angle_factor

        # Add noise to signal
        signal_strength = add_gaussian_noise(signal_strength, 0.05, self.rng)
        signal_strength = max(0.0, min(1.0, signal_strength))

        # Threshold for tracking
        min_signal = 0.1
        can_track = signal_strength > min_signal

        return (can_track, signal_strength)

    def compute_seeker_angles(
        self,
        target_position: Vector3
    ) -> Tuple[float, float]:
        """
        Compute seeker bearing and elevation to target.

        Args:
            target_position: Target position [m]

        Returns:
            (bearing_deg, elevation_deg) relative to missile heading
        """
        los = target_position - self.state.position
        range_m = los.magnitude()

        if range_m < 1.0:
            return (0.0, 0.0)

        # Convert to angles relative to missile orientation
        missile_heading = math.radians(self.state.orientation.heading_deg)
        missile_pitch = math.radians(self.state.orientation.pitch_deg)

        # Target bearing (horizontal angle)
        target_bearing = math.atan2(los.x, los.z)
        relative_bearing = target_bearing - missile_heading
        bearing_deg = math.degrees(relative_bearing)

        # Target elevation
        ground_range = math.sqrt(los.x ** 2 + los.z ** 2)
        target_elevation = math.atan2(los.y, ground_range)
        relative_elevation = target_elevation - missile_pitch
        elevation_deg = math.degrees(relative_elevation)

        return (bearing_deg, elevation_deg)

    def update_seeker(
        self,
        target_position: Vector3,
        target_velocity: Vector3,
        target_ir_signature: float,
        flare_positions: List[Vector3] = None,
        flare_intensities: List[float] = None,
        dt: float = 0.01
    ):
        """
        Update seeker state based on target and countermeasures.

        Args:
            target_position: Target position [m]
            target_velocity: Target velocity [m/s]
            target_ir_signature: Target IR intensity
            flare_positions: List of active flare positions
            flare_intensities: List of flare IR intensities
            dt: Time step [s]
        """
        if self.state.status != MissileStatus.FLYING:
            return

        # Check if we can still track target
        can_track, signal = self.can_track_target(
            target_position, target_ir_signature)

        # Handle flare distraction
        if flare_positions and len(flare_positions) > 0:
            flare_attraction = self._compute_flare_attraction(
                target_position, target_ir_signature,
                flare_positions, flare_intensities or [
                    1.0] * len(flare_positions)
            )

            if flare_attraction > 0.5:  # High probability of being decoyed
                # Roll dice for flare effectiveness
                if self.rng.random() < flare_attraction:
                    self.state.seeker.status = SeekerStatus.TRACKING_FLARE
                    self.state.status = MissileStatus.DECOYED
                    return

        if can_track:
            self.state.seeker.status = SeekerStatus.LOCKED
            self.state.seeker.signal_strength = signal
            self.state.seeker.lock_time_s += dt

            bearing, elevation = self.compute_seeker_angles(target_position)
            self.state.seeker.target_bearing_deg = bearing
            self.state.seeker.target_elevation_deg = elevation
        else:
            # Lost track
            self.state.seeker.status = SeekerStatus.LOST
            self.state.seeker.signal_strength = 0
            self.state.status = MissileStatus.LOST_TRACK

    def _compute_flare_attraction(
        self,
        target_position: Vector3,
        target_ir: float,
        flare_positions: List[Vector3],
        flare_intensities: List[float]
    ) -> float:
        """
        Compute probability of being decoyed by flares.

        Based on missile IRCCM level and flare/target IR ratio.
        """
        # Base effectiveness based on IRCCM level
        irccm_factors = {
            IRCCMLevel.NONE: 0.95,
            IRCCMLevel.LOW: 0.7,
            IRCCMLevel.MEDIUM: 0.4,
            IRCCMLevel.HIGH: 0.1
        }
        base_susceptibility = irccm_factors.get(self.config.irccm_level, 0.5)

        # Find strongest flare attraction
        max_attraction = 0.0

        for flare_pos, flare_ir in zip(flare_positions, flare_intensities):
            # Distance factors
            flare_range = (flare_pos - self.state.position).magnitude()
            target_range = (target_position - self.state.position).magnitude()

            if flare_range < 1.0:
                flare_range = 1.0

            # Flare is more attractive if closer and brighter
            flare_signal = flare_ir / (flare_range / 1000.0)
            target_signal = target_ir / (target_range / 1000.0)

            if target_signal > 0:
                ir_ratio = flare_signal / target_signal
            else:
                ir_ratio = 10.0

            # Attraction probability
            attraction = base_susceptibility * min(1.0, ir_ratio)
            max_attraction = max(max_attraction, attraction)

        return max_attraction

    def compute_guidance(
        self,
        target_position: Vector3,
        target_velocity: Vector3
    ) -> Vector3:
        """
        Compute guidance acceleration command.

        Args:
            target_position: Target position [m]
            target_velocity: Target velocity [m/s]

        Returns:
            Commanded acceleration [m/s²]
        """
        if self.state.seeker.status != SeekerStatus.LOCKED:
            # No guidance if not locked
            return Vector3(0, -GRAVITY_M_S2, 0)  # Just gravity

        # Add noise to target position/velocity (simulates tracking errors)
        noise_scale = 20.0 / (1.0 + self.state.seeker.signal_strength * 10)
        noisy_target_pos = add_vector_noise(
            target_position, noise_scale, self.rng)
        noisy_target_vel = add_vector_noise(
            target_velocity, noise_scale * 0.1, self.rng)

        # Proportional navigation
        accel_cmd = compute_proportional_navigation(
            self.state.position,
            self.state.velocity,
            noisy_target_pos,
            noisy_target_vel,
            nav_gain=self.nav_gain,
            max_accel_g=self.config.max_g
        )

        return accel_cmd

    def update(
        self,
        target_position: Vector3,
        target_velocity: Vector3,
        target_ir_signature: float = 1.0,
        flare_positions: List[Vector3] = None,
        flare_intensities: List[float] = None,
        dt: float = 0.01
    ) -> MissileStatus:
        """
        Update missile state for one timestep.

        Args:
            target_position: Target position [m]
            target_velocity: Target velocity [m/s]
            target_ir_signature: Target IR signature strength
            flare_positions: Active flare positions
            flare_intensities: Active flare intensities
            dt: Time step [s]

        Returns:
            Current missile status
        """
        if self.state.status != MissileStatus.FLYING:
            return self.state.status

        # Update time
        self.state.time_since_launch += dt

        # Check fuel/coast time
        max_flight_time = self.config.burn_time_s + self.config.coast_time_s
        if self.state.time_since_launch > max_flight_time:
            self.state.status = MissileStatus.FUEL_OUT
            return self.state.status

        # Update seeker
        self.update_seeker(
            target_position, target_velocity, target_ir_signature,
            flare_positions, flare_intensities, dt
        )

        if self.state.status != MissileStatus.FLYING:
            return self.state.status

        # Compute propulsion state for this timestep
        prop_state = compute_thrust(
            self.state.time_since_launch,
            self.config.burn_time_s
        )
        self.state.is_motor_burning = prop_state.is_burning
        self.state.current_thrust = prop_state.thrust_n
        self.state.fuel_remaining = prop_state.fuel_remaining_kg

        # Calculate current mass based on fuel burned
        # fuel_remaining is 1.0 at launch, 0.0 when all fuel is burned
        fuel_burned = (1.0 - self.state.fuel_remaining) * \
            self.config.fuel_mass_kg
        current_mass = self.config.mass_kg - fuel_burned
        self.state.current_mass = current_mass

        # Calculate mass at end of timestep for RK4 mass interpolation
        prop_state_end = compute_thrust(
            self.state.time_since_launch + dt,
            self.config.burn_time_s
        )
        fuel_burned_end = (1.0 - prop_state_end.fuel_remaining_kg) * \
            self.config.fuel_mass_kg
        new_mass = self.config.mass_kg - fuel_burned_end

        # Create EntityState for RK4 integration
        entity_state = EntityState(
            position=self.state.position,
            velocity=self.state.velocity,
            acceleration=self.state.acceleration,
            orientation=self.state.orientation,
            time=self.state.time_since_launch,
            mass=current_mass
        )

        # Define acceleration function for RK4 that computes total acceleration at any state
        # This closure captures target info for guidance computation
        def compute_acceleration(state: EntityState) -> Vector3:
            altitude = state.position.y
            speed = state.velocity.magnitude()

            # Thrust force in direction of travel
            # Use thrust from current propulsion state (approximation for RK4 substeps)
            if speed > 1.0:
                thrust_direction = state.velocity.normalized()
            else:
                thrust_direction = state.orientation.to_direction_vector()
            thrust_force = thrust_direction * prop_state.thrust_n

            # Drag force (opposes velocity)
            drag_force = compute_drag(
                state.velocity,
                altitude,
                self.reference_area,
                cd=0.3
            )

            # Gravity force (F = mg)
            gravity_force = Vector3(0, -state.mass * GRAVITY_M_S2, 0)

            # Base physics forces
            base_force = thrust_force + drag_force + gravity_force

            # Convert base force to acceleration (a = F/m)
            base_accel = base_force * (1.0 / state.mass)

            # Guidance acceleration
            # Compute guidance based on current state position/velocity
            if self.state.seeker.status == SeekerStatus.LOCKED:
                guidance_accel = compute_proportional_navigation(
                    state.position,
                    state.velocity,
                    target_position,
                    target_velocity,
                    nav_gain=self.nav_gain,
                    max_accel_g=self.config.max_g
                )
            else:
                guidance_accel = Vector3(0, 0, 0)

            # Total acceleration
            total_accel = base_accel + guidance_accel

            # Limit to max G-load
            max_accel = self.config.max_g * GRAVITY_M_S2
            accel_magnitude = total_accel.magnitude()
            if accel_magnitude > max_accel:
                total_accel = total_accel.normalized() * max_accel

            return total_accel

        # Propagate using RK4 integration
        new_entity_state = propagate_trajectory_rk4(
            entity_state,
            compute_acceleration,
            dt,
            new_mass=new_mass
        )

        # Update missile state from RK4 result
        self.state.position = new_entity_state.position
        self.state.velocity = new_entity_state.velocity
        self.state.acceleration = new_entity_state.acceleration
        self.state.orientation = new_entity_state.orientation

        # Update performance metrics
        speed = self.state.velocity.magnitude()
        altitude = self.state.position.y
        self.state.current_g_load = compute_g_load(self.state.acceleration)
        self.state.current_mach = mps_to_mach(speed, altitude)

        # Store trajectory
        if len(self.trajectory_history) < self.max_history_length:
            self.trajectory_history.append(
                Vector3(self.state.position.x,
                        self.state.position.y, self.state.position.z)
            )

        # Check for intercept
        range_to_target = (target_position - self.state.position).magnitude()
        kill_radius = 5.0  # meters - proximity fuze radius

        if range_to_target < kill_radius:
            self.state.status = MissileStatus.HIT
            return self.state.status

        # Check for ground impact
        if self.state.position.y < 0:
            self.state.status = MissileStatus.MISS
            return self.state.status

        return self.state.status

    def get_trajectory(self) -> List[Tuple[float, float, float]]:
        """Get trajectory as list of (x, y, z) tuples."""
        return [pos.to_tuple() for pos in self.trajectory_history]

    def get_state_dict(self) -> dict:
        """Get current state as dictionary."""
        return {
            'position': self.state.position.to_tuple(),
            'velocity': self.state.velocity.to_tuple(),
            'status': self.state.status.value,
            'mach': self.state.current_mach,
            'g_load': self.state.current_g_load,
            'fuel': self.state.fuel_remaining,
            'mass': self.state.current_mass,
            'time': self.state.time_since_launch
        }
