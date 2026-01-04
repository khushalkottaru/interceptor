"""
Physics engine for missile flight dynamics and kinematics.

Provides core computations for:
- Aerodynamic forces (drag, lift)
- Propulsion modeling
- Trajectory integration (RK4)
- Intercept geometry calculations
- Proportional navigation guidance
"""

import math
from dataclasses import dataclass
from typing import Tuple, Optional

from core.config import Vector3, Orientation, MissileConfig


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

# Atmospheric constants (ISA - International Standard Atmosphere)
SEA_LEVEL_PRESSURE_PA = 101325.0
SEA_LEVEL_TEMPERATURE_K = 288.15
SEA_LEVEL_DENSITY_KG_M3 = 1.225
TEMPERATURE_LAPSE_RATE = 0.0065  # K/m
GAS_CONSTANT = 287.05  # J/(kg·K)
GRAVITY_M_S2 = 9.80665
GAMMA = 1.4  # Ratio of specific heats for air

# Speed of sound at sea level
SPEED_OF_SOUND_SEA_LEVEL_MPS = 340.29


# =============================================================================
# ATMOSPHERIC MODEL
# =============================================================================

def get_air_density(altitude_m: float) -> float:
    """
    Calculate air density at given altitude using ISA model.

    Valid for troposphere (0-11km).

    Args:
        altitude_m: Altitude in meters

    Returns:
        Air density in kg/m³
    """
    if altitude_m < 0:
        altitude_m = 0

    # Troposphere model (valid up to ~11km)
    if altitude_m <= 11000:
        temperature = SEA_LEVEL_TEMPERATURE_K - TEMPERATURE_LAPSE_RATE * altitude_m
        pressure_ratio = (temperature / SEA_LEVEL_TEMPERATURE_K) ** (
            GRAVITY_M_S2 / (TEMPERATURE_LAPSE_RATE * GAS_CONSTANT))
        density = SEA_LEVEL_DENSITY_KG_M3 * pressure_ratio * \
            (SEA_LEVEL_TEMPERATURE_K / temperature)
        return density
    else:
        # Simplified stratosphere (constant temperature)
        # Density continues to decrease exponentially
        tropo_density = get_air_density(11000)
        scale_height = 6500  # meters
        return tropo_density * math.exp(-(altitude_m - 11000) / scale_height)


def get_speed_of_sound(altitude_m: float) -> float:
    """
    Calculate speed of sound at given altitude.

    Args:
        altitude_m: Altitude in meters

    Returns:
        Speed of sound in m/s
    """
    if altitude_m < 0:
        altitude_m = 0

    if altitude_m <= 11000:
        temperature = SEA_LEVEL_TEMPERATURE_K - TEMPERATURE_LAPSE_RATE * altitude_m
    else:
        temperature = 216.65  # Constant in stratosphere

    return math.sqrt(GAMMA * GAS_CONSTANT * temperature)


def mps_to_mach(velocity_mps: float, altitude_m: float) -> float:
    """Convert velocity in m/s to Mach number at given altitude."""
    return velocity_mps / get_speed_of_sound(altitude_m)


def mach_to_mps(mach: float, altitude_m: float) -> float:
    """Convert Mach number to velocity in m/s at given altitude."""
    return mach * get_speed_of_sound(altitude_m)


# =============================================================================
# AERODYNAMIC FORCES
# =============================================================================

@dataclass
class AeroCoefficients:
    """Aerodynamic coefficients for a missile."""
    cd_0: float = 0.3        # Zero-lift drag coefficient
    cd_alpha: float = 2.0    # Drag due to angle of attack (per radian²)
    cl_alpha: float = 10.0   # Lift curve slope (per radian)


def compute_drag(
    velocity: Vector3,
    altitude_m: float,
    reference_area_m2: float,
    cd: float = 0.3
) -> Vector3:
    """
    Compute aerodynamic drag force.

    Args:
        velocity: Velocity vector [m/s]
        altitude_m: Altitude [m]
        reference_area_m2: Reference cross-sectional area [m²]
        cd: Drag coefficient

    Returns:
        Drag force vector [N] (opposite to velocity)
    """
    speed = velocity.magnitude()
    if speed < 1e-6:
        return Vector3(0, 0, 0)

    rho = get_air_density(altitude_m)
    dynamic_pressure = 0.5 * rho * speed ** 2
    drag_magnitude = cd * reference_area_m2 * dynamic_pressure

    # Drag opposes velocity
    drag_direction = velocity.normalized() * -1
    return drag_direction * drag_magnitude


def compute_mach_drag_factor(mach: float) -> float:
    """
    Compute drag coefficient multiplier based on Mach number.

    Accounts for wave drag in transonic/supersonic regime.

    Args:
        mach: Mach number

    Returns:
        Drag multiplier (1.0 = subsonic, higher for transonic)
    """
    if mach < 0.8:
        return 1.0
    elif mach < 1.2:
        # Transonic drag rise
        return 1.0 + 2.0 * (mach - 0.8) ** 2
    else:
        # Supersonic
        return 1.0 + 0.5 / mach


# =============================================================================
# PROPULSION
# =============================================================================

@dataclass
class PropulsionState:
    """Current state of missile propulsion."""
    is_burning: bool
    thrust_n: float
    fuel_remaining_kg: float


def compute_thrust(
    time_since_launch: float,
    burn_time_s: float,
    total_impulse_ns: float = 15000.0,  # Typical for Sidewinder
) -> PropulsionState:
    """
    Compute thrust at given time.

    Uses simple constant-thrust model during burn.

    Args:
        time_since_launch: Time since motor ignition [s]
        burn_time_s: Total burn time [s]
        total_impulse_ns: Total impulse [N·s]

    Returns:
        PropulsionState with current thrust
    """
    if time_since_launch < 0:
        return PropulsionState(is_burning=False, thrust_n=0, fuel_remaining_kg=1.0)

    if time_since_launch < burn_time_s:
        # Constant thrust during burn
        thrust = total_impulse_ns / burn_time_s
        fuel_remaining = 1.0 - (time_since_launch / burn_time_s)
        return PropulsionState(is_burning=True, thrust_n=thrust, fuel_remaining_kg=fuel_remaining)
    else:
        return PropulsionState(is_burning=False, thrust_n=0, fuel_remaining_kg=0)


# =============================================================================
# KINEMATICS AND TRAJECTORY
# =============================================================================

@dataclass
class EntityState:
    """Complete state of a flying entity."""
    position: Vector3
    velocity: Vector3
    acceleration: Vector3
    orientation: Orientation
    time: float
    mass: float = 1.0  # Current mass in kg (decreases as fuel burns)


def compute_net_force(
    state: EntityState,
    thrust_n: float,
    reference_area_m2: float,
    cd: float = 0.3
) -> Vector3:
    """
    Compute net force on a flying entity.

    This function properly couples the entity's mass with force calculations.
    The gravity force is computed using the entity's current mass (F = mg).

    Args:
        state: Current entity state (includes mass, position, velocity)
        thrust_n: Thrust force magnitude [N]
        reference_area_m2: Reference cross-sectional area for drag [m²]
        cd: Drag coefficient

    Returns:
        Net force vector [N]
    """
    # Thrust force in direction of travel
    speed = state.velocity.magnitude()
    if speed > 1.0:
        thrust_direction = state.velocity.normalized()
    else:
        thrust_direction = state.orientation.to_direction_vector()
    thrust_force = thrust_direction * thrust_n

    # Drag force (opposes velocity)
    altitude = state.position.y
    drag_force = compute_drag(state.velocity, altitude, reference_area_m2, cd)

    # Gravity force - uses entity's current mass (F = mg)
    gravity_force = Vector3(0, -state.mass * GRAVITY_M_S2, 0)

    return thrust_force + drag_force + gravity_force


def propagate_trajectory_euler(
    state: EntityState,
    total_force: Vector3,
    dt: float,
    new_mass: Optional[float] = None
) -> EntityState:
    """
    Propagate trajectory using Euler integration.

    Computes acceleration from force and instantaneous mass (F = ma -> a = F/m).
    This properly couples kinematics with propulsion as mass changes from fuel burn.

    Args:
        state: Current entity state (includes mass)
        total_force: Total force acting on entity [N]
        dt: Time step [s]
        new_mass: Optional new mass for next state (if fuel was consumed)

    Returns:
        New EntityState after dt
    """
    # Compute acceleration from force and current mass (a = F/m)
    if state.mass <= 0:
        raise ValueError("Entity mass must be positive")
    acceleration = total_force * (1.0 / state.mass)

    new_velocity = state.velocity + acceleration * dt
    new_position = state.position + state.velocity * \
        dt + acceleration * (0.5 * dt * dt)

    # Update orientation to match velocity
    new_orientation = velocity_to_orientation(new_velocity)

    # Use new mass if provided (fuel consumption), otherwise keep current mass
    final_mass = new_mass if new_mass is not None else state.mass

    return EntityState(
        position=new_position,
        velocity=new_velocity,
        acceleration=acceleration,
        orientation=new_orientation,
        time=state.time + dt,
        mass=final_mass
    )


def propagate_trajectory_rk4(
    state: EntityState,
    accel_func,
    dt: float,
    new_mass: Optional[float] = None
) -> EntityState:
    """
    Propagate trajectory using 4th-order Runge-Kutta integration.

    Args:
        state: Current entity state (includes mass)
        accel_func: Function(EntityState) -> Vector3 acceleration [m/s²]
                    Receives full state so it can compute forces and divide by mass
        dt: Time step [s]
        new_mass: Optional new mass for next state (if fuel was consumed)

    Returns:
        New EntityState after dt
    """
    if state.mass <= 0:
        raise ValueError("Entity mass must be positive")

    # For RK4 with varying mass, we linearly interpolate mass during the timestep
    # This is an approximation - mass changes continuously during burn
    final_mass = new_mass if new_mass is not None else state.mass
    mass_rate = (final_mass - state.mass) / dt if dt > 0 else 0.0

    # k1 - state at start
    s1 = state
    a1 = accel_func(s1)
    v1 = s1.velocity

    # k2 - state at midpoint using k1 derivatives
    s2 = EntityState(
        position=state.position + v1 * (dt / 2),
        velocity=state.velocity + a1 * (dt / 2),
        acceleration=a1,
        orientation=state.orientation,
        time=state.time + dt / 2,
        mass=state.mass + mass_rate * (dt / 2)
    )
    a2 = accel_func(s2)
    v2 = s2.velocity

    # k3 - state at midpoint using k2 derivatives
    s3 = EntityState(
        position=state.position + v2 * (dt / 2),
        velocity=state.velocity + a2 * (dt / 2),
        acceleration=a2,
        orientation=state.orientation,
        time=state.time + dt / 2,
        mass=state.mass + mass_rate * (dt / 2)
    )
    a3 = accel_func(s3)
    v3 = s3.velocity

    # k4 - state at end using k3 derivatives
    s4 = EntityState(
        position=state.position + v3 * dt,
        velocity=state.velocity + a3 * dt,
        acceleration=a3,
        orientation=state.orientation,
        time=state.time + dt,
        mass=final_mass
    )
    a4 = accel_func(s4)

    # Combine using RK4 weights
    new_velocity = state.velocity + (a1 + a2 * 2 + a3 * 2 + a4) * (dt / 6)
    new_position = state.position + \
        (v1 + v2 * 2 + v3 * 2 + s4.velocity) * (dt / 6)
    new_acceleration = a4

    new_orientation = velocity_to_orientation(new_velocity)

    return EntityState(
        position=new_position,
        velocity=new_velocity,
        acceleration=new_acceleration,
        orientation=new_orientation,
        time=state.time + dt,
        mass=final_mass
    )


def velocity_to_orientation(velocity: Vector3) -> Orientation:
    """
    Convert velocity vector to orientation angles.

    Args:
        velocity: Velocity vector

    Returns:
        Orientation with heading and pitch (roll = 0)
    """
    speed = velocity.magnitude()
    if speed < 1e-6:
        return Orientation(0, 0, 0)

    # Heading (azimuth from north)
    heading_rad = math.atan2(velocity.x, velocity.z)
    heading_deg = math.degrees(heading_rad)
    if heading_deg < 0:
        heading_deg += 360

    # Pitch (elevation)
    ground_speed = math.sqrt(velocity.x ** 2 + velocity.z ** 2)
    pitch_rad = math.atan2(velocity.y, ground_speed)
    pitch_deg = math.degrees(pitch_rad)

    return Orientation(heading_deg, pitch_deg, 0)


# =============================================================================
# INTERCEPT GEOMETRY
# =============================================================================

@dataclass
class InterceptSolution:
    """Solution for missile-target intercept."""
    time_to_intercept: float     # Estimated time to intercept [s]
    intercept_point: Vector3     # Predicted intercept location
    closing_velocity: float      # Closure rate [m/s]
    aspect_angle_deg: float      # Angle off target tail (0=tail, 180=nose)
    lead_angle_deg: float        # Required lead angle for intercept
    is_valid: bool               # Whether a valid solution exists


def compute_intercept_geometry(
    missile_pos: Vector3,
    missile_vel: Vector3,
    target_pos: Vector3,
    target_vel: Vector3
) -> InterceptSolution:
    """
    Compute intercept geometry for missile and target.

    Uses simple linear extrapolation (assumes constant velocities).

    Args:
        missile_pos: Missile position [m]
        missile_vel: Missile velocity [m/s]
        target_pos: Target position [m]
        target_vel: Target velocity [m/s]

    Returns:
        InterceptSolution with geometry data
    """
    # Line-of-sight vector
    los = target_pos - missile_pos
    range_m = los.magnitude()

    if range_m < 1.0:
        # Already at intercept
        return InterceptSolution(
            time_to_intercept=0,
            intercept_point=target_pos,
            closing_velocity=0,
            aspect_angle_deg=0,
            lead_angle_deg=0,
            is_valid=True
        )

    los_unit = los.normalized()

    # Relative velocity
    rel_vel = missile_vel - target_vel
    closing_velocity = -rel_vel.dot(los_unit)  # Positive when closing

    if closing_velocity <= 0:
        # Not closing - no valid intercept
        return InterceptSolution(
            time_to_intercept=float('inf'),
            intercept_point=Vector3(0, 0, 0),
            closing_velocity=closing_velocity,
            aspect_angle_deg=0,
            lead_angle_deg=0,
            is_valid=False
        )

    # Simple time to intercept
    time_to_intercept = range_m / closing_velocity

    # Intercept point (linear extrapolation)
    intercept_point = target_pos + target_vel * time_to_intercept

    # Aspect angle (angle between target velocity and LOS)
    target_speed = target_vel.magnitude()
    if target_speed > 1e-6:
        target_dir = target_vel.normalized()
        # Angle between target heading and line from target to missile
        los_from_target = (missile_pos - target_pos).normalized()
        cos_aspect = target_dir.dot(los_from_target)
        aspect_angle_deg = math.degrees(math.acos(max(-1, min(1, cos_aspect))))
    else:
        aspect_angle_deg = 0

    # Lead angle
    lead_vector = intercept_point - missile_pos
    if lead_vector.magnitude() > 1e-6 and los.magnitude() > 1e-6:
        lead_unit = lead_vector.normalized()
        cos_lead = los_unit.dot(lead_unit)
        lead_angle_deg = math.degrees(math.acos(max(-1, min(1, cos_lead))))
    else:
        lead_angle_deg = 0

    return InterceptSolution(
        time_to_intercept=time_to_intercept,
        intercept_point=intercept_point,
        closing_velocity=closing_velocity,
        aspect_angle_deg=aspect_angle_deg,
        lead_angle_deg=lead_angle_deg,
        is_valid=True
    )


# =============================================================================
# PROPORTIONAL NAVIGATION GUIDANCE
# =============================================================================

def compute_proportional_navigation(
    missile_pos: Vector3,
    missile_vel: Vector3,
    target_pos: Vector3,
    target_vel: Vector3,
    nav_gain: float = 4.0,
    max_accel_g: float = 35.0
) -> Vector3:
    """
    Compute acceleration command using Proportional Navigation.

    PN Law: a = N * Vc * dλ/dt

    Where:
    - N = navigation constant (typically 3-5)
    - Vc = closing velocity
    - dλ/dt = line-of-sight rate

    Args:
        missile_pos: Missile position [m]
        missile_vel: Missile velocity [m/s]
        target_pos: Target position [m]
        target_vel: Target velocity [m/s]
        nav_gain: Navigation constant N
        max_accel_g: Maximum acceleration limit [G]

    Returns:
        Commanded acceleration vector [m/s²]
    """
    # Line of sight
    los = target_pos - missile_pos
    range_m = los.magnitude()

    if range_m < 1.0:
        return Vector3(0, 0, 0)

    los_unit = los.normalized()

    # Relative velocity
    rel_vel = target_vel - missile_vel  # Target motion relative to missile
    closing_velocity = -rel_vel.dot(los_unit)

    # Line of sight rate (angular velocity)
    # ω = (V_r - (V_r · LOS) * LOS) / R
    los_rate = (rel_vel - los_unit * rel_vel.dot(los_unit)) * (1.0 / range_m)

    # PN acceleration command
    # a_cmd = N * Vc * ω
    accel_cmd = los_rate * (nav_gain * abs(closing_velocity))

    # Add gravity compensation for better accuracy
    accel_cmd = accel_cmd + Vector3(0, GRAVITY_M_S2, 0)

    # Limit to maximum G
    max_accel = max_accel_g * GRAVITY_M_S2
    accel_magnitude = accel_cmd.magnitude()

    if accel_magnitude > max_accel:
        accel_cmd = accel_cmd.normalized() * max_accel

    return accel_cmd


def compute_lead_pursuit(
    missile_pos: Vector3,
    missile_vel: Vector3,
    target_pos: Vector3,
    target_vel: Vector3,
    missile_speed: float,
    max_accel_g: float = 35.0
) -> Vector3:
    """
    Compute acceleration for lead pursuit guidance.

    Simpler than PN, aims at predicted intercept point.

    Args:
        missile_pos: Missile position [m]
        missile_vel: Missile velocity [m/s]
        target_pos: Target position [m]
        target_vel: Target velocity [m/s]
        missile_speed: Current missile speed [m/s]
        max_accel_g: Maximum acceleration [G]

    Returns:
        Commanded acceleration [m/s²]
    """
    intercept = compute_intercept_geometry(
        missile_pos, missile_vel, target_pos, target_vel)

    if not intercept.is_valid:
        # Pure pursuit if no valid intercept
        los = target_pos - missile_pos
        desired_direction = los.normalized()
    else:
        # Lead pursuit to intercept point
        lead_vector = intercept.intercept_point - missile_pos
        desired_direction = lead_vector.normalized()

    # Desired velocity
    desired_velocity = desired_direction * missile_speed

    # Required acceleration to achieve desired velocity
    # Using simple proportional control
    velocity_error = desired_velocity - missile_vel
    accel_cmd = velocity_error * 2.0  # Proportional gain

    # Limit acceleration
    max_accel = max_accel_g * GRAVITY_M_S2
    accel_magnitude = accel_cmd.magnitude()

    if accel_magnitude > max_accel:
        accel_cmd = accel_cmd.normalized() * max_accel

    return accel_cmd


# =============================================================================
# G-FORCE CALCULATIONS
# =============================================================================

def compute_g_load(acceleration: Vector3) -> float:
    """
    Compute G-load from acceleration vector.

    Args:
        acceleration: Acceleration vector [m/s²]

    Returns:
        G-load (magnitude relative to gravity)
    """
    return acceleration.magnitude() / GRAVITY_M_S2


def limit_g_load(
    acceleration: Vector3,
    max_g: float
) -> Vector3:
    """
    Limit acceleration to maximum G-load.

    Args:
        acceleration: Desired acceleration [m/s²]
        max_g: Maximum allowed G-load

    Returns:
        Limited acceleration vector [m/s²]
    """
    max_accel = max_g * GRAVITY_M_S2
    accel_mag = acceleration.magnitude()

    if accel_mag > max_accel:
        return acceleration.normalized() * max_accel
    return acceleration


# =============================================================================
# PROBABILITY / NOISE MODELS
# =============================================================================

def add_gaussian_noise(
    value: float,
    std_dev: float,
    rng=None
) -> float:
    """
    Add Gaussian noise to a value.

    Args:
        value: Base value
        std_dev: Standard deviation of noise
        rng: Random number generator (optional)

    Returns:
        Value with noise added
    """
    import random
    if rng is None:
        rng = random
    return value + rng.gauss(0, std_dev)


def add_vector_noise(
    vector: Vector3,
    std_dev: float,
    rng=None
) -> Vector3:
    """
    Add Gaussian noise to each component of a vector.

    Args:
        vector: Base vector
        std_dev: Standard deviation of noise for each component
        rng: Random number generator (optional)

    Returns:
        Vector with noise added
    """
    return Vector3(
        add_gaussian_noise(vector.x, std_dev, rng),
        add_gaussian_noise(vector.y, std_dev, rng),
        add_gaussian_noise(vector.z, std_dev, rng)
    )


def add_angular_noise(
    orientation: Orientation,
    std_dev_deg: float,
    rng=None
) -> Orientation:
    """
    Add Gaussian noise to orientation angles.

    Args:
        orientation: Base orientation
        std_dev_deg: Standard deviation in degrees
        rng: Random number generator

    Returns:
        Orientation with noise
    """
    return Orientation(
        heading_deg=add_gaussian_noise(
            orientation.heading_deg, std_dev_deg, rng),
        pitch_deg=add_gaussian_noise(orientation.pitch_deg, std_dev_deg, rng),
        roll_deg=add_gaussian_noise(orientation.roll_deg, std_dev_deg, rng)
    )
