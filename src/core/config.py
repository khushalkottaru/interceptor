"""
Fox-2 Missile Simulation Configuration

Contains missile presets based on declassified specifications and configurable
parameters for targets, flares, and simulation settings.

DATA SOURCES:
- AIM-9B/D/E/G specs: DCS World documentation, War Thunder wiki (cross-referenced)
- AIM-9L/M specs: USAF Fact Sheet, HistoryNet, Navy.mil
- General sidewinder data: Wikipedia (AIM-9 Sidewinder), af.mil

Values marked with [PLACEHOLDER] are estimated and should be verified/adjusted.
Values marked with [DECLASSIFIED] come from publicly available military documents.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import math


class SeekerType(Enum):
    """IR seeker engagement capability."""
    REAR_ASPECT = "rear_aspect"      # Can only lock from behind target
    ALL_ASPECT = "all_aspect"        # Can lock from any angle


class IRCCMLevel(Enum):
    """Infrared Counter-Countermeasures capability."""
    NONE = "none"           # No flare resistance
    LOW = "low"             # Basic flare rejection
    MEDIUM = "medium"       # Improved signal processing
    HIGH = "high"           # Advanced IRCCM (imaging seeker)


@dataclass
class MissileConfig:
    """
    Missile configuration parameters.

    All values are annotated with their data source reliability.
    """
    name: str

    # Performance parameters
    max_range_km: float          # Maximum effective range [km]
    min_range_km: float          # Minimum arming distance [km]
    max_speed_mach: float        # Maximum velocity [Mach]
    max_g: float                 # Maximum G load

    # Propulsion
    burn_time_s: float           # Motor burn duration [seconds]
    coast_time_s: float          # Unpowered flight capability [seconds]

    # Seeker parameters
    seeker_type: SeekerType      # Engagement aspect capability
    seeker_fov_deg: float        # Seeker field of view [degrees]
    seeker_gimbal_deg: float     # Maximum off-boresight angle [degrees]
    track_rate_deg_s: float      # Seeker tracking rate [deg/sec]

    # Countermeasure resistance
    irccm_level: IRCCMLevel      # Flare resistance capability

    # Physical properties
    mass_kg: float               # Launch mass [kg]
    # Propellant mass [kg] - mass decreases as fuel burns
    fuel_mass_kg: float
    diameter_m: float            # Body diameter [m]
    length_m: float              # Total length [m]

    # Data source notes
    source_notes: str = ""       # Documentation of data sources


# =============================================================================
# MISSILE PRESETS - Ordered by declassification confidence
# =============================================================================

# AIM-9B (1956) - First production Sidewinder
# Most data is declassified due to age
AIM_9B = MissileConfig(
    name="AIM-9B",
    # [DECLASSIFIED] Range data from multiple sources
    max_range_km=4.8,            # ~3 miles effective, 5km max
    min_range_km=0.9,            # ~900m minimum
    max_speed_mach=2.5,          # [DECLASSIFIED]
    max_g=11.0,                  # [DECLASSIFIED] Limited maneuverability

    burn_time_s=2.2,             # [DECLASSIFIED] Mk 17 motor
    coast_time_s=15.0,           # [PLACEHOLDER] Estimated

    seeker_type=SeekerType.REAR_ASPECT,
    seeker_fov_deg=4.0,          # [DECLASSIFIED] Very narrow cone
    seeker_gimbal_deg=12.0,      # [PLACEHOLDER] Limited gimbal
    track_rate_deg_s=11.0,       # [DECLASSIFIED]

    irccm_level=IRCCMLevel.NONE,  # No flare resistance

    # [DECLASSIFIED] Physical specs
    mass_kg=70.4,
    fuel_mass_kg=6.5,            # [ESTIMATED] Mk 17 motor propellant
    diameter_m=0.127,
    length_m=2.83,

    source_notes="DCS World, War Thunder wiki, Navy historical docs"
)

# AIM-9D (1965) - Navy improved version
AIM_9D = MissileConfig(
    name="AIM-9D",
    max_range_km=18.0,           # [DECLASSIFIED] 0.6-22 miles
    min_range_km=1.0,            # [DECLASSIFIED]
    max_speed_mach=2.5,          # [DECLASSIFIED]
    max_g=18.0,                  # [DECLASSIFIED] at sea level

    burn_time_s=5.0,             # [DECLASSIFIED] Mk 36 motor
    coast_time_s=20.0,           # [PLACEHOLDER]

    seeker_type=SeekerType.REAR_ASPECT,
    seeker_fov_deg=25.0,         # [DECLASSIFIED] Wider FOV
    seeker_gimbal_deg=25.0,      # [DECLASSIFIED]
    track_rate_deg_s=12.0,       # [DECLASSIFIED]

    irccm_level=IRCCMLevel.LOW,  # Nitrogen cooling, basic rejection

    mass_kg=88.5,                # [DECLASSIFIED]
    fuel_mass_kg=9.0,            # [ESTIMATED] Mk 36 motor propellant
    diameter_m=0.127,
    length_m=2.87,

    source_notes="Wikipedia, Navy.mil historical, YouTube analysis"
)

# AIM-9G (1970) - Improved D with better seeker
AIM_9G = MissileConfig(
    name="AIM-9G",
    max_range_km=18.0,           # [DECLASSIFIED]
    min_range_km=0.6,            # [DECLASSIFIED]
    max_speed_mach=2.5,          # [DECLASSIFIED]
    max_g=18.0,                  # [DECLASSIFIED]

    burn_time_s=5.0,             # [DECLASSIFIED] Mk 36 motor
    coast_time_s=25.0,           # [PLACEHOLDER]

    seeker_type=SeekerType.REAR_ASPECT,
    seeker_fov_deg=25.0,         # [DECLASSIFIED]
    seeker_gimbal_deg=25.0,      # [DECLASSIFIED]
    track_rate_deg_s=16.5,       # [DECLASSIFIED] SEAM capability

    irccm_level=IRCCMLevel.LOW,  # Improved flare rejection

    mass_kg=88.5,                # [DECLASSIFIED]
    fuel_mass_kg=9.0,            # [ESTIMATED] Mk 36 motor propellant
    diameter_m=0.127,
    length_m=2.87,

    source_notes="War Thunder wiki, DCS, fandom wikis (cross-referenced)"
)

# AIM-9L (1977) - First all-aspect Sidewinder
AIM_9L = MissileConfig(
    name="AIM-9L",
    max_range_km=18.0,           # [DECLASSIFIED]
    min_range_km=0.3,            # [DECLASSIFIED] Improved arming
    max_speed_mach=2.5,          # [DECLASSIFIED]
    max_g=35.0,                  # [DECLASSIFIED] Major improvement

    burn_time_s=5.2,             # [DECLASSIFIED] Mk 36 Mod 7/8
    coast_time_s=30.0,           # [PLACEHOLDER]

    seeker_type=SeekerType.ALL_ASPECT,  # [DECLASSIFIED] Key improvement
    seeker_fov_deg=25.0,         # [DECLASSIFIED]
    seeker_gimbal_deg=40.0,      # [PLACEHOLDER] Improved gimbal
    track_rate_deg_s=20.0,       # [PLACEHOLDER]

    irccm_level=IRCCMLevel.MEDIUM,

    mass_kg=85.3,                # [DECLASSIFIED]
    fuel_mass_kg=9.0,            # [ESTIMATED] Mk 36 Mod 7/8 motor propellant
    diameter_m=0.127,
    length_m=2.87,

    source_notes="USAF Fact Sheet, HistoryNet, af.mil"
)

# AIM-9M (1982) - Improved L with better IRCCM
AIM_9M = MissileConfig(
    name="AIM-9M",
    max_range_km=18.0,           # [DECLASSIFIED]
    min_range_km=0.3,            # [DECLASSIFIED]
    max_speed_mach=2.5,          # [DECLASSIFIED]
    max_g=35.0,                  # [DECLASSIFIED]

    burn_time_s=5.2,             # [DECLASSIFIED] Mk 36 Mod 9
    coast_time_s=30.0,           # [PLACEHOLDER]

    seeker_type=SeekerType.ALL_ASPECT,
    seeker_fov_deg=25.0,         # [DECLASSIFIED]
    seeker_gimbal_deg=40.0,      # [PLACEHOLDER]
    track_rate_deg_s=20.0,       # [PLACEHOLDER]

    irccm_level=IRCCMLevel.HIGH,  # [DECLASSIFIED] Key improvement

    mass_kg=85.3,                # [DECLASSIFIED]
    fuel_mass_kg=9.0,            # [ESTIMATED] Mk 36 Mod 9 motor propellant
    diameter_m=0.127,
    length_m=2.87,

    source_notes="USAF Fact Sheet, af.mil"
)

# AIM-9X (2003) - Modern imaging seeker
# Most specs are classified - use placeholders with modernization factors
AIM_9X = MissileConfig(
    name="AIM-9X",
    max_range_km=35.0,           # [PLACEHOLDER] ~2x improvement estimated
    min_range_km=0.2,            # [PLACEHOLDER]
    max_speed_mach=2.5,          # [PLACEHOLDER] Similar airframe
    max_g=60.0,                  # [ESTIMATED] Thrust vectoring >60G

    burn_time_s=5.5,             # [ESTIMATED] Modernized Mk 36
    coast_time_s=40.0,           # [PLACEHOLDER]

    seeker_type=SeekerType.ALL_ASPECT,
    seeker_fov_deg=90.0,         # [PLACEHOLDER] High off-boresight
    seeker_gimbal_deg=90.0,      # [PLACEHOLDER] Helmet-mounted cueing
    track_rate_deg_s=60.0,       # [PLACEHOLDER] Imaging seeker

    irccm_level=IRCCMLevel.HIGH,  # Imaging seeker, best IRCCM

    mass_kg=85.3,                # [PLACEHOLDER]
    fuel_mass_kg=9.5,            # [PLACEHOLDER] Modernized motor
    diameter_m=0.127,
    length_m=3.0,                # [PLACEHOLDER]

    source_notes="MOSTLY PLACEHOLDER - Apply modernization factors from older models"
)

# Dictionary of all presets for easy access
MISSILE_PRESETS = {
    "AIM-9B": AIM_9B,
    "AIM-9D": AIM_9D,
    "AIM-9G": AIM_9G,
    "AIM-9L": AIM_9L,
    "AIM-9M": AIM_9M,
    "AIM-9X": AIM_9X,
}


# =============================================================================
# TARGET CONFIGURATION
# =============================================================================

@dataclass
class TargetConfig:
    """Target aircraft configuration."""
    name: str

    # Performance
    max_speed_mach: float        # Maximum speed [Mach]
    max_g: float                 # Maximum G load
    turn_rate_deg_s: float       # Instantaneous turn rate [deg/sec]

    # IR signature (relative scale 0-1)
    ir_signature: float          # Base IR signature strength
    afterburner_multiplier: float  # IR boost when in afterburner

    # Countermeasures
    flare_count: int             # Number of flares carried
    # Number of chaff bundles (for future radar sim)
    chaff_count: int


# Example target presets
FIGHTER_JET = TargetConfig(
    name="Generic Fighter",
    max_speed_mach=2.0,
    max_g=9.0,
    turn_rate_deg_s=20.0,
    ir_signature=0.8,
    afterburner_multiplier=2.5,
    flare_count=60,
    chaff_count=60,
)

TRANSPORT = TargetConfig(
    name="Transport Aircraft",
    max_speed_mach=0.85,
    max_g=2.5,
    turn_rate_deg_s=5.0,
    ir_signature=0.9,
    afterburner_multiplier=1.0,  # No afterburner
    flare_count=0,
    chaff_count=0,
)

TARGET_PRESETS = {
    "fighter": FIGHTER_JET,
    "transport": TRANSPORT,
}


# =============================================================================
# SIMULATION CONFIGURATION
# =============================================================================

@dataclass
class SimulationConfig:
    """Monte Carlo simulation parameters."""

    # Time stepping
    dt: float = 0.01             # Simulation timestep [seconds]
    max_time_s: float = 60.0     # Maximum simulation duration [seconds]

    # Monte Carlo
    num_runs: int = 100          # Number of trajectories to simulate
    random_seed: Optional[int] = None  # For reproducibility

    # Noise models (standard deviations)
    seeker_noise_deg: float = 0.5      # Seeker tracking noise [degrees]
    guidance_noise_deg: float = 1.0    # Guidance computation noise [degrees]
    # Target motion prediction error [fraction]
    target_prediction_noise: float = 0.02

    # Initial conditions noise (for Monte Carlo variation)
    launch_angle_noise_deg: float = 2.0  # Launch angle variation [degrees]
    launch_speed_noise_pct: float = 0.05  # Launch speed variation [fraction]


@dataclass
class Vector3:
    """3D vector for positions and velocities."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> 'Vector3':
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> 'Vector3':
        return self.__mul__(scalar)

    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalized(self) -> 'Vector3':
        mag = self.magnitude()
        if mag == 0:
            return Vector3(0, 0, 0)
        return Vector3(self.x / mag, self.y / mag, self.z / mag)

    def dot(self, other: 'Vector3') -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: 'Vector3') -> 'Vector3':
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def to_tuple(self) -> tuple:
        return (self.x, self.y, self.z)


@dataclass
class Orientation:
    """
    Aircraft/missile orientation using Euler angles.

    All angles in degrees:
    - heading: 0-360, 0=North, 90=East
    - pitch: -90 to 90, positive=nose up
    - roll: -180 to 180, positive=right wing down
    """
    heading_deg: float = 0.0     # Yaw/azimuth
    pitch_deg: float = 0.0       # Elevation
    roll_deg: float = 0.0        # Bank angle

    def to_direction_vector(self) -> Vector3:
        """Convert orientation to unit direction vector."""
        heading_rad = math.radians(self.heading_deg)
        pitch_rad = math.radians(self.pitch_deg)

        # Convert spherical to Cartesian (Y-up convention)
        x = math.cos(pitch_rad) * math.sin(heading_rad)
        y = math.sin(pitch_rad)
        z = math.cos(pitch_rad) * math.cos(heading_rad)

        return Vector3(x, y, z)


@dataclass
class InitialConditions:
    """
    Initial conditions for missile and target.

    Coordinate system:
    - X: East
    - Y: Up (altitude)
    - Z: North
    """
    # Missile initial state
    missile_position: Vector3 = field(
        default_factory=lambda: Vector3(0, 5000, 0))
    missile_velocity_mps: float = 300.0  # Launch platform speed [m/s]
    missile_orientation: Orientation = field(default_factory=Orientation)

    # Target initial state
    target_position: Vector3 = field(
        default_factory=lambda: Vector3(5000, 5000, 10000))
    target_velocity_mps: float = 250.0   # Target speed [m/s]
    target_orientation: Orientation = field(
        default_factory=lambda: Orientation(heading_deg=180))

    # Engagement geometry
    aspect_angle_deg: float = 0.0  # 0=tail, 180=head-on


@dataclass
class FlareConfig:
    """Countermeasure flare configuration."""
    ir_intensity: float = 1.0    # Relative to aircraft signature
    burn_time_s: float = 3.0     # Flare burn duration
    deploy_interval_s: float = 0.5  # Time between flare releases

    # Effectiveness vs IRCCM
    effectiveness_vs_none: float = 0.95   # Pk reduction vs no IRCCM
    effectiveness_vs_low: float = 0.7     # Pk reduction vs low IRCCM
    effectiveness_vs_medium: float = 0.4  # Pk reduction vs medium IRCCM
    effectiveness_vs_high: float = 0.1    # Pk reduction vs high IRCCM


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

DEFAULT_SIMULATION = SimulationConfig()
DEFAULT_INITIAL_CONDITIONS = InitialConditions()
DEFAULT_FLARE = FlareConfig()
