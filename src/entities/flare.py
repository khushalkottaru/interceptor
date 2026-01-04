"""
Flare countermeasure simulation.

Models infrared flares used to decoy IR-guided missiles.
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum

from core.config import Vector3, FlareConfig
from core.physics import compute_drag, GRAVITY_M_S2


class FlareStatus(Enum):
    """Current status of a flare."""
    DEPLOYED = "deployed"    # Active and burning
    EXPIRED = "expired"      # Burn complete


@dataclass
class FlareState:
    """State of a single flare."""
    position: Vector3
    velocity: Vector3
    ir_intensity: float      # Current IR output (decreases over time)
    time_deployed: float     # Time since deployment
    burn_time: float         # Total burn duration
    status: FlareStatus = FlareStatus.DEPLOYED


class Flare:
    """
    Single IR countermeasure flare.

    Models:
    - IR intensity profile over time
    - Ballistic trajectory (falls from aircraft)
    - Expiration when burned out
    """

    def __init__(
        self,
        initial_position: Vector3,
        initial_velocity: Vector3,
        config: FlareConfig = None,
        rng: Optional[random.Random] = None
    ):
        """
        Initialize flare.

        Args:
            initial_position: Deployment position [m]
            initial_velocity: Initial velocity (from aircraft) [m/s]
            config: Flare configuration
            rng: Random number generator
        """
        self.config = config or FlareConfig()
        self.rng = rng or random.Random()

        # Add some randomness to initial velocity (ejection dispersion)
        dispersion = 5.0  # m/s
        ejection_vel = Vector3(
            self.rng.gauss(0, dispersion),
            self.rng.gauss(-10, dispersion),  # Ejected downward
            self.rng.gauss(0, dispersion)
        )

        self.state = FlareState(
            position=initial_position,
            velocity=initial_velocity + ejection_vel,
            ir_intensity=self.config.ir_intensity,
            time_deployed=0.0,
            burn_time=self.config.burn_time_s
        )

        # Flare physical properties
        self.mass_kg = 0.2           # Typical pyrotechnic flare mass [kg]
        self.drag_coefficient = 1.0  # High drag (tumbling flare)
        self.reference_area = 0.01   # m² (small cross-section)

    def update(self, dt: float) -> FlareStatus:
        """
        Update flare state for one timestep.

        Args:
            dt: Time step [s]

        Returns:
            Current flare status
        """
        if self.state.status == FlareStatus.EXPIRED:
            return self.state.status

        self.state.time_deployed += dt

        # Check if burned out
        if self.state.time_deployed >= self.state.burn_time:
            self.state.status = FlareStatus.EXPIRED
            self.state.ir_intensity = 0.0
            return self.state.status

        # Update IR intensity (peaks quickly, then decays)
        burn_fraction = self.state.time_deployed / self.state.burn_time
        # Intensity curve: rises quickly to peak, decays slowly
        if burn_fraction < 0.1:
            intensity_factor = burn_fraction / 0.1  # Rise to peak
        else:
            intensity_factor = 1.0 - \
                ((burn_fraction - 0.1) / 0.9) ** 0.5  # Decay

        self.state.ir_intensity = self.config.ir_intensity * \
            max(0, intensity_factor)

        # Ballistic trajectory with proper aerodynamics
        altitude = self.state.position.y
        gravity_accel = Vector3(0, -GRAVITY_M_S2, 0)

        # Compute drag force using proper ISA atmosphere model
        drag_force = compute_drag(
            self.state.velocity,
            altitude,
            self.reference_area,
            self.drag_coefficient
        )
        # Convert drag force to acceleration (a = F/m)
        drag_accel = drag_force * (1.0 / self.mass_kg)

        total_accel = gravity_accel + drag_accel

        # Integrate
        self.state.velocity = self.state.velocity + total_accel * dt
        self.state.position = self.state.position + self.state.velocity * dt

        # Check ground impact
        if self.state.position.y < 0:
            self.state.status = FlareStatus.EXPIRED
            self.state.ir_intensity = 0.0

        return self.state.status

    def get_position(self) -> Vector3:
        """Get current position."""
        return self.state.position

    def get_ir_intensity(self) -> float:
        """Get current IR intensity."""
        return self.state.ir_intensity

    def is_active(self) -> bool:
        """Check if flare is still active."""
        return self.state.status == FlareStatus.DEPLOYED


class FlareDispenser:
    """
    Manages flare deployment from an aircraft.

    Handles deployment patterns and timing.
    """

    def __init__(
        self,
        total_flares: int = 60,
        config: FlareConfig = None,
        rng: Optional[random.Random] = None
    ):
        """
        Initialize dispenser.

        Args:
            total_flares: Total flares available
            config: Flare configuration
            rng: Random number generator
        """
        self.config = config or FlareConfig()
        self.total_flares = total_flares
        self.flares_remaining = total_flares
        self.rng = rng or random.Random()

        # Active flares
        self.active_flares: List[Flare] = []

        # Deployment timing
        self.last_deploy_time = -float('inf')
        self.deploy_interval = self.config.deploy_interval_s

    def deploy(
        self,
        aircraft_position: Vector3,
        aircraft_velocity: Vector3,
        current_time: float
    ) -> bool:
        """
        Deploy a flare if available and cooldown elapsed.

        Args:
            aircraft_position: Aircraft position [m]
            aircraft_velocity: Aircraft velocity [m/s]
            current_time: Current simulation time [s]

        Returns:
            True if flare was deployed
        """
        if self.flares_remaining <= 0:
            return False

        if current_time - self.last_deploy_time < self.deploy_interval:
            return False

        # Deploy flare
        flare = Flare(
            initial_position=aircraft_position,
            initial_velocity=aircraft_velocity,
            config=self.config,
            rng=self.rng
        )

        self.active_flares.append(flare)
        self.flares_remaining -= 1
        self.last_deploy_time = current_time

        return True

    def deploy_burst(
        self,
        aircraft_position: Vector3,
        aircraft_velocity: Vector3,
        current_time: float,
        num_flares: int = 4
    ) -> int:
        """
        Deploy a burst of flares.

        Args:
            aircraft_position: Aircraft position [m]
            aircraft_velocity: Aircraft velocity [m/s]
            current_time: Current simulation time [s]
            num_flares: Number of flares to deploy

        Returns:
            Number of flares actually deployed
        """
        deployed = 0
        for i in range(num_flares):
            if self.flares_remaining <= 0:
                break

            # Offset position slightly for each flare
            offset = Vector3(
                self.rng.gauss(0, 2),
                self.rng.gauss(0, 2),
                self.rng.gauss(0, 2)
            )

            flare = Flare(
                initial_position=aircraft_position + offset,
                initial_velocity=aircraft_velocity,
                config=self.config,
                rng=self.rng
            )

            self.active_flares.append(flare)
            self.flares_remaining -= 1
            deployed += 1

        if deployed > 0:
            self.last_deploy_time = current_time

        return deployed

    def update(self, dt: float):
        """
        Update all active flares.

        Args:
            dt: Time step [s]
        """
        for flare in self.active_flares:
            flare.update(dt)

        # Remove expired flares
        self.active_flares = [f for f in self.active_flares if f.is_active()]

    def get_active_positions(self) -> List[Vector3]:
        """Get positions of all active flares."""
        return [f.get_position() for f in self.active_flares]

    def get_active_intensities(self) -> List[float]:
        """Get IR intensities of all active flares."""
        return [f.get_ir_intensity() for f in self.active_flares]

    def get_active_count(self) -> int:
        """Get number of active flares."""
        return len(self.active_flares)
