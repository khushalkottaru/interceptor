# Fox 2 Interceptor Simulation

## Abstract
This project is a python-based high fidelity model for simulating infraed guided missile engagments. In 3-DOF, the project moves beyond basic logic to implement professional-grade gudiance and numerical integration standards.

## Technical Architecture
The simulation utilizes a **4th-Order Runge-Kutta (RK4)** integrator to propagate the entity state vector $S = [x, y, z, v_x, v_y, v_z, m]$.

Unlike Euler integration, which has a local truncation error of $O(\Delta t^2)$, RK4 samples the derivatives at four points to achieve $O(\Delta t^5)$ local error, ensuring stability during **high-G terminal maneuvers**.

$\rightarrow y_{n+1} = y_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)$

## Guidance & Navigation
The interceptor utilizes **Gravity-Compensated Proportional Navigation (PN)** analgous to an operational sidewinder missile to prevent "sagging" due to underestimating gravity. The commanded acceleration $a_c$ is derived from the Line-of-Sight (LOS) rate $\dot{\lambda}$ and closing velocity $V_c$.

$\rightarrow a_c = N \cdot V_c \cdot \dot{\lambda}$

## Propulsion/Dynamic Mass
The model couples **kinematics** with a **variable-mass propulsion system**. Mass $m$ is treated as a **dynamic variable**, depleting according to the rocket equation based on thrust $T$ and specific impulse $I_{sp}$.
$\rightarrow \dot{m} = -\frac{T}{g_0 I_{sp}}$
This ensures the $F=ma$ calculations account for the significantly increased maneuverability of the interceptor in the post-burn "dart" phase.

## Aerodynamics & Atmospheric Modeling
- **ISA Model**: Air density $\rho$ is calculated using the International Standard Atmosphere model for accurate drag profiles across altitudes.
- **Transonic Drag Rise**: Implements Mach-dependent drag coefficients to simulate wave drag increases between Mach 0.8 and 1.2.
- $\rightarrow F_d = \frac{1}{2} \rho v^2 C_d A$

 ## Seeker + Countermeasure (flare) logic
 The IR seeker model includes:
    - **Gimbal Limits**: Rigid FOV and seeker head constraints
    - **Signal-to-Noise Modeling**: Detection probability based on target IR signature, range, and aspect angle.
    - **IRCCM**: Flare susceptibility logic based on IR intensity ratios and spatial separation/
 
   ## Monte Carlo Probability Analysis & $P_k$
  The effectiveness of each engagement is evaluated via Monte Carlo iteration.  by injecting Gaussian noise into initial launch conditions and target state estimates, the system generates a statistical **probability of kill ($P_k$)**.
  $\rightarrow P_k = \frac{\sum \text{Intercepts}}{\sum \text{Total Runs}}$

  Technical Note: This is a **3-DOF** point-mass simulation. Future iterations may focus on 6-DOF by implementing rotation mechanics.
