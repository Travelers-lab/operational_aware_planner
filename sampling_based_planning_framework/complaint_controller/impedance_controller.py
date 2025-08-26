import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import warnings


class TwoDImpedanceController:
    """
    A 2D impedance controller with multiple control modes.

    This controller implements three different impedance control modes:
    1. Position-only control (spring-damper)
    2. Position-velocity control (PD control)
    3. Position-velocity-acceleration control (PID-like with inertia)

    The controller calculates the required force/torque based on the current
    robot state and the desired trajectory.
    """

    def __init__(self):
        """
        Initialize the 2D impedance controller.

        The controller parameters are set when specific control modes are initialized.
        """
        self.control_mode = None
        self.parameters = {}
        self.state_history = []
        self.max_history_size = 1000

    def initialize_position_control(self, K: List[float]) -> None:
        """
        Initialize position-only control mode.

        This mode uses only spring elements to generate forces based on position errors.
        F = K * (x_desired - x_current)

        Args:
            K: Spring stiffness coefficients [Kx, Ky] in N/m
        """
        if len(K) != 2:
            raise ValueError("K must be a list of 2 values [Kx, Ky]")

        self.control_mode = "position_only"
        self.parameters = {
            'K': np.array(K, dtype=np.float64),
            'D': np.array([0.0, 0.0]),  # Damping (not used)
            'M': np.array([0.0, 0.0])  # Mass (not used)
        }

    def initialize_position_velocity_control(self, K: List[float], D: List[float]) -> None:
        """
        Initialize position-velocity control mode.

        This mode uses spring and damper elements (PD control).
        F = K * (x_desired - x_current) + D * (v_desired - v_current)

        Args:
            K: Spring stiffness coefficients [Kx, Ky] in N/m
            D: Damping coefficients [Dx, Dy] in N·s/m
        """
        if len(K) != 2 or len(D) != 2:
            raise ValueError("K and D must be lists of 2 values [Kx, Ky], [Dx, Dy]")

        self.control_mode = "position_velocity"
        self.parameters = {
            'K': np.array(K, dtype=np.float64),
            'D': np.array(D, dtype=np.float64),
            'M': np.array([0.0, 0.0])  # Mass (not used)
        }

    def initialize_position_velocity_acceleration_control(self,
                                                          K: List[float],
                                                          D: List[float],
                                                          M: List[float]) -> None:
        """
        Initialize position-velocity-acceleration control mode.

        This mode uses spring, damper, and mass elements (impedance control).
        F = M * (a_desired - a_current) + D * (v_desired - v_current) + K * (x_desired - x_current)

        Args:
            K: Spring stiffness coefficients [Kx, Ky] in N/m
            D: Damping coefficients [Dx, Dy] in N·s/m
            M: Virtual mass coefficients [Mx, My] in kg
        """
        if len(K) != 2 or len(D) != 2 or len(M) != 2:
            raise ValueError("K, D, and M must be lists of 2 values")

        self.control_mode = "position_velocity_acceleration"
        self.parameters = {
            'K': np.array(K, dtype=np.float64),
            'D': np.array(D, dtype=np.float64),
            'M': np.array(M, dtype=np.float64)
        }

    def compute_force(self,
                      current_position: List[float],
                      desired_position: List[float],
                      current_velocity: Optional[List[float]] = None,
                      desired_velocity: Optional[List[float]] = None,
                      current_acceleration: Optional[List[float]] = None,
                      desired_acceleration: Optional[List[float]] = None,
                      dt: float = 0.001) -> List[float]:
        """
        Compute the control force based on the current control mode.

        Args:
            current_position: Current robot position [x, y] in meters
            desired_position: Desired position [x_desired, y_desired] in meters
            current_velocity: Current robot velocity [vx, vy] in m/s (optional)
            desired_velocity: Desired velocity [vx_desired, vy_desired] in m/s (optional)
            current_acceleration: Current robot acceleration [ax, ay] in m/s² (optional)
            desired_acceleration: Desired acceleration [ax_desired, ay_desired] in m/s² (optional)
            dt: Time step for numerical differentiation (used if velocities/accelerations not provided)

        Returns:
            Control force [Fx, Fy] in Newtons

        Raises:
            ValueError: If controller is not initialized or inputs are invalid
            RuntimeError: If required state information is missing for the current control mode
        """
        if self.control_mode is None:
            raise ValueError("Controller must be initialized with a control mode first")

        # Convert inputs to numpy arrays
        x_current = np.array(current_position, dtype=np.float64)
        x_desired = np.array(desired_position, dtype=np.float64)

        # Calculate position error
        position_error = x_desired - x_current

        # Store current state for history
        self._update_state_history(x_current, current_velocity, current_acceleration, dt)

        # Compute force based on control mode
        if self.control_mode == "position_only":
            return self._compute_position_only_force(position_error)

        elif self.control_mode == "position_velocity":
            return self._compute_position_velocity_force(position_error, current_velocity,
                                                         desired_velocity, dt)

        elif self.control_mode == "position_velocity_acceleration":
            return self._compute_position_velocity_acceleration_force(
                position_error, current_velocity, desired_velocity,
                current_acceleration, desired_acceleration, dt
            )

        else:
            raise ValueError(f"Unknown control mode: {self.control_mode}")

    def _compute_position_only_force(self, position_error: np.ndarray) -> List[float]:
        """
        Compute force for position-only control mode.

        Args:
            position_error: Position error vector [ex, ey]

        Returns:
            Control force [Fx, Fy]
        """
        force = self.parameters['K'] * position_error
        return force.tolist()

    def _compute_position_velocity_force(self,
                                         position_error: np.ndarray,
                                         current_velocity: Optional[List[float]],
                                         desired_velocity: Optional[List[float]],
                                         dt: float) -> List[float]:
        """
        Compute force for position-velocity control mode.

        Args:
            position_error: Position error vector [ex, ey]
            current_velocity: Current velocity [vx, vy] or None
            desired_velocity: Desired velocity [vx_desired, vy_desired] or None
            dt: Time step for numerical differentiation

        Returns:
            Control force [Fx, Fy]
        """
        # Handle velocity inputs
        if current_velocity is None or desired_velocity is None:
            # Estimate velocities from position history
            v_current = self._estimate_current_velocity(dt)
            v_desired = np.array([0.0, 0.0])  # Assume zero desired velocity if not provided
        else:
            v_current = np.array(current_velocity, dtype=np.float64)
            v_desired = np.array(desired_velocity, dtype=np.float64) if desired_velocity else np.array([0.0, 0.0])

        # Calculate velocity error
        velocity_error = v_desired - v_current

        # Compute PD force
        force = self.parameters['K'] * position_error + self.parameters['D'] * velocity_error
        return force.tolist()

    def _compute_position_velocity_acceleration_force(self,
                                                      position_error: np.ndarray,
                                                      current_velocity: Optional[List[float]],
                                                      desired_velocity: Optional[List[float]],
                                                      current_acceleration: Optional[List[float]],
                                                      desired_acceleration: Optional[List[float]],
                                                      dt: float) -> List[float]:
        """
        Compute force for position-velocity-acceleration control mode.

        Args:
            position_error: Position error vector [ex, ey]
            current_velocity: Current velocity [vx, vy] or None
            desired_velocity: Desired velocity [vx_desired, vy_desired] or None
            current_acceleration: Current acceleration [ax, ay] or None
            desired_acceleration: Desired acceleration [ax_desired, ay_desired] or None
            dt: Time step for numerical differentiation

        Returns:
            Control force [Fx, Fy]
        """
        # Handle velocity inputs
        if current_velocity is None or desired_velocity is None:
            v_current = self._estimate_current_velocity(dt)
            v_desired = np.array([0.0, 0.0])
        else:
            v_current = np.array(current_velocity, dtype=np.float64)
            v_desired = np.array(desired_velocity, dtype=np.float64) if desired_velocity else np.array([0.0, 0.0])

        # Handle acceleration inputs
        if current_acceleration is None or desired_acceleration is None:
            a_current = self._estimate_current_acceleration(dt)
            a_desired = np.array([0.0, 0.0])
        else:
            a_current = np.array(current_acceleration, dtype=np.float64)
            a_desired = np.array(desired_acceleration, dtype=np.float64) if desired_acceleration else np.array(
                [0.0, 0.0])

        # Calculate errors
        velocity_error = v_desired - v_current
        acceleration_error = a_desired - a_current

        # Compute impedance force
        force = (self.parameters['M'] * acceleration_error +
                 self.parameters['D'] * velocity_error +
                 self.parameters['K'] * position_error)

        return force.tolist()

    def _update_state_history(self,
                              position: np.ndarray,
                              velocity: Optional[List[float]],
                              acceleration: Optional[List[float]],
                              dt: float) -> None:
        """
        Update the state history for numerical differentiation.

        Args:
            position: Current position
            velocity: Current velocity (if available)
            acceleration: Current acceleration (if available)
            dt: Time step
        """
        state = {
            'position': position,
            'timestamp': len(self.state_history) * dt if self.state_history else 0,
            'velocity': np.array(velocity) if velocity is not None else None,
            'acceleration': np.array(acceleration) if acceleration is not None else None
        }

        self.state_history.append(state)

        # Limit history size to prevent memory issues
        if len(self.state_history) > self.max_history_size:
            self.state_history.pop(0)

    def _estimate_current_velocity(self, dt: float) -> np.ndarray:
        """
        Estimate current velocity from position history.

        Args:
            dt: Time step

        Returns:
            Estimated velocity [vx, vy]
        """
        if len(self.state_history) < 2:
            return np.array([0.0, 0.0])

        current_pos = self.state_history[-1]['position']
        prev_pos = self.state_history[-2]['position']

        # Simple backward difference
        velocity = (current_pos - prev_pos) / dt
        return velocity

    def _estimate_current_acceleration(self, dt: float) -> np.ndarray:
        """
        Estimate current acceleration from velocity history.

        Args:
            dt: Time step

        Returns:
            Estimated acceleration [ax, ay]
        """
        if len(self.state_history) < 3:
            return np.array([0.0, 0.0])

        # Estimate velocities
        v_current = self._estimate_current_velocity(dt)
        v_prev = (self.state_history[-2]['position'] - self.state_history[-3]['position']) / dt

        # Simple backward difference for acceleration
        acceleration = (v_current - v_prev) / dt
        return acceleration

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current controller parameters.

        Returns:
            Dictionary containing controller parameters
        """
        return self.parameters.copy()

    def get_control_mode(self) -> str:
        """
        Get current control mode.

        Returns:
            Current control mode string
        """
        return self.control_mode

    def reset(self) -> None:
        """
        Reset the controller state and history.
        """
        self.state_history = []
        # Parameters remain unchanged


# Example usage and testing
if __name__ == "__main__":
    # Create controller instance
    controller = TwoDImpedanceController()

    # Test position-only control
    print("Testing Position-Only Control:")
    controller.initialize_position_control(K=[100.0, 100.0])

    current_pos = [0.5, 0.5]
    desired_pos = [1.0, 1.0]

    force = controller.compute_force(current_pos, desired_pos)
    print(f"Position error: {[desired_pos[0] - current_pos[0], desired_pos[1] - current_pos[1]]}")
    print(f"Control force: {force}")

    # Test position-velocity control
    print("\nTesting Position-Velocity Control:")
    controller.initialize_position_velocity_control(K=[100.0, 100.0], D=[10.0, 10.0])

    current_vel = [0.1, 0.1]
    desired_vel = [0.2, 0.2]

    force = controller.compute_force(current_pos, desired_pos, current_vel, desired_vel)
    print(f"Control force: {force}")

    # Test full impedance control
    print("\nTesting Position-Velocity-Acceleration Control:")
    controller.initialize_position_velocity_acceleration_control(
        K=[100.0, 100.0], D=[10.0, 10.0], M=[1.0, 1.0]
    )

    current_acc = [0.01, 0.01]
    desired_acc = [0.02, 0.02]

    force = controller.compute_force(
        current_pos, desired_pos,
        current_vel, desired_vel,
        current_acc, desired_acc
    )
    print(f"Control force: {force}")