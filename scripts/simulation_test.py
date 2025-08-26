import numpy as np
import pybullet as p
import pybullet_data as pd
import time
import json
import os
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import random
from dataclasses import dataclass
import matplotlib.pyplot as plt

# Import required modules
try:
    from robot_environment.robot_environment import Robot
    from sampling_based_planning_framework.interactive_planning_pipeline import IntegratedMotionPlanningPipeline
    from sampling_based_planning_framework.complaint_controller.impedance_controller import TwoDImpedanceController
except ImportError as e:
    print(f"Import warning: {e}")


    # Create dummy classes for testing
    class Robot:
        def __init__(self, robot_urdf, table_urdf):
            pass

        def load_environment(self, obstacle_positions, obstacle_probability=0.5):
            pass

        def get_effector_states(self):
            return {"pos": [0.5, 0.3, 0.1], "vel": [0.0, 0.0, 0.0]}

        def torque_control_step(self, force):
            pass

        def reset_simulation(self):
            pass

        def close(self):
            pass


    class IntegratedMotionPlanningPipeline:
        def __init__(self):
            pass

        def run_full_pipeline(self, objects_dict, history_point_cloud, grid_map):
            return {'planning_success': True, 'planned_path': [[0, 0], [1, 1], [2, 2]]}


    class TwoDImpedanceController:
        def __init__(self):
            pass

        def initialize_position_control(self, K):
            pass

        def compute_force(self, current_position, desired_position):
            return [0.0, 0.0]


@dataclass
class TestResult:
    """Data class to store test results"""
    test_id: str
    planning_success: bool
    execution_success: bool
    planning_time: float
    path_length: int
    path_cost: float
    start_position: List[float]
    target_position: List[float]
    obstacles_count: int
    execution_time: float
    final_error: float


class RobotPlanningTester:
    """
    A class for testing robot motion planning algorithms using PyBullet.

    This class provides functionality to:
    - Load robot simulation environment
    - Generate test tasks
    - Initialize key variables
    - Execute test tasks
    - Save simulation results
    """

    def __init__(self,config):
        """
        Initialize the robot planning tester.

        Args:
            results_dir: Directory to save test results
            workspace_bounds: Workspace boundaries [[x_min, y_min], [x_max, y_max]]
            grid_resolution: Resolution for grid mapping
        """
        self.config = config
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.teme_step = 1/200.0
        self.workspace_bounds = config.workspace_bounds
        self.grid_resolution = config.grid_resolution
        self.robot_env = None
        self.planning_pipeline = None
        self.impedance_controller = None

        self.client1 = None
        self.client2 = None

        # Initialize key variables
        self.objects_dict = {}
        self.history_point_cloud = []
        self.grid_map = None
        self.test_results = []

        # Initialize grid mapping
        self._initialize_grid_map()

    def load_environment(self,
                         obstacle_count: int = 3,
                         min_obstacle_distance: float = 0.1) -> None:
        """
        Load the robot simulation environment with obstacles.

        Args:
            robot_urdf_path: Path to robot URDF file
            table_urdf_path: Path to table URDF file
            obstacle_count: Number of obstacles to generate
            min_obstacle_distance: Minimum distance between obstacles
        """
        print("Loading robot environment...")
        self.client1 = p.connect(p.GUI)
        p.setAdditionalSearchPath(pd.getDataPath())
        p.setPhysicsEngineParameter(numSolverIterations=50)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client1)
        p.setTimeStep(self.teme_step)

        # Initialize robot environment
        self.robot_env = Robot(p, self.config.robot_urdf_path, self.client1)

        # Generate random obstacle positions
        obstacle_positions = self._generate_obstacle_positions(
            obstacle_count, min_obstacle_distance
        )

        # Load environment with obstacles
        self.robot_env.load_environment(obstacle_positions, obstacle_probability=0.5)

        print(f"Environment loaded with {len(obstacle_positions)} obstacles")

    def init_planning_component(self):
        """
           Initialize pipeline component for motion planning.
        """
        self.planning_pipeline = IntegratedMotionPlanningPipeline(sensor_transform_matrix=self.robot_env.transformMatrix(self.config.body_link),
                                                                  robots=self.robot_env.robot,
                                                                  objectsId=self.robot_env.object,
                                                                  model_checkpoint_path=self.config.)

    def _generate_obstacle_positions(self,
                                     count: int,
                                     min_distance: float) -> List[List[float]]:
        """
        Generate random obstacle positions within workspace.

        Args:
            count: Number of obstacles to generate
            min_distance: Minimum distance between obstacles

        Returns:
            List of obstacle positions [x, y, z]
        """
        positions = []
        attempts = 0
        max_attempts = 100

        while len(positions) < count and attempts < max_attempts:
            # Generate random position within workspace
            x = random.uniform(self.workspace_bounds[0][0], self.workspace_bounds[1][0])
            y = random.uniform(self.workspace_bounds[0][1], self.workspace_bounds[1][1])
            z = 0.0  # Assuming obstacles are on table surface

            new_pos = [x, y, z]

            # Check minimum distance from existing obstacles
            valid_position = True
            for existing_pos in positions:
                distance = np.sqrt((new_pos[0] - existing_pos[0]) ** 2 +
                                   (new_pos[1] - existing_pos[1]) ** 2)
                if distance < min_distance:
                    valid_position = False
                    break

            if valid_position:
                positions.append(new_pos)

            attempts += 1

        return positions

    def _initialize_grid_map(self) -> None:
        """
        Initialize the grid map by discretizing the workspace.
        """
        x_min, y_min = self.workspace_bounds[0]
        x_max, y_max = self.workspace_bounds[1]

        # Create empty grid map
        self.grid_map = np.zeros((self.grid_resolution, self.grid_resolution), dtype=np.float32)

        # Store grid parameters for coordinate conversion
        self.grid_params = {
            'x_min': x_min,
            'y_min': y_min,
            'x_max': x_max,
            'y_max': y_max,
            'x_range': x_max - x_min,
            'y_range': y_max - y_min,
            'resolution': self.grid_resolution
        }

    def workspace_to_grid(self, workspace_coord: List[float]) -> List[int]:
        """
        Convert workspace coordinates to grid coordinates.

        Args:
            workspace_coord: Workspace coordinates [x, y]

        Returns:
            Grid coordinates [grid_x, grid_y]
        """
        x, y = workspace_coord
        grid_x = int((x - self.grid_params['x_min']) / self.grid_params['x_range'] * self.grid_params['resolution'])
        grid_y = int((y - self.grid_params['y_min']) / self.grid_params['y_range'] * self.grid_params['resolution'])

        # Clamp to grid boundaries
        grid_x = max(0, min(self.grid_params['resolution'] - 1, grid_x))
        grid_y = max(0, min(self.grid_params['resolution'] - 1, grid_y))

        return [grid_x, grid_y]

    def grid_to_workspace(self, grid_coord: List[int]) -> List[float]:
        """
        Convert grid coordinates to workspace coordinates.

        Args:
            grid_coord: Grid coordinates [grid_x, grid_y]

        Returns:
            Workspace coordinates [x, y]
        """
        grid_x, grid_y = grid_coord
        x = self.grid_params['x_min'] + (grid_x / self.grid_params['resolution']) * self.grid_params['x_range']
        y = self.grid_params['y_min'] + (grid_y / self.grid_params['resolution']) * self.grid_params['y_range']

        return [x, y]

    def generate_test_task(self) -> List[float]:
        """
        Generate a random target point within the workspace.

        Returns:
            Target position in workspace coordinates [x, y, z]
        """
        x = random.uniform(self.workspace_bounds[0][0], self.workspace_bounds[1][0])
        y = random.uniform(self.workspace_bounds[0][1], self.workspace_bounds[1][1])
        z = 0.0  # Assuming target is on table surface

        return [x, y, z]

    def get_robot_state(self) -> Dict[str, List[float]]:
        """
        Get the current robot effector state.

        Returns:
            Dictionary containing position and velocity
        """
        if self.robot_env is None:
            raise RuntimeError("Robot environment not loaded")

        return self.robot_env.get_effector_states()

    def create_motion_mission(self, target_position: List[float]) -> Dict[str, Any]:
        """
        Create a motion mission dictionary from target position.

        Args:
            target_position: Target position in workspace coordinates

        Returns:
            Motion mission dictionary
        """
        robot_state = self.get_robot_state()
        current_pos = robot_state["pos"][:2]  # Use only x, y coordinates

        # Convert to grid coordinates
        grid_start = self.workspace_to_grid(current_pos)
        grid_target = self.workspace_to_grid(target_position[:2])

        motion_mission = {
            'start_position': grid_start,
            'target_position': grid_target,
            'mission_type': 'navigation',
            'constraints': {
                'max_velocity': 2.0,
                'max_acceleration': 5.0,
                'timeout': 30.0
            }
        }

        return motion_mission

    def plan_motion(self, motion_mission: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute motion planning pipeline.

        Args:
            motion_mission: Motion mission specification

        Returns:
            Planning results dictionary
        """
        if self.planning_pipeline is None:
            self.planning_pipeline = IntegratedMotionPlanningPipeline()

        # Run planning pipeline
        results = self.planning_pipeline.run_full_pipeline(
            objects_dict=self.objects_dict,
            history_point_cloud=self.history_point_cloud,
            grid_map=self.grid_map
        )

        return results

    def evaluate_planning_result(self,
                                 results: Dict[str, Any],
                                 motion_mission: Dict[str, Any]) -> bool:
        """
        Evaluate planning results and decide whether to execute.

        Args:
            results: Planning results
            motion_mission: Motion mission

        Returns:
            True if planning is successful and should be executed
        """
        if not results.get('planning_success', False):
            print("Planning failed - stopping execution")
            return False

        # Check if start and target are too close
        start_pos = motion_mission['start_position']
        target_pos = motion_mission['target_position']
        distance = np.sqrt((target_pos[0] - start_pos[0]) ** 2 +
                           (target_pos[1] - start_pos[1]) ** 2)

        if distance < 2:  # Less than 2 grid cells
            print("Start and target are too close - planning successful")
            return True

        return True

    def execute_planned_motion(self,
                               planned_path: List[List[float]],
                               max_points: int = 50) -> bool:
        """
        Execute the planned motion using impedance control.

        Args:
            planned_path: Planned path in grid coordinates
            max_points: Maximum number of trajectory points to execute

        Returns:
            True if execution was successful
        """
        if self.impedance_controller is None:
            self.impedance_controller = TwoDImpedanceController()
            self.impedance_controller.initialize_position_control(K=[100.0, 100.0])

        # Convert path to workspace coordinates and limit points
        workspace_path = [self.grid_to_workspace(point) for point in planned_path[:max_points]]

        execution_success = True
        start_time = time.time()

        try:
            for i, target_point in enumerate(workspace_path):
                # Get current robot state
                robot_state = self.get_robot_state()
                current_pos = robot_state["pos"][:2]  # Use only x, y

                # Compute control force
                force = self.impedance_controller.compute_force(current_pos, target_point)

                # Apply control force
                self.robot_env.torque_control_step(force)

                # Small delay for simulation stability
                time.sleep(0.01)

                # Check if we're close to target
                if i == len(workspace_path) - 1:
                    final_error = np.sqrt((current_pos[0] - target_point[0]) ** 2 +
                                          (current_pos[1] - target_point[1]) ** 2)
                    if final_error > 0.05:  # 5cm error threshold
                        execution_success = False

        except Exception as e:
            print(f"Error during motion execution: {e}")
            execution_success = False

        execution_time = time.time() - start_time
        return execution_success, execution_time

    def run_test(self, test_id: str = None) -> TestResult:
        """
        Run a complete test case.

        Args:
            test_id: Optional test identifier

        Returns:
            Test result object
        """
        if test_id is None:
            test_id = f"test_{len(self.test_results) + 1:04d}"

        print(f"Running test: {test_id}")

        # Generate test task
        target_position = self.generate_test_task()

        # Create motion mission
        motion_mission = self.create_motion_mission(target_position)

        # Plan motion
        planning_start_time = time.time()
        planning_results = self.plan_motion(motion_mission)
        planning_time = time.time() - planning_start_time

        # Evaluate planning result
        should_execute = self.evaluate_planning_result(planning_results, motion_mission)

        execution_success = False
        execution_time = 0.0
        final_error = 0.0

        if should_execute and planning_results.get('planned_path'):
            # Execute planned motion
            execution_success, execution_time = self.execute_planned_motion(
                planning_results['planned_path']
            )

            # Calculate final error
            final_state = self.get_robot_state()
            final_pos = final_state["pos"][:2]
            final_error = np.sqrt((final_pos[0] - target_position[0]) ** 2 +
                                  (final_pos[1] - target_position[1]) ** 2)

        # Create test result
        test_result = TestResult(
            test_id=test_id,
            planning_success=planning_results.get('planning_success', False),
            execution_success=execution_success,
            planning_time=planning_time,
            path_length=len(planning_results.get('planned_path', [])),
            path_cost=planning_results.get('path_cost', float('inf')),
            start_position=motion_mission['start_position'],
            target_position=motion_mission['target_position'],
            obstacles_count=len(self.objects_dict),
            execution_time=execution_time,
            final_error=final_error
        )

        self.test_results.append(test_result)
        self._save_test_result(test_result)

        return test_result

    def _save_test_result(self, test_result: TestResult) -> None:
        """
        Save individual test result to file.

        Args:
            test_result: Test result to save
        """
        result_file = self.results_dir / f"{test_result.test_id}.json"

        result_dict = {
            'test_id': test_result.test_id,
            'planning_success': test_result.planning_success,
            'execution_success': test_result.execution_success,
            'planning_time': test_result.planning_time,
            'path_length': test_result.path_length,
            'path_cost': test_result.path_cost,
            'start_position': test_result.start_position,
            'target_position': test_result.target_position,
            'obstacles_count': test_result.obstacles_count,
            'execution_time': test_result.execution_time,
            'final_error': test_result.final_error
        }

        with open(result_file, 'w') as f:
            json.dump(result_dict, f, indent=2)

    def run_test_suite(self, num_tests: int = 10) -> None:
        """
        Run a suite of tests.

        Args:
            num_tests: Number of tests to run
        """
        print(f"Running test suite with {num_tests} tests...")

        for i in range(num_tests):
            test_result = self.run_test()
            print(f"Test {test_result.test_id}: "
                  f"Planning: {'Success' if test_result.planning_success else 'Fail'}, "
                  f"Execution: {'Success' if test_result.execution_success else 'Fail'}, "
                  f"Final error: {test_result.final_error:.3f}m")

        self._generate_summary_report()

    def _generate_summary_report(self) -> None:
        """
        Generate a summary report of all test results.
        """
        if not self.test_results:
            return

        summary = {
            'total_tests': len(self.test_results),
            'planning_success_rate': sum(1 for r in self.test_results if r.planning_success) / len(self.test_results),
            'execution_success_rate': sum(1 for r in self.test_results if r.execution_success) / len(self.test_results),
            'avg_planning_time': np.mean([r.planning_time for r in self.test_results]),
            'avg_execution_time': np.mean([r.execution_time for r in self.test_results]),
            'avg_final_error': np.mean([r.final_error for r in self.test_results]),
            'test_results': [r.__dict__ for r in self.test_results]
        }

        summary_file = self.results_dir / "test_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Summary report saved to {summary_file}")

    def cleanup(self) -> None:
        """
        Clean up resources and close simulation.
        """
        if self.robot_env:
            self.robot_env.close()
        print("Simulation cleanup completed")


# Example usage
if __name__ == "__main__":
    # Initialize tester
    tester = RobotPlanningTester(results_dir="test_results")

    try:
        # Load environment (replace with actual URDF paths)
        tester.load_environment(
            robot_urdf_path="path/to/robot.urdf",
            table_urdf_path="path/to/table.urdf",
            obstacle_count=3
        )

        # Run test suite
        tester.run_test_suite(num_tests=5)

    except Exception as e:
        print(f"Error during testing: {e}")

    finally:
        # Cleanup
        tester.cleanup()

