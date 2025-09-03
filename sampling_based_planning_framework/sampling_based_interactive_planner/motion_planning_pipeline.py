import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
import time
from sampling_based_planning_framework.sampling_based_interactive_planner.motion_intent import MotionIntentRecognizer
from sampling_based_planning_framework.sampling_based_interactive_planner.planner import RRTConnectPlanner
from sampling_based_planning_framework.sampling_based_interactive_planner.trajectory_optimization import SamplingPointOptimizer



class MotionPlanningPipeline:
    """
    A motion planning pipeline class that integrates motion intent recognition,
    sampling-based path planning, and trajectory optimization.

    This class orchestrates the complete workflow from intent recognition to
    smooth trajectory generation for robotic motion planning.
    """

    def __init__(self,
                 interpolation_distance: float = 0.0002,
                 interpolation_method: str = 'cubic',
                 rrt_connect_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the Motion Planning Pipeline.

        Args:
            interpolation_distance: Desired Euclidean distance between interpolated points
            interpolation_method: Interpolation method for trajectory optimization
            rrt_connect_params: Additional parameters for RRT-Connect planner
        """
        # Initialize components
        self.motion_intent_recognizer = MotionIntentRecognizer()
        self.planner = RRTConnectPlanner()
        self.trajectory_optimizer = SamplingPointOptimizer(
            interpolation_distance=interpolation_distance,
            interpolation_method=interpolation_method
        )

        # Store planning results
        self.planning_results = {}
        self.last_target_point = None
        self.last_start_position = None

    def set_planner_parameters(self, **kwargs) -> None:
        """
        Set parameters for the RRT-Connect planner.

        Args:
            **kwargs: Parameters to pass to the RRT-Connect planner
        """
        # This method allows dynamic configuration of the planner
        # Note: Actual implementation depends on RRTConnectPlanner's interface
        pass

    def recognize_motion_intent(self,
                                objects_dict: Dict[str, Dict[str, Any]],
                                motion_target: List[float]) -> np.ndarray:
        """
        Recognize motion intent and compute the target point.

        Args:
            objects_dict: Dictionary containing object information
            motion_target: Initial motion target coordinates

        Returns:
            Computed target point coordinates
        """
        target_point = self.motion_intent_recognizer.get_target_point(
            objects_dict=objects_dict,
            motion_target=motion_target
        )
        self.last_target_point = target_point
        return target_point

    def plan_path(self, cost_map: np.ndarray, motion_mission: Dict[str, Any]) -> Dict[str, Any]:
        """
        Plan a path using RRT-Connect planner.

        Args:
            start_position: Starting point coordinates (x, y)
            goal_position: Goal point coordinates (x, y)

        Returns:
            Dictionary containing planning results
        """
        planning_result = self.planner.plan(
            cost_map=cost_map,
            motion_mission=motion_mission
        )
        return planning_result

    def optimize_trajectory(self, sampled_points: List[np.ndarray]) -> List[np.ndarray]:
        """
        Optimize the sampled trajectory points.

        Args:
            sampled_points: List of sampled points from the planner

        Returns:
            List of optimized and interpolated points
        """
        optimized_path = self.trajectory_optimizer.optimize_points(sampled_points)
        return optimized_path

    def execute_full_pipeline(self,
                              cost_map: np.ndarray,
                              objects_dict: Dict[str, Dict[str, Any]],
                              motion_mission: Dict[str, Any],
                              map_to_grid) -> Dict[str, Any]:
        """
        Execute the complete motion planning pipeline.

        Args:
            cost_map: Starting point coordinates (x, y)
            objects_dict: Dictionary containing object information
            motion_mission: Initial motion target coordinates
            map_to_grid: Function for workspace to grid map transformation

        Returns:
            Dictionary containing all planning results and metrics
        """
        self.last_start_position = motion_mission.get('start_position', [])
        motion_target = motion_mission.get('target_position', [])
        results = {}

        # Step 1: Motion Intent Recognition
        try:
            motion_mission['target_position'] = self.recognize_motion_intent(objects_dict, motion_target)
        except Exception as e:
            print(e)
            results['success'] = False
            return results

        motion_mission['start_position'] = map_to_grid.world_to_grid_discrete(motion_mission['start_position'])
        motion_mission['target_position'] = map_to_grid.world_to_grid_discrete(motion_mission['target_position'])

        # Step 2: Path Planning with RRT-Connect
        start_time_planning = time.time()

        try:
            planning_result = self.plan_path(cost_map=cost_map, motion_mission=motion_mission)
            planning_time = time.time() - start_time_planning
            results['path'] = planning_result['path']
            results['success'] = planning_result['success']
            results['planning_time'] = planning_time

        except Exception as e:
            print(e)
            results['success'] = False
            return results

        if not planning_result['success']:
            results['overall_success'] = False
            return results

        self.planning_results = results
        return results

    def get_optimized_path(self) -> Optional[List[np.ndarray]]:
        """
        Get the optimized path from the last successful planning.

        Returns:
            Optimized path points or None if not available
        """
        if (self.planning_results.get('overall_success') and
                'trajectory_optimization' in self.planning_results and
                self.planning_results['trajectory_optimization']['success']):
            return self.planning_results['trajectory_optimization']['optimized_path']
        return None

    def get_planning_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the last planning execution.

        Returns:
            Dictionary containing planning summary
        """
        summary = {
            'success': self.planning_results.get('overall_success', False),
            'total_time': self.planning_results.get('total_computation_time', 0),
            'path_length': len(self.get_optimized_path() or [])
        }

        if 'quality_metrics' in self.planning_results:
            summary.update(self.planning_results['quality_metrics'])

        return summary

    def reset(self) -> None:
        """
        Reset the pipeline state and clear previous results.
        """
        self.planning_results = {}
        self.last_target_point = None
        self.last_start_position = None


# Example usage
if __name__ == "__main__":
    # Example usage of the motion planning pipeline
    pipeline = MotionPlanningPipeline(interpolation_distance=0.0002)

    # Example inputs
    start_position = (0.0, 0.0)
    motion_target = np.array([1.0, 1.0])
    objects_dict = {
        'object1': {'position': [0.5, 0.5], 'type': 'obstacle', 'size': 0.1},
        'object2': {'position': [0.8, 0.8], 'type': 'target', 'size': 0.05}
    }

    # Execute the complete pipeline
    results = pipeline.execute_full_pipeline(
        start_position=start_position,
        objects_dict=objects_dict,
        motion_target=motion_target
    )

    # Display results
    print(f"Planning successful: {results['overall_success']}")
    print(f"Total computation time: {results['total_computation_time']:.4f}s")

    if results['overall_success']:
        optimized_path = pipeline.get_optimized_path()
        print(f"Optimized path points: {len(optimized_path)}")
        print(f"Quality metrics: {results.get('quality_metrics', {})}")