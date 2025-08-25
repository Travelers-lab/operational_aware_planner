import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import time
from motion_intent import MotionIntentRecognizer
from planner import RRTConnectPlanner
from trajectory_optimization import SamplingPointOptimizer


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
        self.planner = RRTConnectPlanner(**(rrt_connect_params or {}))
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
                                motion_target: np.ndarray) -> np.ndarray:
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

    def plan_path(self,
                  start_position: Tuple[float, float],
                  goal_position: Tuple[float, float]) -> Dict[str, Any]:
        """
        Plan a path using RRT-Connect planner.

        Args:
            start_position: Starting point coordinates (x, y)
            goal_position: Goal point coordinates (x, y)

        Returns:
            Dictionary containing planning results
        """
        planning_result = self.planner.plan_path(
            start=start_position,
            goal=goal_position
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
                              start_position: Tuple[float, float],
                              objects_dict: Dict[str, Dict[str, Any]],
                              motion_target: np.ndarray) -> Dict[str, Any]:
        """
        Execute the complete motion planning pipeline.

        Args:
            start_position: Starting point coordinates (x, y)
            objects_dict: Dictionary containing object information
            motion_target: Initial motion target coordinates

        Returns:
            Dictionary containing all planning results and metrics
        """
        self.last_start_position = start_position
        results = {}

        # Step 1: Motion Intent Recognition
        start_time_intent = time.time()
        try:
            target_point = self.recognize_motion_intent(objects_dict, motion_target)
            intent_time = time.time() - start_time_intent
            results['intent_recognition'] = {
                'success': True,
                'target_point': target_point,
                'computation_time': intent_time
            }
        except Exception as e:
            intent_time = time.time() - start_time_intent
            results['intent_recognition'] = {
                'success': False,
                'error': str(e),
                'computation_time': intent_time
            }
            results['overall_success'] = False
            return results

        # Step 2: Path Planning with RRT-Connect
        start_time_planning = time.time()
        try:
            planning_result = self.plan_path(start_position, tuple(target_point[:2]))
            planning_time = time.time() - start_time_planning
            results['path_planning'] = {
                'success': planning_result['success'],
                'path': planning_result['path'],
                'planning_time': planning_result['planning_time'],
                'iterations': planning_result['iterations'],
                'computation_time': planning_time
            }
        except Exception as e:
            planning_time = time.time() - start_time_planning
            results['path_planning'] = {
                'success': False,
                'error': str(e),
                'computation_time': planning_time
            }
            results['overall_success'] = False
            return results

        if not planning_result['success']:
            results['overall_success'] = False
            return results

        # Step 3: Trajectory Optimization
        start_time_optimization = time.time()
        try:
            optimized_path = self.optimize_trajectory(planning_result['path'])
            optimization_time = time.time() - start_time_optimization
            results['trajectory_optimization'] = {
                'success': True,
                'optimized_path': optimized_path,
                'original_points': len(planning_result['path']),
                'optimized_points': len(optimized_path),
                'computation_time': optimization_time
            }
        except Exception as e:
            optimization_time = time.time() - start_time_optimization
            results['trajectory_optimization'] = {
                'success': False,
                'error': str(e),
                'computation_time': optimization_time
            }
            results['overall_success'] = False
            return results

        # Calculate quality metrics for the optimized path
        try:
            quality_metrics = self.trajectory_optimizer.calculate_path_quality(optimized_path)
            results['quality_metrics'] = quality_metrics
        except Exception as e:
            results['quality_metrics'] = {'error': str(e)}

        # Overall results
        total_time = intent_time + planning_time + optimization_time
        results['overall_success'] = True
        results['total_computation_time'] = total_time
        results['start_position'] = start_position
        results['target_position'] = target_point.tolist()

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