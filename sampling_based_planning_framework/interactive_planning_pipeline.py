
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from multi_dimension_perception.environmental_sensing_pipeline import MultiModalTactilePerceptionPipeline
from sampling_based_interactive_planner.motion_planning_pipeline import MotionPlanningPipeline
from model.operational_aware_map_pipeline import MapGenerationPipeline



class IntegratedMotionPlanningPipeline:
    """
    Integrated motion planning pipeline that combines perception, map generation, and motion planning.

    This pipeline integrates:
    1. Multi-modal tactile perception for object tracking and point cloud generation
    2. Operational cost map generation based on object properties
    3. Motion planning for robot trajectory generation
    """

    def __init__(self,
                 config,
                 sensor_transform_matrix: Optional[np.ndarray] = None,
                 robots: Optional[Union[str, List[str]]] = None,
                 objectsId: Optional[Union[int, List[int]]] = None,
                 model_checkpoint_path: str = None,
                 operation_limits: List[float] = [400, 400, 20],
                 model_config_path: Optional[str] = None,
                 device: Optional[str] = None,
                 ):
        """
        Initialize the integrated motion planning pipeline.

        Args:
            sensor_transform_matrix: Transformation matrix for sensor calibration
            robots: Robot identifiers
            objectsId: Object identifiers to track
            model_checkpoint_path: Path to the model checkpoint for map generation
            operation_limits: Operation limits for material properties
            model_config_path: Path to model configuration file
            device: Device to run the model on (e.g., 'cuda', 'cpu')
        """
        # Initialize sub-pipeline
        self.perception_pipeline = MultiModalTactilePerceptionPipeline(config.perception_config)
        self.map_generation_pipeline = MapGenerationPipeline(config.map_generation_config)
        self.motion_planning_pipeline = MotionPlanningPipeline()

        # Set sensor parameters for perception pipeline
        if sensor_transform_matrix is not None or robots is not None or objectsId is not None:
            self.set_sensor_parameters(sensor_transform_matrix, robots, objectsId)

        # Initialize state variables
        self.objects_dict = {}
        self.history_point_cloud = []
        self.cost_map = None
        self.completed_map = None
        self.planning_results = None

    def set_sensor_parameters(self,
                              sensor_transform_matrix: Optional[np.ndarray] = None,
                              robots: Optional[Union[str, List[str]]] = None,
                              objects_id: Optional[Union[int, List[int]]] = None):
        """
        Set sensor parameters for the perception pipeline.

        Args:
            sensor_transform_matrix: Transformation matrix for sensor calibration
            robots: Robot identifiers
            objects_id: Object identifiers to track
        """
        self.perception_pipeline.set_sensor_parameters(
            sensor_transform_matrix=sensor_transform_matrix,
            robots=robots,
            objectsId=objects_id
        )

    def set_grid_map(self, grid_map: np.ndarray):
        """
        Set the binary occupancy grid map for the perception pipeline.

        Args:
            grid_map: Binary occupancy map (0=free, 1=occupied)
        """
        self.perception_pipeline.set_grid_map(grid_map)

    def run_full_pipeline(self,
                          objects_dict: Dict[str, Dict[str, Any]],
                          binary_grid_map: np.ndarray,
                          motion_mission: Dict[str, Any],
                          history_point_cloud: List[List[float]],
                          return_intermediate: bool = False) -> Dict[str, Any]:
        """
        Execute the complete motion planning pipeline.

        Args:
            objects_dict: Dictionary containing object data with structure:
                {
                    'object_id': {
                        'material_property': [float, float, float],
                        'contour_equation': {
                            'shape': str,
                            'params': dict (shape-specific parameters)
                        },
                        'interaction_pairs': list,
                        'contact_status': str
                    }
                }
            binary_grid_map: Binary occupancy map (0=free, 1=occupied)
            motion_mission: Motion mission specification with structure:
                {
                    'start_position': [x, y],
                    'target_position': [x, y],
                    'mission_type': str,  # e.g., 'navigation', 'manipulation'
                    'constraints': dict   # optional constraints
                }
            history_point_cloud: point cloud [List]
            return_intermediate: Whether to return intermediate results

        Returns:
            Dictionary containing planning results and optionally intermediate data
        """
        # Store initial objects dictionary
        self.objects_dict = objects_dict.copy()

        # Step 1: Run perception pipeline
        # print("Running perception pipeline...")
        point_cloud = self.perception_pipeline.run_full_pipeline(
            objects_dict=self.objects_dict,
            binary_grid_map=binary_grid_map,
            history_point_cloud=history_point_cloud
        )

        grid_point_cloud = [self.perception_pipeline.object_manager.coordinate_transform(pose, to_grid=True) for pose in point_cloud]

        # Update objects dictionary and get point cloud
        # self.objects_dict = perception_results.get('objects_dict', self.objects_dict)
        # point_cloud = perception_results.get('point_cloud', [])
        # self.history_point_cloud = perception_results.get('history_point_cloud', [])

        # Step 2: Run map generation pipeline
        # print("Running map generation pipeline...")
        map_generation_results = self.map_generation_pipeline.generate_operational_cost_map(
            point_cloud_coordinates=grid_point_cloud,
            objects_dict=objects_dict,
            return_intermediate=return_intermediate
        )

        # Extract cost maps
        if return_intermediate and isinstance(map_generation_results, dict):
            self.cost_map = map_generation_results.get('cost_map')
            self.completed_map = map_generation_results.get('completed_map')
        elif isinstance(map_generation_results, (list, tuple)) and len(map_generation_results) >= 2:
            self.cost_map, self.completed_map = map_generation_results[:2]
        else:
            self.cost_map = map_generation_results
            self.completed_map = self.cost_map  # Fallback if only one map is returned

        # Step 3: Run motion planning pipeline
        # print("Running motion planning pipeline...")
        self.planning_results = self.motion_planning_pipeline.execute_full_pipeline(
            cost_map=self.completed_map,
            objects_dict=self.objects_dict,
            motion_mission=motion_mission
        )

        # Trajectory Optimization
        planned_trajectory = [self.perception_pipeline.object_manager.coordinate_transform(grid, to_grid=False) for grid in self.planning_results['path_planning']['path']]
        optimized_path = self.motion_planning_pipeline.optimize_trajectory(planned_trajectory)

        # Prepare final results
        results = {
            'planning_success': self.planning_results.get('success', False),
            'planned_path': optimized_path,
            'planning_time': self.planning_results.get('planning_time', 0),
            'iterations': self.planning_results.get('iterations', 0)
        }


        # Add intermediate results if requested
        if return_intermediate:
            results.update({
                'point_cloud': point_cloud,
                'history_point_cloud': self.history_point_cloud,
                'map_generation_results': map_generation_results,
                'motion_planning_results': self.planning_results
            })

        return results

    def get_current_state(self) -> Dict[str, Any]:
        """
        Get the current state of the pipeline.

        Returns:
            Dictionary containing current pipeline state
        """
        return {
            'objects_dict': self.objects_dict,
            'history_point_cloud': self.history_point_cloud,
            'cost_map': self.cost_map,
            'completed_map': self.completed_map,
            'planning_results': self.planning_results
        }

    def reset(self):
        """
        Reset the pipeline state.
        """
        self.objects_dict = {}
        self.history_point_cloud = []
        self.cost_map = None
        self.completed_map = None
        self.planning_results = None

        # Reset sub-pipelines if they have reset methods
        if hasattr(self.perception_pipeline, 'reset'):
            self.perception_pipeline.reset()
        if hasattr(self.map_generation_pipeline, 'reset'):
            self.map_generation_pipeline.reset()
        if hasattr(self.motion_planning_pipeline, 'reset'):
            self.motion_planning_pipeline.reset()

    def visualize_results(self, save_path: Optional[str] = None):
        """
        Visualize the planning results.

        Args:
            save_path: Optional path to save the visualization
        """
        # Check if motion planning pipeline has visualization capability
        if (hasattr(self.motion_planning_pipeline, 'visualize') and
                self.completed_map is not None and
                self.planning_results is not None):

            # Set the cost map for visualization
            self.motion_planning_pipeline.cost_map = self.completed_map

            # Visualize the results
            self.motion_planning_pipeline.visualize(
                show_path=True,
                show_trees=False,
                save_path=save_path
            )
        else:
            print("Visualization not available or required data not present")

    def update_mission(self, new_mission: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the motion mission and re-run planning if needed.

        Args:
            new_mission: New motion mission specification

        Returns:
            Updated planning results
        """
        # Check if we have all required data for planning
        if (self.completed_map is not None and
                self.objects_dict is not None):

            # Run only the motion planning step with new mission
            self.planning_results = self.motion_planning_pipeline.execute_full_pipeline(
                cost_map=self.completed_map,
                objects_dict=self.objects_dict,
                motion_mission=new_mission
            )

            return {
                'planning_success': self.planning_results.get('success', False),
                'planned_path': self.planning_results.get('path', []),
                'path_cost': self.planning_results.get('path_cost', float('inf')),
                'planning_time': self.planning_results.get('planning_time', 0),
                'iterations': self.planning_results.get('iterations', 0)
            }
        else:
            raise ValueError("Cannot update mission: Required data (map or objects) not available")