import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from multi_modal_sensing import TactileDataManager
from memory_pool import ObjectRecognitionAndStorage
from interactive_characteristics_inference import MechanicalPropertyInferrer
from point_clond_extraction import PointCloudExtractor


class MultiModalTactilePerceptionPipeline:
    """
    A pipeline for multi-modal tactile perception that integrates tactile data acquisition,
    object dictionary maintenance, mechanical property inference, and point cloud extraction.

    This class orchestrates the workflow of acquiring tactile data, processing sensory information,
    inferring mechanical properties, and extracting point clouds from the processed data.
    """

    def __init__(self, config):
        """
        Initialize the multi-modal tactile perception pipeline with all necessary components.
        """
        self.tactile_data_manager = TactileDataManager()
        self.object_manager = ObjectRecognitionAndStorage(config)
        self.mechanical_inferrer = MechanicalPropertyInferrer()
        self.point_cloud_extractor = PointCloudExtractor()

        # Initialize data structures
        self.objects_dict = {}  # Dictionary to store object information
        self.history_point_cloud = []  # List to maintain historical point clouds
        self.grid_map = None  # Binary grid map for spatial representation

    def set_sensor_parameters(self,
                              sensor_transform_matrix: Optional[np.ndarray] = None,
                              robots: Optional[Union[str, List[str]]] = None,
                              objectsId: Optional[Union[int, List[int]]] = None) -> None:
        """
        Configure sensor parameters for tactile data acquisition.

        Args:
            sensor_transform_matrix: Optional transformation matrix for sensor calibration
            robots: Optional single robot identifier or list of robot identifiers
            objectsId: Optional single object ID or list of object IDs to focus on
        """
        self.tactile_data_manager.set_sensor_parameters(
            sensor_transform_matrix=sensor_transform_matrix,
            robots=robots,
            objectsId=objectsId
        )

    def set_grid_map(self, grid_map: np.ndarray) -> None:
        """
        Set the binary grid map for spatial context.

        Args:
            grid_map: Binary grid map as numpy array for spatial representation
        """
        self.grid_map = grid_map

    def acquire_tactile_data(self) -> Dict[str, List]:
        """
        Acquire both approaching and contact tactile data from sensors.

        Returns:
            Dictionary containing 'approaching_data' and 'contact_data' lists
        """
        tactile_data = self.tactile_data_manager.get_tactile_data()
        return tactile_data

    def acquire_approaching_data(self) -> List:
        """
        Acquire only approaching tactile data from sensors.

        Returns:
            List of approaching tactile data
        """
        return self.tactile_data_manager.get_approaching_data()

    def acquire_contact_data(self) -> List:
        """
        Acquire only contact tactile data from sensors.

        Returns:
            List of contact tactile data
        """
        return self.tactile_data_manager.get_contact_data()

    def process_sensory_data(self, tactile_data: Dict[str, List],
                             objects_dict: Dict[str, Dict[str, Any]],
                             binary_grid_map: np.ndarray,
                             history_point_cloud: List[List[float]]):
        """
        Process acquired tactile data to update object dictionary and point cloud history.

        Args:
            tactile_data: Dictionary containing 'approaching_data' and 'contact_data'
        """
        # Process sensory data to update objects_dict and history_point_cloud
        objects_dict, history_point_cloud = self.object_manager.process_sensory_data(
            sensing_data=tactile_data,
            binary_grid_map=binary_grid_map,
            objects_dict=objects_dict,
            history_point_cloud=history_point_cloud
        )

    def infer_mechanical_properties(self) -> None:
        """
        Infer mechanical properties of objects and update the objects dictionary.
        """
        # Infer mechanical properties and update objects_dict
        updated_objects_dict = self.mechanical_inferrer.infer_properties(
            objects_dict=self.objects_dict
        )

        # Update the objects dictionary with inferred properties
        self.objects_dict = updated_objects_dict

    def extract_point_cloud(self) -> List[List[float]]:
        """
        Extract point cloud from current object dictionary and historical point clouds.

        Returns:
            List of point clouds as numpy arrays (n*2 format)
        """
        point_cloud = self.point_cloud_extractor.extract_point_cloud(
            objects_dict=self.objects_dict,
            history_point_cloud=self.history_point_cloud
        )

        # Add the newly extracted point cloud to history
        if point_cloud is not None and len(point_cloud) > 0:
            self.history_point_cloud.append(point_cloud)

        return point_cloud

    def run_full_pipeline(self,
                          objects_dict: Dict[str, Dict[str, Any]],
                          binary_grid_map: np.ndarray,
                          history_point_cloud: List[List[float]]) -> List[List[float]]:
        """
        Execute the complete multi-modal tactile perception pipeline.

        Returns:
            List of extracted point clouds as numpy arrays
        """
        # Step 1: Acquire tactile data
        tactile_data = self.acquire_tactile_data()

        # Step 2: Process sensory data to update object dictionary and point cloud history
        self.process_sensory_data(tactile_data,
                                  objects_dict,
                                  binary_grid_map,
                                  history_point_cloud)

        # Step 3: Infer mechanical properties
        self.infer_mechanical_properties()

        # Step 4: Extract point cloud
        point_cloud = self.extract_point_cloud()

        return point_cloud

    def get_objects_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the current objects dictionary.

        Returns:
            Dictionary containing information about all recognized objects
        """
        return self.objects_dict

    def get_history_point_cloud(self) -> List[np.ndarray]:
        """
        Get the historical point cloud data.

        Returns:
            List of historical point clouds
        """
        return self.history_point_cloud

    def clear_history(self) -> None:
        """
        Clear the historical point cloud data while preserving current object dictionary.
        """
        self.history_point_cloud = []

    def reset_pipeline(self) -> None:
        """
        Reset the entire pipeline by clearing all stored data.
        """
        self.objects_dict = {}
        self.history_point_cloud = []
        self.grid_map = None