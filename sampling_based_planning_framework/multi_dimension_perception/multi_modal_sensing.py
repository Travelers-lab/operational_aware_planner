from enum import Enum, auto
from typing import Dict, Any, Optional, Union
import numpy as np
from sampling_based_planning_framework.multi_dimension_perception.tactile_perception import approaching_sensing, contact_sensing


class TactileDataTypes(Enum):
    """Enumeration of available tactile data types"""
    APPROACHING_SENSING = auto()
    CONTACT_SENSING = auto()
    ALL = auto()


class TactileDataManager:
    """
    A class to manage tactile data acquisition from different sensing modalities.

    This class provides a unified interface for retrieving tactile perception data
    from both approaching sensing (proximity) and contact sensing (touch) modules.
    """

    def __init__(self):
        """
        Initialize the Tactile Data Manager.

        Attributes:
            sensor_transform_matrix: Transformation matrix for approaching sensor
            robots: Robot identifiers for contact sensing
            objectsId: Object identifiers for contact sensing
        """
        self.sensor_transform_matrix = None
        self.robots = None
        self.objectsId = None

    def set_sensor_parameters(self,
                              sensor_transform_matrix: Optional[np.ndarray] = None,
                              robots: Optional[Union[str, list]] = None,
                              objectsId: Optional[Union[int, list]] = None) -> None:
        """
        Set parameters for tactile sensing modules.

        Args:
            sensor_transform_matrix: Transformation matrix for approaching sensor
            robots: Robot identifier(s) for contact sensing. Can be string or list of strings
            objectsId: Object identifier(s) for contact sensing. Can be int or list of ints
        """
        if sensor_transform_matrix is not None:
            self.sensor_transform_matrix = sensor_transform_matrix

        if robots is not None:
            self.robots = robots if isinstance(robots, int) else robots[0]

        if objectsId is not None:
            self.objectsId = objectsId if isinstance(objectsId, list) else [objectsId]

    def validate_parameters(self, data_type: TactileDataTypes) -> bool:
        """
        Validate if required parameters are set for the requested data type.

        Args:
            data_type: The type of tactile data to validate parameters for

        Returns:
            bool: True if parameters are valid, False otherwise

        Raises:
            ValueError: If required parameters are missing
        """
        if data_type in [TactileDataTypes.APPROACHING_SENSING, TactileDataTypes.ALL]:
            if self.sensor_transform_matrix is None:
                raise ValueError("sensor_transform_matrix is required for approaching sensing")

        if data_type in [TactileDataTypes.CONTACT_SENSING, TactileDataTypes.ALL]:
            if self.robots is None:
                raise ValueError("robots parameter is required for contact sensing")
            if self.objectsId is None:
                raise ValueError("objectsId parameter is required for contact sensing")

        return True

    def get_tactile_data(self,
                         data_type: TactileDataTypes = TactileDataTypes.ALL,
                         **kwargs) -> Dict[str, Any]:
        """
        Retrieve tactile perception data based on specified data type.

        Args:
            data_type: Type of tactile data to retrieve. Defaults to ALL
            **kwargs: Additional parameters that will be passed to sensing functions

        Returns:
            Dict[str, Any]: Dictionary containing the requested tactile data.
                           Keys are 'approaching_data' and/or 'contact_data'

        Raises:
            ValueError: If invalid data type is specified or parameters are missing
        """
        # Update parameters if provided via kwargs
        self.set_sensor_parameters(**kwargs)

        # Validate parameters for the requested data type
        self.validate_parameters(data_type)

        result = {}

        # Get approaching sensing data if requested
        if data_type in [TactileDataTypes.APPROACHING_SENSING, TactileDataTypes.ALL]:
            approaching_data = self._get_approaching_data()
            result['approaching_data'] = approaching_data

        # Get contact sensing data if requested
        if data_type in [TactileDataTypes.CONTACT_SENSING, TactileDataTypes.ALL]:
            contact_data = self._get_contact_data()
            result['contact_data'] = contact_data

        return result

    def _get_approaching_data(self) -> Any:
        """
        Retrieve approaching sensing data.

        Returns:
            Any: Data returned by approaching_sensing function.
                 Return type matches the original function's return type.
        """
        return approaching_sensing(self.sensor_transform_matrix)

    def _get_contact_data(self) -> Any:
        """
        Retrieve contact sensing data.

        Returns:
            Any: Data returned by contact_sensing function.
                 Return type matches the original function's return type.
        """
        return contact_sensing(self.robots, self.objectsId)

    def get_approaching_data(self, sensor_transform_matrix: Optional[np.ndarray] = None) -> Any:
        """
        Convenience method to get only approaching sensing data.

        Args:
            sensor_transform_matrix: Optional transformation matrix. If not provided,
                                   uses the previously set value.

        Returns:
            Any: Approaching sensing data
        """
        if sensor_transform_matrix is not None:
            self.sensor_transform_matrix = sensor_transform_matrix

        self.validate_parameters(TactileDataTypes.APPROACHING_SENSING)
        return self._get_approaching_data()

    def get_contact_data(self,
                         robots: Optional[Union[str, list]] = None,
                         objectsId: Optional[Union[int, list]] = None) -> Any:
        """
        Convenience method to get only contact sensing data.

        Args:
            robots: Optional robot identifier(s). If not provided, uses previously set value.
            objectsId: Optional object identifier(s). If not provided, uses previously set value.

        Returns:
            Any: Contact sensing data
        """
        if robots is not None:
            self.robots = robots if isinstance(robots, list) else [robots]
        if objectsId is not None:
            self.objectsId = objectsId if isinstance(objectsId, list) else [objectsId]

        self.validate_parameters(TactileDataTypes.CONTACT_SENSING)
        return self._get_contact_data()

    def clear_parameters(self) -> None:
        """Clear all stored sensor parameters."""
        self.sensor_transform_matrix = None
        self.robots = None
        self.objectsId = None

    def get_current_parameters(self) -> Dict[str, Any]:
        """
        Get currently set sensor parameters.

        Returns:
            Dict[str, Any]: Dictionary containing current parameter values
        """
        return {
            'sensor_transform_matrix': self.sensor_transform_matrix,
            'robots': self.robots,
            'objectsId': self.objectsId
        }


# Example usage
if __name__ == "__main__":
    # Initialize the tactile data manager
    tactile_manager = TactileDataManager()

    # Set parameters
    transform_matrix = np.eye(4)  # Example transformation matrix
    robots_list = ["robot_arm_1"]
    objects_list = [101, 102]

    tactile_manager.set_sensor_parameters(
        sensor_transform_matrix=transform_matrix,
        robots=robots_list,
        objectsId=objects_list
    )

    # Get all tactile data
    all_data = tactile_manager.get_tactile_data(TactileDataTypes.ALL)
    print("All tactile data:", all_data.keys())

    # Get only approaching data
    approaching_data = tactile_manager.get_tactile_data(TactileDataTypes.APPROACHING_SENSING)
    print("Approaching data:", approaching_data.keys())

    # Get only contact data
    contact_data = tactile_manager.get_tactile_data(TactileDataTypes.CONTACT_SENSING)
    print("Contact data:", contact_data.keys())

    # Use convenience methods
    approaching_only = tactile_manager.get_approaching_data()
    contact_only = tactile_manager.get_contact_data()

    # Get current parameters
    current_params = tactile_manager.get_current_parameters()
    print("Current parameters:", current_params)