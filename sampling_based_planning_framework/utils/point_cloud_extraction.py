import numpy as np
from typing import List, Dict, Any, Union


def extract_point_clouds(objects_dict: Dict[str, Dict[str, Any]],
                         history_point_cloud: List[np.ndarray]) -> List[np.ndarray]:
    """
    Extract point cloud data from object contours and combine with historical point cloud data.

    This function extracts sample points from the contour equations of all objects in the objects dictionary,
    then combines these points with the historical point cloud data to form a complete point cloud representation.

    Args:
        objects_dict (Dict): Dictionary containing multiple object sub-dictionaries. Each object dictionary
                            should contain keys such as 'material_property', 'contour_equation',
                            'interaction_pairs', and 'contact_status'.
        history_point_cloud (List[np.ndarray]): List of historical 2D point clouds.

    Returns:
        List[np.ndarray]: Combined list of 2D point clouds including both object contour samples
                         and historical point clouds.
    """
    # Initialize list to store all point clouds
    all_point_clouds = []

    # Extract sample points from each object's contour equation
    for obj_name, obj_data in objects_dict.items():
        if 'contour_equation' in obj_data and obj_data['contour_equation'] is not None:
            contour_eq = obj_data['contour_equation']
            if 'sample_points' in contour_eq and contour_eq['sample_points'] is not None:
                # Ensure sample points are in the correct format
                sample_points = np.array(contour_eq['sample_points'])
                if sample_points.ndim == 2 and sample_points.shape[1] == 2:
                    all_point_clouds.append(sample_points)

    # Add historical point clouds
    for point_cloud in history_point_cloud:
        point_cloud_array = np.array(point_cloud)
        if point_cloud_array.ndim == 2 and point_cloud_array.shape[1] == 2:
            all_point_clouds.append(point_cloud_array)

    return all_point_clouds


# Example usage
if __name__ == "__main__":
    # Create a mock objects dictionary
    objects_dict = {
        "object0": {
            "material_property": [0.5, 0.3, 0.2],
            "contour_equation": {
                "type": "circle",
                "params": [1.0, 1.0, 0.5],
                "sample_points": np.array([[1.5, 1.0], [1.0, 1.5], [0.5, 1.0], [1.0, 0.5]])
            },
            "interaction_pairs": [],
            "contact_status": "active"
        },
        "object1": {
            "material_property": [0.8, 0.1, 0.1],
            "contour_equation": {
                "type": "square",
                "params": [2.0, 2.0, 1.0],
                "sample_points": np.array([[1.5, 1.5], [2.5, 1.5], [2.5, 2.5], [1.5, 2.5]])
            },
            "interaction_pairs": [],
            "contact_status": "active"
        }
    }

    # Create mock historical point cloud
    history_point_cloud = [
        np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
        np.array([[3.0, 3.0], [3.1, 3.2], [3.3, 3.4]])
    ]

    # Extract point clouds
    combined_point_clouds = extract_point_clouds(objects_dict, history_point_cloud)

    # Print results
    print("Number of point clouds:", len(combined_point_clouds))
    for i, cloud in enumerate(combined_point_clouds):
        print(f"Point cloud {i} shape: {cloud.shape}")
        print(f"Point cloud {i}:\n{cloud}")