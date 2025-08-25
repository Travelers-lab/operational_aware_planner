import numpy as np
from typing import Dict, List, Any, Optional
from enum import Enum
import math
import matplotlib.pyplot as plt

class ShapeType(Enum):
    """Enumeration of supported shape types"""
    CIRCLE = "circle"
    SQUARE = "square"
    TRIANGLE = "triangle"
    UNKNOWN = "unknown"


class PointCloudExtractor:
    """
    A class for extracting point clouds from objects dictionary and history point cloud.

    This class generates a complete point cloud by combining historical points
    with points sampled from object contours.
    """

    def __init__(self, num_contour_points: int = 100):
        """
        Initialize the PointCloudExtractor.

        Args:
            num_contour_points: Number of points to sample from each object contour
        """
        self.num_contour_points = num_contour_points

    def extract_point_cloud(self,
                            objects_dict: Dict[str, Dict[str, Any]],
                            history_point_cloud: List[np.ndarray]) -> List[np.ndarray]:
        """
        Extract complete point cloud from objects and history data.

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
            history_point_cloud: List of historical 2D points (n*2)

        Returns:
            Complete point cloud list containing both history points and object contour points
        """
        # Initialize with history point cloud
        complete_point_cloud = history_point_cloud.copy()

        # Process each object to extract contour points
        for obj_id, obj_data in objects_dict.items():
            contour_equation = obj_data.get('contour_equation')

            if contour_equation is not None and isinstance(contour_equation, dict):
                contour_points = self._extract_contour_points(contour_equation)
                complete_point_cloud.extend(contour_points)

        return complete_point_cloud

    def _extract_contour_points(self, contour_equation: Dict[str, Any]) -> List[np.ndarray]:
        """
        Extract points from object contour based on its shape equation.

        Args:
            contour_equation: Dictionary containing shape parameters

        Returns:
            List of points sampled from the object contour
        """
        if not contour_equation or 'params' not in contour_equation:
            return []

        shape_type = contour_equation.get('shape')
        params = contour_equation.get('params', {})

        try:
            if shape_type == ShapeType.CIRCLE.value:
                return self._sample_circle_points(params)
            elif shape_type == ShapeType.SQUARE.value:
                return self._sample_square_points(params)
            elif shape_type == ShapeType.TRIANGLE.value:
                return self._sample_triangle_points(params)
            else:
                print(f"Unsupported shape type: {shape_type}")
                return []

        except Exception as e:
            print(f"Error extracting contour points for shape {shape_type}: {e}")
            return []

    def _sample_circle_points(self, params: Dict[str, Any]) -> List[np.ndarray]:
        """
        Sample points from a circle contour.

        Args:
            params: Circle parameters {'center': [x, y], 'radius': r}

        Returns:
            List of points on the circle circumference
        """
        center = np.array(params.get('center', [0.0, 0.0]))
        radius = params.get('radius', 1.0)

        points = []
        for i in range(self.num_contour_points):
            angle = 2 * math.pi * i / self.num_contour_points
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            points.append(np.array([x, y], dtype=np.float32))

        return points

    def _sample_square_points(self, params: Dict[str, Any]) -> List[np.ndarray]:
        """
        Sample points from a square contour.

        Args:
            params: Square parameters {'center': [x, y], 'side_length': l}

        Returns:
            List of points on the square perimeter
        """
        center = np.array(params.get('center', [0.0, 0.0]))
        side_length = params.get('side_length', 1.0)
        half_side = side_length / 2.0

        points = []
        points_per_side = self.num_contour_points // 4

        # Top side
        for i in range(points_per_side):
            x = center[0] - half_side + (side_length * i / points_per_side)
            y = center[1] - half_side
            points.append(np.array([x, y], dtype=np.float32))

        # Right side
        for i in range(points_per_side):
            x = center[0] + half_side
            y = center[1] - half_side + (side_length * i / points_per_side)
            points.append(np.array([x, y], dtype=np.float32))

        # Bottom side
        for i in range(points_per_side):
            x = center[0] + half_side - (side_length * i / points_per_side)
            y = center[1] + half_side
            points.append(np.array([x, y], dtype=np.float32))

        # Left side
        for i in range(points_per_side):
            x = center[0] - half_side
            y = center[1] + half_side - (side_length * i / points_per_side)
            points.append(np.array([x, y], dtype=np.float32))

        # Add remaining points to complete the total count
        remaining_points = self.num_contour_points - len(points)
        for i in range(remaining_points):
            points.append(points[i % len(points)].copy())

        return points

    def _sample_triangle_points(self, params: Dict[str, Any]) -> List[np.ndarray]:
        """
        Sample points from a triangle contour.

        Args:
            params: Triangle parameters {'vertices': [[x1,y1], [x2,y2], [x3,y3]]}

        Returns:
            List of points on the triangle perimeter
        """
        vertices = params.get('vertices', [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        if len(vertices) != 3:
            print("Triangle requires exactly 3 vertices")
            return []

        v0 = np.array(vertices[0], dtype=np.float32)
        v1 = np.array(vertices[1], dtype=np.float32)
        v2 = np.array(vertices[2], dtype=np.float32)

        points = []
        points_per_side = self.num_contour_points // 3

        # Edge v0-v1
        for i in range(points_per_side):
            t = i / points_per_side
            point = v0 + t * (v1 - v0)
            points.append(point)

        # Edge v1-v2
        for i in range(points_per_side):
            t = i / points_per_side
            point = v1 + t * (v2 - v1)
            points.append(point)

        # Edge v2-v0
        for i in range(points_per_side):
            t = i / points_per_side
            point = v2 + t * (v0 - v2)
            points.append(point)

        # Add remaining points to complete the total count
        remaining_points = self.num_contour_points - len(points)
        for i in range(remaining_points):
            points.append(points[i % len(points)].copy())

        return points

    def visualize_point_cloud(self, point_cloud: List[np.ndarray], title: str = "Point Cloud"):
        """
        Visualize the extracted point cloud.

        Args:
            point_cloud: List of 2D points to visualize
            title: Title for the plot
        """
        if not point_cloud:
            print("No points to visualize")
            return

        # Convert to numpy array for plotting
        points_array = np.array(point_cloud)

        plt.figure(figsize=(10, 8))
        plt.scatter(points_array[:, 0], points_array[:, 1], s=10, alpha=0.6)
        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.axis('equal')
        plt.show()

        print(f"Visualized {len(point_cloud)} points")


# Example usage
if __name__ == "__main__":
    # Create sample objects dictionary
    sample_objects = {
        'object_1': {
            'material_property': [0, 0, 0],
            'contour_equation': {
                'shape': 'circle',
                'params': {
                    'center': [2.0, 2.0],
                    'radius': 1.5
                }
            },
            'interaction_pairs': [],
            'contact_status': 'active'
        },
        'object_2': {
            'material_property': [0, 0, 0],
            'contour_equation': {
                'shape': 'square',
                'params': {
                    'center': [5.0, 5.0],
                    'side_length': 2.0
                }
            },
            'interaction_pairs': [],
            'contact_status': 'active'
        },
        'object_3': {
            'material_property': [0, 0, 0],
            'contour_equation': {
                'shape': 'triangle',
                'params': {
                    'vertices': [[8.0, 8.0], [10.0, 8.0], [9.0, 10.0]]
                }
            },
            'interaction_pairs': [],
            'contact_status': 'active'
        }
    }

    # Sample history point cloud
    sample_history = [
        np.array([1.0, 1.0]),
        np.array([1.5, 1.2]),
        np.array([2.0, 0.8]),
        np.array([6.0, 6.0]),
        np.array([7.0, 7.0])
    ]

    # Initialize point cloud extractor
    extractor = PointCloudExtractor(num_contour_points=100)

    # Extract complete point cloud
    complete_point_cloud = extractor.extract_point_cloud(sample_objects, sample_history)

    print(f"Total points in complete point cloud: {len(complete_point_cloud)}")
    print(f"History points: {len(sample_history)}")
    print(f"Contour points: {len(complete_point_cloud) - len(sample_history)}")

    # Visualize the results
    extractor.visualize_point_cloud(complete_point_cloud, "Complete Point Cloud")

    # Display first few points
    print("\nFirst 10 points:")
    for i, point in enumerate(complete_point_cloud[:10]):
        print(f"Point {i}: {point}")