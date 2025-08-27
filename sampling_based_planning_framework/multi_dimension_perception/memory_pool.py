import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from scipy import spatial, ndimage
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import copy
from dataclasses import dataclass, field

@dataclass
class Config:
    """Configuration class for object recognition and data storage"""
    work_space: List[List[float]] = field(default_factory=lambda: [[0.30, 0.05], [0.85, 0.6]])
    grid_resolution: int = 100
    contact_threshold: float = 0.005  # 5mm threshold for contact detection
    min_object_points: int = 10  # Minimum points to form an object


class ObjectRecognitionAndStorage:
    """
    A class for object recognition and data storage management.

    This class handles:
    - Coordinate transformation between grid and workspace
    - Tactile data processing and object recognition
    - Object dictionary maintenance and updating
    - Point cloud history management
    - Object contour inference from binary grid maps
    """

    def __init__(self, config: Config):
        """
        Initialize the object recognition and data manager.

        Args:
            config: Configuration object containing workspace and resolution parameters
        """
        self.config = config
        self.work_space = config.workspace_bounds
        self.grid_resolution = config.grid_resolution
        self.contact_threshold = config.contact_threshold

        # Calculate grid parameters
        self.x_min, self.y_min = self.work_space[0]
        self.x_max, self.y_max = self.work_space[1]
        self.x_range = self.x_max - self.x_min
        self.y_range = self.y_max - self.y_min

    def coordinate_transform(self,
                             coord: List[float],
                             to_grid: bool = True) -> List[float]:
        """
        Transform coordinates between workspace and grid space.

        Args:
            coord: Input coordinates [x, y] or [grid_x, grid_y]
            to_grid: If True, transform workspace to grid; else grid to workspace

        Returns:
            Transformed coordinates
        """
        if to_grid:
            # Workspace to grid coordinates
            x, y = coord
            grid_x = int((x - self.x_min) / self.x_range * self.grid_resolution)
            grid_y = int((y - self.y_min) / self.y_range * self.grid_resolution)

            # Clamp to grid boundaries
            grid_x = max(0, min(self.grid_resolution - 1, grid_x))
            grid_y = max(0, min(self.grid_resolution - 1, grid_y))

            return [grid_x, grid_y]
        else:
            # Grid to workspace coordinates
            grid_x, grid_y = coord
            x = self.x_min + (grid_x / self.grid_resolution) * self.x_range
            y = self.y_min + (grid_y / self.grid_resolution) * self.y_range

            return [x, y]

    def is_point_in_contour(self,
                            point: List[float],
                            contour_equation: Dict[str, Any]) -> bool:
        """
        Check if a point is inside or near a contour.

        Args:
            point: Point coordinates [x, y]
            contour_equation: Contour equation dictionary

        Returns:
            True if point is inside or near the contour
        """
        shape = contour_equation.get('shape', '')
        params = contour_equation.get('params', {})

        if shape == 'circle':
            center = params.get('center', [0, 0])
            radius = params.get('radius', 0)
            distance = np.sqrt((point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2)
            return distance <= (radius + self.contact_threshold)

        elif shape == 'square' or shape == 'rectangle':
            center = params.get('center', [0, 0])
            if 'side_length' in params:
                side = params['side_length']
                half_side = side / 2
                x_min, x_max = center[0] - half_side, center[0] + half_side
                y_min, y_max = center[1] - half_side, center[1] + half_side
            else:
                width = params.get('width', 0)
                height = params.get('height', 0)
                half_width, half_height = width / 2, height / 2
                x_min, x_max = center[0] - half_width, center[0] + half_width
                y_min, y_max = center[1] - half_height, center[1] + half_height

            # Check if point is inside expanded bounding box
            expanded_x_min = x_min - self.contact_threshold
            expanded_x_max = x_max + self.contact_threshold
            expanded_y_min = y_min - self.contact_threshold
            expanded_y_max = y_max + self.contact_threshold

            return (expanded_x_min <= point[0] <= expanded_x_max and
                    expanded_y_min <= point[1] <= expanded_y_max)

        elif shape == 'triangle':
            vertices = params.get('vertices', [])
            if len(vertices) < 3:
                return False

            # Simple bounding box check for triangles
            x_coords = [v[0] for v in vertices]
            y_coords = [v[1] for v in vertices]

            expanded_x_min = min(x_coords) - self.contact_threshold
            expanded_x_max = max(x_coords) + self.contact_threshold
            expanded_y_min = min(y_coords) - self.contact_threshold
            expanded_y_max = max(y_coords) + self.contact_threshold

            return (expanded_x_min <= point[0] <= expanded_x_max and
                    expanded_y_min <= point[1] <= expanded_y_max)

        return False

    def object_contour_inference(self,
                                 grid_map: np.ndarray,
                                 contact_point: List[float]) -> Dict[str, Any]:
        """
        Infer object contour from binary grid map and contact point.

        Args:
            grid_map: 100x100 binary occupancy grid map
            contact_point: Contact point coordinates [x, y] in workspace

        Returns:
            Contour equation dictionary
        """
        # Convert contact point to grid coordinates
        grid_contact = self.coordinate_transform(contact_point, to_grid=True)
        grid_x, grid_y = grid_contact

        # Find connected obstacle region around contact point
        labeled_map, num_features = ndimage.label(grid_map)
        contact_label = labeled_map[grid_y, grid_x]

        if contact_label == 0:  # No obstacle at contact point
            return {'shape': 'unknown', 'params': {}}

        # Get all points belonging to this obstacle
        obstacle_points = np.argwhere(labeled_map == contact_label)

        if len(obstacle_points) < self.config.min_object_points:
            return {'shape': 'unknown', 'params': {}}

        # Convert grid points to workspace coordinates
        workspace_points = [self.coordinate_transform([x, y], to_grid=False)
                            for y, x in obstacle_points]

        # Calculate bounding box and basic shape properties
        x_coords = [p[0] for p in workspace_points]
        y_coords = [p[1] for p in workspace_points]

        center_x = (min(x_coords) + max(x_coords)) / 2
        center_y = (min(y_coords) + max(y_coords)) / 2
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)

        # Simple shape classification
        aspect_ratio = width / height if height > 0 else 1
        area = width * height
        filled_area = len(obstacle_points) * (self.x_range / self.grid_resolution) * (
                    self.y_range / self.grid_resolution)
        fill_ratio = filled_area / area if area > 0 else 0

        # Shape inference logic
        if 0.8 <= aspect_ratio <= 1.2 and fill_ratio > 0.6:
            # Likely circle or square
            if fill_ratio > 0.7:
                # Circle
                radius = max(width, height) / 2
                return {
                    'shape': 'circle',
                    'params': {
                        'center': [center_x, center_y],
                        'radius': radius
                    }
                }
            else:
                # Square
                side_length = max(width, height)
                return {
                    'shape': 'square',
                    'params': {
                        'center': [center_x, center_y],
                        'side_length': side_length
                    }
                }
        elif fill_ratio > 0.5:
            # Rectangle
            return {
                'shape': 'rectangle',
                'params': {
                    'center': [center_x, center_y],
                    'width': width,
                    'height': height
                }
            }
        else:
            # Triangle or complex shape - use convex hull or bounding polygon
            try:
                # Use DBSCAN to find main clusters
                clustering = DBSCAN(eps=2, min_samples=5).fit(workspace_points)
                unique_labels = set(clustering.labels_)

                if len(unique_labels) == 2:  # Main cluster + noise
                    main_cluster_points = [p for p, l in zip(workspace_points, clustering.labels_)
                                           if l == 0]
                    if len(main_cluster_points) >= 3:
                        # Fit triangle to main cluster
                        hull = spatial.ConvexHull(main_cluster_points)
                        vertices = [main_cluster_points[i] for i in hull.vertices[:3]]

                        return {
                            'shape': 'triangle',
                            'params': {
                                'vertices': vertices
                            }
                        }
            except:
                pass

            # Fallback to rectangle
            return {
                'shape': 'rectangle',
                'params': {
                    'center': [center_x, center_y],
                    'width': width,
                    'height': height
                }
            }

    def process_tactile_data(self,
                             tactile_data: Dict[str, Any],
                             history_point_cloud: List[List[float]],
                             grid_map: np.ndarray,
                             object_dict: Dict[str, Dict[str, Any]]) -> Tuple[
        Dict[str, Dict[str, Any]], List[List[float]]]:
        """
        Process tactile data and update object dictionary and point cloud history.

        Args:
            tactile_data: Tactile perception data
            history_point_cloud: Historical point cloud data
            grid_map: Binary occupancy grid map
            object_dict: Object dictionary

        Returns:
            Updated object dictionary and point cloud history
        """
        # Create deep copies to avoid modifying inputs
        updated_object_dict = copy.deepcopy(object_dict)
        updated_history_point_cloud = copy.deepcopy(history_point_cloud)

        # Initialize object counter if not exists
        if not any('object' in key for key in updated_object_dict.keys()):
            object_counter = 0
        else:
            object_counter = max([int(key.replace('object', ''))
                                  for key in updated_object_dict.keys()
                                  if key.startswith('object')], default=-1) + 1

        # Process contact data if available
        contact_data = tactile_data.get('contact_data', [])
        for contact_entry in contact_data:
            if len(contact_entry) < 2:
                continue

            contact_position = contact_entry[0]
            object_data = contact_entry[1] if len(contact_entry) > 1 else []

            # Find existing object that contains contact position
            matched_object_id = None
            for obj_id, obj_data in updated_object_dict.items():
                contour_eq = obj_data.get('contour_equation', {})
                if self.is_point_in_contour(contact_position, contour_eq):
                    matched_object_id = obj_id
                    break

            if matched_object_id:
                # Update existing object
                if object_data and len(object_data) >= 3:
                    interaction_data = {
                        'object_position': object_data[0],
                        'object_velocity': object_data[1],
                        'contact_force': object_data[2]
                    }

                    # Add to interaction pairs
                    if 'interaction_pairs' not in updated_object_dict[matched_object_id]:
                        updated_object_dict[matched_object_id]['interaction_pairs'] = []
                    updated_object_dict[matched_object_id]['interaction_pairs'].append(interaction_data)

                # Update object center if contact position is significantly different
                contour_eq = updated_object_dict[matched_object_id].get('contour_equation', {})
                if contour_eq.get('shape') in ['circle', 'square', 'rectangle']:
                    current_center = contour_eq.get('params', {}).get('center', [0, 0])
                    if np.linalg.norm(np.array(contact_position) - np.array(current_center)) > self.contact_threshold:
                        contour_eq['params']['center'] = contact_position

                # Set contact status
                updated_object_dict[matched_object_id]['contact_status'] = True

            else:
                # Create new object
                object_id = f"object{object_counter}"
                object_counter += 1

                # Infer contour from grid map
                contour_equation = self.object_contour_inference(grid_map, contact_position)

                # Create new object entry
                new_object = {
                    'material_property': [0.0, 0.0, 0.0],
                    'contour_equation': contour_equation,
                    'interaction_pairs': [],
                    'contact_status': True
                }

                # Add interaction data if available
                if object_data and len(object_data) >= 3:
                    interaction_data = {
                        'object_position': object_data[0],
                        'object_velocity': object_data[1],
                        'contact_force': object_data[2]
                    }
                    new_object['interaction_pairs'].append(interaction_data)

                updated_object_dict[object_id] = new_object

                # Remove points near the new object from history point cloud
                updated_history_point_cloud = [
                    point for point in updated_history_point_cloud
                    if not self.is_point_in_contour(point, contour_equation)
                ]

        # Process approaching data
        approaching_data = tactile_data.get('approaching_data', [])
        for point in approaching_data:
            point_near_object = False

            # Check if point is near any existing object
            for obj_data in updated_object_dict.values():
                contour_eq = obj_data.get('contour_equation', {})
                if self.is_point_in_contour(point, contour_eq):
                    point_near_object = True
                    break

            # Add to history if not near any object
            if not point_near_object:
                updated_history_point_cloud.append(point)

        # Update contact status for all objects
        contact_positions = [entry[0] for entry in contact_data if len(entry) > 0]

        for obj_id, obj_data in updated_object_dict.items():
            contour_eq = obj_data.get('contour_equation', {})
            has_contact = False

            # Check if any contact position is near this object
            for contact_pos in contact_positions:
                if self.is_point_in_contour(contact_pos, contour_eq):
                    has_contact = True
                    break

            # Update contact status
            obj_data['contact_status'] = has_contact

        # Clean up history point cloud (remove duplicates and limit size)
        updated_history_point_cloud = self._clean_point_cloud(updated_history_point_cloud)

        return updated_object_dict, updated_history_point_cloud

    def _clean_point_cloud(self, point_cloud: List[List[float]],
                           max_points: int = 1000) -> List[List[float]]:
        """
        Clean and deduplicate point cloud data.

        Args:
            point_cloud: Input point cloud
            max_points: Maximum number of points to keep

        Returns:
            Cleaned point cloud
        """
        if not point_cloud:
            return []

        # Remove duplicates using tolerance
        unique_points = []
        tolerance = self.contact_threshold / 2

        for point in point_cloud:
            is_duplicate = False
            for unique_point in unique_points:
                if (abs(point[0] - unique_point[0]) < tolerance and
                        abs(point[1] - unique_point[1]) < tolerance):
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_points.append(point)

        # Limit size
        if len(unique_points) > max_points:
            unique_points = unique_points[-max_points:]

        return unique_points

    def visualize_objects(self,
                          object_dict: Dict[str, Dict[str, Any]],
                          history_point_cloud: List[List[float]] = None,
                          title: str = "Object Recognition Results"):
        """
        Visualize objects and point cloud.

        Args:
            object_dict: Object dictionary to visualize
            history_point_cloud: Point cloud history to visualize
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot workspace boundaries
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        ax.set_xlabel('X Coordinate (m)')
        ax.set_ylabel('Y Coordinate (m)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # Plot objects
        colors = plt.cm.tab10.colors
        for i, (obj_id, obj_data) in enumerate(object_dict.items()):
            color = colors[i % len(colors)]
            contour_eq = obj_data.get('contour_equation', {})
            shape = contour_eq.get('shape', '')
            params = contour_eq.get('params', {})

            if shape == 'circle':
                center = params.get('center', [0, 0])
                radius = params.get('radius', 0)
                circle = plt.Circle(center, radius, fill=False, color=color, linewidth=2)
                ax.add_patch(circle)
                ax.plot(center[0], center[1], 'o', color=color, markersize=4)

            elif shape in ['square', 'rectangle']:
                center = params.get('center', [0, 0])
                if 'side_length' in params:
                    side = params['side_length']
                    half_side = side / 2
                    x_min, x_max = center[0] - half_side, center[0] + half_side
                    y_min, y_max = center[1] - half_side, center[1] + half_side
                else:
                    width = params.get('width', 0)
                    height = params.get('height', 0)
                    half_width, half_height = width / 2, height / 2
                    x_min, x_max = center[0] - half_width, center[0] + half_width
                    y_min, y_max = center[1] - half_height, center[1] + half_height

                rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                     fill=False, color=color, linewidth=2)
                ax.add_patch(rect)
                ax.plot(center[0], center[1], 'o', color=color, markersize=4)

            elif shape == 'triangle':
                vertices = params.get('vertices', [])
                if len(vertices) >= 3:
                    triangle = plt.Polygon(vertices[:3], fill=False, color=color, linewidth=2)
                    ax.add_patch(triangle)
                    center = np.mean(vertices[:3], axis=0)
                    ax.plot(center[0], center[1], 'o', color=color, markersize=4)

            # Add object ID label
            if 'center' in locals():
                ax.text(center[0], center[1] + 0.01, obj_id,
                        fontsize=8, ha='center', color=color)

        # Plot history point cloud
        if history_point_cloud:
            points_array = np.array(history_point_cloud)
            if len(points_array) > 0:
                ax.scatter(points_array[:, 0], points_array[:, 1],
                           s=10, alpha=0.6, color='gray', label='History Points')

        ax.legend()
        plt.tight_layout()
        return fig, ax


# Example usage
if __name__ == "__main__":
    # Initialize configuration
    config = Config(
        work_space=[[0.30, 0.05], [0.85, 0.6]],
        grid_resolution=100,
        contact_threshold=0.005
    )

    # Initialize object recognition manager
    manager = ObjectRecognitionAndStorage(config)

    # Example grid map (binary occupancy)
    grid_map = np.zeros((100, 100))
    # Add some obstacles
    grid_map[30:50, 40:60] = 1  # Square obstacle
    grid_map[70:80, 20:40] = 1  # Rectangular obstacle

    # Example tactile data
    tactile_data = {
        'approaching_data': [
            [0.4, 0.3],
            [0.5, 0.4],
            [0.6, 0.5]
        ],
        'contact_data': [
            [[0.45, 0.35], [[0.45, 0.35], [0.1, 0.1], [5.0, 3.0]]],
            [[0.75, 0.3], [[0.75, 0.3], [0.2, -0.1], [7.0, 4.0]]]
        ]
    }

    # Initialize empty object dictionary and point cloud
    object_dict = {}
    history_point_cloud = []

    # Process tactile data
    updated_objects, updated_point_cloud = manager.process_tactile_data(
        tactile_data, history_point_cloud, grid_map, object_dict
    )

    print(f"Detected objects: {list(updated_objects.keys())}")
    print(f"Updated point cloud size: {len(updated_point_cloud)}")

    # Visualize results
    fig, ax = manager.visualize_objects(updated_objects, updated_point_cloud)
    plt.show()