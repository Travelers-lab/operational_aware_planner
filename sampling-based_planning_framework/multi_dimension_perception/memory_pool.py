import numpy as np
from scipy.optimize import least_squares
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


class ObjectManager:
    def __init__(self):
        self.objects_dict = {}  # Dictionary to store object information
        self.history_point_cloud = []  # List to store historical point clouds
        self.object_counter = 0  # Counter for naming new objects

    def process_sensory_data(self, point_cloud, tactile_info, objects_dict, history_point_cloud):
        """
        Process sensory data including point cloud and tactile information.
        Updates object dictionary and history point cloud based on the input data.

        Args:
            point_cloud (list or numpy.ndarray): n x 2 array of 2D point cloud data.
            tactile_info (list of tuples): Each tuple contains a 1x2 contact point and a 3x2 interaction info.
            objects_dict (dict): Dictionary of objects, each containing material properties,
                                contour equations, interaction pairs, and contact status.
            history_point_cloud (list): List of historical point clouds.

        Returns:
            tuple: Updated objects_dict and history_point_cloud.
        """
        self.objects_dict = objects_dict
        self.history_point_cloud = history_point_cloud

        # If tactile information is provided
        if tactile_info:
            for contact_point, interaction_info in tactile_info:
                contact_point = np.array(contact_point).flatten()
                interaction_info = np.array(interaction_info)
                # Convert interaction info to 1x3 Euclidean distances
                interaction_dist = np.linalg.norm(interaction_info, axis=1)

                found_object = False
                # Check if contact point is near any existing object
                for obj_name, obj_data in self.objects_dict.items():
                    contour_eq = obj_data['contour_equation']
                    if self._is_point_near_contour(contact_point, contour_eq):
                        # Store interaction info
                        obj_data['interaction_pairs'].append(interaction_info)
                        # Update contour by shifting centroid (simplified update)
                        # Here, we shift the centroid by the mean of the first row of interaction_info
                        shift_vector = np.mean(interaction_info[:, 0])
                        obj_data['contour_equation'] = self._update_contour_centroid(contour_eq, shift_vector)
                        found_object = True
                        break

                if not found_object:
                    # Create new object
                    obj_name = f"object{self.object_counter}"
                    self.object_counter += 1
                    self.objects_dict[obj_name] = {
                        'material_property': [0, 0, 0],
                        'contour_equation': None,  # To be updated
                        'interaction_pairs': [interaction_dist],
                        'contact_status': 'active'
                    }
                    # Update contour using the contact point
                    # For simplicity, we assume a grid map is available; in practice, this should be passed
                    grid_map = np.zeros((100, 100))  # Placeholder grid map
                    contour_eq = self.update_object_contour(grid_map, contact_point)
                    self.objects_dict[obj_name]['contour_equation'] = contour_eq

                # Update history point cloud: remove points near the object's contour
                self.history_point_cloud = [
                    pt for pt in self.history_point_cloud
                    if not self._is_point_near_contour(pt, self.objects_dict[obj_name]['contour_equation'])
                ]

        # Process point cloud data
        if point_cloud is not None:
            point_cloud = np.array(point_cloud)
            for point in point_cloud:
                point_near_object = False
                for obj_data in self.objects_dict.values():
                    if self._is_point_near_contour(point, obj_data['contour_equation']):
                        point_near_object = True
                        break
                if not point_near_object:
                    self.history_point_cloud.append(point)

        return self.objects_dict, self.history_point_cloud

    def update_object_contour(self, grid_map, contact_point):
        """
        Update the contour equation of an object based on the grid map and contact point.

        Args:
            grid_map (numpy.ndarray): 100x100 binary occupancy grid map.
            contact_point (list or numpy.ndarray): 1x2 contact point.

        Returns:
            dict: Dictionary containing contour type, parameters, and sample points.
        """
        # Find nearest obstacle in the grid map
        obstacle_points = np.argwhere(grid_map == 1)
        if obstacle_points.size == 0:
            return None

        # Cluster obstacles to find the one nearest to the contact point
        clustering = DBSCAN(eps=3, min_samples=1).fit(obstacle_points)
        labels = clustering.labels_
        unique_labels = set(labels)

        min_dist = float('inf')
        nearest_cluster = None
        contact_point = np.array(contact_point).flatten()

        for label in unique_labels:
            cluster_points = obstacle_points[labels == label]
            centroid = np.mean(cluster_points, axis=0)
            dist = np.linalg.norm(centroid - contact_point)
            if dist < min_dist:
                min_dist = dist
                nearest_cluster = cluster_points

        if nearest_cluster is None:
            return None

        # Fit contour to the cluster points
        contour_type, params = self._fit_contour(nearest_cluster)
        sample_points = self._generate_sample_points(contour_type, params)

        return {
            'type': contour_type,
            'params': params,
            'sample_points': sample_points
        }

    def _is_point_near_contour(self, point, contour_eq, threshold=0.005):
        """
        Check if a point is near the contour of an object.

        Args:
            point (list or numpy.ndarray): Point to check.
            contour_eq (dict): Contour equation dictionary.
            threshold (float): Distance threshold.

        Returns:
            bool: True if point is near the contour.
        """
        if contour_eq is None:
            return False

        point = np.array(point).flatten()
        if contour_eq['type'] == 'circle':
            center = np.array(contour_eq['params'][:2])
            radius = contour_eq['params'][2]
            dist = np.linalg.norm(point - center) - radius
        elif contour_eq['type'] == 'square':
            center = np.array(contour_eq['params'][:2])
            side = contour_eq['params'][2]
            # Calculate distance to square (simplified)
            dist = np.max(np.abs(point - center)) - side / 2
        elif contour_eq['type'] == 'triangle':
            # For triangle, use sample points to compute distance
            sample_points = contour_eq['sample_points']
            dist = min([np.linalg.norm(point - sp) for sp in sample_points])
        else:
            return False

        return abs(dist) < threshold

    def _fit_contour(self, points):
        """
        Fit a contour (circle, square, triangle) to the given points.

        Args:
            points (numpy.ndarray): Points to fit.

        Returns:
            tuple: Contour type and parameters.
        """
        # Normalize points
        scaler = StandardScaler()
        points_scaled = scaler.fit_transform(points)

        # Try fitting circle
        def circle_residuals(params, pts):
            center = params[:2]
            radius = params[2]
            return np.linalg.norm(pts - center, axis=1) - radius

        initial_guess = np.mean(points_scaled, axis=0).tolist() + [1.0]
        res_circle = least_squares(circle_residuals, initial_guess, args=(points_scaled,))
        circle_error = np.mean(np.abs(res_circle.fun))

        # Try fitting square (simplified as bounding box)
        min_vals = np.min(points_scaled, axis=0)
        max_vals = np.max(points_scaled, axis=0)
        side_lengths = max_vals - min_vals
        square_error = np.std(side_lengths)  # Measure of how square it is

        # Try fitting triangle (simplified as convex hull with 3 vertices)
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points_scaled)
        if len(hull.vertices) >= 3:
            triangle_vertices = points_scaled[hull.vertices[:3]]
            # Calculate area of triangle
            area = 0.5 * np.abs(np.cross(triangle_vertices[1] - triangle_vertices[0],
                                         triangle_vertices[2] - triangle_vertices[0]))
            triangle_error = area / len(points)  # Error based on area coverage
        else:
            triangle_error = float('inf')

        # Choose the best fit
        errors = {'circle': circle_error, 'square': square_error, 'triangle': triangle_error}
        best_type = min(errors, key=errors.get)

        if best_type == 'circle':
            params = res_circle.x
            params[:2] = scaler.inverse_transform([params[:2]])[0]  # Rescale center
            params[2] *= scaler.scale_[0]  # Rescale radius
        elif best_type == 'square':
            center = scaler.inverse_transform([(min_vals + max_vals) / 2])[0]
            side = np.mean(scaler.inverse_transform([side_lengths])[0])
            params = np.array([center[0], center[1], side])
        else:  # triangle
            vertices = scaler.inverse_transform(triangle_vertices)
            params = vertices.flatten()

        return best_type, params

    def _generate_sample_points(self, contour_type, params, n_samples=100):
        """
        Generate sample points for the contour.

        Args:
            contour_type (str): Type of contour.
            params (numpy.ndarray): Parameters of the contour.
            n_samples (int): Number of sample points.

        Returns:
            numpy.ndarray: Sample points.
        """
        if contour_type == 'circle':
            center = params[:2]
            radius = params[2]
            angles = np.linspace(0, 2 * np.pi, n_samples)
            x = center[0] + radius * np.cos(angles)
            y = center[1] + radius * np.sin(angles)
            return np.vstack((x, y)).T
        elif contour_type == 'square':
            center = params[:2]
            side = params[2]
            half_side = side / 2
            x = np.array([center[0] - half_side, center[0] + half_side,
                          center[0] + half_side, center[0] - half_side])
            y = np.array([center[1] - half_side, center[1] - half_side,
                          center[1] + half_side, center[1] + half_side])
            return np.vstack((x, y)).T
        else:  # triangle
            vertices = params.reshape(3, 2)
            return vertices

    def _update_contour_centroid(self, contour_eq, shift_value):
        """
        Update the centroid of the contour equation by shifting it.

        Args:
            contour_eq (dict): Contour equation dictionary.
            shift_value (float): Value to shift the centroid.

        Returns:
            dict: Updated contour equation.
        """
        if contour_eq['type'] == 'circle':
            contour_eq['params'][0] += shift_value
            contour_eq['params'][1] += shift_value
        elif contour_eq['type'] == 'square':
            contour_eq['params'][0] += shift_value
            contour_eq['params'][1] += shift_value
        else:  # triangle
            contour_eq['params'][::2] += shift_value  # Shift x coordinates
            contour_eq['params'][1::2] += shift_value  # Shift y coordinates

        # Regenerate sample points
        contour_eq['sample_points'] = self._generate_sample_points(
            contour_eq['type'], contour_eq['params'])
        return contour_eq


import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from scipy import spatial, optimize
import matplotlib.pyplot as plt
from enum import Enum


class ShapeType(Enum):
    """Enumeration of supported shape types"""
    CIRCLE = "circle"
    SQUARE = "square"
    TRIANGLE = "triangle"
    UNKNOWN = "unknown"


class ObjectRecognitionAndStorage:
    """
    A class for object recognition, data storage, and maintenance.

    This class handles processing of sensory data, fitting object contours,
    and updating object feature parameters in an environment.
    """

    def __init__(self, grid_resolution: float = 0.01, distance_threshold: float = 0.005):
        """
        Initialize the Object Recognition and Storage system.

        Args:
            grid_resolution: Resolution of the grid map in meters per cell
            distance_threshold: Distance threshold for point association in meters
        """
        self.grid_resolution = grid_resolution
        self.distance_threshold = distance_threshold
        self.objects_dict = {}
        self.history_point_cloud = []
        self.object_counter = 0

    def process_sensory_data(self,
                             sensing_data: Dict[str, Any],
                             binary_grid_map: np.ndarray,
                             objects_dict: Dict[str, Dict[str, Any]],
                             history_point_cloud: List[List[float]]) -> Tuple[Dict[str, Any], List[np.ndarray]]:
        """
        Process sensory data and update objects dictionary and point cloud.

        Args:
            sensing_data: Dictionary containing approaching and contact data
            binary_grid_map: 100x100 binary occupancy grid map
            objects_dict: Dictionary containing object descriptions
            history_point_cloud: List of historical point cloud data

        Returns:
            Updated objects dictionary and history point cloud
        """
        self.objects_dict = objects_dict.copy()
        self.history_point_cloud = history_point_cloud.copy()

        # Process contact data if available
        contact_data_list = sensing_data.get('contact_data', [])
        if contact_data_list:
            self._process_contact_data(contact_data_list, binary_grid_map)

        # Process approaching data
        if sensing_data.get('approaching_data'):
            self._process_approaching_data(sensing_data['approaching_data'])

        # Update contact states for all objects based on current contact data
        self._update_contact_states(contact_data_list)

        # Clean up history point cloud
        self._cleanup_point_cloud()

        return self.objects_dict, self.history_point_cloud

    def _process_contact_data(self, contact_data_list: List, binary_grid_map: np.ndarray):
        """
        Process multiple contact data signals.

        Args:
            contact_data_list: List of contact data signals
            binary_grid_map: Binary occupancy grid map for contour fitting
        """
        # Track which objects have been updated in this processing cycle
        updated_objects = set()

        for contact_signal in contact_data_list:
            if len(contact_signal) != 2:
                print(f"Warning: Invalid contact signal format: {contact_signal}")
                continue

            contact_position_data = contact_signal[0]
            object_data_list = contact_signal[1]

            # Validate contact position
            if not isinstance(contact_position_data, (list, np.ndarray)) or len(contact_position_data) != 2:
                print(f"Warning: Invalid contact position: {contact_position_data}")
                continue

            contact_position = np.array(contact_position_data)

            # Validate object data
            if (not isinstance(object_data_list, list) or len(object_data_list) != 3 or
                    not all(isinstance(item, (list, np.ndarray)) and len(item) == 2 for item in object_data_list)):
                print(f"Warning: Invalid object data: {object_data_list}")
                continue

            object_position = np.array(object_data_list[0])
            object_velocity = np.array(object_data_list[1])
            contact_force = np.array(object_data_list[2])

            # Create complete object data package
            complete_object_data = [object_position, object_velocity, contact_force]

            # Find if contact point belongs to any existing object
            associated_object_id = self._find_associated_object(contact_position)

            if associated_object_id is None:
                # Create new object
                new_obj_id = self._create_new_object(contact_position, complete_object_data, binary_grid_map)
                if new_obj_id:
                    updated_objects.add(new_obj_id)
            else:
                # Update existing object
                self._update_existing_object(associated_object_id, contact_position, complete_object_data)
                updated_objects.add(associated_object_id)

        # For objects not updated in this cycle, mark them as potentially inactive
        # (final contact status will be determined in _update_contact_states)
        all_object_ids = set(self.objects_dict.keys())
        non_updated_objects = all_object_ids - updated_objects
        for obj_id in non_updated_objects:
            # Only mark as potentially inactive if we had contact data to process
            if contact_data_list:
                self.objects_dict[obj_id]['contact_status'] = False

    def _find_associated_object(self, point: np.ndarray) -> Optional[str]:
        """
        Find if a point is associated with any existing object.

        Args:
            point: 2D point coordinates

        Returns:
            Object ID if associated, None otherwise
        """
        for obj_id, obj_data in self.objects_dict.items():
            contour_eq = obj_data.get('contour_equation', {})
            if self._is_point_near_contour(point, contour_eq):
                return obj_id
        return None

    def _is_point_near_contour(self, point: np.ndarray, contour_eq: Dict[str, Any]) -> bool:
        """
        Check if a point is inside or near a contour.

        Args:
            point: 2D point coordinates
            contour_eq: Contour equation dictionary

        Returns:
            True if point is inside or near contour, False otherwise
        """
        if not contour_eq or 'params' not in contour_eq:
            return False

        shape_type = contour_eq.get('shape')
        params = contour_eq.get('params', {})

        try:
            if shape_type == ShapeType.CIRCLE.value:
                center = np.array(params.get('center', [0, 0]))
                radius = params.get('radius', 0)
                distance = np.linalg.norm(point - center)
                return distance <= radius + self.distance_threshold

            elif shape_type == ShapeType.SQUARE.value:
                center = np.array(params.get('center', [0, 0]))
                side_length = params.get('side_length', 0)
                half_side = side_length / 2

                # Check if point is inside extended square
                dx = abs(point[0] - center[0])
                dy = abs(point[1] - center[1])
                return dx <= half_side + self.distance_threshold and dy <= half_side + self.distance_threshold

            elif shape_type == ShapeType.TRIANGLE.value:
                vertices = params.get('vertices', [])
                if len(vertices) != 3:
                    return False

                # Calculate distance to triangle using barycentric coordinates
                vertices_array = np.array(vertices)
                return self._point_near_triangle(point, vertices_array)

        except Exception as e:
            print(f"Error in point contour check: {e}")
            return False

        return False

    def _point_near_triangle(self, point: np.ndarray, vertices: np.ndarray) -> bool:
        """
        Check if point is near a triangle using barycentric coordinates.

        Args:
            point: Point to check
            vertices: Triangle vertices (3x2 array)

        Returns:
            True if point is near triangle, False otherwise
        """
        # Calculate barycentric coordinates
        v0 = vertices[1] - vertices[0]
        v1 = vertices[2] - vertices[0]
        v2 = point - vertices[0]

        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        # Compute barycentric coordinates
        inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom

        # Check if point is in triangle
        if (u >= 0) and (v >= 0) and (u + v <= 1):
            return True

        # Check distance to edges if not inside
        min_distance = float('inf')
        for i in range(3):
            j = (i + 1) % 3
            edge = vertices[j] - vertices[i]
            edge_length = np.linalg.norm(edge)
            if edge_length > 0:
                t = max(0, min(1, np.dot(point - vertices[i], edge) / (edge_length * edge_length)))
                projection = vertices[i] + t * edge
                distance = np.linalg.norm(point - projection)
                min_distance = min(min_distance, distance)

        return min_distance <= self.distance_threshold

    def _create_new_object(self, contact_position: np.ndarray, object_data: List, binary_grid_map: np.ndarray) -> \
    Optional[str]:
        """
        Create a new object based on contact data.

        Args:
            contact_position: Contact position coordinates
            object_data: Object data including position, velocity, and force
            binary_grid_map: Binary occupancy grid map

        Returns:
            Object ID if created successfully, None otherwise
        """
        try:
            obj_id = f"object{self.object_counter}"
            self.object_counter += 1

            # Fit contour to the object
            contour_equation = self.fit_contour_features(binary_grid_map, contact_position)

            # Create new object dictionary
            new_object = {
                'material_property': [0, 0, 0],
                'contour_equation': contour_equation,
                'interaction_pairs': [object_data],
                'contact_status': True
            }

            self.objects_dict[obj_id] = new_object
            print(f"Created new object: {obj_id}")
            return obj_id

        except Exception as e:
            print(f"Error creating new object: {e}")
            return None

    def _update_existing_object(self, obj_id: str, contact_position: np.ndarray, object_data: List):
        """
        Update an existing object with new contact data.

        Args:
            obj_id: Object identifier
            contact_position: Contact position coordinates
            object_data: Object data including position, velocity, and force
        """
        try:
            if obj_id in self.objects_dict:
                # Add interaction pair
                self.objects_dict[obj_id]['interaction_pairs'].append(object_data)

                # Update contour equation with new object position
                object_position = object_data[0]  # object_position from object_data
                self.update_contour_features(obj_id, object_position)

                # Update contact status
                self.objects_dict[obj_id]['contact_status'] = True

                print(f"Updated object: {obj_id} with new interaction data")

        except Exception as e:
            print(f"Error updating object {obj_id}: {e}")

    def _process_approaching_data(self, approaching_data: List):
        """
        Process approaching data and update point cloud.

        Args:
            approaching_data: List of approaching data points
        """
        for point in approaching_data:
            try:
                point_array = np.array(point)
                if point_array.shape != (2,):
                    continue

                is_near_object = False

                # Check if point is near any object
                for obj_data in self.objects_dict.values():
                    contour_eq = obj_data.get('contour_equation', {})
                    if self._is_point_near_contour(point_array, contour_eq):
                        is_near_object = True
                        break

                # If not near any object, add to history point cloud
                if not is_near_object:
                    self.history_point_cloud.append(point_array)

            except Exception as e:
                print(f"Error processing approaching data point {point}: {e}")

    def _update_contact_states(self, contact_data_list: List):
        """
        Update contact states for all objects based on current contact data.

        Args:
            contact_data_list: List of contact data signals
        """
        # Extract all contact positions from current data
        current_contact_positions = []
        for contact_signal in contact_data_list:
            if len(contact_signal) >= 1:
                contact_position = contact_signal[0]
                if isinstance(contact_position, (list, np.ndarray)) and len(contact_position) == 2:
                    current_contact_positions.append(np.array(contact_position))

        # Update contact status for each object
        for obj_id, obj_data in self.objects_dict.items():
            has_current_contact = False

            # Check if any current contact position is near this object
            for contact_pos in current_contact_positions:
                if self._is_point_near_contour(contact_pos, obj_data.get('contour_equation', {})):
                    has_current_contact = True
                    break

            obj_data['contact_status'] = has_current_contact

    def _cleanup_point_cloud(self):
        """Remove points from history point cloud that are near objects."""
        new_point_cloud = []

        for point in self.history_point_cloud:
            point_near_object = False

            for obj_data in self.objects_dict.values():
                contour_eq = obj_data.get('contour_equation', {})
                if self._is_point_near_contour(point, contour_eq):
                    point_near_object = True
                    break

            if not point_near_object:
                new_point_cloud.append(point)

        self.history_point_cloud = new_point_cloud

    # [Previous methods for contour fitting and shape analysis remain the same]
    # fit_contour_features, _find_connected_component, _fit_shape_to_points,
    # _fit_circle, _fit_square, _fit_triangle, _calculate_circle_error,
    # _calculate_square_error, _calculate_triangle_error, update_contour_features


# Example usage
if __name__ == "__main__":
    # Initialize the recognition system
    recognizer = ObjectRecognitionAndStorage()

    # Example sensory data with multiple contact signals
    sensing_data = {
        "approaching_data": [[0.1, 0.2], [0.3, 0.4], [0.8, 0.9]],
        "contact_data": [
            [
                [0.5, 0.6],  # contact_position 1
                [[0.55, 0.65], [0.01, 0.02], [1.0, 0.5]]  # object data 1
            ],
            [
                [0.7, 0.8],  # contact_position 2
                [[0.75, 0.85], [0.02, -0.01], [0.8, 0.6]]  # object data 2
            ]
        ]
    }

    # Example binary grid map
    binary_grid_map = np.zeros((100, 100))
    binary_grid_map[40:60, 40:60] = 1  # Square obstacle
    binary_grid_map[20:30, 70:80] = 1  # Another obstacle

    # Empty initial objects and point cloud
    objects_dict = {}
    history_point_cloud = []

    # Process sensory data
    updated_objects, updated_point_cloud = recognizer.process_sensory_data(
        sensing_data, binary_grid_map, objects_dict, history_point_cloud
    )

    print("Objects created:", list(updated_objects.keys()))
    for obj_id, obj_data in updated_objects.items():
        print(f"  {obj_id}: {len(obj_data['interaction_pairs'])} interaction pairs, "
              f"contact status: {obj_data['contact_status']}")

    print("Point cloud size:", len(updated_point_cloud))