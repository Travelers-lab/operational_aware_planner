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