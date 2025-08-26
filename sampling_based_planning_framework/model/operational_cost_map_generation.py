import numpy as np
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt


class OperationCostMapGenerator:
    """
    A class for generating operation cost maps based on object properties and binary occupancy maps.

    This class processes binary occupancy maps and object dictionaries to create
    operation cost maps where each grid cell has a cost value based on the material
    properties of objects occupying that cell.
    """

    def __init__(self, operation_limits: List[float] = [400, 400, 20]):
        """
        Initialize the OperationCostMapGenerator.

        Args:
            operation_limits: List of operation limits for material properties [limit1, limit2, limit3]
        """
        self.operation_limits = np.array(operation_limits, dtype=np.float32)
        self.map_size = (100, 100)

    def generate_cost_map(self,
                          binary_map: np.ndarray,
                          objects_dict: Dict[str, Dict[str, Any]]) -> np.ndarray:
        """
        Generate an operation cost map from binary map and object properties.

        Args:
            binary_map: 100x100 binary occupancy map (0=free, 1=occupied)
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

        Returns:
            Operation cost map with same dimensions as input (100x100)
        """
        # Validate input
        self._validate_input(binary_map)

        # Initialize cost map with zeros (free space has zero cost)
        cost_map = np.zeros_like(binary_map, dtype=np.float32)

        # Process each object
        for obj_id, obj_data in objects_dict.items():
            try:
                # Extract object properties
                material_property = obj_data.get('material_property', [0, 0, 0])
                contour_equation = obj_data.get('contour_equation', {})

                # Calculate operation cost for this object
                object_cost = self._calculate_object_cost(material_property)

                # Get object mask from contour equation
                object_mask = self._get_object_mask_from_contour(contour_equation)

                if object_mask is not None:
                    # Apply cost to object region in the cost map
                    # Only apply cost where binary map indicates occupancy
                    object_region = (object_mask > 0) & (binary_map > 0.5)
                    cost_map[object_region] = object_cost

                    print(f"Applied cost {object_cost:.3f} to object {obj_id}")

            except Exception as e:
                print(f"Error processing object {obj_id}: {str(e)}")
                continue

        return cost_map

    def _validate_input(self, binary_map: np.ndarray):
        """
        Validate input binary map.

        Args:
            binary_map: Input binary occupancy map

        Raises:
            ValueError: If input map is invalid
        """
        if binary_map.shape != self.map_size:
            raise ValueError(f"Binary map must be {self.map_size}, got {binary_map.shape}")

        if binary_map.min() < 0 or binary_map.max() > 1:
            raise ValueError("Binary map values must be in range [0, 1]")

    def _calculate_object_cost(self, material_property: List[float]) -> float:
        """
        Calculate operation cost based on material properties.

        Args:
            material_property: List of 3 material property values

        Returns:
            Operation cost value between 0 and 1
        """
        if not isinstance(material_property, (list, np.ndarray)) or len(material_property) != 3:
            raise ValueError("material_property must be a list of 3 values")

        # Convert to numpy array
        material_array = np.array(material_property, dtype=np.float32)

        # Check if any property exceeds operation limits
        normalized_properties = material_array / self.operation_limits

        if np.any(normalized_properties > 1.0):
            return 1.0  # Cannot operate

        # Calculate average normalized cost
        cost = np.mean(normalized_properties)

        # Ensure cost is in [0, 1] range
        return min(max(cost, 0.0), 1.0)

    def _get_object_mask_from_contour(self, contour_equation: Dict[str, Any]) -> np.ndarray:
        """
        Create a binary mask for an object based on its contour equation.

        Args:
            contour_equation: Dictionary containing contour parameters

        Returns:
            Binary mask indicating object region, or None if invalid
        """
        if not contour_equation or 'params' not in contour_equation:
            return None

        shape_type = contour_equation.get('shape', '').lower()
        params = contour_equation.get('params', {})

        try:
            # Create empty mask
            mask = np.zeros(self.map_size, dtype=np.uint8)

            if shape_type == 'square':
                mask = self._create_square_mask(params)
            elif shape_type == 'circle':
                mask = self._create_circle_mask(params)
            elif shape_type == 'triangle':
                mask = self._create_triangle_mask(params)
            elif shape_type == 'rectangle':
                mask = self._create_rectangle_mask(params)
            elif shape_type == 'polygon':
                mask = self._create_polygon_mask(params)
            else:
                print(f"Unsupported shape type: {shape_type}")
                return None

            return mask.astype(np.float32)

        except Exception as e:
            print(f"Error creating mask for shape {shape_type}: {str(e)}")
            return None

    def _create_square_mask(self, params: Dict[str, Any]) -> np.ndarray:
        """
        Create a square mask based on parameters.

        Args:
            params: Dictionary containing square parameters:
                - center: [x, y] coordinates of center
                - side_length: length of square side

        Returns:
            Binary mask with square region filled
        """
        mask = np.zeros(self.map_size, dtype=np.uint8)

        center = params.get('center', [0, 0])
        side_length = params.get('side_length', 0)

        if side_length <= 0:
            return mask

        # Calculate bounding box coordinates
        half_side = side_length / 2
        x_min = max(0, int(center[0] - half_side))
        x_max = min(self.map_size[1], int(center[0] + half_side))
        y_min = max(0, int(center[1] - half_side))
        y_max = min(self.map_size[0], int(center[1] + half_side))

        # Fill the square region
        mask[y_min:y_max, x_min:x_max] = 1

        return mask

    def _create_circle_mask(self, params: Dict[str, Any]) -> np.ndarray:
        """
        Create a circle mask based on parameters.

        Args:
            params: Dictionary containing circle parameters:
                - center: [x, y] coordinates of center
                - radius: radius of circle

        Returns:
            Binary mask with circular region filled
        """
        mask = np.zeros(self.map_size, dtype=np.uint8)

        center = params.get('center', [0, 0])
        radius = params.get('radius', 0)

        if radius <= 0:
            return mask

        # Create coordinate grids
        y_coords, x_coords = np.ogrid[:self.map_size[0], :self.map_size[1]]

        # Calculate distance from center
        distance = np.sqrt((x_coords - center[0]) ** 2 + (y_coords - center[1]) ** 2)

        # Fill circle region
        mask[distance <= radius] = 1

        return mask

    def _create_triangle_mask(self, params: Dict[str, Any]) -> np.ndarray:
        """
        Create a triangle mask based on parameters.

        Args:
            params: Dictionary containing triangle parameters:
                - vertices: list of three [x, y] coordinates

        Returns:
            Binary mask with triangular region filled
        """
        mask = np.zeros(self.map_size, dtype=np.uint8)

        vertices = params.get('vertices', [])

        if len(vertices) != 3:
            return mask

        # Convert vertices to numpy array
        triangle_vertices = np.array(vertices, dtype=np.float32)

        # Create coordinate grids
        x_coords, y_coords = np.meshgrid(np.arange(self.map_size[1]), np.arange(self.map_size[0]))
        points = np.vstack((x_coords.ravel(), y_coords.ravel())).T

        # Check if points are inside triangle
        def point_in_triangle(point, triangle):
            # Barycentric coordinate method
            v0 = triangle[2] - triangle[0]
            v1 = triangle[1] - triangle[0]
            v2 = point - triangle[0]

            dot00 = np.dot(v0, v0)
            dot01 = np.dot(v0, v1)
            dot02 = np.dot(v0, v2)
            dot11 = np.dot(v1, v1)
            dot12 = np.dot(v1, v2)

            inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
            u = (dot11 * dot02 - dot01 * dot12) * inv_denom
            v = (dot00 * dot12 - dot01 * dot02) * inv_denom

            return (u >= 0) and (v >= 0) and (u + v <= 1)

        # Vectorize the point checking
        inside_points = []
        for point in points:
            if point_in_triangle(point, triangle_vertices):
                inside_points.append(point)

        if inside_points:
            inside_points = np.array(inside_points, dtype=int)
            mask[inside_points[:, 1], inside_points[:, 0]] = 1

        return mask

    def _create_rectangle_mask(self, params: Dict[str, Any]) -> np.ndarray:
        """
        Create a rectangle mask based on parameters.

        Args:
            params: Dictionary containing rectangle parameters:
                - top_left: [x, y] coordinates of top-left corner
                - width: width of rectangle
                - height: height of rectangle
                OR
                - center: [x, y] coordinates of center
                - width: width of rectangle
                - height: height of rectangle

        Returns:
            Binary mask with rectangular region filled
        """
        mask = np.zeros(self.map_size, dtype=np.uint8)

        if 'top_left' in params:
            top_left = params.get('top_left', [0, 0])
            width = params.get('width', 0)
            height = params.get('height', 0)

            x_min = max(0, int(top_left[0]))
            y_min = max(0, int(top_left[1]))
            x_max = min(self.map_size[1], int(top_left[0] + width))
            y_max = min(self.map_size[0], int(top_left[1] + height))

        elif 'center' in params:
            center = params.get('center', [0, 0])
            width = params.get('width', 0)
            height = params.get('height', 0)

            half_width = width / 2
            half_height = height / 2

            x_min = max(0, int(center[0] - half_width))
            x_max = min(self.map_size[1], int(center[0] + half_width))
            y_min = max(0, int(center[1] - half_height))
            y_max = min(self.map_size[0], int(center[1] + half_height))

        else:
            return mask

        if x_min >= x_max or y_min >= y_max:
            return mask

        # Fill the rectangle region
        mask[y_min:y_max, x_min:x_max] = 1

        return mask

    def _create_polygon_mask(self, params: Dict[str, Any]) -> np.ndarray:
        """
        Create a polygon mask based on parameters.

        Args:
            params: Dictionary containing polygon parameters:
                - vertices: list of [x, y] coordinates for polygon vertices

        Returns:
            Binary mask with polygonal region filled
        """
        mask = np.zeros(self.map_size, dtype=np.uint8)

        vertices = params.get('vertices', [])

        if len(vertices) < 3:
            return mask

        # Convert vertices to numpy array
        polygon_vertices = np.array(vertices, dtype=np.float32)

        # Create coordinate grids
        x_coords, y_coords = np.meshgrid(np.arange(self.map_size[1]), np.arange(self.map_size[0]))
        points = np.vstack((x_coords.ravel(), y_coords.ravel())).T

        # Use matplotlib Path to check if points are inside polygon
        from matplotlib.path import Path
        polygon_path = Path(polygon_vertices)

        # Check which points are inside the polygon
        inside_points = polygon_path.contains_points(points)
        inside_points = points[inside_points].astype(int)

        # Fill polygon region
        if len(inside_points) > 0:
            mask[inside_points[:, 1], inside_points[:, 0]] = 1

        return mask

    def visualize_cost_map(self,
                           binary_map: np.ndarray,
                           cost_map: np.ndarray,
                           objects_dict: Dict[str, Dict[str, Any]] = None,
                           save_path: str = None):
        """
        Visualize the binary map and cost map.

        Args:
            binary_map: Input binary occupancy map
            cost_map: Generated operation cost map
            objects_dict: Optional objects dictionary for additional visualization
            save_path: Optional path to save the visualization
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot binary map
        im1 = axes[0].imshow(binary_map, cmap='binary', vmin=0, vmax=1)
        axes[0].set_title('Binary Occupancy Map')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        plt.colorbar(im1, ax=axes[0])

        # Plot cost map
        im2 = axes[1].imshow(cost_map, cmap='hot', vmin=0, vmax=1)
        axes[1].set_title('Operation Cost Map')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')
        plt.colorbar(im2, ax=axes[1])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")

        plt.show()


# Example usage
if __name__ == "__main__":
    # Create sample binary map
    binary_map = np.zeros((100, 100))
    binary_map[20:40, 30:50] = 1  # Square obstacle
    binary_map[60:80, 70:90] = 1  # Another obstacle

    # Create sample objects dictionary
    sample_objects = {
        'object_1': {
            'material_property': [200, 150, 8],  # Normalized: [0.5, 0.375, 0.4] -> avg ~0.425
            'contour_equation': {
                'shape': 'square',
                'params': {
                    'center': [35, 35],
                    'side_length': 20
                }
            },
            'interaction_pairs': [],
            'contact_status': 'active'
        },
        'object_2': {
            'material_property': [500, 300, 25],  # Exceeds limits -> cost=1.0
            'contour_equation': {
                'shape': 'circle',
                'params': {
                    'center': [75, 75],
                    'radius': 10
                }
            },
            'interaction_pairs': [],
            'contact_status': 'active'
        }
    }

    # Initialize cost map generator
    cost_generator = OperationCostMapGenerator()

    # Generate cost map
    cost_map = cost_generator.generate_cost_map(binary_map, sample_objects)

    print(f"Cost map shape: {cost_map.shape}")
    print(f"Cost range: [{cost_map.min():.3f}, {cost_map.max():.3f}]")
    print(f"Number of non-zero cost cells: {np.sum(cost_map > 0)}")

    # Visualize results
    cost_generator.visualize_cost_map(binary_map, cost_map, sample_objects, "cost_map_visualization.png")