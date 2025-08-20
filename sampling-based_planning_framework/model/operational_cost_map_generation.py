import numpy as np
from typing import Dict, List, Any, Tuple


def generate_operation_cost_map(binary_map: np.ndarray,
                                objects_dict: Dict[str, Dict[str, Any]]) -> np.ndarray:
    """
    Generate an operation cost map based on object properties and binary occupancy map.

    This function creates a cost map where each grid cell has an operation cost value
    based on the material properties of objects occupying that cell. The cost is calculated
    by comparing object material properties with operation limits.

    Args:
        binary_map (np.ndarray): 100x100 binary occupancy map where 1 represents obstacles
                                and 0 represents free space.
        objects_dict (Dict): Dictionary containing multiple object sub-dictionaries. Each
                            object dictionary should contain 'material_property' and
                            'contour_equation' keys.

    Returns:
        np.ndarray: 100x100 operation cost map with values between 0 and 1, where 1
                    represents impossible to operate and 0 represents no cost.
    """
    # Initialize operation cost map with zeros
    operation_cost_map = np.zeros_like(binary_map, dtype=float)

    # Operation limits for material properties
    operation_limits = np.array([400, 400, 20])

    # Create a grid of coordinates for the map
    x_coords, y_coords = np.meshgrid(np.arange(100), np.arange(100))
    grid_points = np.column_stack((x_coords.ravel(), y_coords.ravel()))

    # Process each object in the dictionary
    for obj_name, obj_data in objects_dict.items():
        # Check if object has contour equation
        if 'contour_equation' not in obj_data or obj_data['contour_equation'] is None:
            continue

        contour_eq = obj_data['contour_equation']
        params = contour_eq.get('params', None)

        # Check if object has material properties
        if 'material_property' not in obj_data:
            continue

        material_property = np.array(obj_data['material_property'])

        # Calculate operation cost for this object
        ratios = material_property / operation_limits
        if np.any(ratios > 1):
            obj_cost = 1.0  # Impossible to operate
        else:
            obj_cost = np.mean(ratios)  # Average cost

        # Determine which grid cells are inside this object's contour
        if params is not None:
            mask = np.zeros(10000, dtype=bool)  # 100x100 = 10000 cells

            if contour_eq.get('type') == 'circle':
                # Circle: params = [center_x, center_y, radius]
                center = np.array([params[0], params[1]])
                radius = params[2]
                distances = np.linalg.norm(grid_points - center, axis=1)
                mask = distances <= radius

            elif contour_eq.get('type') == 'square':
                # Square: params = [center_x, center_y, side_length]
                center = np.array([params[0], params[1]])
                half_side = params[2] / 2
                min_corner = center - half_side
                max_corner = center + half_side

                # Check if points are inside the square
                in_x = np.logical_and(grid_points[:, 0] >= min_corner[0],
                                      grid_points[:, 0] <= max_corner[0])
                in_y = np.logical_and(grid_points[:, 1] >= min_corner[1],
                                      grid_points[:, 1] <= max_corner[1])
                mask = np.logical_and(in_x, in_y)

            elif contour_eq.get('type') == 'triangle':
                # Triangle: params = [x1, y1, x2, y2, x3, y3]
                vertices = np.array(params).reshape(3, 2)

                # Use barycentric coordinate method to check if point is inside triangle
                def point_in_triangle(point, v1, v2, v3):
                    # Compute vectors
                    v0 = v3 - v1
                    v1_vec = v2 - v1
                    v2_vec = point - v1

                    # Compute dot products
                    dot00 = np.dot(v0, v0)
                    dot01 = np.dot(v0, v1_vec)
                    dot02 = np.dot(v0, v2_vec)
                    dot11 = np.dot(v1_vec, v1_vec)
                    dot12 = np.dot(v1_vec, v2_vec)

                    # Compute barycentric coordinates
                    inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
                    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
                    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

                    # Check if point is in triangle
                    return (u >= 0) and (v >= 0) and (u + v <= 1)

                # Vectorize the function for all points
                point_in_triangle_vec = np.vectorize(
                    lambda x, y: point_in_triangle(np.array([x, y]),
                                                   vertices[0], vertices[1], vertices[2]),
                    otypes=[bool]
                )

                mask = point_in_triangle_vec(grid_points[:, 0], grid_points[:, 1])

            # Update operation cost for cells inside this object
            if np.any(mask):
                # Reshape mask to 100x100 and update cost map
                mask_2d = mask.reshape(100, 100)

                # Only update cells that are obstacles in the binary map
                obstacle_cells = np.logical_and(mask_2d, binary_map == 1)
                operation_cost_map[obstacle_cells] = obj_cost

    return operation_cost_map


# Example usage
if __name__ == "__main__":
    # Create a mock 100x100 binary map
    binary_map = np.zeros((100, 100))

    # Add some obstacles
    binary_map[20:40, 30:50] = 1  # A square obstacle
    binary_map[60:80, 70:90] = 1  # Another square obstacle

    # Create a mock objects dictionary
    objects_dict = {
        "object0": {
            "material_property": [300, 350, 15],  # Will result in cost < 1
            "contour_equation": {
                "type": "square",
                "params": [35, 30, 20]  # Center at (35, 30) with side length 20
            },
            "interaction_pairs": [],
            "contact_status": "active"
        },
        "object1": {
            "material_property": [450, 400, 25],  # One value > limit, cost = 1
            "contour_equation": {
                "type": "square",
                "params": [75, 65, 20]  # Center at (75, 65) with side length 20
            },
            "interaction_pairs": [],
            "contact_status": "active"
        }
    }

    # Generate operation cost map
    cost_map = generate_operation_cost_map(binary_map, objects_dict)

    # Print some information
    print("Binary map shape:", binary_map.shape)
    print("Objects in dictionary:", len(objects_dict))
    print("Cost map shape:", cost_map.shape)
    print("Unique cost values:", np.unique(cost_map))

    # Display cost for the first object's area
    obj0_mask = np.zeros_like(binary_map, dtype=bool)
    obj0_mask[20:40, 30:50] = True
    obj0_cost = cost_map[obj0_mask]
    print(f"Object 0 cost: {np.unique(obj0_cost)}")

    # Display cost for the second object's area
    obj1_mask = np.zeros_like(binary_map, dtype=bool)
    obj1_mask[60:80, 70:90] = True
    obj1_cost = cost_map[obj1_mask]
    print(f"Object 1 cost: {np.unique(obj1_cost)}")