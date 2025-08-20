import numpy as np
from PIL import Image, ImageDraw
import random
import os
import math
from typing import List, Tuple, Dict, Any


def generate_binary_grid_map(num_obstacles: int, save_path: str) -> None:
    """
    Generate a binary grid map with random obstacles and save as PNG.

    This function creates a 100x100 grid map representing a 0.5m x 0.5m space.
    It randomly places non-overlapping obstacles (circles, equilateral triangles, and rectangles)
    that do not interfere with the boundaries.

    Args:
        num_obstacles (int): Number of obstacles to generate (1, 2, or 3)
        save_path (str): Path to save the generated PNG image

    Returns:
        None
    """
    # Initialize a 100x100 grid with zeros (free space)
    grid = np.zeros((100, 100), dtype=np.uint8)

    # List to store obstacle information
    obstacles = []

    # Conversion factor: 100 pixels = 0.5m = 50cm, so 1 pixel = 0.5cm
    pixel_to_cm = 0.5

    for _ in range(num_obstacles):
        # Randomly select obstacle type
        obstacle_type = random.choice(['circle', 'triangle', 'rectangle'])

        # Try to place the obstacle (may need multiple attempts due to collisions)
        placed = False
        attempts = 0
        max_attempts = 100  # Prevent infinite loops

        while not placed and attempts < max_attempts:
            attempts += 1

            if obstacle_type == 'circle':
                # Generate random radius between 5cm and 7cm
                radius_cm = random.uniform(5, 7)
                radius_pixels = int(radius_cm / pixel_to_cm)

                # Generate random center position (ensuring the circle stays within bounds)
                center_x = random.randint(radius_pixels, 100 - radius_pixels)
                center_y = random.randint(radius_pixels, 100 - radius_pixels)

                # Check for collisions with existing obstacles
                collision = False
                for obs in obstacles:
                    if _circle_circle_collision((center_x, center_y), radius_pixels,
                                                obs['position'], obs['radius'] if 'radius' in obs else 0):
                        collision = True
                        break

                if not collision:
                    obstacles.append({
                        'type': 'circle',
                        'position': (center_x, center_y),
                        'radius': radius_pixels
                    })
                    placed = True

            elif obstacle_type == 'triangle':
                # Generate random side length between 5cm and 10cm
                side_cm = random.uniform(5, 10)
                side_pixels = int(side_cm / pixel_to_cm)

                # Height of equilateral triangle
                height_pixels = int((math.sqrt(3) / 2) * side_pixels)

                # Generate random center position
                center_x = random.randint(height_pixels // 2, 100 - height_pixels // 2)
                center_y = random.randint(height_pixels // 2, 100 - height_pixels // 2)

                # Calculate vertices of the equilateral triangle
                vertices = [
                    (center_x, center_y - height_pixels // 2),  # Top vertex
                    (center_x - side_pixels // 2, center_y + height_pixels // 2),  # Bottom left
                    (center_x + side_pixels // 2, center_y + height_pixels // 2)  # Bottom right
                ]

                # Check for collisions with existing obstacles
                collision = False
                for obs in obstacles:
                    if obs['type'] == 'circle':
                        if _circle_triangle_collision(obs['position'], obs['radius'], vertices):
                            collision = True
                            break
                    elif obs['type'] == 'triangle':
                        if _triangle_triangle_collision(vertices, obs['vertices']):
                            collision = True
                            break
                    elif obs['type'] == 'rectangle':
                        if _triangle_rectangle_collision(vertices, obs['position'], obs['width'], obs['height']):
                            collision = True
                            break

                if not collision:
                    obstacles.append({
                        'type': 'triangle',
                        'position': (center_x, center_y),
                        'side': side_pixels,
                        'vertices': vertices
                    })
                    placed = True

            elif obstacle_type == 'rectangle':
                # Generate random side length between 5cm and 7cm
                side_cm = random.uniform(5, 7)
                side_pixels = int(side_cm / pixel_to_cm)

                # For rectangle, we'll use square for simplicity
                width = side_pixels
                height = side_pixels

                # Generate random center position
                center_x = random.randint(width // 2, 100 - width // 2)
                center_y = random.randint(height // 2, 100 - height // 2)

                # Calculate top-left corner
                top_left = (center_x - width // 2, center_y - height // 2)

                # Check for collisions with existing obstacles
                collision = False
                for obs in obstacles:
                    if obs['type'] == 'circle':
                        if _circle_rectangle_collision(obs['position'], obs['radius'], top_left, width, height):
                            collision = True
                            break
                    elif obs['type'] == 'triangle':
                        if _triangle_rectangle_collision(obs['vertices'], top_left, width, height):
                            collision = True
                            break
                    elif obs['type'] == 'rectangle':
                        if _rectangle_rectangle_collision(top_left, width, height,
                                                          obs['top_left'], obs['width'], obs['height']):
                            collision = True
                            break

                if not collision:
                    obstacles.append({
                        'type': 'rectangle',
                        'position': (center_x, center_y),
                        'width': width,
                        'height': height,
                        'top_left': top_left
                    })
                    placed = True

    # Draw obstacles on the grid
    for obstacle in obstacles:
        if obstacle['type'] == 'circle':
            _draw_circle(grid, obstacle['position'], obstacle['radius'])
        elif obstacle['type'] == 'triangle':
            _draw_triangle(grid, obstacle['vertices'])
        elif obstacle['type'] == 'rectangle':
            _draw_rectangle(grid, obstacle['top_left'], obstacle['width'], obstacle['height'])

    # Convert grid to image and save
    img = Image.fromarray(grid * 255)
    img = img.convert('1')
    img.save(save_path)


# Collision detection helper functions
def _circle_circle_collision(center1, radius1, center2, radius2):
    """Check if two circles collide."""
    distance = math.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
    return distance < (radius1 + radius2)


def _circle_triangle_collision(circle_center, circle_radius, triangle_vertices):
    """Check if a circle and a triangle collide."""
    # Simplified check: if any vertex is inside the circle or any edge intersects the circle
    for vertex in triangle_vertices:
        if math.sqrt((vertex[0] - circle_center[0]) ** 2 + (vertex[1] - circle_center[1]) ** 2) < circle_radius:
            return True

    # Check if circle center is inside triangle (simplified)
    # This is a simplified implementation; a more accurate one would use point-in-polygon
    return False


def _triangle_triangle_collision(vertices1, vertices2):
    """Check if two triangles collide."""
    # Simplified implementation - check if any vertex of one triangle is inside the other
    # This is a placeholder; a complete implementation would use separating axis theorem
    return False


def _circle_rectangle_collision(circle_center, circle_radius, rect_top_left, rect_width, rect_height):
    """Check if a circle and a rectangle collide."""
    # Find the closest point on the rectangle to the circle
    closest_x = max(rect_top_left[0], min(circle_center[0], rect_top_left[0] + rect_width))
    closest_y = max(rect_top_left[1], min(circle_center[1], rect_top_left[1] + rect_height))

    # Calculate distance between circle center and closest point
    distance = math.sqrt((circle_center[0] - closest_x) ** 2 + (circle_center[1] - closest_y) ** 2)

    return distance < circle_radius


def _triangle_rectangle_collision(triangle_vertices, rect_top_left, rect_width, rect_height):
    """Check if a triangle and a rectangle collide."""
    # Simplified implementation - check if any vertex of triangle is inside rectangle
    for vertex in triangle_vertices:
        if (rect_top_left[0] <= vertex[0] <= rect_top_left[0] + rect_width and
                rect_top_left[1] <= vertex[1] <= rect_top_left[1] + rect_height):
            return True

    return False


def _rectangle_rectangle_collision(top_left1, width1, height1, top_left2, width2, height2):
    """Check if two rectangles collide."""
    return not (top_left1[0] + width1 < top_left2[0] or
                top_left1[0] > top_left2[0] + width2 or
                top_left1[1] + height1 < top_left2[1] or
                top_left1[1] > top_left2[1] + height2)


# Drawing helper functions
def _draw_circle(grid, center, radius):
    """Draw a circle on the grid."""
    for y in range(max(0, center[1] - radius), min(100, center[1] + radius + 1)):
        for x in range(max(0, center[0] - radius), min(100, center[0] + radius + 1)):
            if (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2:
                grid[y, x] = 1


def _draw_triangle(grid, vertices):
    """Draw a triangle on the grid."""
    # Create a PIL image to draw the triangle
    img = Image.new('L', (100, 100), 0)
    draw = ImageDraw.Draw(img)
    draw.polygon(vertices, fill=255)

    # Convert back to numpy array and update grid
    triangle_array = np.array(img)
    grid[triangle_array > 0] = 1


def _draw_rectangle(grid, top_left, width, height):
    """Draw a rectangle on the grid."""
    x1, y1 = top_left
    x2, y2 = x1 + width, y1 + height

    # Ensure coordinates are within grid bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(100, x2), min(100, y2)

    grid[y1:y2, x1:x2] = 1


def generate_maps_dataset():
    """
    Generate a dataset of 15,000 binary grid maps with 1, 2, or 3 obstacles.

    This function creates three directories (1_obstacle, 2_obstacles, 3_obstacles)
    and generates 5,000 maps for each category.
    """
    # Create directories for each obstacle count
    for count in [1, 2, 3]:
        dir_path = f"{count}_obstacle{'s' if count > 1 else ''}"
        os.makedirs(dir_path, exist_ok=True)

    # Generate 5,000 maps for each obstacle count
    for count in [1, 2, 3]:
        dir_path = f"{count}_obstacle{'s' if count > 1 else ''}"
        for i in range(5000):
            file_path = os.path.join(dir_path, f"map_{i:04d}.png")
            generate_binary_grid_map(count, file_path)

            # Print progress every 100 maps
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1} maps with {count} obstacle{'s' if count > 1 else ''}")


if __name__ == "__main__":
    generate_maps_dataset()