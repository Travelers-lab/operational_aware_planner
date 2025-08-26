"""
Binary Grid Map Generator
This module generates binary grid maps with random obstacles (triangles, circles, rectangles)
and saves them as PNG images. It also creates dataset splits for training, validation and testing.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
import random
import os
from typing import List, Tuple, Dict
import math
import json
from os.path import join, dirname, abspath

class BinaryMapGenerator:
    def __init__(self, grid_size: int = 100, physical_size: float = 0.5):
        """
        Initialize the map generator

        Parameters:
        grid_size (int): Number of grid cells per dimension (default: 100)
        physical_size (float): Physical size of the grid in meters (default: 0.5m)
        """
        self.grid_size = grid_size
        self.physical_size = physical_size
        self.cell_size = physical_size / grid_size  # Size of each grid cell in meters

        # Conversion factors from cm to grid cells
        self.cm_to_cells = 0.01 / self.cell_size

        # Obstacle parameter ranges (converted to grid cells)
        self.triangle_size_range = [5 * self.cm_to_cells, 10 * self.cm_to_cells]
        self.circle_radius_range = [2.5 * self.cm_to_cells, 5 * self.cm_to_cells]
        self.rectangle_size_range = [5 * self.cm_to_cells, 7 * self.cm_to_cells]

        # Minimum distance from obstacles to boundary (in grid cells)
        self.boundary_margin = max(
            self.triangle_size_range[1] * math.sqrt(3) / 2,  # Triangle height
            self.circle_radius_range[1],
            self.rectangle_size_range[1]
        )

    def generate_obstacle(self, obstacle_type: str) -> Dict:
        """
        Generate parameters for a single obstacle

        Parameters:
        obstacle_type (str): Type of obstacle ('triangle', 'circle', 'rectangle')

        Returns:
        Dict: Obstacle parameters
        """
        if obstacle_type == 'triangle':
            size = random.uniform(*self.triangle_size_range)
            height = size * math.sqrt(3) / 2  # Height of equilateral triangle
            return {
                'type': 'triangle',
                'size': size,
                'height': height
            }
        elif obstacle_type == 'circle':
            return {
                'type': 'circle',
                'radius': random.uniform(*self.circle_radius_range)
            }
        elif obstacle_type == 'rectangle':
            width = random.uniform(*self.rectangle_size_range)
            height = random.uniform(*self.rectangle_size_range)
            return {
                'type': 'rectangle',
                'width': width,
                'height': height
            }

    def is_valid_position(self, obstacle: Dict, position: Tuple[float, float],
                          existing_obstacles: List[Tuple[Dict, Tuple[float, float]]]) -> bool:
        """
        Check if obstacle can be placed at given position without overlapping

        Parameters:
        obstacle (Dict): Obstacle parameters
        position (Tuple): Proposed (x, y) position
        existing_obstacles (List): List of already placed obstacles

        Returns:
        bool: True if position is valid, False otherwise
        """
        x, y = position

        # Check boundary constraints
        if obstacle['type'] == 'triangle':
            if (x - obstacle['size'] / 2 < self.boundary_margin or
                    x + obstacle['size'] / 2 > self.grid_size - self.boundary_margin or
                    y - obstacle['height'] / 2 < self.boundary_margin or
                    y + obstacle['height'] / 2 > self.grid_size - self.boundary_margin):
                return False

        elif obstacle['type'] == 'circle':
            if (x - obstacle['radius'] < self.boundary_margin or
                    x + obstacle['radius'] > self.grid_size - self.boundary_margin or
                    y - obstacle['radius'] < self.boundary_margin or
                    y + obstacle['radius'] > self.grid_size - self.boundary_margin):
                return False

        elif obstacle['type'] == 'rectangle':
            if (x - obstacle['width'] / 2 < self.boundary_margin or
                    x + obstacle['width'] / 2 > self.grid_size - self.boundary_margin or
                    y - obstacle['height'] / 2 < self.boundary_margin or
                    y + obstacle['height'] / 2 > self.grid_size - self.boundary_margin):
                return False

        # Check overlap with existing obstacles
        for existing_obstacle, existing_pos in existing_obstacles:
            if self.check_overlap(obstacle, position, existing_obstacle, existing_pos):
                return False

        return True

    def check_overlap(self, obs1: Dict, pos1: Tuple[float, float],
                      obs2: Dict, pos2: Tuple[float, float]) -> bool:
        """
        Check if two obstacles overlap

        Parameters:
        obs1, obs2: Obstacle parameters
        pos1, pos2: Obstacle positions

        Returns:
        bool: True if obstacles overlap, False otherwise
        """
        # Simplified overlap check using bounding circles
        x1, y1 = pos1
        x2, y2 = pos2

        # Calculate approximate radii for different shapes
        if obs1['type'] == 'triangle':
            r1 = obs1['size'] * 0.6  # Approximate bounding circle radius
        elif obs1['type'] == 'circle':
            r1 = obs1['radius']
        else:  # rectangle
            r1 = math.sqrt(obs1['width'] ** 2 + obs1['height'] ** 2) / 2

        if obs2['type'] == 'triangle':
            r2 = obs2['size'] * 0.6
        elif obs2['type'] == 'circle':
            r2 = obs2['radius']
        else:  # rectangle
            r2 = math.sqrt(obs2['width'] ** 2 + obs2['height'] ** 2) / 2

        # Check if bounding circles overlap
        distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        return distance < (r1 + r2)

    def place_obstacle(self, obstacle: Dict, existing_obstacles: List) -> Tuple[float, float]:
        """
        Find a valid position for an obstacle

        Parameters:
        obstacle (Dict): Obstacle parameters
        existing_obstacles (List): List of already placed obstacles

        Returns:
        Tuple: (x, y) position for the obstacle
        """
        max_attempts = 100
        for _ in range(max_attempts):
            # Generate random position within valid bounds
            if obstacle['type'] == 'triangle':
                x = random.uniform(self.boundary_margin + obstacle['size'] / 2,
                                   self.grid_size - self.boundary_margin - obstacle['size'] / 2)
                y = random.uniform(self.boundary_margin + obstacle['height'] / 2,
                                   self.grid_size - self.boundary_margin - obstacle['height'] / 2)
            elif obstacle['type'] == 'circle':
                x = random.uniform(self.boundary_margin + obstacle['radius'],
                                   self.grid_size - self.boundary_margin - obstacle['radius'])
                y = random.uniform(self.boundary_margin + obstacle['radius'],
                                   self.grid_size - self.boundary_margin - obstacle['radius'])
            else:  # rectangle
                x = random.uniform(self.boundary_margin + obstacle['width'] / 2,
                                   self.grid_size - self.boundary_margin - obstacle['width'] / 2)
                y = random.uniform(self.boundary_margin + obstacle['height'] / 2,
                                   self.grid_size - self.boundary_margin - obstacle['height'] / 2)

            if self.is_valid_position(obstacle, (x, y), existing_obstacles):
                return (x, y)

        raise ValueError("Cannot find valid position for obstacle")

    def draw_obstacle(self, grid: np.ndarray, obstacle: Dict, position: Tuple[float, float]):
        """
        Draw an obstacle on the grid

        Parameters:
        grid (np.ndarray): Binary grid
        obstacle (Dict): Obstacle parameters
        position (Tuple): (x, y) position
        """
        x, y = position

        if obstacle['type'] == 'triangle':
            # Calculate triangle vertices
            size = obstacle['size']
            height = obstacle['height']
            vertices = [
                (x, y + height / 2),  # Top vertex
                (x - size / 2, y - height / 2),  # Bottom left
                (x + size / 2, y - height / 2)  # Bottom right
            ]
            self.draw_polygon(grid, vertices)

        elif obstacle['type'] == 'circle':
            self.draw_circle(grid, (x, y), obstacle['radius'])

        elif obstacle['type'] == 'rectangle':
            width = obstacle['width']
            height = obstacle['height']
            self.draw_rectangle(grid, (x - width / 2, y - height / 2), width, height)

    def draw_circle(self, grid: np.ndarray, center: Tuple[float, float], radius: float):
        """
        Draw a circle on the grid using midpoint circle algorithm

        Parameters:
        grid (np.ndarray): Binary grid
        center (Tuple): (x, y) center coordinates
        radius (float): Circle radius
        """
        cx, cy = center
        r = int(radius)

        # Iterate through all grid cells
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Calculate distance from center
                distance = math.sqrt((i - cx) ** 2 + (j - cy) ** 2)
                if distance <= radius:
                    grid[int(i), int(j)] = 1

    def draw_rectangle(self, grid: np.ndarray, origin: Tuple[float, float], width: float, height: float):
        """
        Draw a rectangle on the grid

        Parameters:
        grid (np.ndarray): Binary grid
        origin (Tuple): (x, y) of bottom-left corner
        width (float): Rectangle width
        height (float): Rectangle height
        """
        x0, y0 = origin

        # Iterate through all grid cells
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (x0 <= i <= x0 + width) and (y0 <= j <= y0 + height):
                    grid[int(i), int(j)] = 1

    def draw_polygon(self, grid: np.ndarray, vertices: List[Tuple[float, float]]):
        """
        Draw a polygon on the grid using ray casting algorithm

        Parameters:
        grid (np.ndarray): Binary grid
        vertices (List): List of (x, y) vertices
        """
        # Find bounding box
        min_x = min(v[0] for v in vertices)
        max_x = max(v[0] for v in vertices)
        min_y = min(v[1] for v in vertices)
        max_y = max(v[1] for v in vertices)

        # Iterate through bounding box
        for i in range(int(min_x), int(max_x) + 1):
            for j in range(int(min_y), int(max_y) + 1):
                if self.is_point_in_polygon((i, j), vertices):
                    grid[i, j] = 1

    def is_point_in_polygon(self, point: Tuple[float, float], vertices: List[Tuple[float, float]]) -> bool:
        """
        Check if a point is inside a polygon using ray casting algorithm

        Parameters:
        point (Tuple): (x, y) point to check
        vertices (List): List of polygon vertices

        Returns:
        bool: True if point is inside polygon
        """
        x, y = point
        n = len(vertices)
        inside = False

        p1x, p1y = vertices[0]
        for i in range(n + 1):
            p2x, p2y = vertices[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def generate_map(self, num_obstacles: int) -> np.ndarray:
        """
        Generate a single binary map with specified number of obstacles

        Parameters:
        num_obstacles (int): Number of obstacles to place

        Returns:
        np.ndarray: Binary grid map
        """
        # Initialize empty grid (0 = free, 1 = obstacle)
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        obstacles = []
        positions = []

        for _ in range(num_obstacles):
            # Randomly select obstacle type
            obstacle_type = random.choice(['triangle', 'circle', 'rectangle'])
            obstacle = self.generate_obstacle(obstacle_type)

            # Find valid position
            position = self.place_obstacle(obstacle, list(zip(obstacles, positions)))

            # Draw obstacle
            self.draw_obstacle(grid, obstacle, position)

            obstacles.append(obstacle)
            positions.append(position)

        return grid

    def save_map(self, grid: np.ndarray, filename: str):
        """
        Save grid as PNG image

        Parameters:
        grid (np.ndarray): Binary grid
        filename (str): Output filename
        """
        plt.figure(figsize=(5, 5))
        plt.imshow(grid, cmap='binary', origin='lower')
        plt.axis('off')
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()

    def generate_dataset(self, output_dir: str, num_maps: int = 15000):
        """
        Generate the complete dataset with specified splits

        Parameters:
        output_dir (str): Directory to save dataset
        num_maps (int): Total number of maps to generate
        """
        # Create output directories
        splits = {
            'train': 0.7,
            'val': 0.15,
            'test': 0.15
        }

        # Count maps for each obstacle count
        obstacle_counts = {
            1: 5000,
            2: 5000,
            3: 5000
        }

        # Create directories
        for split in splits:
            for count in obstacle_counts:
                os.makedirs(os.path.join(output_dir, split, str(count)), exist_ok=True)

        # Generate maps
        metadata = {}
        map_id = 0

        for count, num_maps_count in obstacle_counts.items():
            for i in range(num_maps_count):
                # Determine split
                if i < num_maps_count * splits['train']:
                    split = 'train'
                elif i < num_maps_count * (splits['train'] + splits['val']):
                    split = 'val'
                else:
                    split = 'test'

                # Generate map
                grid = self.generate_map(count)

                # Save map
                filename = os.path.join(output_dir, split, str(count), f'map_{map_id:06d}.png')
                self.save_map(grid, filename)

                # Store metadata
                metadata[map_id] = {
                    'split': split,
                    'obstacle_count': count,
                    'filename': filename
                }

                map_id += 1
                if map_id % 100 == 0:
                    print(f"Generated {map_id}/{num_maps} maps")

        # Save metadata
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Dataset generation complete. Saved to {output_dir}")


# Pseudocode:
"""
CLASS BinaryMapGenerator:
    INIT(grid_size, physical_size):
        SET grid parameters
        CONVERT obstacle size ranges from cm to grid cells
        CALCULATE boundary margin

    FUNCTION generate_obstacle(obstacle_type):
        IF triangle: GENERATE random size, calculate height
        IF circle: GENERATE random radius
        IF rectangle: GENERATE random width and height
        RETURN obstacle parameters

    FUNCTION is_valid_position(obstacle, position, existing_obstacles):
        CHECK if obstacle fits within boundaries
        CHECK if obstacle overlaps with existing obstacles
        RETURN validity

    FUNCTION place_obstacle(obstacle, existing_obstacles):
        FOR multiple attempts:
            GENERATE random position
            IF position is valid: RETURN position
        RAISE error if no valid position found

    FUNCTION draw_obstacle(grid, obstacle, position):
        IF triangle: CALCULATE vertices and draw polygon
        IF circle: DRAW circle using midpoint algorithm
        IF rectangle: DRAW rectangle

    FUNCTION generate_map(num_obstacles):
        INITIALIZE empty grid
        FOR each obstacle:
            SELECT random type
            GENERATE obstacle parameters
            FIND valid position
            DRAW obstacle on grid
        RETURN grid

    FUNCTION generate_dataset(output_dir, num_maps):
        CREATE directory structure
        FOR each obstacle count (1, 2, 3):
            FOR specified number of maps:
                DETERMINE dataset split
                GENERATE map
                SAVE as PNG
                STORE metadata
        SAVE metadata file
"""

# Example usage:
if __name__ == "__main__":
    generator = BinaryMapGenerator(grid_size=100, physical_size=0.5)
    generator.generate_dataset(dirname(abspath(__file__)))