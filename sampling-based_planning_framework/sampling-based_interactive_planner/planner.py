import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import math
import random
from collections import deque
import time
import matplotlib
matplotlib.use('TkAgg')

class RRTConnectPlanner:
    def __init__(self, cost_map, start, goal):
        """
        Initialize RRT-Connect path planner

        Args:
            cost_map (numpy.ndarray): 100x100 operation cost map with values in [0, 1]
            start (tuple): Starting point coordinates (x, y)
            goal (tuple): Goal point coordinates (x, y)
        """
        self.cost_map = cost_map
        self.map_size = cost_map.shape
        self.start = start
        self.goal = goal

        # Initialize two trees
        self.tree_start = {start: None}  # key: node, value: parent node
        self.tree_goal = {goal: None}

        # Path planning results
        self.path = []
        self.success = False

        # Visualization parameters
        self.fig, self.ax = None, None

    def is_valid_point(self, point, check_radius=10):
        """
        Check if a point is valid (not near obstacles and within boundaries)

        Args:
            point (tuple): Point to check (x, y)
            check_radius (int): Check radius

        Returns:
            bool: Whether the point is valid
        """
        x, y = point

        # Check if out of map boundaries
        if x < 0 or x >= self.map_size[0] or y < 0 or y >= self.map_size[1]:
            return False

        # Check if point is in obstacle (cost = 1)
        if self.cost_map[int(x), int(y)] == 1:
            return False

        # Check surrounding area for obstacles
        for dx in range(-check_radius, check_radius + 1):
            for dy in range(-check_radius, check_radius + 1):
                nx, ny = int(x + dx), int(y + dy)
                # Check if within map boundaries
                if 0 <= nx < self.map_size[0] and 0 <= ny < self.map_size[1]:
                    # Check if obstacle
                    if self.cost_map[nx, ny] == 1:
                        return False

        return True

    def is_valid_path(self, point1, point2, step_size=1, check_radius=10):
        """
        Check if the path between two points is valid

        Args:
            point1 (tuple): First point (x, y)
            point2 (tuple): Second point (x, y)
            step_size (int): Step size for interpolation
            check_radius (int): Check radius

        Returns:
            bool: Whether the path is valid
        """
        x1, y1 = point1
        x2, y2 = point2

        # Calculate distance and direction
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if distance == 0:
            return True

        # Normalize direction vector
        dx = (x2 - x1) / distance
        dy = (y2 - y1) / distance

        # Check intermediate points
        num_steps = int(distance / step_size) + 1
        for i in range(num_steps + 1):
            # Calculate intermediate point
            t = min(i * step_size / distance, 1.0)
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)

            # Check if intermediate point is valid
            if not self.is_valid_point((x, y), check_radius):
                return False

        return True

    def get_nearest_node(self, tree, point):
        """
        Find the nearest node in the tree to the given point

        Args:
            tree (dict): Tree to search
            point (tuple): Target point (x, y)

        Returns:
            tuple: Nearest node coordinates
        """
        min_dist = float('inf')
        nearest_node = None

        for node in tree.keys():
            dist = math.sqrt((node[0] - point[0]) ** 2 + (node[1] - point[1]) ** 2)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node

        return nearest_node

    def extend_tree(self, tree, target_point, step_size=2, max_attempts=50):
        """
        Extend tree towards target point

        Args:
            tree (dict): Tree to extend
            target_point (tuple): Target point to extend towards
            step_size (int): Step size for extension
            max_attempts (int): Maximum number of extension attempts

        Returns:
            tuple: New node if successful, None otherwise
        """
        nearest_node = self.get_nearest_node(tree, target_point)

        # Calculate direction vector
        dx = target_point[0] - nearest_node[0]
        dy = target_point[1] - nearest_node[1]
        distance = math.sqrt(dx ** 2 + dy ** 2)

        if distance == 0:
            return None

        # Normalize direction
        dx /= distance
        dy /= distance

        # Calculate new point
        new_point = (
            nearest_node[0] + dx * min(step_size, distance),
            nearest_node[1] + dy * min(step_size, distance)
        )

        # Check if path is valid
        if self.is_valid_path(nearest_node, new_point):
            tree[new_point] = nearest_node
            return new_point

        return None

    def connect_trees(self, step_size=2):
        """
        Connect the two trees using RRT-Connect algorithm

        Args:
            step_size (int): Step size for tree extension

        Returns:
            tuple: Connection point if successful, None otherwise
        """
        for _ in range(1000):  # Maximum iterations
            # Generate random point
            if random.random() < 0.1:  # 10% chance to sample goal
                rand_point = self.goal
            else:
                rand_point = (random.uniform(0, self.map_size[0]),
                              random.uniform(0, self.map_size[1]))

            # Extend start tree towards random point
            new_start_node = self.extend_tree(self.tree_start, rand_point, step_size)

            if new_start_node:
                # Try to extend goal tree towards new start node
                new_goal_node = self.extend_tree(self.tree_goal, new_start_node, step_size)

                # Check if trees are connected
                if new_goal_node and math.sqrt(
                        (new_start_node[0] - new_goal_node[0]) ** 2 +
                        (new_start_node[1] - new_goal_node[1]) ** 2
                ) < step_size * 1.5:
                    return new_start_node, new_goal_node

        return None, None

    def calculate_path_cost(self, path):
        """
        Calculate the total cost of a path

        Args:
            path (list): List of path points

        Returns:
            float: Total path cost
        """
        total_cost = 0

        for i in range(len(path) - 1):
            point1 = path[i]
            point2 = path[i + 1]

            # Distance cost
            distance = math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

            # Operation cost (average cost along the segment)
            mid_x = int((point1[0] + point2[0]) / 2)
            mid_y = int((point1[1] + point2[1]) / 2)
            if 0 <= mid_x < self.map_size[0] and 0 <= mid_y < self.map_size[1]:
                op_cost = self.cost_map[mid_x, mid_y]
            else:
                op_cost = 1.0  # Maximum cost if out of bounds

            # Weighted sum (adjust weights as needed)
            segment_cost = distance * 0.7 + op_cost * 0.3
            total_cost += segment_cost

        return total_cost

    def extract_path(self, connection_point_start, connection_point_goal):
        """
        Extract complete path from start to goal

        Args:
            connection_point_start (tuple): Connection point in start tree
            connection_point_goal (tuple): Connection point in goal tree

        Returns:
            list: Complete path from start to goal
        """
        # Extract path from start to connection point
        path_from_start = []
        node = connection_point_start
        while node is not None:
            path_from_start.append(node)
            node = self.tree_start[node]
        path_from_start.reverse()

        # Extract path from connection point to goal
        path_from_goal = []
        node = connection_point_goal
        while node is not None:
            path_from_goal.append(node)
            node = self.tree_goal[node]

        # Combine paths
        full_path = path_from_start + path_from_goal
        return full_path

    def plan_path(self, max_iterations=1000, step_size=2):
        """
        Plan path using RRT-Connect algorithm

        Args:
            max_iterations (int): Maximum number of iterations
            step_size (int): Step size for tree extension

        Returns:
            dict: Dictionary containing success status and path
        """
        start_time = time.time()

        for iteration in range(max_iterations):
            # Try to connect trees
            connection_start, connection_goal = self.connect_trees(step_size)

            if connection_start and connection_goal:
                # Extract complete path
                self.path = self.extract_path(connection_start, connection_goal)
                self.success = True
                break

        # If no path found, return best effort path
        if not self.success:
            # Find closest points between trees
            min_dist = float('inf')
            best_start, best_goal = None, None

            for start_node in self.tree_start.keys():
                for goal_node in self.tree_goal.keys():
                    dist = math.sqrt((start_node[0] - goal_node[0]) ** 2 +
                                     (start_node[1] - goal_node[1]) ** 2)
                    if dist < min_dist and self.is_valid_path(start_node, goal_node):
                        min_dist = dist
                        best_start, best_goal = start_node, goal_node

            if best_start and best_goal:
                self.path = self.extract_path(best_start, best_goal)

        planning_time = time.time() - start_time

        return {
            'success': self.success,
            'path': self.path,
            'planning_time': planning_time,
            'iterations': iteration + 1 if self.success else max_iterations
        }

    def visualize(self, show_path=True, show_trees=False):
        """
        Visualize the map and planned path

        Args:
            show_path (bool): Whether to show the planned path
            show_trees (bool): Whether to show the search trees
        """
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 10))

        self.ax.clear()

        # Display cost map with colormap
        im = self.ax.imshow(self.cost_map.T, origin='lower',
                            cmap='viridis', alpha=0.7,
                            extent=[0, self.map_size[0], 0, self.map_size[1]])

        # Add colorbar
        cbar = self.fig.colorbar(im, ax=self.ax, shrink=0.8)
        cbar.set_label('Operation Cost')

        # Mark start and goal points
        self.ax.plot(self.start[0], self.start[1], 'go', markersize=10, label='Start')
        self.ax.plot(self.goal[0], self.goal[1], 'ro', markersize=10, label='Goal')

        if show_trees:
            # Draw start tree
            for node, parent in self.tree_start.items():
                if parent:
                    self.ax.plot([parent[0], node[0]], [parent[1], node[1]], 'b-', alpha=0.3)

            # Draw goal tree
            for node, parent in self.tree_goal.items():
                if parent:
                    self.ax.plot([parent[0], node[0]], [parent[1], node[1]], 'r-', alpha=0.3)

        if show_path and self.path:
            # Draw planned path
            path_x = [p[0] for p in self.path]
            path_y = [p[1] for p in self.path]
            self.ax.plot(path_x, path_y, 'y-', linewidth=3, label='Planned Path')
            self.ax.plot(path_x, path_y, 'yo', markersize=4, alpha=0.7)

        self.ax.set_xlim(0, self.map_size[0])
        self.ax.set_ylim(0, self.map_size[1])
        self.ax.set_xlabel('X Coordinate')
        self.ax.set_ylabel('Y Coordinate')
        self.ax.set_title('RRT-Connect Path Planning')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Create a sample cost map
    cost_map = np.zeros((100, 100))

    # Add some obstacles (cost = 1)
    cost_map[20:40, 30:50] = 1  # Square obstacle
    cost_map[60:80, 20:40] = 1  # Another square
    cost_map[40:60, 70:90] = 1  # Third obstacle

    # Add some high-cost areas (0.5-0.9)
    cost_map[10:20, 10:20] = 0.7
    cost_map[80:90, 80:90] = 0.5

    # Define start and goal points
    start_point = (10, 10)
    goal_point = (90, 90)

    # Create planner and plan path
    planner = RRTConnectPlanner(cost_map, start_point, goal_point)
    result = planner.plan_path(max_iterations=500, step_size=2)

    # Print results
    print(f"Planning successful: {result['success']}")
    print(f"Planning time: {result['planning_time']:.3f} seconds")
    print(f"Path length: {len(result['path'])} points")
    print(f"Path cost: {planner.calculate_path_cost(result['path']):.3f}")

    # Visualize results
    planner.visualize(show_path=True, show_trees=True)