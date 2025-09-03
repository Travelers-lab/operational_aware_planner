
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Any


class RRTConnectPlanner:
    """
    RRT-Connect path planner for grid-based environments.

    This class implements the RRT-Connect algorithm for path planning on a 2D grid map.
    It supports configurable planning parameters, robust input validation, modular design,
    and optional visualization.

    Attributes:
        step_size (int): Expansion step size in grid units.
        max_attempts (int): Maximum number of sampling attempts.
        radius (int): Collision checking radius around sample points.
        length_weight (float): Weight for path length in cost calculation.
        cost_weight (float): Weight for operational cost in cost calculation.
        random_seed (Optional[int]): Seed for reproducible sampling.
    """

    def __init__(
        self,
        step_size: int = 2,
        max_attempts: int = 5000,
        radius: int = 10,
        length_weight: float = 1.0,
        cost_weight: float = 1.0,
        random_seed: Optional[int] = None
    ) -> None:
        """
        Initialize planner parameters.

        Args:
            step_size: Expansion step size in grid units.
            max_attempts: Maximum number of sampling attempts.
            radius: Collision checking radius.
            length_weight: Weight for path length.
            cost_weight: Weight for operational cost.
            random_seed: Optional random seed for reproducibility.
        """
        self.step_size = step_size
        self.max_attempts = max_attempts
        self.radius = radius
        self.length_weight = length_weight
        self.cost_weight = cost_weight
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

        # Internal state for visualization
        self._sampled_points: List[Tuple[int, int]] = []
        self._edges: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        self._cost_map: Optional[np.ndarray] = None

    def plan(
        self,
        cost_map: np.ndarray,
        motion_mission: Dict[str, List[int]],
        visualize: bool = False
    ) -> Dict[str, Any]:
        """
        Plan a path using RRT-Connect algorithm.

        Args:
            cost_map: 128x128 numpy array, values in [0, 1], 1=obstacle.
            motion_mission: Dict with 'start_position' and 'target_position' as [x, y].
            visualize: If True, generates visualization data.

        Returns:
            Dict with keys:
                'success': True if path found, else False.
                'path': List of waypoints [(x, y), ...].
                If visualize=True, also:
                    'sampled_points': List of sampled points.
                    'edges': List of edges as ((x1, y1), (x2, y2)).
        Raises:
            ValueError: On invalid inputs.
            RuntimeError: On planning failure.
        """
        # Input validation
        self._validate_inputs(cost_map, motion_mission, visualize)
        self._cost_map = cost_map
        start = tuple(motion_mission["start_position"])
        goal = tuple(motion_mission["target_position"])
        obstacle_mask = (cost_map == 1)

        # Trees: node -> parent
        tree_start: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        tree_goal: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {goal: None}
        nodes_start: List[Tuple[int, int]] = [start]
        nodes_goal: List[Tuple[int, int]] = [goal]

        self._sampled_points = []
        self._edges = []

        success = False
        path: List[Tuple[int, int]] = []

        try:
            for attempt in range(self.max_attempts):
                # Uniformly sample a random point
                rand_point = (
                    int(np.random.randint(0, cost_map.shape[0])),
                    int(np.random.randint(0, cost_map.shape[1]))
                )
                self._sampled_points.append(rand_point)

                # Extend start tree towards sampled point
                nearest_start = self._nearest_node(rand_point, nodes_start)
                new_start = self._steer(nearest_start, rand_point)
                if self._is_valid_sample(new_start, obstacle_mask, cost_map.shape):
                    if self._check_line_segment(nearest_start, new_start, obstacle_mask, cost_map.shape):
                        tree_start[new_start] = nearest_start
                        nodes_start.append(new_start)
                        self._edges.append((nearest_start, new_start))

                        # Try to connect goal tree to new_start
                        nearest_goal = self._nearest_node(new_start, nodes_goal)
                        new_goal = self._steer(nearest_goal, new_start)
                        if self._is_valid_sample(new_goal, obstacle_mask, cost_map.shape):
                            if self._check_line_segment(nearest_goal, new_goal, obstacle_mask, cost_map.shape):
                                tree_goal[new_goal] = nearest_goal
                                nodes_goal.append(new_goal)
                                self._edges.append((nearest_goal, new_goal))

                                # Check if trees are connected
                                if self._is_connected(new_start, new_goal, obstacle_mask, cost_map.shape):
                                    path = self._reconstruct_path(tree_start, tree_goal, new_start, new_goal)
                                    success = True
                                    break

                # Swap trees for bidirectional growth
                tree_start, tree_goal = tree_goal, tree_start
                nodes_start, nodes_goal = nodes_goal, nodes_start

            if not success:
                # No connection found, return best partial path
                best_cost = float('inf')
                best_path: List[Tuple[int, int]] = []
                for node in nodes_start:
                    partial_path = self._reconstruct_partial_path(tree_start, node)
                    cost = self._calculate_path_cost(partial_path)
                    if cost < best_cost:
                        best_cost = cost
                        best_path = partial_path
                path = best_path

            result: Dict[str, Any] = {
                "success": success,
                "path": path
            }
            if visualize:
                result["sampled_points"] = self._sampled_points
                result["edges"] = self._edges
                self._visualize_result(cost_map, path, self._sampled_points, self._edges)
            return result

        except Exception as e:
            raise RuntimeError(f"Planning failed: {e}")

    def _validate_inputs(
        self,
        cost_map: np.ndarray,
        motion_mission: Dict[str, List[int]],
        visualize: bool
    ) -> None:
        """
        Validate input arguments.

        Args:
            cost_map: Grid map.
            motion_mission: Mission dict.
            visualize: Visualization flag.

        Raises:
            ValueError: On invalid inputs.
        """
        if not isinstance(cost_map, np.ndarray):
            raise ValueError("cost_map must be a numpy array.")
        if cost_map.shape != (128, 128):
            raise ValueError("cost_map must have shape (128, 128).")
        if not np.all((0 <= cost_map) & (cost_map <= 1)):
            raise ValueError("cost_map values must be in [0, 1].")
        if not isinstance(motion_mission, dict):
            raise ValueError("motion_mission must be a dictionary.")
        for key in ["start_position", "target_position"]:
            if key not in motion_mission:
                raise ValueError(f"motion_mission missing key: {key}")
            pos = motion_mission[key]
            if (not isinstance(pos, (list, tuple)) or len(pos) != 2 or
                not all(isinstance(x, int) for x in pos)):
                raise ValueError(f"{key} must be a list of two integers.")
            if not (0 <= pos[0] < 128 and 0 <= pos[1] < 128):
                raise ValueError(f"{key} coordinates out of bounds.")
        if not isinstance(visualize, bool):
            raise ValueError("visualize must be a boolean.")

    def _is_valid_sample(
        self,
        point: Tuple[int, int],
        obstacle_mask: np.ndarray,
        map_shape: Tuple[int, int]
    ) -> bool:
        """
        Check if a sample point is valid (collision-free).

        Args:
            point: (x, y) coordinates.
            obstacle_mask: Boolean mask of obstacles.
            map_shape: Shape of the map.

        Returns:
            True if valid, False otherwise.
        """
        x, y = point
        if not (0 <= x < map_shape[0] and 0 <= y < map_shape[1]):
            return False
        x_min = max(0, x - self.radius)
        x_max = min(map_shape[0] - 1, x + self.radius)
        y_min = max(0, y - self.radius)
        y_max = min(map_shape[1] - 1, y + self.radius)
        region = obstacle_mask[x_min:x_max + 1, y_min:y_max + 1]
        if np.any(region):
            return False
        return True

    def _check_line_segment(
        self,
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        obstacle_mask: np.ndarray,
        map_shape: Tuple[int, int]
    ) -> bool:
        """
        Check if the line segment between p1 and p2 is collision-free.

        Args:
            p1: Start point.
            p2: End point.
            obstacle_mask: Boolean mask of obstacles.
            map_shape: Shape of the map.

        Returns:
            True if collision-free, False otherwise.
        """
        line_pts = np.linspace(p1, p2, 11)
        for pt in line_pts:
            xi, yi = int(round(pt[0])), int(round(pt[1]))
            if not (0 <= xi < map_shape[0] and 0 <= yi < map_shape[1]):
                return False
            if obstacle_mask[xi, yi]:
                return False
        return True

    def _nearest_node(
        self,
        pt: Tuple[int, int],
        nodes: List[Tuple[int, int]]
    ) -> Tuple[int, int]:
        """
        Find the nearest node in nodes to pt.

        Args:
            pt: Query point.
            nodes: List of nodes.

        Returns:
            Nearest node.
        """
        arr = np.array(nodes)
        dists = np.linalg.norm(arr - np.array(pt), axis=1)
        idx = np.argmin(dists)
        return nodes[idx]

    def _steer(
        self,
        from_pt: Tuple[int, int],
        to_pt: Tuple[int, int]
    ) -> Tuple[int, int]:
        """
        Generate a new node toward to_pt from from_pt by step_size.

        Args:
            from_pt: Starting point.
            to_pt: Target point.

        Returns:
            New node coordinates.
        """
        vec = np.array(to_pt) - np.array(from_pt)
        dist = np.linalg.norm(vec)
        if dist == 0:
            return from_pt
        direction = vec / dist
        step = min(self.step_size, dist)
        new_pt = np.array(from_pt) + direction * step
        new_pt = np.clip(new_pt, 0, 127)
        return (int(round(new_pt[0])), int(round(new_pt[1])))

    def _is_connected(
        self,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        obstacle_mask: np.ndarray,
        map_shape: Tuple[int, int]
    ) -> bool:
        """
        Check if two nodes are within step_size and collision-free.

        Args:
            pt1: First node.
            pt2: Second node.
            obstacle_mask: Boolean mask of obstacles.
            map_shape: Shape of the map.

        Returns:
            True if connected, False otherwise.
        """
        dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
        if dist <= self.step_size:
            if self._check_line_segment(pt1, pt2, obstacle_mask, map_shape):
                return True
        return False

    def _reconstruct_path(
        self,
        tree_start: Dict[Tuple[int, int], Optional[Tuple[int, int]]],
        tree_goal: Dict[Tuple[int, int], Optional[Tuple[int, int]]],
        connect_start: Tuple[int, int],
        connect_goal: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """
        Reconstruct full path from start to goal via connection.

        Args:
            tree_start: Start tree.
            tree_goal: Goal tree.
            connect_start: Connection node in start tree.
            connect_goal: Connection node in goal tree.

        Returns:
            List of waypoints.
        """
        path_start = []
        node = connect_start
        while node is not None:
            path_start.append(node)
            node = tree_start[node]
        path_start = path_start[::-1]

        path_goal = []
        node = connect_goal
        while node is not None:
            path_goal.append(node)
            node = tree_goal[node]
        return path_start + path_goal

    def _reconstruct_partial_path(
        self,
        tree: Dict[Tuple[int, int], Optional[Tuple[int, int]]],
        node: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """
        Reconstruct path from start to node.

        Args:
            tree: Tree dictionary.
            node: End node.

        Returns:
            List of waypoints.
        """
        path = []
        while node is not None:
            path.append(node)
            node = tree[node]
        return path[::-1]

    def _calculate_path_cost(
        self,
        path: List[Tuple[int, int]]
    ) -> float:
        """
        Compute weighted sum of path length and operational cost.

        Args:
            path: List of waypoints.

        Returns:
            Total cost.
        """
        if not path or len(path) < 2 or self._cost_map is None:
            return float('inf')
        length = 0.0
        op_cost = 0.0
        for i in range(1, len(path)):
            prev, curr = path[i - 1], path[i]
            length += np.linalg.norm(np.array(curr) - np.array(prev))
            op_cost += self._cost_map[curr[0], curr[1]]
        return self.length_weight * length + self.cost_weight * op_cost

    def _visualize_result(
        self,
        cost_map: np.ndarray,
        path: List[Tuple[int, int]],
        sampled_points: List[Tuple[int, int]],
        edges: List[Tuple[Tuple[int, int], Tuple[int, int]]]
    ) -> None:
        """
        Generate visualization using matplotlib.

        Args:
            cost_map: Grid map.
            path: Planned path.
            sampled_points: List of sampled points.
            edges: List of edges.
        """
        plt.figure(figsize=(8, 8))
        plt.imshow(cost_map.T, origin='lower', cmap='viridis')
        cbar = plt.colorbar(fraction=0.03, pad=0.04)
        # Position colorbar at top-right, height=50% of image
        ax = plt.gca()
        fig = plt.gcf()
        bbox = ax.get_position()
        cbar.ax.set_position([bbox.x1 + 0.02, bbox.y1 - 0.25 * bbox.height, 0.03, 0.5 * bbox.height])

        # Plot sampled points
        if sampled_points:
            pts = np.array(sampled_points)
            plt.scatter(pts[:, 0], pts[:, 1], s=5, c='gray', alpha=0.5, label='Sampled Points')

        # Plot edges
        for edge in edges:
            x = [edge[0][0], edge[1][0]]
            y = [edge[0][1], edge[1][1]]
            plt.plot(x, y, color='orange', linewidth=0.5, alpha=0.7)

        # Plot path
        if path and len(path) > 1:
            px = [p[0] for p in path]
            py = [p[1] for p in path]
            plt.plot(px, py, color='red', linewidth=2, label='Planned Path')
            plt.scatter([px[0], px[-1]], [py[0], py[-1]], c=['green', 'blue'], s=50, label='Start/Goal')

        plt.legend(loc='upper left')
        plt.title("RRT-Connect Path Planning")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    cost_map = np.zeros((128, 128))
    cost_map[20:30, 20:30] = 0.5

    cost_map[30:45, 50:65] = 1

    start = [10, 10]
    end = [75, 75]
    motion_mission = {"start_position": start, "target_position": end}

    planner = RRTConnectPlanner()
    planner.plan(cost_map=cost_map, motion_mission=motion_mission, visualize=True)
