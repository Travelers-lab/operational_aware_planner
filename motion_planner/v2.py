import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import heapq
import random
import matplotlib
matplotlib.use('TkAgg')

class RRTConnectPlanner:
    def __init__(self, cost_map, start, goal, max_iter=1000, step_size=5):
        self.cost_map = cost_map
        self.map_shape = cost_map.shape
        self.start = start
        self.goal = goal
        self.max_iter = max_iter
        self.step_size = step_size
        self.tree_start = {start: None}
        self.tree_goal = {goal: None}
        self.path = []

    def plan(self):
        for i in range(self.max_iter):
            rand_point = self.sample_free()
            nearest_start = self.nearest_vertex(self.tree_start, rand_point)
            new_start = self.steer(nearest_start, rand_point)
            if not self.is_valid_point(new_start) or not self.check_line_validity(nearest_start, new_start):
                continue

            self.tree_start[new_start] = nearest_start

            nearest_goal = self.nearest_vertex(self.tree_goal, new_start)
            new_goal = self.steer(nearest_goal, new_start)
            if not self.is_valid_point(new_goal) or not self.check_line_validity(nearest_goal, new_goal):
                continue

            self.tree_goal[new_goal] = nearest_goal

            if np.linalg.norm(np.array(new_start) - np.array(new_goal)) <= self.step_size:
                if self.check_line_validity(new_start, new_goal):
                    self.tree_goal[new_start] = new_goal
                    self.path = self.extract_path()
                    return {'path': self.path, 'success': True}

        return {'path': self.path, 'success': False}

    def sample_free(self):
        while True:
            x = random.randint(0, self.map_shape[0] - 1)
            y = random.randint(0, self.map_shape[1] - 1)
            if self.cost_map[x, y] < 1.0:
                return (x, y)

    def nearest_vertex(self, tree, point):
        return min(tree.keys(), key=lambda p: np.linalg.norm(np.array(p) - np.array(point)))

    def steer(self, from_point, to_point):
        vec = np.array(to_point) - np.array(from_point)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return from_point
        vec = vec / norm
        new_point = np.array(from_point) + self.step_size * vec
        new_point = tuple(np.round(new_point).astype(int))
        return self.clip_point(new_point)

    def clip_point(self, point):
        x, y = point
        x = min(max(x, 0), self.map_shape[0] - 1)
        y = min(max(y, 0), self.map_shape[1] - 1)
        return (x, y)

    def is_valid_point(self, point):
        x, y = point
        if not (0 <= x < self.map_shape[0] and 0 <= y < self.map_shape[1]):
            return False
        if self.cost_map[x, y] >= 1.0:
            return False
        # Check 1-unit radius neighborhood
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.map_shape[0] and 0 <= ny < self.map_shape[1]:
                    if self.cost_map[nx, ny] >= 1.0:
                        return False
                else:
                    return False  # Out of bounds
        return True

    def check_line_validity(self, p1, p2):
        points = self.bresenham_line(p1, p2)
        for idx, (x, y) in enumerate(points):
            if idx == 0: continue  # skip start point
            if self.cost_map[x, y] >= 1.0:
                return False
        return True

    def bresenham_line(self, p0, p1):
        """Returns list of points between p0 and p1 using Bresenham's algorithm."""
        x0, y0 = p0
        x1, y1 = p1
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                points.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                points.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        points.append((x1, y1))
        return points

    def extract_path(self):
        path_start = []
        node = list(self.tree_goal.keys())[-1]
        while node is not None:
            path_start.append(node)
            node = self.tree_goal[node]
        path_start.reverse()

        node = self.tree_start[path_start[0]]
        while node is not None:
            path_start.insert(0, node)
            node = self.tree_start[node]

        return path_start

    def visualize(self, title="RRT-Connect Path"):
        fig, ax = plt.subplots()
        norm = Normalize(vmin=0, vmax=1)
        cmap = plt.cm.viridis
        ax.imshow(self.cost_map.T, origin='lower', cmap=cmap, norm=norm)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.05)
        cbar.set_label("Cost")

        if self.path:
            path_arr = np.array(self.path)
            ax.plot(path_arr[:, 0], path_arr[:, 1], color='red', linewidth=2)

        ax.scatter(*self.start, c='green', s=50, label="Start")
        ax.scatter(*self.goal, c='blue', s=50, label="Goal")
        ax.legend(loc='upper right')
        ax.set_title(title)
        plt.show()
if __name__ =="__main__":
    # 创建地图（0表示可通行，1表示障碍）
    cost_map = np.random.rand(100, 100)
    cost_map[cost_map > 0.7] = 1.0  # 障碍区域

    start = (5, 5)
    goal = (90, 90)

    planner = RRTConnectPlanner(cost_map, start, goal)
    result = planner.plan()

    print("Success:", result["success"], result["path"])
    if result["success"]:
        print("Path length:", len(result["path"]))

    planner.visualize()
