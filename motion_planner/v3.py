import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import heapq
import random
import matplotlib
matplotlib.use('TkAgg')

class RRTConnectPlanner:
    def __init__(self, cost_map, max_iter=500, step_size=5):
        self.cost_map = cost_map
        self.height, self.width = cost_map.shape
        self.max_iter = max_iter
        self.step_size = step_size

    def is_valid_point(self, point):
        x, y = int(point[0]), int(point[1])
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        if self.cost_map[y, x] >= 1.0:
            return False

        # Check neighbors within 1 unit radius
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    return False
                if self.cost_map[ny, nx] >= 1.0:
                    return False
        return True

    def is_valid_connection(self, p1, p2):
        """检查p1和p2之间的连接是否安全"""
        num_steps = int(np.linalg.norm(np.array(p2) - np.array(p1)))
        if num_steps == 0:
            return True
        for i in range(1, num_steps + 1):
            alpha = i / num_steps
            x = int(p1[0] + alpha * (p2[0] - p1[0]))
            y = int(p1[1] + alpha * (p2[1] - p1[1]))
            if not self.is_valid_point((x, y)):
                return False
        return True

    def get_random_point(self):
        for _ in range(100):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if self.is_valid_point((x, y)):
                return (x, y)
        return None

    def nearest(self, tree, point):
        return min(tree, key=lambda p: np.linalg.norm(np.array(p) - np.array(point)))

    def steer(self, from_point, to_point):
        vec = np.array(to_point) - np.array(from_point)
        dist = np.linalg.norm(vec)
        if dist == 0:
            return from_point
        vec = vec / dist
        new_point = np.array(from_point) + vec * min(self.step_size, dist)
        return tuple(new_point.astype(int))

    def plan(self, start, goal):
        if not self.is_valid_point(start) or not self.is_valid_point(goal):
            return {'success': False, 'path': [start]}

        tree_start = [start]
        tree_goal = [goal]
        parents_start = {start: None}
        parents_goal = {goal: None}

        for _ in range(self.max_iter):
            rand_point = self.get_random_point()
            if rand_point is None:
                break

            # Extend start tree
            nearest_start = self.nearest(tree_start, rand_point)
            new_start = self.steer(nearest_start, rand_point)
            if not self.is_valid_point(new_start) or not self.is_valid_connection(nearest_start, new_start):
                continue
            tree_start.append(new_start)
            parents_start[new_start] = nearest_start

            # Extend goal tree
            nearest_goal = self.nearest(tree_goal, new_start)
            new_goal = self.steer(nearest_goal, new_start)
            if not self.is_valid_point(new_goal) or not self.is_valid_connection(nearest_goal, new_goal):
                continue
            tree_goal.append(new_goal)
            parents_goal[new_goal] = nearest_goal

            if np.linalg.norm(np.array(new_start) - np.array(new_goal)) <= self.step_size:
                # Path found
                path = self.reconstruct_path(parents_start, new_start)
                path += self.reconstruct_path(parents_goal, new_goal)[::-1]
                return {'success': True, 'path': path}

        # Failed
        partial_path = self.reconstruct_path(parents_start, tree_start[-1])
        return {'success': False, 'path': partial_path}

    def reconstruct_path(self, parents, end):
        path = []
        while end is not None:
            path.append(end)
            end = parents[end]
        return path[::-1]

    def visualize(self, path):
        fig, ax = plt.subplots(figsize=(8, 8))
        cmap = cm.viridis
        im = ax.imshow(self.cost_map, cmap=cmap, origin='lower')

        # Draw path
        if len(path) > 1:
            xs, ys = zip(*path)
            ax.plot(xs, ys, color='red', linewidth=2, marker='o', markersize=4)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.05, pad=0.01)
        cbar.ax.set_title('Cost')

        plt.title('RRT-Connect Path Planning')
        plt.axis('on')
        plt.show()
if __name__ == "__main__":
    # 构造地图
    map_size = 50
    energy_map = np.random.rand(map_size, map_size) * 0.8
    energy_map[15:25, 10:20] = 1.0  # 障碍物1
    energy_map[30:40, 30:40] = 1.0  # 障碍物2
    y, x = np.ogrid[:50, :50]
    mask2 = (x - 40) ** 2 + (y - 30) ** 2 <= 64  # 半径8的圆
    energy_map[mask2] = 1
    mask2 = (x - 9) ** 2 + (y - 35) ** 2 <= 64  # 半径8的圆
    energy_map[mask2] = 1
    mask2 = (x - 23) ** 2 + (y - 35) ** 2 <= 49  # 半径8的圆
    energy_map[mask2] = 1

    # 初始化规划器
    planner = RRTConnectPlanner(energy_map)

    # 设定起点与终点
    start = (2, 2)
    goal = (45, 45)

    # 执行路径规划
    result = planner.plan(start, goal)
    print("规划成功:", result['success'])

    # 可视化路径
    planner.visualize(result['path'])
