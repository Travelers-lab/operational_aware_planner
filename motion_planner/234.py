import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from heapq import heappop, heappush
import math
import random
import matplotlib
matplotlib.use('TkAgg')

class SafeZonePlanner:
    def __init__(self, energy_map: np.ndarray, resolution: float = 1.0):
        self.energy_map = energy_map
        self.resolution = resolution
        self.height, self.width = energy_map.shape

    def is_valid_point(self, grid_pos: Tuple[int, int]) -> bool:
        """检查点及周围2单位范围内是否安全"""
        x, y = grid_pos
        if x <= 1 or x >= self.width - 2 or y <= 1 or y >= self.height - 2:
            return False
        if self.energy_map[y, x] >= 1.0:
            return False
        for dx, dy in self.extended_neighborhood:
            if self.energy_map[y + dy, x + dx] >= 1.0:
                return False
        return True

    def plan(self, start: Tuple[float, float], goal: Tuple[float, float],
             alpha: float = 0.5, max_samples: int = 2000) -> Tuple[Optional[List[Tuple[float, float]]], str]:
        """安全走廊路径规划"""
        start_grid = self._world_to_grid(start)
        goal_grid = self._world_to_grid(goal)

        if not self.is_valid_point(start_grid):
            return None, "起点位于危险区域"
        if not self.is_valid_point(goal_grid):
            return None, "目标点位于危险区域"

        # 初始化采样点和路线图
        samples = [start_grid, goal_grid]
        roadmap = {0: [], 1: []}
        safe_path_cache = {}  # 缓存已验证的安全路径段

        # 采样阶段
        for _ in range(max_samples):
            # 有偏采样策略
            if random.random() < 0.7:
                sample = self._biased_sample(samples, goal_grid)
            else:
                sample = (random.randint(2, self.width - 3),
                          random.randint(2, self.height - 3))

            if self.is_valid_point(sample):
                node_id = len(samples)
                samples.append(sample)
                roadmap[node_id] = []

                # 连接最近的5个节点
                nearest = self._find_nearest(sample, samples[:-1], 5)
                for neighbor in nearest:
                    if (neighbor, node_id) not in safe_path_cache:
                        path_segment, is_complete = self._check_safe_corridor(
                            samples[neighbor], sample)
                        safe_path_cache[(neighbor, node_id)] = (path_segment, is_complete)
                        safe_path_cache[(node_id, neighbor)] = (path_segment[::-1], is_complete)

                    path_segment, is_complete = safe_path_cache[(neighbor, node_id)]
                    if path_segment:
                        cost = self._calculate_path_cost(path_segment, alpha)
                        roadmap[node_id].append((neighbor, cost, is_complete))
                        roadmap[neighbor].append((node_id, cost, is_complete))

        # 路径搜索（优先选择完整连接的路径段）
        path = self._find_best_path(roadmap, samples, start_grid, goal_grid)

        if path:
            world_path = [self._grid_to_world(p) for p in path]
            return world_path, f"规划成功，保留{len(path)}个路径点"
        return None, "规划失败"

    def _check_safe_corridor(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> Tuple[List[Tuple[int, int]], bool]:
        """检查安全走廊并返回已验证的安全路径点"""
        safe_points = [p1]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        steps = max(abs(dx), abs(dy)) * 2

        for t in np.linspace(0, 1, max(steps, 2)):
            x = int(round(p1[0] + t * dx))
            y = int(round(p1[1] + t * dy))

            # 检查当前点及周围1单位范围
            for dx2 in [-1, 0, 1]:
                for dy2 in [-1, 0, 1]:
                    nx, ny = x + dx2, y + dy2
                    if not (0 <= nx < self.width and 0 <= ny < self.height):
                        return safe_points, False
                    if self.energy_map[ny, nx] >= 1.0:
                        return safe_points, False

            if (x, y) != safe_points[-1]:
                safe_points.append((x, y))

        return safe_points, True

    def _calculate_path_cost(self, path: List[Tuple[int, int]], alpha: float) -> float:
        """计算路径段总代价"""
        total_cost = 0.0
        for i in range(1, len(path)):
            dx = path[i][0] - path[i - 1][0]
            dy = path[i][1] - path[i - 1][1]
            dist = math.sqrt(dx * dx + dy * dy)

            # 取路径中点能量值作为代表
            mid_x = (path[i][0] + path[i - 1][0]) // 2
            mid_y = (path[i][1] + path[i - 1][1]) // 2
            energy = self.energy_map[mid_y, mid_x]

            total_cost += (1 - alpha) * dist + alpha * energy * dist * 5
        return total_cost

    def _find_best_path(self, roadmap, samples, start, goal):
        """优先使用完整连接的路径段进行搜索"""
        start_idx = 0
        goal_idx = 1

        open_set = [(0, start_idx, [start_idx])]
        visited = set()

        while open_set:
            cost, current, path = heappop(open_set)

            if current == goal_idx:
                return [samples[i] for i in path]

            if current in visited:
                continue
            visited.add(current)

            for neighbor, edge_cost, is_complete in roadmap[current]:
                if is_complete:  # 优先选择完整路径段
                    new_cost = cost + edge_cost * 0.8  # 完整路径奖励
                else:
                    new_cost = cost + edge_cost * 1.2  # 不完整路径惩罚

                heappush(open_set, (new_cost, neighbor, path + [neighbor]))

        return None

    def _sample_near(self, point: Tuple[int, int], radius: int) -> Tuple[int, int]:
        x, y = point
        return (random.randint(max(0, x - radius), min(self.width - 1, x + radius)),
                random.randint(max(0, y - radius), min(self.height - 1, y + radius)))

    def _world_to_grid(self, pos: Tuple[float, float]) -> Tuple[int, int]:
        return (int(pos[0] / self.resolution), int(pos[1] / self.resolution))

    def _grid_to_world(self, pos: Tuple[int, int]) -> Tuple[float, float]:
        return (pos[0] * self.resolution, pos[1] * self.resolution)

    def visualize(self, path: List[Tuple[float, float]],
                  verified: List[Tuple[int, int]]):
        """可视化结果"""
        plt.figure(figsize=(12, 10))

        # 显示能量地图
        plt.imshow(self.energy_map, cmap='RdYlGn_r', origin='lower',
                   vmin=0, vmax=1)

        # 显示已验证点
        if verified:
            vx, vy = zip(*verified)
            plt.scatter(vx, vy, c='blue', s=5, alpha=0.3, label='Verified Points')

        # 显示路径
        if path:
            px, py = zip(*path)
            plt.plot(px, py, 'r-', linewidth=2, label='Path')
            plt.scatter(px, py, c='red', s=50)
            plt.scatter([px[0]], [py[0]], c='green', s=100, marker='*', label='Start')
            plt.scatter([px[-1]], [py[-1]], c='black', s=100, marker='*', label='Goal')

        plt.legend()
        plt.colorbar(label='Energy Cost')
        plt.title("Safe Zone Path Planning")
        plt.show()


# 使用示例
if __name__ == "__main__":
    # 创建测试地图
    map_size = 50
    energy_map = np.random.rand(map_size, map_size) * 0.8
    energy_map[20:30, 20:30] = 1.0  # 障碍物
    energy_map[15:25, 10:20] = 1.0  # 障碍物1
    energy_map[30:40, 30:40] = 1.0  # 障碍物2
    y, x = np.ogrid[:50, :50]
    mask2 = (x - 40) ** 2 + (y - 30) ** 2 <= 64  # 半径8的圆
    energy_map[mask2] = 1
    mask2 = (x - 9) ** 2 + (y - 35) ** 2 <= 64  # 半径8的圆
    energy_map[mask2] = 1
    mask2 = (x - 23) ** 2 + (y - 35) ** 2 <= 49  # 半径8的圆
    energy_map[mask2] = 1

    planner = SafeZonePlanner(energy_map)
    start, goal = (5, 5), (45, 45)

    path, verified = planner.plan(start, goal)
    if path:
        print(f"Found path with {len(path) - 2} waypoints")
        planner.visualize(path, verified)
    else:
        print("Planning failed")
        planner.visualize([], verified)  # 显示已验证区域