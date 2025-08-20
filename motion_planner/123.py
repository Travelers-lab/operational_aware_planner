import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import random
import math
from collections import deque
import matplotlib
matplotlib.use('TkAgg')

class EnergyAwareSamplingPlanner:
    def __init__(self, energy_map: np.ndarray, resolution: float = 1.0):
        """
        初始化基于能量代价的采样运动规划器

        参数:
            energy_map: 操作能量代价地图 (0-1连续值，1表示不可通行)
            resolution: 地图网格分辨率(单位：米/格)
        """
        self.energy_map = energy_map
        self.resolution = resolution
        self.height, self.width = energy_map.shape

        # 8连通邻域检查
        self.neighborhood = [(-1, -1), (-1, 0), (-1, 1),
                             (0, -1), (0, 1),
                             (1, -1), (1, 0), (1, 1)]

    def is_valid_point(self, grid_pos: Tuple[int, int]) -> bool:
        """
        检查采样点是否有效（扩展版）：
        1. 点本身能量值 < 1
        2. 点周围1单位范围内没有能量值=1的网格
        3. 不超出地图边界
        """
        x, y = grid_pos

        # 检查是否在地图边界内
        if x <= 0 or x >= self.width - 1 or y <= 0 or y >= self.height - 1:
            return False

        # 检查点本身是否可通行
        if self.energy_map[y, x] >= 1.0:
            return False

        # 检查周围8邻域
        for dx, dy in self.neighborhood:
            nx, ny = x + dx, y + dy
            if self.energy_map[ny, nx] >= 1.0:
                return False

        return True

    def plan(self, start: Tuple[float, float], goal: Tuple[float, float],
             alpha: float = 0.5, max_samples: int = 1000) -> Tuple[Optional[List[Tuple[float, float]]], str]:
        """
        基于采样的路径规划

        参数:
            start: 起始位置(x,y)
            goal: 目标位置(x,y)
            alpha: 代价权衡系数(0-1, 0=仅路径长度, 1=仅能量代价)
            max_samples: 最大采样次数

        返回:
            (路径点列表, 状态消息)
        """
        # 转换为网格坐标
        start_grid = self._world_to_grid(start)
        goal_grid = self._world_to_grid(goal)

        # 验证起点和目标点
        if not self.is_valid_point(start_grid):
            return None, "起点位于无效区域"
        if not self.is_valid_point(goal_grid):
            return None, "目标点位于无效区域"

        # 初始化
        samples = [start_grid, goal_grid]
        path = []
        valid_samples_found = False

        # 采样循环
        for _ in range(max_samples):
            # 随机采样
            if random.random() < 0.1:  # 10%概率采样目标点附近
                sample = self._sample_near(goal_grid, radius=5)
            else:
                sample = (random.randint(1, self.width - 2),
                          random.randint(1, self.height - 2))

            # 检查采样点有效性
            if self.is_valid_point(sample):
                valid_samples_found = True
                samples.append(sample)

                # 尝试连接路径
                path = self._find_path(samples, alpha)
                if path and self._is_path_complete(path, start_grid, goal_grid):
                    return [self._grid_to_world(p) for p in path], "规划成功"

        # 最终尝试连接已有采样点
        if valid_samples_found:
            path = self._find_path(samples, alpha)
            if path:
                return [self._grid_to_world(p) for p in path], "部分成功，未达目标"

        return None, "规划失败，无有效路径"

    def _sample_near(self, point: Tuple[int, int], radius: int) -> Tuple[int, int]:
        """在指定点附近采样"""
        x, y = point
        return (random.randint(max(1, x - radius), min(self.width - 2, x + radius)),
                random.randint(max(1, y - radius), min(self.height - 2, y + radius)))

    def _find_path(self, samples: List[Tuple[int, int]], alpha: float) -> List[Tuple[int, int]]:
        """在采样点中寻找最优路径（Dijkstra算法）"""
        # 构建图结构（简化版）
        graph = {i: [] for i in range(len(samples))}
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                if self._is_connectable(samples[i], samples[j]):
                    cost = self._calculate_cost(samples[i], samples[j], alpha)
                    graph[i].append((j, cost))
                    graph[j].append((i, cost))

        # Dijkstra算法实现（简化版）
        start_idx = 0
        goal_idx = 1
        distances = {i: float('inf') for i in range(len(samples))}
        distances[start_idx] = 0
        prev = {}
        queue = [(0, start_idx)]

        while queue:
            current_dist, u = queue.pop(0)
            if u == goal_idx:
                break

            for v, cost in graph[u]:
                if distances[v] > current_dist + cost:
                    distances[v] = current_dist + cost
                    prev[v] = u
                    queue.append((distances[v], v))

        # 重建路径
        if goal_idx in prev:
            path = []
            u = goal_idx
            while u in prev:
                path.append(samples[u])
                u = prev[u]
            path.append(samples[start_idx])
            return path[::-1]
        return []

    def _is_connectable(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> bool:
        """检查两点之间是否可直线连接"""
        # Bresenham直线算法检查路径上的点
        x1, y1 = p1
        x2, y2 = p2
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        sx = -1 if x1 > x2 else 1
        sy = -1 if y1 > y2 else 1

        if dx > dy:
            err = dx / 2.0
            while x != x2:
                if not self.is_valid_point((x, y)):
                    return False
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y2:
                if not self.is_valid_point((x, y)):
                    return False
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy

        return True

    def _calculate_cost(self, p1: Tuple[int, int], p2: Tuple[int, int], alpha: float) -> float:
        """计算两点间的混合代价"""
        # 路径长度代价
        path_cost = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

        # 能量代价（取路径上的平均能量）
        line_points = self._get_line_points(p1, p2)
        energy_sum = sum(self.energy_map[y, x] for x, y in line_points)
        energy_cost = energy_sum / len(line_points)

        return (1 - alpha) * path_cost + alpha * energy_cost * path_cost

    def _get_line_points(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> List[Tuple[int, int]]:
        """获取两点间直线经过的所有网格点"""
        points = []
        x1, y1 = p1
        x2, y2 = p2
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        sx = -1 if x1 > x2 else 1
        sy = -1 if y1 > y2 else 1

        if dx > dy:
            err = dx / 2.0
            while x != x2:
                points.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y2:
                points.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy

        points.append((x2, y2))
        return points

    def _is_path_complete(self, path: List[Tuple[int, int]],
                          start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
        """检查路径是否连接起点和目标点"""
        return path[0] == start and path[-1] == goal

    def _world_to_grid(self, world_pos: Tuple[float, float]) -> Tuple[int, int]:
        """世界坐标转网格坐标"""
        x, y = world_pos
        return (int(x / self.resolution), int(y / self.resolution))

    def _grid_to_world(self, grid_pos: Tuple[int, int]) -> Tuple[float, float]:
        """网格坐标转世界坐标"""
        x, y = grid_pos
        return (x * self.resolution, y * self.resolution)

    def visualize(self, path: List[Tuple[float, float]] = None):
        """可视化地图和路径"""
        plt.figure(figsize=(10, 10))

        # 显示能量地图
        plt.imshow(self.energy_map, cmap='RdYlGn_r', origin='lower',
                   vmin=0, vmax=1, interpolation='none')
        plt.colorbar(label='Energy Cost', fraction=0.046, pad=0.04)

        # 显示路径
        if path:
            path_x, path_y = zip(*path)
            plt.plot(path_x, path_y, 'b-', linewidth=2)
            plt.scatter(path_x, path_y, c='red', s=30)
            plt.scatter([path_x[0], path_x[-1]], [path_y[0], path_y[-1]],
                        c=['green', 'blue'], s=100, marker='*')

        plt.title("Energy-Aware Sampling-Based Planning")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.grid(True, alpha=0.3)
        plt.show()


# 示例用法
if __name__ == "__main__":
    # 创建测试地图
    energy_map = np.random.rand(50, 50) * 0.6

    # 添加障碍物
    energy_map[20:30, 20:30] = 1.0  # 矩形障碍物
    energy_map[5:15, 35:45] = 1.0  # 第二个障碍物

    # 添加高能量区域
    for x in range(10, 40):
        for y in range(10, 40):
            if (x - 25) ** 2 + (y - 35) ** 2 < 100:  # 圆形区域
                energy_map[y, x] = 0.8

    # 创建规划器
    planner = EnergyAwareSamplingPlanner(energy_map, resolution=1)

    # 设置起点和终点
    start = (1.0, 1.0)
    goal = (45, 45)

    # 执行规划
    path, status = planner.plan(start, goal, alpha=0.6, max_samples=5000)
    print(f"规划状态: {status}")

    # 可视化
    planner.visualize(path)