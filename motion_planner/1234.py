import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict
import random
import math
from heapq import heappop, heappush
import matplotlib
matplotlib.use('TkAgg')

class EnhancedSamplingPlanner:
    def __init__(self, energy_map: np.ndarray, resolution: float = 1.0):
        self.energy_map = energy_map
        self.resolution = resolution
        self.height, self.width = energy_map.shape

        # 8邻域检查
        self.neighborhood = [(-1, -1), (-1, 0), (-1, 1),
                             (0, -1), (0, 1),
                             (1, -1), (1, 0), (1, 1)]

    def is_valid_point(self, grid_pos: Tuple[int, int]) -> bool:
        """检查点及周围1单位范围内是否安全"""
        x, y = grid_pos
        if x <= 0 or x >= self.width - 1 or y <= 0 or y >= self.height - 1:
            return False
        if self.energy_map[y, x] >= 1.0:
            return False
        for dx, dy in self.neighborhood:
            if self.energy_map[y + dy, x + dx] >= 1.0:
                return False
        return True

    def plan(self, start: Tuple[float, float], goal: Tuple[float, float],
             alpha: float = 0.5, max_samples: int = 2000,
             min_waypoints: int = 5) -> Tuple[Optional[List[Tuple[float, float]]], str]:
        """改进的采样规划算法"""
        start_grid = self._world_to_grid(start)
        goal_grid = self._world_to_grid(goal)

        if not self.is_valid_point(start_grid):
            return None, "起点无效"
        if not self.is_valid_point(goal_grid):
            return None, "目标点无效"

        # 初始化采样点（包含起点和目标点）
        samples = [start_grid, goal_grid]
        roadmap = {0: [], 1: []}  # 邻接表

        # 采样阶段
        for _ in range(max_samples):
            # 有偏采样：70%随机，20%目标区域，10%起点区域
            rand_val = random.random()
            if rand_val < 0.2:
                sample = self._sample_near(goal_grid, radius=5)
            elif rand_val < 0.3:
                sample = self._sample_near(start_grid, radius=5)
            else:
                sample = (random.randint(1, self.width - 2),
                          random.randint(1, self.height - 2))

            if self.is_valid_point(sample):
                samples.append(sample)
                node_id = len(samples) - 1
                roadmap[node_id] = []

                # 连接最近的k个有效节点
                k = min(10, len(samples) - 1)
                nearest = self._find_nearest(sample, samples[:-1], k)

                for neighbor in nearest:
                    if self._is_connectable(sample, samples[neighbor]):
                        cost = self._calculate_cost(sample, samples[neighbor], alpha)
                        roadmap[node_id].append((neighbor, cost))
                        roadmap[neighbor].append((node_id, cost))

        # 路径搜索（A*算法）
        path = self._astar_search(roadmap, samples, start_grid, goal_grid, alpha)

        if path:
            # 确保有足够中间点
            if len(path) < min_waypoints:
                path = self._add_intermediate_points(path, samples, roadmap,
                                                     min_waypoints, alpha)

            world_path = [self._grid_to_world(p) for p in path]
            return world_path, f"规划成功，找到{len(path) - 2}个中间点"

        return None, "规划失败"

    def _sample_near(self, point: Tuple[int, int], radius: int) -> Tuple[int, int]:
        """在指定点附近采样"""
        x, y = point
        return (random.randint(max(1, x - radius), min(self.width - 2, x + radius)),
                random.randint(max(1, y - radius), min(self.height - 2, y + radius)))

    def _find_nearest(self, point: Tuple[int, int],
                      samples: List[Tuple[int, int]], k: int) -> List[int]:
        """找到最近的k个点"""
        distances = []
        for i, sample in enumerate(samples):
            dist = math.sqrt((point[0] - sample[0]) ** 2 + (point[1] - sample[1]) ** 2)
            distances.append((dist, i))
        distances.sort()
        return [idx for _, idx in distances[:k]]

    def _is_connectable(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> bool:
        """改进的直线连接检查"""
        # 增加中间点检查密度
        steps = max(abs(p2[0] - p1[0]), abs(p2[1] - p1[1])) * 2
        for t in np.linspace(0, 1, steps):
            x = int(p1[0] + t * (p2[0] - p1[0]))
            y = int(p1[1] + t * (p2[1] - p1[1]))
            if not self.is_valid_point((x, y)):
                return False
        return True

    def _calculate_cost(self, p1: Tuple[int, int], p2: Tuple[int, int], alpha: float) -> float:
        """改进的代价计算（防除零错误版）"""
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]

        # 处理相同点的情况
        if dx == 0 and dy == 0:
            return 0.0

        dist = math.sqrt(dx * dx + dy * dy)

        # 确保至少采样2个点（起点和终点）
        steps = max(abs(dx), abs(dy)) * 2
        steps = max(steps, 2)  # 保证至少有2步

        # 路径能量积分
        energy_sum = 0
        valid_points = 0
        for t in np.linspace(0, 1, steps):
            x = int(p1[0] + t * dx)
            y = int(p1[1] + t * dy)
            if 0 <= x < self.width and 0 <= y < self.height:
                energy_sum += self.energy_map[y, x]
                valid_points += 1

        # 处理无效路径段
        if valid_points == 0:
            return float('inf')

        energy_cost = energy_sum / valid_points
        return (1 - alpha) * dist + alpha * energy_cost * dist * 5

    def _astar_search(self, roadmap: Dict[int, List],
                      samples: List[Tuple[int, int]],
                      start: Tuple[int, int], goal: Tuple[int, int],
                      alpha: float) -> List[Tuple[int, int]]:
        """A*路径搜索"""
        start_idx = 0
        goal_idx = 1

        open_set = [(0, start_idx)]
        came_from = {}
        g_score = {i: float('inf') for i in range(len(samples))}
        g_score[start_idx] = 0

        f_score = {i: float('inf') for i in range(len(samples))}
        f_score[start_idx] = self._heuristic(samples[start_idx], samples[goal_idx])

        while open_set:
            _, current = heappop(open_set)

            if current == goal_idx:
                path = [samples[goal_idx]]
                while current in came_from:
                    current = came_from[current]
                    path.append(samples[current])
                path.reverse()
                return path

            for neighbor, cost in roadmap[current]:
                tentative_g = g_score[current] + cost
                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(
                        samples[neighbor], samples[goal_idx])
                    heappush(open_set, (f_score[neighbor], neighbor))

        return []

    def _heuristic(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """启发式函数"""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def _add_intermediate_points(self, path: List[Tuple[int, int]],
                                 samples: List[Tuple[int, int]],
                                 roadmap: Dict[int, List],
                                 min_points: int, alpha: float) -> List[Tuple[int, int]]:
        """添加中间点以满足最小数量要求"""
        if len(path) >= min_points:
            return path

        # 在现有路径段中寻找可以插入的点
        new_path = [path[0]]
        for i in range(1, len(path)):
            seg_length = math.sqrt(
                (path[i][0] - path[i - 1][0]) ** 2 + (path[i][1] - path[i - 1][1]) ** 2)

            if seg_length > 5:  # 长线段需要分割
                mid_points = self._find_best_midpoints(
                    path[i - 1], path[i], samples, roadmap, alpha)
                new_path.extend(mid_points)
            new_path.append(path[i])

        return new_path if len(new_path) >= min_points else path

    def _find_best_midpoints(self, p1: Tuple[int, int], p2: Tuple[int, int],
                             samples: List[Tuple[int, int]],
                             roadmap: Dict[int, List],
                             alpha: float) -> List[Tuple[int, int]]:
        """在两个点之间寻找最佳中间点"""
        # 查找两个点之间的候选连接点
        candidates = []
        for node in roadmap:
            if node >= len(samples):
                continue
            if (self._is_connectable(p1, samples[node]) and
                    self._is_connectable(samples[node], p2)):
                cost1 = self._calculate_cost(p1, samples[node], alpha)
                cost2 = self._calculate_cost(samples[node], p2, alpha)
                total_cost = cost1 + cost2
                candidates.append((total_cost, node))

        # 选择代价最低的3个候选点
        candidates.sort()
        return [samples[idx] for _, idx in candidates[:3]]

    def _world_to_grid(self, world_pos: Tuple[float, float]) -> Tuple[int, int]:
        x, y = world_pos
        return (int(x / self.resolution), int(y / self.resolution))

    def _grid_to_world(self, grid_pos: Tuple[int, int]) -> Tuple[float, float]:
        x, y = grid_pos
        return (x * self.resolution, y * self.resolution)

    def visualize(self, file_path, path: List[Tuple[float, float]] = None):
        """增强的可视化"""
        plt.figure(figsize=(5,5))

        # 显示能量地图
        plt.imshow(self.energy_map, cmap='RdYlGn_r', origin='lower',
                   vmin=0, vmax=1, interpolation='none')
        cbar = plt.colorbar(label='Operational cost', fraction=0.046, pad=0.04)
        cbar.set_ticks([0, 0.5, 1])
        # cbar.set_ticklabels(['安全', '中等', '危险'])

        # 显示路径
        if path:
            path_x, path_y = zip(*path)
            plt.plot(path_x, path_y, 'b-', linewidth=15, zorder=3, label="path")
            # plt.scatter(path_x[:24], path_y[:24], c='red', s=40, zorder=4)
            plt.scatter([path_x], [path_y], c='lime', s=200,
                        marker='o', edgecolor='black', zorder=5)
            plt.scatter([path_x[-1]], [path_y[-1]], c='magenta', s=200,
                        marker='o', edgecolor='black', zorder=5, label="target")

            # 标记中间点
            # for i, (x, y) in enumerate(path[1:-1], 1):
            #     plt.text(x, y, str(i), color='white', ha='center', va='center',
            #              bbox=dict(facecolor='black', alpha=0.7, pad=1), zorder=6)

        plt.title("Operational aware sampling-based planner", fontsize=10)
        plt.xlabel("X (m)", fontsize=8)
        plt.ylabel("Y (m)", fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.legend()
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.show()


# 示例用法
if __name__ == "__main__":
    # 创建复杂能量地图
    energy_map = np.random.rand(50, 50) * 0.6
    energy_map[15:25, 10:20] = 1.0  # 障碍物1
    energy_map[30:40, 30:40] = 1.0  # 障碍物2
    y, x = np.ogrid[:50, :50]
    mask2 = (x - 40) ** 2 + (y - 30) ** 2 <= 64  # 半径8的圆
    energy_map[mask2] = 1
    mask2 = (x - 9) ** 2 + (y - 35) ** 2 <= 64  # 半径8的圆
    energy_map[mask2] = 1
    mask2 = (x - 23) ** 2 + (y - 35) ** 2 <= 49  # 半径8的圆
    energy_map[mask2] = 1

    # 添加高能量区域
    for x in range(50):
        for y in range(50):
            if 10 < x < 40 and 10 < y < 40:
                if (x - 25) ** 2 + (y - 35) ** 2 < 64:  # 圆形高能量区
                    energy_map[y, x] = min(energy_map[y, x] + 0.4, 0.95)
                if (x - 15) ** 2 + (y - 15) ** 2 < 25:  # 小圆形高能量区
                    energy_map[y, x] = min(energy_map[y, x] + 0.3, 0.95)

    planner = EnhancedSamplingPlanner(energy_map, resolution=1)

    # 设置起点和终点
    start = (1.0, 1.0)
    goal = (45, 45)

    # 执行规划（要求至少5个中间点）
    path, status = planner.plan(start, goal, alpha=0.67, min_waypoints=5)
    print(status)
    file_path = "operational_cost.png"

    # 可视化
    if path:
        planner.visualize(file_path, path)
    else:
        print("无法找到有效路径")