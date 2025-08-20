# import numpy as np
# from scipy.spatial.distance import euclidean
# from heapq import heappush, heappop
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')
#
# class EnergyAwareSamplingPlanner:
#     def __init__(self, energy_map, alpha=0.5):
#         """
#         初始化运动规划器
#
#         参数:
#             energy_map: 操作能量代价地图 (0-1的二维数组)
#             alpha: 路径长度与能量代价的权衡系数 (0-1)
#                    0: 仅考虑路径长度
#                    1: 仅考虑能量代价
#         """
#         self.energy_map = energy_map
#         self.alpha = alpha
#         self.height, self.width = energy_map.shape
#
#     def plan(self, start, goal, max_samples=1000):
#         """
#         执行基于采样的运动规划
#
#         参数:
#             start: 起始位置 (x, y)
#             goal: 目标位置 (x, y)
#             max_samples: 最大采样次数
#
#         返回:
#             路径 (包含起点和终点的离散点列表)
#         """
#         # 转换为网格坐标
#         start_grid = (int(start[0]), int(start[1]))
#         goal_grid = (int(goal[0]), int(goal[1]))
#
#         # 初始化开放集和封闭集
#         open_set = []
#         closed_set = set()
#
#         # 存储节点信息: {位置: (g_cost, h_cost, parent)}
#         node_info = {}
#
#         # 添加起点
#         h_start = self._heuristic(start_grid, goal_grid)
#         heappush(open_set, (h_start, start_grid))
#         node_info[start_grid] = (0, h_start, None)
#
#         path_found = False
#         samples = 0
#
#         while open_set and samples < max_samples:
#             _, current = heappop(open_set)
#             samples += 1
#
#             if current == goal_grid:
#                 path_found = True
#                 break
#
#             if current in closed_set:
#                 continue
#
#             closed_set.add(current)
#
#             # 生成邻近节点
#             for neighbor in self._get_neighbors(current):
#                 if neighbor in closed_set:
#                     continue
#
#                 # 计算新的g_cost (路径长度 + 能量代价)
#                 step_cost = self._get_cost(current, neighbor)
#                 new_g_cost = node_info[current][0] + step_cost
#
#                 # 如果节点未探索或找到更优路径
#                 if neighbor not in node_info or new_g_cost < node_info[neighbor][0]:
#                     h_cost = self._heuristic(neighbor, goal_grid)
#                     total_cost = new_g_cost + h_cost
#                     heappush(open_set, (total_cost, neighbor))
#                     node_info[neighbor] = (new_g_cost, h_cost, current)
#
#         # 回溯路径
#         path = []
#         if path_found:
#             current = goal_grid
#             while current is not None:
#                 path.append(current)
#                 current = node_info[current][2]
#             path.reverse()
#
#         return path
#
#     def _get_neighbors(self, pos):
#         """获取8邻域邻居"""
#         x, y = pos
#         neighbors = []
#
#         for dx in [-1, 0, 1]:
#             for dy in [-1, 0, 1]:
#                 if dx == 0 and dy == 0:
#                     continue
#
#                 nx, ny = x + dx, y + dy
#                 if 0 <= nx < self.width and 0 <= ny < self.height:
#                     neighbors.append((nx, ny))
#
#         return neighbors
#
#     def _heuristic(self, a, b):
#         """启发式函数 (欧氏距离)"""
#         return euclidean(a, b)
#
#     def _get_cost(self, from_pos, to_pos):
#         """
#         计算从from_pos到to_pos的代价
#         权衡路径长度和能量代价
#         """
#         # 路径长度代价
#         dist_cost = euclidean(from_pos, to_pos)
#
#         # 能量代价 (使用目标位置的能耗)
#         energy_cost = self.energy_map[to_pos[1], to_pos[0]]
#
#         # 加权总和
#         return (1 - self.alpha) * dist_cost + self.alpha * energy_cost
#
#     def visualize_path(self, path):
#         """可视化路径"""
#         plt.figure(figsize=(10, 10))
#         plt.imshow(self.energy_map, cmap='viridis', origin='lower')
#         plt.colorbar(label='Energy Cost')
#
#         if path:
#             path_x = [p[0] for p in path]
#             path_y = [p[1] for p in path]
#             plt.plot(path_x, path_y, 'r-', linewidth=2)
#             plt.scatter(path_x, path_y, c='red', s=20)
#
#         plt.scatter([path[0][0]], [path[0][1]], c='green', s=100, label='Start')
#         plt.scatter([path[-1][0]], [path[-1][1]], c='blue', s=100, label='Goal')
#         plt.legend()
#         plt.title('Energy-Aware Path Planning')
#         plt.show()
#
#
# # 示例用法
# if __name__ == "__main__":
#     # 创建测试能量地图 (0-1)
#     np.random.seed(42)
#     energy_map = np.random.rand(50, 50) * 0.8 + 0.2  # 能量值在0.2-1.0之间
#
#     # 添加一些高能耗障碍物
#     energy_map[20:30, 15:25] = 1.0
#     energy_map[10:15, 30:40] = 1.0
#
#     # 创建规划器
#     planner = EnergyAwareSamplingPlanner(energy_map, alpha=0.7)  # 更注重能量代价
#
#     # 设置起点和终点
#     start = (5, 5)
#     goal = (45, 45)
#
#     # 执行规划
#     path = planner.plan(start, goal)
#
#     # 可视化结果
#     print(f"规划路径点数量: {len(path)}")
#     planner.visualize_path(path)


# import numpy as np
# from heapq import heappush, heappop
# from typing import List, Tuple, Optional
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')
#
#
def add_circular_region(map_array, center, radius, energy_value):
    """
    在地图中添加圆形区域并设置能量值

    参数:
        map_array: 原始地图数组
        center: 圆心坐标(行,列)
        radius: 半径(格数)
        energy_value: 要设置的能量值(0-1)
    """
    rows, cols = map_array.shape
    y, x = center

    # 创建网格坐标
    yy, xx = np.ogrid[:rows, :cols]

    # 计算每个点到圆心的距离
    distance = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)

    # 设置圆形区域内的能量值
    map_array[distance <= radius] = energy_value
    return map_array
#
# class EnergyAwareRRTPlanner:
#     def __init__(self, energy_map: np.ndarray, resolution: float = 1.0):
#         """
#         初始化基于能量代价的运动规划器
#
#         参数:
#             energy_map: 操作能量代价地图 (0-1连续值，1表示不可通行)
#             resolution: 地图网格分辨率(单位：米/格)
#         """
#         self.energy_map = energy_map
#         self.resolution = resolution
#         self.height, self.width = energy_map.shape
#         # 运动模型（8连通邻域）
#         self.motions = [
#             (1, 0, 1.0), (0, 1, 1.0), (-1, 0, 1.0), (0, -1, 1.0),  # 4方向
#             (1, 1, np.sqrt(2)), (-1, 1, np.sqrt(2)),  # 对角线
#             (1, -1, np.sqrt(2)), (-1, -1, np.sqrt(2))
#         ]
#
#     def plan(self, start: Tuple[float, float], goal: Tuple[float, float],
#              alpha: float = 0.5, max_iter: int = 1000) -> Optional[List[Tuple[float, float]]]:
#         """
#         基于能量和路径长度权衡的路径规划
#
#         参数:
#             start: 起始位置(x,y)
#             goal: 目标位置(x,y)
#             alpha: 代价权衡系数(0-1, 0=仅路径长度, 1=仅能量代价)
#             max_iter: 最大迭代次数
#
#         返回:
#             路径点列表[(x1,y1), (x2,y2), ...] 或 None(规划失败)
#         """
#         # 转换为网格坐标
#         start_grid = self._world_to_grid(start)
#         goal_grid = self._world_to_grid(goal)
#
#         # 验证起点和目标点有效性
#         if not self._is_valid(start_grid) or not self._is_valid(goal_grid):
#             raise ValueError("起点或目标点位于无效区域")
#
#         # 混合代价函数
#         def cost_func(from_pos, to_pos):
#             path_cost = np.linalg.norm(np.subtract(to_pos, from_pos))  # 路径长度代价
#             energy_cost = self._get_energy_cost(to_pos)  # 能量代价
#             return (1 - alpha) * path_cost + alpha * energy_cost
#
#         # 优先队列 (总代价, 当前位置, 路径)
#         open_set = []
#         heappush(open_set, (0, start_grid, [start_grid]))
#
#         visited = set()
#         visited.add(start_grid)
#
#         for _ in range(max_iter):
#             if not open_set:
#                 break
#
#             current_cost, current_pos, path = heappop(open_set)
#
#             # 到达目标
#             if self._is_goal(current_pos, goal_grid):
#                 return [self._grid_to_world(p) for p in path]
#
#             # 探索邻域
#             for dx, dy, motion_cost in self.motions:
#                 next_pos = (current_pos[0] + dx, current_pos[1] + dy)
#
#                 if not self._is_valid(next_pos) or next_pos in visited:
#                     continue
#
#                 visited.add(next_pos)
#                 new_cost = current_cost + cost_func(current_pos, next_pos)
#                 new_path = path + [next_pos]
#                 heappush(open_set, (new_cost, next_pos, new_path))
#
#         return None  # 规划失败
#
#     def _is_valid(self, grid_pos: Tuple[int, int]) -> bool:
#         """检查网格位置是否有效"""
#         x, y = grid_pos
#         return (0 <= x < self.width and 0 <= y < self.height
#                 and self.energy_map[y, x] < 1.0)  # 能量代价<1表示可通行
#
#     def _is_goal(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> bool:
#         """检查是否到达目标附近"""
#         return np.linalg.norm(np.subtract(pos, goal)) < 2  # 2格范围内视为到达
#
#     def _get_energy_cost(self, grid_pos: Tuple[int, int]) -> float:
#         """获取网格位置的能量代价"""
#         x, y = grid_pos
#         return self.energy_map[y, x]
#
#     def _world_to_grid(self, world_pos: Tuple[float, float]) -> Tuple[int, int]:
#         """世界坐标转网格坐标"""
#         x, y = world_pos
#         return (int(x / self.resolution), int(y / self.resolution))
#
#     def _grid_to_world(self, grid_pos: Tuple[int, int]) -> Tuple[float, float]:
#         """网格坐标转世界坐标"""
#         x, y = grid_pos
#         return (x * self.resolution, y * self.resolution)
#
#     def visualize_path(self,file_path, path: List[Tuple[float, float]]):
#         """可视化路径（调整颜色条位置和大小）
#
#         参数:
#             path: 路径点列表[(x1,y1), (x2,y2), ...]
#         """
#         # 创建图形并调整布局
#         fig = plt.figure(figsize=(10, 10))
#
#         # 调整主图像位置（为颜色条留出空间）
#         # [left, bottom, width, height] 单位是图形比例（0-1）
#         main_ax = fig.add_axes([0.1, 0.1, 0.7, 0.7])  # 右侧留出15%空间
#
#         # 显示能量地图
#         img = main_ax.imshow(self.energy_map, cmap='RdYlGn_r', origin='lower',
#                              vmin=0, vmax=1, interpolation='none')
#
#         # 添加路径
#         if path:
#             path_x, path_y = zip(*path)
#             main_ax.plot(path_x, path_y, 'b-', linewidth=2, zorder=2)
#             main_ax.scatter(path_x, path_y, c='red', s=30, zorder=3)
#
#         # 添加颜色条（高度1/2，右上角）
#         cax = fig.add_axes([0.85, 0.5, 0.02, 0.4])  # [left, bottom, width, height]
#         cbar = fig.colorbar(img, cax=cax, label='Energy Cost')
#
#         # 设置图形属性
#         main_ax.set_title("Energy-Aware Path Planning")
#         main_ax.set_xlabel("X (m)")
#         main_ax.set_ylabel("Y (m)")
#         main_ax.grid(True, color='lightgray', linestyle='--', alpha=0.7)
#         plt.savefig(file_path, dpi=300, bbox_inches='tight')
#         plt.show()
#
# # 示例用法
# if __name__ == "__main__":
#     # 创建测试地图 (0=低能量代价, 1=不可通行)
#     energy_map = np.random.rand(50, 50) * 0.8  # 80%区域可通行
#     energy_map[20:30, 20:30] = 1.0  # 添加障碍物区域
#     energy_map[20:30, 20:30] = 1.0
#     energy_map = add_circular_region(energy_map, center=(10, 15), radius=5, energy_value=1)
#     energy_map = add_circular_region(energy_map, center=(39, 38), radius=8, energy_value=1)
#     energy_map = add_circular_region(energy_map, center=(40, 17), radius=5, energy_value=1)
#
#     planner = EnergyAwareRRTPlanner(energy_map, resolution=1)
#
#
#     # 设置起点和终点
#     start = (1.0, 1.0)
#     goal = (46, 46)
#
#     # 路径规划 (alpha=0.7表示更注重能量代价)
#     path = planner.plan(start, goal, alpha=0.6, max_iter=5000)
#     file_path = "occupied_cost.png"
#
#     if path:
#         print("规划成功! 路径点:", path)
#         planner.visualize_path(file_path, path)
#     else:
#         print("规划失败!")


import numpy as np
from heapq import heappush, heappop
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from math import sqrt, dist
import matplotlib
matplotlib.use('TkAgg')


class EnergyAwareSamplingPlanner:
    def __init__(self, energy_map: np.ndarray, resolution: float = 1.0):
        """
        初始化基于能量代价的采样规划器

        参数:
            energy_map: 操作能量代价地图 (0-1连续值，1表示不可通行)
            resolution: 地图网格分辨率(单位：米/格)
        """
        self.energy_map = energy_map
        self.resolution = resolution
        self.height, self.width = energy_map.shape

        # 定义路径点直径（2单位）的安全检查范围
        self.safety_radius = 1  # 直径2意味着半径1的检查范围

    def plan(self, start: Tuple[float, float],
             goal: Tuple[float, float],
             alpha: float = 0.5,
             max_samples: int = 5000) -> Tuple[Optional[List[Tuple[float, float]]], str]:
        """
        基于采样的路径规划

        参数:
            start: 起始位置(x,y)
            goal: 目标位置(x,y)
            alpha: 代价权衡系数(0=仅路径长度, 1=仅能量代价)
            max_samples: 最大采样次数

        返回:
            (路径点列表, 状态消息)
        """
        # 转换坐标到网格系
        start_grid = self._world_to_grid(start)
        goal_grid = self._world_to_grid(goal)

        # 验证起点和终点有效性
        if not self._is_valid_config(start_grid):
            return None, "起点位于无效区域"
        if not self._is_valid_config(goal_grid):
            return None, "目标点位于无效区域"

        # 初始化数据结构
        samples = []
        visited = set()
        visited.add(start_grid)

        # 将起点加入样本集
        samples.append(start_grid)

        # 混合代价函数
        def cost_func(from_pos, to_pos):
            path_cost = sqrt((to_pos[0] - from_pos[0]) ** 2 + (to_pos[1] - from_pos[1]) ** 2)
            energy_cost = self._get_region_energy(to_pos)
            return (1 - alpha) * path_cost + alpha * energy_cost * path_cost  # 能量代价按路径长度加权

        # 主采样循环
        for _ in range(max_samples):
            # 90%概率采样朝向目标，10%随机采样
            if np.random.random() < 0.9:
                # 朝向目标的偏向采样
                direction = np.array(goal_grid) - np.array(samples[-1])
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                rand_offset = np.random.normal(0, 2, size=2)  # 加入随机扰动
                sample = tuple((np.array(samples[-1]) + direction * 5 + rand_offset).astype(int))
            else:
                # 完全随机采样
                sample = (np.random.randint(0, self.width), np.random.randint(0, self.height))

            # 确保采样点在网格范围内
            sample = (np.clip(sample[0], 0, self.width - 1), np.clip(sample[1], 0, self.height - 1))

            # 检查采样点有效性
            if not self._is_valid_config(sample) or sample in visited:
                continue

            visited.add(sample)
            samples.append(sample)

            # 检查是否到达目标
            if self._is_goal_reached(sample, goal_grid):
                break

        # 构建路径（简化版：直接连接有效样本点）
        path = []
        for sample in samples:
            if self._is_valid_config(sample):
                path.append(self._grid_to_world(sample))

        # 检查是否到达目标
        if len(path) >= 2 and self._is_goal_reached(self._world_to_grid(path[-1]), goal_grid):
            return path, "规划成功"
        else:
            return path, "规划失败：无法找到有效路径"

    def _is_valid_config(self, grid_pos: Tuple[int, int]) -> bool:
        """检查配置是否有效（考虑路径点直径）"""
        x, y = grid_pos
        radius = self.safety_radius

        # 检查边界
        if (x - radius < 0 or x + radius >= self.width or
                y - radius < 0 or y + radius >= self.height):
            return False

        # 检查区域内是否有不可通行点
        region = self.energy_map[y - radius:y + radius + 1, x - radius:x + radius + 1]
        return not np.any(region >= 1.0)

    def _get_region_energy(self, grid_pos: Tuple[int, int]) -> float:
        """获取区域平均能量代价"""
        x, y = grid_pos
        radius = self.safety_radius
        region = self.energy_map[y - radius:y + radius + 1, x - radius:x + radius + 1]
        return np.mean(region)

    def _is_goal_reached(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> bool:
        """检查是否到达目标附近（2格范围内）"""
        return sqrt((pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2) <= 2

    def _world_to_grid(self, world_pos: Tuple[float, float]) -> Tuple[int, int]:
        """世界坐标转网格坐标"""
        x, y = world_pos
        return (int(x / self.resolution), int(y / self.resolution))

    def _grid_to_world(self, grid_pos: Tuple[int, int]) -> Tuple[float, float]:
        """网格坐标转世界坐标"""
        x, y = grid_pos
        return (x * self.resolution, y * self.resolution)

    def visualize(self, path: List[Tuple[float, float]], status: str):
        """可视化结果"""
        fig, ax = plt.subplots(figsize=(10, 10))

        # 显示能量地图
        img = ax.imshow(self.energy_map, cmap='RdYlGn_r', origin='lower',
                        vmin=0, vmax=1, interpolation='none')

        # 添加路径
        if path:
            path_x, path_y = zip(*path)
            ax.plot(path_x, path_y, 'b-', linewidth=2)
            ax.scatter(path_x, path_y, c='red', s=30)
            ax.scatter([path[0][0], path[-1][0]], [path[0][1], path[-1][1]],
                       c=['green', 'blue'], s=100, marker='*')  # 起点绿色，终点蓝色

        # 添加颜色条
        cax = fig.add_axes([0.85, 0.5, 0.02, 0.4])
        fig.colorbar(img, cax=cax, label='Energy Cost')

        ax.set_title(f"Energy-Aware Sampling Path\nStatus: {status}")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        plt.show()


# 示例用法
if __name__ == "__main__":
    # 创建测试地图
    energy_map = np.random.rand(50, 50) * 0.6

    # 添加障碍物
    energy_map[20:30, 20:30] = 1.0  # 矩形障碍物

    # 添加高能量区域
    y, x = np.ogrid[:50, :50]
    mask1 = (x - 10) ** 2 + (y - 10) ** 2 <= 25  # 半径5的圆
    energy_map[mask1] = 0.8

    mask2 = (x - 40) ** 2 + (y - 30) ** 2 <= 64  # 半径8的圆
    energy_map[mask2] = 0.5

    # 创建规划器
    planner = EnergyAwareSamplingPlanner(energy_map, resolution=1)

    # 设置起点和终点
    start = (1.0, 1.0)
    goal = (45, 45)

    # 执行规划
    path, status = planner.plan(start, goal, alpha=0.6, max_samples=500)
    print(status)

    # 可视化
    planner.visualize(path, status)