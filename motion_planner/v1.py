import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from math import sqrt, atan2, pi
import random
import matplotlib
matplotlib.use('TkAgg')

class Node:
    """表示树中的节点"""

    def __init__(self, pos, parent=None, cost=0.0):
        self.pos = pos  # 节点位置 (x, y)
        self.parent = parent  # 父节点
        self.cost = cost  # 从根节点到当前节点的累计代价


class RRTConnectPlanner:
    def __init__(self, grid_map, start, goal, step_size=1.0, max_iter=5000, neighbor_radius=1.0):
        """
        初始化路径规划器

        参数:
            grid_map: 二维操作代价网格地图，值在[0,1]之间，1表示障碍
            start: 起点坐标 (x, y)
            goal: 终点坐标 (x, y)
            step_size: 扩展步长
            max_iter: 最大迭代次数
            neighbor_radius: 邻居检查半径
        """
        self.grid_map = np.array(grid_map)
        self.height, self.width = self.grid_map.shape
        self.start = start
        self.goal = goal
        self.step_size = step_size
        self.max_iter = max_iter
        self.neighbor_radius = neighbor_radius

        # 初始化起点树和终点树
        self.start_tree = [Node(start)]
        self.goal_tree = [Node(goal)]

        self.path = []
        self.success = False

    def is_valid_position(self, pos):
        """检查位置是否有效（在边界内且周围没有障碍）"""
        x, y = pos

        # 检查是否在边界内
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False

        # 检查半径范围内的邻居
        radius = int(round(self.neighbor_radius))
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                # 计算实际距离
                distance = sqrt(dx ** 2 + dy ** 2)
                if distance > self.neighbor_radius:
                    continue

                nx, ny = int(round(x + dx)), int(round(y + dy))

                # 检查邻居是否在边界内
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    # 如果邻居是障碍物
                    if self.grid_map[ny, nx] == 1:
                        return False
                # 邻居超出边界
                else:
                    return False
        return True

    def is_valid_path(self, from_pos, to_pos):
        """检查两点之间的路径是否有效（使用1单位距离间隔检查）"""
        # 计算路径上的点
        points = self.interpolate_points_with_step(from_pos, to_pos, 1.0)

        # 检查路径上的每个点（跳过起点）
        for point in points[1:]:
            if not self.is_valid_position(point):
                return False
        return True

    def interpolate_points_with_step(self, from_pos, to_pos, step_size):
        """以指定步长插值两点之间的路径点"""
        x1, y1 = from_pos
        x2, y2 = to_pos

        # 计算两点之间的距离和方向
        dx = x2 - x1
        dy = y2 - y1
        distance = sqrt(dx ** 2 + dy ** 2)

        # 如果距离为0，则返回空列表
        if distance == 0:
            return []

        # 计算方向向量
        dx_unit = dx / distance
        dy_unit = dy / distance

        points = []
        current_dist = 0.0

        # 添加起点
        points.append((x1, y1))

        # 沿路径添加中间点（1单位间隔）
        while current_dist < distance:
            current_dist += step_size
            if current_dist > distance:
                current_dist = distance

            x = x1 + current_dist * dx_unit
            y = y1 + current_dist * dy_unit
            points.append((x, y))

        return points

    def interpolate_points(self, from_pos, to_pos):
        """插值两点之间的路径点（用于可视化）"""
        x1, y1 = from_pos
        x2, y2 = to_pos

        # 计算两点之间的距离和方向
        distance = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        steps = int(distance / self.step_size) + 1

        # 插值路径点
        points = []
        for i in range(steps + 1):
            ratio = i / steps
            x = x1 + ratio * (x2 - x1)
            y = y1 + ratio * (y2 - y1)
            points.append((x, y))
        return points

    def find_nearest_node(self, tree, target_pos):
        """在树中找到距离目标位置最近的节点"""
        min_dist = float('inf')
        nearest_node = None

        for node in tree:
            dist = sqrt((node.pos[0] - target_pos[0]) ** 2 +
                        (node.pos[1] - target_pos[1]) ** 2)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node

        return nearest_node

    def steer(self, from_node, to_pos):
        """从节点向目标位置扩展"""
        from_x, from_y = from_node.pos
        to_x, to_y = to_pos

        # 计算方向向量
        dx = to_x - from_x
        dy = to_y - from_y
        dist = sqrt(dx ** 2 + dy ** 2)

        # 计算新位置
        if dist <= self.step_size:
            new_pos = to_pos
        else:
            scale = self.step_size / dist
            new_pos = (from_x + dx * scale, from_y + dy * scale)

        # 检查新位置是否有效
        if not self.is_valid_position(new_pos):
            return None

        # 检查路径是否有效（使用1单位距离间隔检查）
        if not self.is_valid_path(from_node.pos, new_pos):
            return None

        # 计算新节点的代价（路径代价 + 操作代价）
        new_cost = from_node.cost + dist

        return Node(new_pos, from_node, new_cost)

    def connect(self, tree, target_pos):
        """尝试从树连接到目标位置"""
        while True:
            nearest_node = self.find_nearest_node(tree, target_pos)
            new_node = self.steer(nearest_node, target_pos)

            if new_node is None:
                break

            tree.append(new_node)

            # 检查是否到达目标位置
            dist_to_target = sqrt((new_node.pos[0] - target_pos[0]) ** 2 +
                                  (new_node.pos[1] - target_pos[1]) ** 2)
            if dist_to_target <= self.step_size:
                return new_node

            target_pos = new_node.pos

        return None

    def plan(self):
        """执行路径规划"""
        for _ in range(self.max_iter):
            # 随机采样
            if random.random() < 0.5:
                rand_pos = (random.uniform(0, self.width),
                            random.uniform(0, self.height))
            else:
                rand_pos = self.goal

            # 从起点树扩展
            new_node = self.connect(self.start_tree, rand_pos)
            if new_node is None:
                continue

            # 从终点树尝试连接到新节点
            connected_node = self.connect(self.goal_tree, new_node.pos)
            if connected_node is not None:
                # 连接成功，构建路径
                self.success = True
                self.construct_path(new_node, connected_node)
                break

        return {'path': self.path, 'success': self.success}

    def construct_path(self, start_node, goal_node):
        """构建最终路径"""
        # 从起点树回溯
        path_from_start = []
        node = start_node
        while node is not None:
            path_from_start.append(node.pos)
            node = node.parent

        # 从终点树回溯
        path_from_goal = []
        node = goal_node
        while node is not None:
            path_from_goal.append(node.pos)
            node = node.parent

        # 合并路径（起点树部分需要反转）
        self.path = list(reversed(path_from_start)) + path_from_goal

    def visualize(self, path=None, title="RRT-Connect Path Planning"):
        """可视化地图和路径"""
        plt.figure(figsize=(10, 8))

        # 创建自定义颜色映射
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'custom', ['white', 'red', 'darkred'])

        # 绘制地图
        plt.imshow(self.grid_map, cmap=cmap, vmin=0, vmax=1,
                   origin='lower', extent=[0, self.width, 0, self.height])

        # 添加颜色条
        cax = plt.axes([0.85, 0.25, 0.05, 0.5])  # 右上角位置
        plt.colorbar(cax=cax, label='Operation Cost')

        # 标记起点和终点
        plt.scatter(*self.start, s=100, c='green', marker='o', label='Start')
        plt.scatter(*self.goal, s=100, c='blue', marker='s', label='Goal')

        # 绘制路径
        if path:
            path = np.array(path)
            plt.plot(path[:, 0], path[:, 1], 'y-', linewidth=2, label='Path')
            plt.scatter(path[:, 0], path[:, 1], s=20, c='orange', marker='o')

        # 设置图形属性
        plt.title(title)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


# 测试示例
if __name__ == "__main__":
    # 创建示例地图 (20x20)
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

    # 添加障碍物
    energy_map[5:15, 10] = 1  # 垂直障碍
    energy_map[10, 5:15] = 1  # 水平障碍

    # 设置起点和终点
    start = (2, 2)
    goal = (45, 45)

    # 创建规划器并执行规划
    planner = RRTConnectPlanner(energy_map, start, goal, step_size=1.0)
    result = planner.plan()

    # 打印规划结果
    print(f"Planning successful: {result['success']}")
    print(f"Path length: {len(result['path'])} points, {result['path']}")

    # 可视化结果
    planner.visualize(result['path'] if result['success'] else None)