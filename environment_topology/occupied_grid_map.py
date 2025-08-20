import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import matplotlib
matplotlib.use('TkAgg')

class GridMap:
    def __init__(self, workspace: List[Tuple[float, float]], resolution: float = 0.1):
        """
        初始化二维栅格地图

        参数:
            workspace: 工作空间的对顶点列表 [(x_min, y_min), (x_max, y_max)]
            resolution: 网格分辨率（每个网格单元的大小）
        """
        self.workspace = workspace
        self.resolution = resolution

        # 计算网格尺寸
        self.x_min, self.y_min = workspace[0]
        self.x_max, self.y_max = workspace[1]

        self.width = self.x_max - self.x_min
        self.height = self.y_max - self.y_min

        self.cols = int(np.ceil(self.width / self.resolution))
        self.rows = int(np.ceil(self.height / self.resolution))

        # 初始化网格地图 (0表示空闲, 1表示障碍物)
        self.grid = np.zeros((self.rows, self.cols), dtype=np.uint8)

    def update_with_circles(self, circles: List[Dict]):
        """
        用圆形障碍物更新地图

        参数:
            circles: 圆形障碍物列表，每个字典包含 'center' 和 'radius'
                   例如: [{'center': (x1, y1), 'radius': r1}, ...]
        """
        # 重置地图
        self.grid.fill(0)

        # 为每个圆形障碍物设置障碍
        for circle in circles:
            center = circle['center']
            radius = circle['radius']
            self._mark_circle(center, radius)

    def _mark_circle(self, center: Tuple[float, float], radius: float):
        """
        标记圆形区域为障碍物（内部方法）
        """
        cx, cy = center

        # 计算圆形在网格中的影响范围
        min_i = max(0, int((cy - radius - self.y_min) / self.resolution))
        max_i = min(self.rows, int((cy + radius - self.y_min) / self.resolution) + 1)

        min_j = max(0, int((cx - radius - self.x_min) / self.resolution))
        max_j = min(self.cols, int((cx + radius - self.x_min) / self.resolution) + 1)

        # 遍历受影响的网格单元
        for i in range(min_i, max_i):
            for j in range(min_j, max_j):
                # 计算网格中心点坐标
                x = self.x_min + (j + 0.5) * self.resolution
                y = self.y_min + (i + 0.5) * self.resolution

                # 检查点是否在圆内
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                    self.grid[i, j] = 1

    def get_obstacles(self) -> np.ndarray:
        """获取障碍物网格"""
        return self.grid

    def visualize(self, path):
        """可视化地图"""
        plt.figure(figsize=(10, 10))
        plt.imshow(self.grid, cmap='binary', origin='lower',
                   extent=[self.x_min, self.x_max, self.y_min, self.y_max])
        plt.title('Grid Map with Obstacles')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True, which='both', color='lightgray', linestyle='--', linewidth=0.5)
        plt.savefig(path, dpi=300)
        plt.show()


# 示例用法
if __name__ == "__main__":
    # 定义工作空间 (x_min, y_min), (x_max, y_max)
    workspace = [(0, 0), (10, 10)]

    # 创建栅格地图
    grid_map = GridMap(workspace, resolution=0.2)

    # 定义圆形障碍物
    circles = [
        {'center': (3, 3), 'radius': 1.5},
        {'center': (7, 7), 'radius': 2.0},
        {'center': (5, 2), 'radius': 1.0}
    ]

    # 更新地图
    grid_map.update_with_circles(circles)

    # 可视化
    grid_map.visualize()

    # 获取障碍物网格
    obstacles = grid_map.get_obstacles()
    print("障碍物网格形状:", obstacles.shape)
    print("障碍物数量:", np.sum(obstacles))