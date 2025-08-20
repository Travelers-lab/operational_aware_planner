import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Tuple, Dict
import matplotlib
matplotlib.use('TkAgg')


class AdvancedGridMap:
    def __init__(self, workspace: List[Tuple[float, float]], max_capabilities: Tuple[float, float, float],
                 resolution: float = 0.1):
        """
        初始化二维栅格地图

        参数:
            workspace: 工作空间的对顶点列表 [(x_min, y_min), (x_max, y_max)]
            max_capabilities: 最大操作能力参数 [K^max, D^max, C^max]
            resolution: 网格分辨率（每个网格单元的大小）
        """
        self.workspace = workspace
        self.resolution = resolution
        self.max_k, self.max_d, self.max_c = max_capabilities

        # 计算网格尺寸
        self.x_min, self.y_min = workspace[0]
        self.x_max, self.y_max = workspace[1]

        self.width = self.x_max - self.x_min
        self.height = self.y_max - self.y_min

        self.cols = int(np.ceil(self.width / self.resolution))
        self.rows = int(np.ceil(self.height / self.resolution))

        # 初始化网格地图 (0表示空闲)
        self.grid = np.zeros((self.rows, self.cols), dtype=np.float32)

        # 定义自定义渐变色
        self.cmap = self._create_custom_colormap()

    def _create_custom_colormap(self):
        """创建自定义渐变色"""
        colors = [
            (0.0, "green"),  # 0.0 - 绿色 (空闲)
            (0.3, "yellow"),  # 0.3 - 黄色 (低能力)
            (0.7, "orange"),  # 0.7 - 橙色 (中能力)
            (1.0, "red")  # 1.0 - 红色 (高能力/障碍)
        ]
        return LinearSegmentedColormap.from_list("capability_map", colors)

    def update_with_capability_circles(self, circles: List[Dict]):
        """
        用带操作能力参数的圆形障碍物更新地图

        参数:
            circles: 圆形障碍物列表，每个字典包含 'center', 'radius' 和 'capabilities' [K, D, C]
                    例如: [{'center': (x1, y1), 'radius': r1, 'capabilities': [k1, d1, c1]}, ...]
        """
        # 重置地图
        self.grid.fill(0.0)

        # 为每个圆形障碍物设置障碍
        for circle in circles:
            center = circle['center']
            radius = circle['radius']
            k, d, c = circle['capabilities']
            self._mark_capability_circle(center, radius, k, d, c)

    def _mark_capability_circle(self, center: Tuple[float, float], radius: float,
                                k: float, d: float, c: float):
        """
        根据操作能力参数标记圆形区域（内部方法）
        """
        cx, cy = center

        # 计算圆形在网格中的影响范围
        min_i = max(0, int((cy - radius - self.y_min) / self.resolution))
        max_i = min(self.rows, int((cy + radius - self.y_min) / self.resolution) + 1)

        min_j = max(0, int((cx - radius - self.x_min) / self.resolution))
        max_j = min(self.cols, int((cx + radius - self.x_min) / self.resolution) + 1)

        # 计算归一化能力值
        k_norm = k / self.max_k
        d_norm = d / self.max_d
        c_norm = c / self.max_c

        # 计算平均能力值
        avg_capability = (k_norm + d_norm + c_norm) / 3.0

        # 如果需要阈值处理
        cell_value = 1.0 if avg_capability >= 1.0 else avg_capability

        # 遍历受影响的网格单元
        for i in range(min_i, max_i):
            for j in range(min_j, max_j):
                # 计算网格中心点坐标
                x = self.x_min + (j + 0.5) * self.resolution
                y = self.y_min + (i + 0.5) * self.resolution

                # 检查点是否在圆内
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                    # 如果当前值小于新值，则更新（处理重叠区域）
                    if self.grid[i, j] < cell_value:
                        self.grid[i, j] = cell_value

    def get_grid(self) -> np.ndarray:
        """获取网格地图"""
        return self.grid

    def get_binary_grid(self, threshold: float = 0.999) -> np.ndarray:
        """获取二值化网格地图"""
        return (self.grid >= threshold).astype(np.uint8)

    def visualize(self, path=None, binary: bool = False):
        """可视化地图

        参数:
            path: 图片保存路径，如果为None则不保存
            binary: 是否显示二值化地图
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        # 自定义颜色映射
        from matplotlib.colors import LinearSegmentedColormap, ListedColormap

        # 主渐变色：白色到藏蓝色
        colors = [(1, 1, 1), (0, 0, 0.4)]  # 从白色 (1,1,1) 到藏蓝色 (0,0,0.4)
        custom_cmap = LinearSegmentedColormap.from_list('white_to_navy', colors)

        # 二值化颜色映射：白色和藏蓝色
        binary_colors = [(1, 1, 1), (0, 0, 0.4)]  # 0=白色, 1=藏蓝色
        binary_cmap = ListedColormap(binary_colors)

        if binary:
            grid_to_show = self.get_grid()
            cmap = binary_cmap  # 使用二值化颜色映射
            title = 'Occupied Grid Map'
            # 确保数据是0或1
            grid_to_show = (grid_to_show > 0).astype(float)
        else:
            grid_to_show = self.grid
            cmap = custom_cmap  # 使用连续渐变色
            title = 'Operational Grid Map'

        img = ax.imshow(grid_to_show, cmap=cmap, origin='lower',
                        extent=[self.x_min, self.x_max, self.y_min, self.y_max],
                        vmin=0, vmax=1)  # 固定颜色范围在0-1之间

        if not binary:
            # 创建颜色条轴，设置位置和大小
            cax = fig.add_axes([0.92, 0.5, 0.02, 0.4])  # [left, bottom, width, height]
            cbar = fig.colorbar(img, cax=cax, label='Capability Value')
            cbar.mappable.set_cmap(custom_cmap)

        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        if path is not None:
            plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.show()


# 示例用法
if __name__ == "__main__":
    # 定义工作空间 (x_min, y_min), (x_max, y_max)
    workspace = [(0, 0), (10, 10)]

    # 定义最大操作能力参数 [K^max, D^max, C^max]
    max_capabilities = (10.0, 5.0, 8.0)

    # 创建高级栅格地图
    grid_map = AdvancedGridMap(workspace, max_capabilities, resolution=0.01)

    # 定义带操作能力的圆形障碍物
    circles = [
        {'center': (3, 3), 'radius': 1.5, 'capabilities': [8.0, 3.0, 6.0]},  # 平均能力 0.766
        {'center': (7, 7), 'radius': 2.0, 'capabilities': [12.0, 6.0, 9.0]},  # 有一个参数超过最大值 (K=12 > K^max=10)
        {'center': (5, 2), 'radius': 1.0, 'capabilities': [5.0, 2.5, 4.0]},  # 平均能力 0.5
        {'center': (2, 8), 'radius': 1.2, 'capabilities': [3.0, 1.0, 2.0]}  # 平均能力 0.25
    ]

    # 更新地图
    grid_map.update_with_capability_circles(circles)

    # 可视化能力地图 (使用渐变色)
    grid_map.visualize(path="capability_map.png", binary=False)

    # 可视化二值地图
    grid_map.visualize(path="binary_map.png", binary=True)

    # 获取网格数据
    capability_grid = grid_map.get_grid()
    binary_grid = grid_map.get_binary_grid()

    print("能力网格:")
    print(capability_grid)
    print("\n二值网格:")
    print(binary_grid)