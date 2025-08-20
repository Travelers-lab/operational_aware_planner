import numpy as np
from itertools import combinations
from collections import defaultdict
import math


def classify_points_to_circles(points, tolerance=0.1):
    """
    将一组二维点分类到不同的圆上

    参数:
        points: 二维坐标列表，格式为 [(x1,y1), (x2,y2), ...]
        tolerance: 判断点是否在同一圆上的容忍度

    返回:
        一个字典列表，每个字典包含圆的圆心和半径
    """
    if len(points) < 3:
        raise ValueError("至少需要3个点才能确定一个圆")

    # 预处理：移除重复点
    unique_points = list(set(points))
    if len(unique_points) < 3:
        raise ValueError("去重后点数不足3个，无法确定圆")

    # 用于存储已分配的点
    assigned = set()
    circles = []

    # 尝试找到包含最多点的圆
    while len(assigned) < len(unique_points):
        best_circle = None
        best_points = []
        best_count = 0

        # 尝试所有可能的3点组合来拟合圆
        for triplet in combinations(unique_points, 3):
            if any(p in assigned for p in triplet):
                continue

            try:
                center, radius = fit_circle(triplet)

                # 计算有多少未分配的点在这个圆上
                count = 0
                current_points = []
                for p in unique_points:
                    if p not in assigned and is_point_on_circle(p, center, radius, tolerance):
                        count += 1
                        current_points.append(p)

                if count > best_count:
                    best_count = count
                    best_points = current_points
                    best_circle = (center, radius)
            except:
                continue

        if best_circle is None:
            # 无法找到更多圆，将剩余点视为噪声或处理为特殊情况
            break

        # 记录这个圆
        center, radius = best_circle
        circles.append({
            'center': center,
            'radius': radius,
            'points': best_points
        })

        # 标记这些点已分配
        assigned.update(best_points)

    # 处理剩余的点（如果有）
    if len(assigned) < len(unique_points):
        remaining = [p for p in unique_points if p not in assigned]
        print(f"警告: {len(remaining)} 个点无法分配到任何圆上")

    # 提取圆的参数
    result = []
    for circle in circles:
        result.append({
            'center': circle['center'],
            'radius': circle['radius']
        })

    return result


def fit_circle(points):
    """
    用三个点拟合一个圆
    返回圆心和半径
    """
    if len(points) != 3:
        raise ValueError("需要恰好3个点来拟合圆")

    A = np.array(points)
    x = A[:, 0]
    y = A[:, 1]

    # 解线性方程组
    A = np.array([
        [2 * (x[1] - x[0]), 2 * (y[1] - y[0])],
        [2 * (x[2] - x[0]), 2 * (y[2] - y[0])]
    ])
    b = np.array([
        x[1] ** 2 - x[0] ** 2 + y[1] ** 2 - y[0] ** 2,
        x[2] ** 2 - x[0] ** 2 + y[2] ** 2 - y[0] ** 2
    ])

    try:
        center = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        raise ValueError("这三个点共线，无法形成圆")

    radius = math.sqrt((x[0] - center[0]) ** 2 + (y[0] - center[1]) ** 2)

    return (center[0], center[1]), radius


def is_point_on_circle(point, center, radius, tolerance):
    """
    检查点是否在圆上（考虑容忍度）
    """
    distance = math.sqrt((point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2)
    return abs(distance - radius) < tolerance


# 示例用法
if __name__ == "__main__":
    # 示例点集（包含两个圆上的点）
    points = [
        (1, 0), (0, 1), (-1, 0), (0, -1),  # 第一个圆上的点
        (3, 2), (4, 1), (3, 0), (2, 1)  # 第二个圆上的点
    ]

    try:
        circles = classify_points_to_circles(points)
        for i, circle in enumerate(circles, 1):
            print(f"圆 {i}:")
            print(f"  圆心: {circle['center']}")
            print(f"  半径: {circle['radius']}")
            print()
    except ValueError as e:
        print(f"错误: {e}")