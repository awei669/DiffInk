import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import cv2
import colorsys
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib import cm
import random

def plot_line(data, save_path, title):
    """
    绘制笔画数据
    :param data: 形状为 (5, sequence_length) 的张量
    :param title: 图表标题
    """
    # 将张量从 GPU 移动到 CPU，并转换为 NumPy 数组
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    if data.shape[0] != 5:
        data = data.transpose(1, 0)

    x_coords = data[0]
    y_coords = data[1]
    is_next = data[2]
    is_new_stroke = data[3]
    is_end = data[4]

    fig, ax = plt.subplots()

    for i in range(data.shape[1] - 1):
        if is_next[i] == 1:  # 当前点和下一个点需要连接
            ax.plot([x_coords[i], x_coords[i + 1]], [y_coords[i], y_coords[i + 1]], 'b-', linewidth=0.5)

    # 绘制每个点
    # ax.plot(x_coords, y_coords, 'ro', markersize=0.5)

    plt.axis('equal')  # 保持横轴和纵轴比例一致
    ax.grid(True)
    plt.title(title)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_line_cv2(data, save_path, canvas_height=64, padding=20, line_thickness=2):
    """
    OpenCV 绘图：高度固定，宽度自适应，保持比例，翻转 y 轴，标记新笔画
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    if data.shape[0] != 5:
        data = data.transpose(1, 0)

    x = data[0]
    y = data[1]
    is_next = data[2]
    is_new_stroke = data[3]

    # 原始范围
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    dx = x_max - x_min
    dy = y_max - y_min + 1e-6  # 防止除 0

    # 缩放比例：固定高度，y 轴填满高度 - 2 * padding
    scale = (canvas_height - 2 * padding) / dy

    # 缩放并翻转 y
    x_scaled = (x - x_min) * scale
    y_scaled = (y_max - y) * scale  # y 轴翻转后缩放

    # 自适应宽度
    scaled_width = int(np.ceil(dx * scale)) + 2 * padding
    x_px = (x_scaled + padding).astype(np.int32)
    y_px = (y_scaled + padding).astype(np.int32)

    # 初始化画布（白底）
    canvas = np.ones((canvas_height, scaled_width, 3), dtype=np.uint8) * 255

    for i in range(len(x_px) - 1):
        if is_next[i] == 1:
            pt1 = (x_px[i], y_px[i])
            pt2 = (x_px[i + 1], y_px[i + 1])
            cv2.line(canvas, pt1, pt2, color=(0, 0, 0), thickness=line_thickness)
        # if is_new_stroke[i] == 1:
        #     cv2.circle(canvas, (x_px[i], y_px[i]), radius=2, color=(0, 0, 255), thickness=-1)

    cv2.imwrite(save_path, canvas)

def plot_line_cv2_new(data, save_path, canvas_height=64, padding=20, line_thickness=2, max_dist=50):
    """
    OpenCV 绘图：绘制到倒数第二个字符为止，重新计算画布宽度，跳过异常点
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    if data.shape[0] != 5:
        data = data.transpose(1, 0)

    x = data[0]
    y = data[1]
    is_next = data[2].astype(np.int32)
    is_new_stroke = data[3].astype(np.int32)
    is_new_char = data[4].astype(np.int32)

    # === 找出所有 xy001 的索引（字符结束点） ===
    char_end_indices = []
    for i in range(len(is_next)):
        if is_next[i] == 0 and is_new_stroke[i] == 0 and is_new_char[i] == 1:
            char_end_indices.append(i)

    if len(char_end_indices) < 2:
        return  # 少于两个字符，跳过绘图

    # 最后一个需要绘制的点位置
    valid_range_end = char_end_indices[-2] + 1  # 注意：+1 是因为 Python 切片右开区间

    # === 裁剪特征序列至倒数第二个字符 ===
    x = x[:valid_range_end]
    y = y[:valid_range_end]
    is_next = is_next[:valid_range_end]
    is_new_stroke = is_new_stroke[:valid_range_end]

    # 原始范围
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    dx = x_max - x_min
    dy = y_max - y_min + 1e-6

    scale = (canvas_height - 2 * padding) / dy

    x_scaled = (x - x_min) * scale
    y_scaled = (y_max - y) * scale  # 翻转 y 轴

    scaled_width = int(np.ceil((x_max - x_min) * scale)) + 2 * padding
    x_px = (x_scaled + padding).astype(np.int32)
    y_px = (y_scaled + padding).astype(np.int32)

    canvas = np.ones((canvas_height, scaled_width, 3), dtype=np.uint8) * 255

    for i in range(len(x_px) - 1):
        if is_next[i] == 1:
            pt1 = np.array([x_px[i], y_px[i]])
            pt2 = np.array([x_px[i + 1], y_px[i + 1]])
            dist = np.linalg.norm(pt1 - pt2)
            if dist < max_dist:
                cv2.line(canvas, tuple(pt1), tuple(pt2), color=(0, 0, 0), thickness=line_thickness)

    cv2.imwrite(save_path, canvas)
