import csv
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

def get_rotated_points(center_x, center_y, angle, width, height):
    theta = np.radians(angle)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    half_width, half_height = width / 2, height / 2
    corner_points = np.array([[-half_width, -half_height], [half_width, -half_height], [half_width, half_height], [-half_width, half_height]])
    rotated_points = np.dot(corner_points, rotation_matrix)
    rotated_points += np.array([center_x, center_y])
    return rotated_points


def draw_gif(data):
    # 初始化动画
    fig, ax = plt.subplots()
    ax.set_xlim(-40, 40)
    ax.set_ylim(-40, 40)
    ax.set_aspect('equal', adjustable='box')
    fixed_point, = ax.plot(0, 0, 'ko')
    frame_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    rect_plots = []

    def init():
        for _ in range(len(data[0])):  # 假设每个时间点的矩形数量相同
            plot, = ax.plot([], [], 'k-')
            rect_plots.append(plot)
        return rect_plots + [fixed_point, frame_text]

    # 更新函数
    def update(frame):
        for plot, rect in zip(rect_plots, data[frame]):
            center_x, center_y, center_z,angle,angle1,angle2, width, height ,high = rect
            points = get_rotated_points(center_x, center_y, angle, width, height)
            points = np.vstack([points, points[0]])  # 连接首尾点以闭合矩形
            plot.set_data(points[:, 0], points[:, 1])
        frame_text.set_text(f'Frame: {frame}')
        return rect_plots + [fixed_point, frame_text]

    # 创建动画
    ani = animation.FuncAnimation(fig, update, frames=len(data), init_func=init, blit=True, interval=200)

    # 显示或保存动画
    plt.show()

root_fn = '/home/fangquan/桌面/tower_crane_ui/tc_data/'
f_lst = os.listdir(root_fn)
max_lists_per_group = 10
for item in f_lst:
    full_path = os.path.join(root_fn,item)
    print(full_path)
    f = csv.reader(open(full_path,'r'))
    grouped_lists = defaultdict(list)
    for lst in list(f)[1:]:
        grouped_lists[lst[1]].append(lst[2:])
    padded_groups = []
    for key, group in grouped_lists.items():
        while len(group) < max_lists_per_group:
            group.append([0] * 9)  # 填充0
        padded_groups.append(group)
    # 构建三维数组
    arr = np.array(padded_groups,dtype=float)
    draw_gif(arr)

