import csv
import os
from collections import defaultdict
from matplotlib.patches import Polygon
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


def draw_gif(data,file_name):
    # 初始化动画
    fig, ax = plt.subplots()
    ax.set_xlim(-40, 40)
    ax.set_ylim(-40, 40)
    ax.set_xticks([])
    ax.set_yticks([])
    # 隐藏轴的边框
    ax.axis('off')
    ax.set_aspect('equal', adjustable='box')
    # fixed_point, = ax.plot(0, 0, 'ko')
    # frame_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    # 存储当前帧矩形的列表
    current_rects = []


    # 初始化函数
    def init():
        # frame_text.set_text('')
        return []


    # 更新函数
    def update(frame):
        global current_rects
        # 移除上一帧的矩形
        for rect in current_rects:
            rect.remove()
        current_rects = []  # 清空列表
        # 对于每一帧中的每个矩形，绘制实心矩形
        for rect in data[frame]:
            center_x, center_y, center_z,angle,angle1,angle2, width, height ,high = rect
            points = get_rotated_points(center_x, center_y, angle, width, height)
            # 创建实心矩形并添加到当前帧矩形列表
            polygon = Polygon(points, closed=True, color='black', fill=True)  # 使用黑色填充
            ax.add_patch(polygon)
            current_rects.append(polygon)

        # 更新帧数显示
        # frame_text.set_text(f'Frame: {frame}')

        return [] + current_rects

    # 创建动画
    ani = animation.FuncAnimation(fig, update, frames=len(data), init_func=init, blit=False, interval=200)
    # 显示或保存动画
    # plt.show()
    ani.save('gif/'+file_name[:-4]+'.gif', writer='pillow', fps=20)  # 可以选择保存动画
current_rects = []
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
    draw_gif(arr,item)