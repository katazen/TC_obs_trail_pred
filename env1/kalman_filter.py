import numpy as np
from filterpy.kalman import KalmanFilter

def create_tracker():
    tracker = KalmanFilter(dim_x=4, dim_z=2)

    # 状态转移矩阵
    dt = 1
    tracker.F = np.array([[1, 0, dt, 0],
                          [0, 1, 0, dt],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])

    # 观测矩阵
    tracker.H = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0]])

    # 状态协方差矩阵
    tracker.P *= 1000

    # 过程噪声协方差
    q = 0.1
    tracker.Q = np.array([[q, 0, 0, 0],
                          [0, q, 0, 0],
                          [0, 0, q, 0],
                          [0, 0, 0, q]])

    # 测量噪声协方差
    tracker.R = np.array([[1, 0],
                          [0, 1]])
    return tracker

tracker = create_tracker()

# 初始状态（初始位置为0,0，初始速度为1,1）
tracker.x = np.array([0, 0, 1, 1])

# 模拟观测数据
observations = [np.array([1, 1]), np.array([2, 2]), np.array([3, 3])]

for obs in observations:
    tracker.predict()  # 预测下一状态
    tracker.update(obs)  # 使用新的观测数据更新状态

    print(f"当前状态: {tracker.x}")
