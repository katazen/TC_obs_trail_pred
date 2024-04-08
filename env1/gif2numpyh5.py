import os
import h5py
import imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def gif2arr(gif_path):
    # 读取GIF文件
    gif = imageio.mimread(gif_path, memtest=False)
    # 将GIF转换为NumPy数组
    frames_array = np.array(gif)
    frames_array = frames_array[:, :, 80:-80, :3].transpose((0, 3, 1, 2))
    frames_array = np.mean(frames_array[..., :], axis=1, dtype=int)
    return frames_array


def arr2gif(frames_array):
    fig, ax = plt.subplots()
    im = ax.imshow(frames_array[0], cmap='gray', vmin=0, vmax=255)

    def update(i):
        im.set_data(frames_array[i])

    ani = FuncAnimation(fig, update, frames=frames_array.shape[0], interval=50)  # interval 控制帧之间的时间
    plt.show()


def creat_h5():
    root_fn = 'gif'
    f_lst = os.listdir(root_fn)
    data_frame = 5
    sum_frame = 0
    index_list = []
    data_arr = []
    frame_arr = []
    for item in f_lst:
        fp = os.path.join(root_fn, item)
        arr = gif2arr(fp)
        data_arr.append(arr)
        arr_len = arr.shape[0]
        frame_arr.append(arr_len)
        st = sum_frame
        ed = sum_frame + arr_len - data_frame
        sum_frame += arr_len
        index = list(range(st, ed))
        index_list += index
    data = np.concatenate(data_arr) / 255
    data = np.round(data)
    label_index = np.array(index_list)
    frame_arr = np.array(frame_arr)
    with h5py.File('data.h5', 'w') as f:
        # 创建一个数据集
        # 'dataset_name' 是您给数据集的名称，可以根据需要命名
        f.create_dataset('data', data=data)
        f.create_dataset('index', data=label_index)
        f.create_dataset('frame', data=frame_arr)


def read_h5():
    with h5py.File('data.h5', 'r') as f:
        # 读取数据集
        data = f['data'][:]
        index = f['index'][:]
        frame = f['frame'][:]
    return data, index, frame


if __name__ == '__main__':
    # creat_h5()
    data, index, frame = read_h5()
    arr2gif(data*255)
    pass