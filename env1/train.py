import h5py
import numpy as np
import torch
import time
import argparse
import random
import os
from torch import nn, optim
from unet import UNet


def idx2data(index):
    data_ranges = index[:, None] + np.arange(5)
    label_ranges = index[:, None] + 5
    # 使用高级索引一次性获取所有切片
    return data[data_ranges], data[label_ranges]


def train_gen():
    while 1:
        random.shuffle(idx)
        for m in range(batch_num):
            d, l = idx2data(idx[batch_size * m:batch_size * (m + 1)])
            yield (d, l)


def train():
    device = torch.device(f"cuda:0")
    net = UNet(5,1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    train_loss_list = []
    for epoch in range(1, all_epoch + 1):
        # train
        net.train()
        train_loss, num_batch_now = 0, 0
        s_time = time.time()
        for x, y in train_gen():
            x = torch.from_numpy(x)
            x = x.float().to(device)
            y = torch.from_numpy(y)
            y = y.float().to(device)
            out = net(x)
            loss = criterion(out, y)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 保存误差和结果
            train_loss += loss.item()
            num_batch_now += 1
            if num_batch_now == batch_num:
                break
        train_epoch_loss = train_loss / batch_num
        content = f'Train Loss: {train_epoch_loss:.4f} - {time.time() - s_time:.0f}s'
        print(content)
        train_loss_list.append(train_epoch_loss)
        # 保存临时模型，用于训练异常中断之后从保存的epoch继续训练
        state = {'model': net.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'epoch': epoch,
                 'train_loss_list': train_loss_list}
        torch.save(state, f'{model_path}/model.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataPath', type=str, default='data.h5', help="data path.")
    parser.add_argument('-m', '--modelSavePath', type=str, default='model/m2d20_k5', help="model save path.")
    opt = parser.parse_args()

    # 1301-1=1300=26*50
    ft = h5py.File(opt.dataPath, 'r')
    data = ft['data'][:]
    idx = ft['index'][:-1]
    ft.close()
    all_epoch = 500
    batch_size = 26
    batch_num = 1300//batch_size
    model_path = opt.modelSavePath
    # shutil.rmtree(model_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path, mode=0o777)
    train()
