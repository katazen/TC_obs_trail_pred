import h5py
import glob
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import argparse
import gc
import random
import os
from threading import Thread
from torch import nn, optim
from model import ECGNet
from sklearn.model_selection import KFold


def set_windows(window_size, x):
    ori_len = x.shape[2]
    batch = x.shape[0]
    arr = np.random.randint(0, ori_len - window_size, batch)
    return np.array([x[idx, :, arr[idx]:arr[idx] + window_size] for idx in range(batch)])


def train_gen(x, y):
    while 1:
        lst = [i for i in range(num_train_data)]
        random.shuffle(lst)
        for m in range(num_train_batch):
            data = np.array([x[i] for i in lst[m * batch_size:(m + 1) * batch_size]])
            label = np.array([y[i] for i in lst[m * batch_size:(m + 1) * batch_size]])
            if window_size == x.shape[2]:
                yield (data, label)
            else:
                yield (set_windows(window_size, data), label)


def val_gen(x, y):
    while 1:
        for m in range(num_val_batch):
            data = x[m * batch_size:(m + 1) * batch_size]
            label = y[m * batch_size:(m + 1) * batch_size]
            if window_size == x.shape[2]:
                yield (data, label)
            else:
                yield (set_windows(window_size, data), label)


def compute_score(p1, t1):
    f_score = []
    for i in range(num_class):
        tp, fp, fn = 0, 0, 0
        for x in range(len(p1)):
            if i in p1[x] and i in t1[x]:
                tp += 1
            elif i in p1[x] and i not in t1[x]:
                fp += 1
            elif i not in p1[x] and i in t1[x]:
                fn += 1
        if (tp + fp == 0) or (tp + fn == 0) or (tp == 0):
            f_score.append(0)
        else:
            pre = tp / (tp + fp)
            rec = tp / (tp + fn)
            f = 2 * pre * rec / (pre + rec)
            f_score.append(round(f, 4))
    f1_str = [str(i) for i in f_score]
    f1_avg = sum(f_score) / num_class
    return f1_str, f1_avg


def get_matrix(p1, t1):
    martix = np.zeros((num_class + 1, num_class + 1), dtype=int)
    for i in range(len(p1)):
        tp = set(p1[i]) & set(t1[i])
        fp = set(p1[i]) - set(t1[i])  # 预测了但并非此类
        fn = set(t1[i]) - set(p1[i])  # 没预测到
        if tp:
            for j in tp:
                martix[j, j] += 1

        if len(fp) == 1 and len(fn) == 1:
            martix[list(fn)[0], list(fp)[0]] += 1

        else:
            if fp:
                for j in fp:
                    martix[-1, j] += 1
            if fn:
                for j in fn:
                    martix[j, -1] += 1
    return martix


def train(thread_id, gpu_id=0, k_id=0, only_save_bestf1=True):
    print(f'线程{thread_id}启动')
    device = torch.device(f"cuda:{gpu_id}")
    train_index, val_index = kfold_list[k_id]
    X_train, X_val = Xt[train_index], Xt[val_index]
    Y_train, Y_val = Yt[train_index], Yt[val_index]
    net = ECGNet(num_classes=num_class).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters())
    # 如果有临时模型则加载临时模型
    if os.path.exists(f'{model_path}/{m}_temp_{k_id}.pt'):
        checkpoint = torch.load(f'{model_path}/{m}_temp_{k_id}.pt')
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        now_epoch = checkpoint['epoch']
        train_f1_list = checkpoint['train_f1_list']
        val_f1_list = checkpoint['val_f1_list']
        train_loss_list = checkpoint['train_loss_list']
        val_loss_list = checkpoint['val_loss_list']
    else:
        now_epoch = 1
        train_f1_list, val_f1_list, train_loss_list, val_loss_list = [], [], [], []
    if glob.glob(f'{model_path}/{m}_{k_id}_*bestloss*.pt'):
        bestloss_file_name = glob.glob(f'{model_path}/{m}_{k_id}_*bestloss*.pt')[0]
        best_loss = float(bestloss_file_name.split('bestloss=')[-1][:4])
    else:
        bestloss_file_name = '0.pt'
        best_loss = 9999
    if glob.glob(f'{model_path}/{m}_{k_id}_*bestf1*.pt'):
        bestf1_file_name = glob.glob(f'{model_path}/{m}_{k_id}_*bestf1*.pt')[0]
        best_f1 = float(bestf1_file_name.split('bestf1=')[-1][:4])
    else:
        bestf1_file_name = '0.pt'
        best_f1 = 0

    for epoch in range(now_epoch, all_epoch + 1):
        # train
        net.train()
        train_loss, num_batch_now = 0, 0
        train_pre, train_ans = [], []
        s_time = time.time()
        # print('\033[31mModel {} Epoch {}/{}\033[m'.format(thread_id,epoch, all_epoch))
        for x, y in train_gen(X_train, Y_train):
            x = torch.from_numpy(x)
            x = x.float().to(device)
            batch_ans = [np.where(i == 1)[0].tolist() for i in y]
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
            batch_pre = [np.where(i == 1)[0].tolist() for i in torch.round(out).cpu().detach().numpy()]
            train_pre += batch_pre
            train_ans += batch_ans
            num_batch_now += 1
            del x, y
            gc.collect()
            if num_batch_now == num_train_batch:
                break
        train_epoch_loss = train_loss / num_train_batch
        f1, fts = compute_score(train_pre, train_ans)
        content = f'Model {thread_id} epoch {epoch} - Train Loss: {train_epoch_loss:.4f} - {time.time() - s_time:.0f}s'
        print(content)
        f1_str = ' - '.join(f1)
        content = f'Model {thread_id} epoch {epoch} - Train F1 score: {f1_str}'
        print(content)

        train_f1_list.append(fts)
        train_loss_list.append(train_epoch_loss)
        # val
        net.eval()
        val_loss, num_batch_now = 0, 0
        val_pre, val_ans = [], []
        s_time = time.time()

        wrong_data = []
        for x, y in val_gen(X_val, Y_val):
            data = x
            x = torch.from_numpy(x)
            x = x.float().to(device)
            batch_ans = [np.where(i == 1)[0].tolist() for i in y]
            y = torch.from_numpy(y)
            y = y.float().to(device)
            out = net(x)
            loss = criterion(out, y)
            # 保存误差和结果
            val_loss += loss.item()
            batch_pre = [np.where(i == 1)[0].tolist() for i in torch.round(out).cpu().detach().numpy()]

            for i in range(len(data)):
                if batch_pre[i] != batch_ans[i]:
                    wrong_data.append([data[i], batch_pre[i], batch_ans[i]])

            val_pre += batch_pre
            val_ans += batch_ans
            num_batch_now += 1
            del x, y
            gc.collect()
            if num_batch_now == num_val_batch:
                break
        val_epoch_loss = val_loss / num_val_batch
        f2, fvs = compute_score(val_pre, val_ans)
        matrix = get_matrix(val_pre, val_ans)
        content = f'Model {thread_id} epoch {epoch} - Val Loss: {val_epoch_loss:.4f} - {time.time() - s_time:.0f}s '
        print(content)
        f2_str = ' - '.join(f2)
        content = f'Model {thread_id} epoch {epoch} - Val F1 score: {f2_str}'
        print(content)

        val_f1_list.append(fvs)
        val_loss_list.append(val_epoch_loss)
        if not only_save_bestf1:
            if val_epoch_loss < best_loss:
                if os.path.exists(bestloss_file_name):
                    os.remove(bestloss_file_name)
                best_loss = val_epoch_loss
                bestloss_file_name = f'{model_path}/{m}_{k_id}_epoch={epoch}_bestloss={val_epoch_loss:.2f}_f1={fvs:.2f}.pt'
                # 保存最佳loss的模型
                state = {'model': net.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'epoch': epoch,
                         'val_f1': f2,
                         'matrix': matrix,
                         'train_f1_list': train_f1_list,
                         'val_f1_list': val_f1_list,
                         'train_loss_list': train_loss_list,
                         'val_loss_list': val_loss_list}
                torch.save(state, bestloss_file_name)
                content = f'Model {thread_id} updata with loss.'
                print(content)

        if fvs > best_f1:

            save_data = np.array([i[0] for i in wrong_data])
            save_pre = np.array([','.join([str(j) for j in i[1]]) if i[1] else '-' for i in wrong_data])
            save_ans = np.array([','.join([str(j) for j in i[2]]) if i[2] else '-' for i in wrong_data])

            if os.path.exists(bestf1_file_name):
                os.remove(bestf1_file_name)
            best_f1 = fvs
            bestf1_file_name = f'{model_path}/{m}_{k_id}_epoch={epoch}_loss={val_epoch_loss:.2f}_bestf1={fvs:.2f}.pt'
            # 保存最佳f1值的模型
            state = {'model': net.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'epoch': epoch,
                     'val_f1': f2,
                     'matrix': matrix,
                     'train_f1_list': train_f1_list,
                     'val_f1_list': val_f1_list,
                     'train_loss_list': train_loss_list,
                     'val_loss_list': val_loss_list}
            torch.save(state, bestf1_file_name)
            content = f'Model {thread_id} updata with f1score.'
            print(content)
            # 保存最佳f1模型中验证集错误的数据
            if not only_save_bestf1:
                np.savez(f'{model_path}/{m}_wrong_valdata_{k_id}.npz',
                         data=save_data,
                         pre=save_pre,
                         ans=save_ans)

        # 保存临时模型，用于训练异常中断之后从保存的epoch继续训练
        state = {'model': net.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'epoch': epoch,
                 'val_f1': f2,
                 'matrix': matrix,
                 'train_f1_list': train_f1_list,
                 'val_f1_list': val_f1_list,
                 'train_loss_list': train_loss_list,
                 'val_loss_list': val_loss_list}
        torch.save(state, f'{model_path}/{m}_temp_{k_id}.pt')
    if not only_save_bestf1:
        # 保存损失-精度图
        plt.figure(figsize=(15, 10))
        plt.subplot(211)
        plt.plot(train_loss_list, 'b', label='tloss')
        plt.plot(val_loss_list, 'r', label='vloss')
        plt.legend()
        plt.subplot(212)
        plt.plot(train_f1_list, 'k', label='tf1')
        plt.plot(val_f1_list, 'y', label='vf1')
        plt.legend()
        plt.savefig(f'{model_path}/{m}_pic_{k_id}.png')
        plt.close()
        print(f'Picture {thread_id} saved.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=2, help="shuffle seed.")
    parser.add_argument('-d', '--dataPath', type=str, default='task0/data/train.h5', help="data path.")
    parser.add_argument('-m', '--modelSavePath', type=str, default='task0/model/cls34_k5', help="model save path.")
    opt = parser.parse_args()

    # 43610 = 5 * 89 * 7 * 7 * 2
    ft = h5py.File(opt.dataPath, 'r')
    Xt = ft['data'][:]
    Yt = ft['label'][:]
    ft.close()

    state = np.random.get_state()
    np.random.shuffle(Xt)
    np.random.set_state(state)
    np.random.shuffle(Yt)

    k_flod = 5
    all_epoch = 500
    batch_size = 89
    num_class = 34
    window_size = 5000

    model_path = opt.modelSavePath
    # shutil.rmtree(model_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path, mode=0o777)
    m = f'seed={opt.seed}_model'
    print(f'This train seed is {opt.seed}')
    np.random.seed(opt.seed)
    state = np.random.get_state()
    np.random.shuffle(Xt)
    np.random.set_state(state)
    np.random.shuffle(Yt)
    num_all_data = Xt.shape[0]
    num_train_data = num_all_data * (k_flod - 1) // k_flod
    num_val_data = num_all_data // k_flod
    num_train_batch = num_train_data // batch_size
    num_val_batch = num_val_data // batch_size
    kf = KFold(n_splits=k_flod)
    kfold_list = list(kf.split(Xt, Yt))
    for i in range(k_flod):
        Thread(target=train, args=(i, i % torch.cuda.device_count(), i)).start()  # 第i折使用第i%N号gpu,N为gpu数量
