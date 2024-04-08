import csv
import os
from collections import defaultdict,Counter
import numpy as np
import matplotlib.pyplot as plt

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
    obs_nums = [len(i) for i in grouped_lists.values()]
    plt.plot(obs_nums)
    plt.show()
    padded_groups = []
    for key, group in grouped_lists.items():
        while len(group) < max_lists_per_group:
            group.append([0] * 9)  # 填充0
        padded_groups.append(group)
    # 构建三维数组
    arr = np.array(padded_groups,dtype=float)

