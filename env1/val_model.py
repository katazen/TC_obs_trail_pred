import torch
from unet import UNet

net = UNet(5,1)
checkpoint = torch.load('/home/fangquan/桌面/a/model.pt')
net.load_state_dict(checkpoint['model'])
pass