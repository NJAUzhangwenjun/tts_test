#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :nn_torch_MaxPool.py
# @Time :2024/7/26 19:11
# @Author :张文军

# 学习torch.nn.MaxPool2d

import torch.nn as nn
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

print(torch.__version__)  # 2.2.2

# 加载CIFAR10数据集
dataSet = torchvision.datasets.CIFAR10(root='/Users/zhangwenjun/Documents/javaFiles/tts_test/data',
                                       transform=torchvision.transforms.ToTensor(), train=False, download=True)

# 创建数据加载器
dataLoader = DataLoader(dataset=dataSet, batch_size=8, shuffle=False)


# 定义图像模型类
class ImgModel(nn.Module):
    def __init__(self):
        super(ImgModel, self).__init__()
        # 定义池化层,通过最大池化层来降低图像的尺寸，从而减少计算量
        self.maxPool = nn.MaxPool2d(kernel_size=3)

    def forward(self, x):
        return self.maxPool(x)


writer = SummaryWriter('/Users/zhangwenjun/Documents/javaFiles/tts_test/log')
model = ImgModel()
step = 0

for data in dataLoader:  # dataLoader是一个迭代器，每次迭代返回一个batch的数据
    images, targets = data
    # 将数据输入模型，并使用tensorboard进行可视化
    writer.add_images('input', images, step)

    # 将数据输入模型，并使用tensorboard进行可视化
    output = model(images)
    writer.add_images('output', model(images), step)
    step += 1

writer.close()  # 关闭tensorboard
# tensorboard --logdir=logs
