#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :nn_torch_sigmoid.py
# @Time :2024/7/26 20:48
# @Author :张文军
import torch
import torch.nn as nn
import torchvision
from torch.nn import Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

dataSet = torchvision.datasets.CIFAR10(root='/Users/zhangwenjun/Documents/javaFiles/tts_test/data', train=False,
                                       download=True, transform=transforms.ToTensor())

dataLoader = DataLoader(dataset=dataSet, batch_size=64, shuffle=True)


class ImgModule(nn.Module):
    def __init__(self):
        super(ImgModule, self).__init__()
        self.sigmoid = Sigmoid()

    def forward(self, x):
        return self.sigmoid(x)


# 创建模型实例
model = ImgModule()

# 创建 SummaryWriter 实例
writer = SummaryWriter('/Users/zhangwenjun/Documents/javaFiles/tts_test/log')

# 只记录一次模型结构
dummy_input = next(iter(dataLoader))[0]  # 获取第一个批次的输入数据作为示例
writer.add_graph(model, dummy_input)

# 开始训练循环
step = 0
for data in dataLoader:
    imgs, targets = data

    # 记录输入图像
    writer.add_images('imgs', imgs, global_step=step)

    # 前向传播
    output_imgs = model(imgs)

    # 记录输出图像
    writer.add_images('output_imgs', output_imgs, global_step=step)

    step += 1

# 关闭 writer
writer.close()
