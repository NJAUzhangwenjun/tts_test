#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :crfar10_moddel.py
# @Time :2024/7/27 00:22
# @Author :张文军
# 设置数据集路径
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

data_path = '/Users/zhangwenjun/Documents/javaFiles/tts_test/data'
log_path = '/Users/zhangwenjun/Documents/javaFiles/tts_test/log'


class CRFAR10Model(nn.Module):
    def __init__(self):
        super(CRFAR10Model, self).__init__()
        # # 定义第一个卷积层，输入通道数为3，输出通道数为32，卷积核大小为5，填充方式为same
        # self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        # # 定义第一个池化层，池化核大小为2
        # self.pool1 = nn.MaxPool2d(kernel_size=2)
        #
        # # 定义第二个卷积层，输入通道数为32，输出通道数为32，卷积核大小为5，填充方式为same
        # self.conv2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        # # 定义第二个池化层，池化核大小为2
        # self.pool2 = nn.MaxPool2d(kernel_size=2)
        #
        # # 定义第三个卷积层，输入通道数为32，输出通道数为64，卷积核大小为5，填充方式为same
        # self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        # # 定义第三个池化层，池化核大小为2
        # self.pool3 = nn.MaxPool2d(kernel_size=2)
        # # 定义展平层
        # self.flat = nn.Flatten()
        #
        # # 定义第一个全连接层，输入维度为64*4*4，输出维度为64
        # self.fc1 = nn.Linear(64 * 4 * 4, 64)
        # # 定义第二个全连接层，输入维度为64，输出维度为10
        # self.fc2 = nn.Linear(64, 10)
        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)

        )

    def forward(self, x):
        # # 第一个卷积层，激活函数为ReLU，池化层，池化核大小为2
        # x = self.pool1(F.relu(self.conv1(x)))
        # # 第二个卷积层，激活函数为ReLU，池化层，池化核大小为2
        # x = self.pool2(F.relu(self.conv2(x)))
        # # 第三个卷积层，激活函数为ReLU，池化层，池化核大小为2
        # x = self.pool3(F.relu(self.conv3(x)))
        # # 展平层
        # x = self.flat(x)
        # # 第一个全连接层，激活函数为ReLU
        # x = F.relu(self.fc1(x))
        # # 第二个全连接层
        # x = self.fc2(x)
        x = self.model1(x)
        return x


model = CRFAR10Model()  # 创建模型实例
print(model)  # 打印模型结构
input = torch.ones((1, 3, 32, 32))  # 创建一个随机输入
output = model(input)  # 通过模型进行前向传播
print(output)  # 打印输出结果

writer = SummaryWriter(log_path)  # 创建一个SummaryWriter实例
writer.add_graph(model, input)  # 将模型和输入添加到SummaryWriter中
writer.close()  # 关闭SummaryWriter
