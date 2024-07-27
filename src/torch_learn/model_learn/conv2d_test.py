#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :conv2d_test.py
# @Time :2024/7/26 23:58
# @Author :张文军
# 设置数据集路径
import torch
import torchvision

data_path = '/Users/zhangwenjun/Documents/javaFiles/tts_test/data'
log_path = '/Users/zhangwenjun/Documents/javaFiles/tts_test/log'

input = torch.randn(1, 3, 32, 32)  # 输入数据，1个样本，1个通道，32x32的图像
print(input.shape)  # 输出：torch.Size([1, 3, 32, 32])

kernel_size = 5  # 卷积核大小
output = torch.nn.functional.conv2d(input, torch.randn(1, 3, kernel_size, kernel_size), padding=2)  # 卷积操作
print(output.shape)  # 输出：torch.Size([1, 1, 28, 28])，输出数据的尺寸为1x1x28x28

Image = torchvision.transforms.ToPILImage()(input.squeeze())  # 将输入数据转换为图像
Image.show("input")  # 显示图像
Image = torchvision.transforms.ToPILImage()(output.squeeze())  # 将输出数据转换为图像
Image.show("output")  # 显示图像
