#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :nn_torch_linear.py
# @Time :2024/7/26 22:41
# @Author :张文军
# 设置数据集路径
import torch

data_path = '/Users/zhangwenjun/Documents/javaFiles/tts_test/data'
log_path = '/Users/zhangwenjun/Documents/javaFiles/tts_test/log'

data = torch.randn(10, 10)  # 生成一个10x10的随机张量作为输入数据
# print(data)  # 打印输入数据
print(data.shape)  # 打印输入数据的形状

# reshape 成一个 1*100 的张量
data = data.reshape(1, 100)  # 将数据重塑为一个1x100的张量
print(data.shape)  # 打印重塑后的数据形状


