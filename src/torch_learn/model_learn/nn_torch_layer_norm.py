#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :nn_torch_layer_norm.py
# @Time :2024/7/26 20:48
# @Author :张文军
import torch
import torch.nn as nn
import torchvision
from torch.nn import LayerNorm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# 设置数据集路径
data_path = '/Users/zhangwenjun/Documents/javaFiles/tts_test/data'
log_path = '/Users/zhangwenjun/Documents/javaFiles/tts_test/log'

# 下载并加载CIFAR-10数据集
data_set = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                        download=True, transform=transforms.ToTensor())

# 加载数据集
data_loader = DataLoader(dataset=data_set, batch_size=64, shuffle=True)
normalized_shape = data_loader.dataset[0][0].shape[1:]


# 定义模型
class ImgModule(nn.Module):
    def __init__(self):
        super(ImgModule, self).__init__()
        # 注意LayerNorm需要指定归一化的维度
        # 在这里，我们对每个样本的每个通道进行归一化
        # 因此，归一化的维度是[1, 2, 3]，即忽略batch size，对剩下的维度进行归一化
        self.layer_norm = LayerNorm(normalized_shape=normalized_shape)

    def forward(self, x):
        # 对输入进行归一化
        return self.layer_norm(x)


# 创建模型实例
model = ImgModule()

# 创建 SummaryWriter 实例
writer = SummaryWriter(log_dir=log_path)

# 只记录一次模型结构
dummy_input = next(iter(data_loader))[0]  # 获取第一个批次的输入数据作为示例
writer.add_graph(model, dummy_input)

# 开始训练循环
step = 0
for data in data_loader:
    imgs, targets = data

    # 记录输入图像
    writer.add_images('input_imgs', imgs, global_step=step)

    # 前向传播
    output_imgs = model(imgs)

    # 记录输出图像
    writer.add_images('output_imgs', output_imgs, global_step=step)

    step += 1

# 关闭 writer
writer.close()
