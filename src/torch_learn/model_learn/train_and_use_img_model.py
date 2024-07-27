#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :train_and_use_img_model.py
# @Time :2024/7/27 01:58
# @Author :张文军
import os
import random

import torch
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# 设置数据集和日志路径
data_path = '/Users/zhangwenjun/Documents/javaFiles/tts_test/data'
log_path = '/Users/zhangwenjun/Documents/javaFiles/tts_test/log'
models_path = '/Users/zhangwenjun/Documents/javaFiles/tts_test/models'

# 创建SummaryWriter对象
writer = SummaryWriter(log_path)

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载数据集

# 创建数据集,都是要测试 数据,比较小
train_dataset = torchvision.datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
val_dataset = torchvision.datasets.CIFAR10(data_path, train=False, download=True, transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


# 定义模型
class ImgModel(nn.Module):
    def __init__(self):
        super(ImgModel, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


def train_model():
    # 初始化模型并记录计算图
    # 判断models_path中是否存在模型文件，如果存在则加载模型，否则创建新的模型
    model = ImgModel()
    if os.path.exists(os.path.join(models_path, 'imgModel.pth')):
        model.load_state_dict(torch.load(os.path.join(models_path, 'imgModel.pth')))
        print('load model')
    else:
        print('create model')

    writer.add_graph(model, torch.randn(64, 3, 32, 32))
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    # 开始训练
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        corrects = 0
        total = 0
        step = 0

        # 训练阶段
        for inputs, labels in train_loader:
            # 前向传播
            outputs = model(inputs)

            # 计算预测结果：argmax(1) 表示沿第二维度（即每一行）选取最大值的索引。
            # 由于 outputs 是一个二维张量，其中每一行代表一个样本，每一列表示一个类别的得分，
            # 因此 argmax(1) 将返回每一行的最大值所在的列索引，即每个样本预测的类别编号。
            """
            如: 下面是两个样本,每个样本有4个类别,预测结果为:
            outputs = [
                        [ 1.0130,  0.9453,  1.7578, -0.4238], // 样本1
                        [-1.4005,  0.2202, -0.0834,  0.5198] // 样本2
                    ]
            
            predicted = outputs.argmax(1)
            # predicted = [2, 3] // 样本1预测为类别2,样本2预测为类别3
            predicted = [2, 3]
            """
            predicted = outputs.argmax(1)  # 返回每一行的最大值所在的列索引，即每个样本预测的类别编号,长度是 batch_size

            # 统计训练样本数量
            total += labels.size(0)
            # 统计训练预测正确的样本数量
            corrects += (predicted == labels).sum().item()  # 计算对应位置相等的个数

            # 计算损失
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            step += 1

        # 调整学习率
        scheduler.step()

        # 计算准确率
        train_accuracy = 100 * corrects / total

        # 验证阶段
        model.eval()
        # 整体测试集损失值
        val_loss = 0.0
        # 整体测试集准确率
        val_corrects = 0
        #
        val_total = 0

        # 取消梯度,防止测试激活调优
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                predicted = outputs.argmax(1)
                val_total += labels.size(0)
                val_corrects += (predicted == labels).sum().item()

                # 累计损失
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        # 计算准确率
        val_accuracy = 100 * val_corrects / val_total

        # 记录日志
        writer.add_scalar('Loss/Train', running_loss / len(train_loader), epoch)
        writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
        writer.add_scalar('Loss/Val', val_loss / len(val_loader), epoch)
        writer.add_scalar('Accuracy/Val', val_accuracy, epoch)

        print(f'Epoch [{epoch + 1}/{num_epochs}] '
              f'Train Loss: {running_loss / len(train_loader):.4f} '
              f'Train Acc: {train_accuracy:.2f}% '
              f'Val Loss: {val_loss / len(val_loader):.4f} '
              f'Val Acc: {val_accuracy:.2f}%')
    # 保存模型
    torch.save(model.state_dict(), f'{models_path}/imgModel.pth')
    print("训练完成")
    writer.close()


# 定义类别名称映射
class_names = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]


def load_model_and_predict():
    # 实例化模型
    model = ImgModel()
    # 加载模型参数
    model.load_state_dict(torch.load(os.path.join(models_path, 'imgModel.pth')))
    # 将模型设置为评估模式
    model.eval()

    # 随机选取一张图片
    random_index = random.randint(0, len(val_dataset) - 1)
    image, label = val_dataset[random_index]

    print(f"Randomly selected index: {random_index}")
    print(f"Image shape: {image.shape}, Label: {label}")

    # 使用模型进行预测
    with torch.no_grad():
        predicted_output = model(image.unsqueeze(0))  # 添加 batch 维度
    # 获取预测结果
    predicted = predicted_output.argmax(1)
    predicted_class = predicted.item()
    print("Predicted class:", class_names[predicted_class])


# 训练模型
train_model()

# 加载模型并测试
load_model_and_predict()
