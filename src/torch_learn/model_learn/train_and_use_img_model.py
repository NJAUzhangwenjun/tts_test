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
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

# 检查是否使用 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 设置数据集和日志路径
data_path = '/Users/zhangwenjun/Documents/javaFiles/tts_test/data'
log_path = '/Users/zhangwenjun/Documents/javaFiles/tts_test/log'
models_path = '/Users/zhangwenjun/Documents/javaFiles/tts_test/models'

# 创建日志目录
os.makedirs(log_path, exist_ok=True)
os.makedirs(models_path, exist_ok=True)

# 创建SummaryWriter对象
writer = SummaryWriter(log_path)

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载数据集
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
    model = ImgModel().to(device)
    if os.path.exists(os.path.join(models_path, 'imgModel.pth')):
        model.load_state_dict(torch.load(os.path.join(models_path, 'imgModel.pth'), map_location=device))
        print('Model loaded from file')
    else:
        print('Creating new model')

    writer.add_graph(model, torch.randn(64, 3, 32, 32).to(device))

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

        # 训练阶段
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            predicted = outputs.argmax(1)

            total += labels.size(0)
            corrects += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        train_accuracy = 100 * corrects / total

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                predicted = outputs.argmax(1)
                val_total += labels.size(0)
                val_corrects += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_accuracy = 100 * val_corrects / val_total

        writer.add_scalar('Loss/Train', running_loss / len(train_loader), epoch)
        writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
        writer.add_scalar('Loss/Val', val_loss / len(val_loader), epoch)
        writer.add_scalar('Accuracy/Val', val_accuracy, epoch)

        print(f'Epoch [{epoch + 1}/{num_epochs}] '
              f'Train Loss: {running_loss / len(train_loader):.4f} '
              f'Train Acc: {train_accuracy:.2f}% '
              f'Val Loss: {val_loss / len(val_loader):.4f} '
              f'Val Acc: {val_accuracy:.2f}%')

    torch.save(model.state_dict(), f'{models_path}/imgModel.pth')
    print("Training complete")
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
    model = ImgModel().to(device)
    model.load_state_dict(torch.load(os.path.join(models_path, 'imgModel.pth'), map_location=device))
    model.eval()

    random_index = random.randint(0, len(val_dataset) - 1)
    image, label = val_dataset[random_index]

    print(f"Randomly selected index: {random_index}")
    print(f"Image shape: {image.shape}, Label: {label}")

    with torch.no_grad():
        image = image.to(device).unsqueeze(0)
        predicted_output = model(image)

    predicted = predicted_output.argmax(1)
    predicted_class = predicted.item()
    print("Predicted class:", class_names[predicted_class])
    print("True class:", class_names[label])


def predict_image(image_path, true_label_name=''):
    # 加载模型
    model = ImgModel().to(device)
    model.load_state_dict(torch.load(os.path.join(models_path, 'imgModel.pth'), map_location=device))
    model.eval()

    # 加载并预处理图像
    response = requests.get(image_path)
    image = Image.open(BytesIO(response.content))
    image = transform(image).unsqueeze(0).to(device)  # 预处理并增加批次维度

    # 进行预测
    with torch.no_grad():
        predicted_output = model(image)
    predicted = predicted_output.argmax(1)
    predicted_class = predicted.item()
    print(f"Predicted class: {class_names[predicted_class]}, True class: {true_label_name}")


# 训练模型
# train_model()

# 加载模型并测试
# load_model_and_predict()

# 使用指定的图像进行预测
# image_url = "https://townsquare.media/site/40/files/2017/03/Dog-.jpg?w=1200&h=0&zc=1&s=0&a=t&q=89" # 狗
image_url = "https://images.pexels.com/photos/244206/pexels-photo-244206.jpeg?cs=srgb&dl=pexels-mike-bird-244206.jpg&fm=jpg" # 汽车
# predict_image(image_url, true_label_name='🐶')
predict_image(image_url, true_label_name='🚗')
