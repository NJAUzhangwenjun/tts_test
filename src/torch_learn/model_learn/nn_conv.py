import torch
import torchvision

# 卷积学习
# 创建一个二维整形tensor
input_data = torch.tensor([[7, 4, 4, 4],
                           [4, 7, 8, 8],
                           [9, 5, 0, 8],
                           [2, 7, 0, 6]])

kernel = torch.tensor([[2, 5, 2],
                       [7, 4, 0],
                       [1, 9, 1]])

print(input_data.shape)
# 输出torch.Size([4, 4])
print(kernel.shape)
# 输出torch.Size([3, 3])
# 计算卷积

# 尺寸变换:
# input_data: [batch_size, channel, height, width]
# 这里需要尺寸变换的原因是;
"""
在处理二维数组（例如图像）时，通道（channel）通常指的是数据的维度之一。在这个例子中，我们处理的是一个二维数组，所以 `channel` 为 1。

1. **灰度图像**：灰度图像只有一个通道，表示图像的亮度信息。在这种情况下，每个像素点只有一个值，表示其亮度。

2. **彩色图像**：彩色图像通常有 3 个通道，分别对应红、绿、蓝（RGB）三个颜色通道。在这种情况下，每个像素点有三个值，分别表示其红色、绿色和蓝色分量。

3. **深度学习模型中的通道**：在卷积神经网络（CNN）中，通道通常用于描述输入数据在某一层或某一时刻的特征。例如，一个 RGB 图像有 3 个通道，而灰度图像有 1 个通道。

在这个例子中，我们处理的是一个二维数组，所以 `channel` 为 1。这是因为二维数组可以看作是一个灰度图像，其中每个像素点只有一个值，表示其亮度。
"""
input_data = input_data.reshape(1, 1, 4, 4)
print(input_data.shape)
print(input_data)
kernel = kernel.reshape(1, 1, 3, 3)
print(kernel)

output = torch.nn.functional.conv2d(input_data, kernel)
print(output)
Image = torchvision.transforms.ToPILImage()(output.squeeze(0))
output = torch.nn.functional.conv2d(input_data, kernel, stride=1, padding=0)
print(output)
Image = torchvision.transforms.ToPILImage()(output.squeeze(0))
