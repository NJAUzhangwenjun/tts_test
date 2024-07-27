import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='/Users/zhangwenjun/Documents/javaFiles/tts_test/data', train=False,
                                       download=True, transform=transforms.ToTensor())

dataload = DataLoader(dataset, batch_size=64)

# 日志
writer = SummaryWriter('/Users/zhangwenjun/Documents/javaFiles/tts_test/log')


class WJModel(torch.nn.Module):

    def __init__(self) -> None:
        super(WJModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=3, stride=1, padding=1)  # 输出通道数改为3

    def forward(self, x):
        x = self.conv1(x)
        # x = self.conv2(x)
        return x


model = WJModel()
writer.add_graph(model, torch.randn(64, 3, 32, 32))  # 添加网络结构到日志中

step = 0
for data in dataload:
    imgs, targets = data
    output = model(imgs)
    print(f'img shape:{imgs.shape}')
    print(f'output shape:{output.shape}')

    writer.add_images('input', imgs, step)  # 添加数据到日志中
    # writer.add_scalar('output', output.shape[0], step)  # 添加数据到日志中

    # 重塑输出为 [-1, 3, 40, 40] 并添加到日志中
    output_reshaped = output.view(-1, 3, 32, 32)
    writer.add_images('output', output_reshaped, step)

    step += 1
    if step == 10:
        break

writer.close()  # 关闭日志

input= torch.randn(1, 3, 32, 32) # print(model(input))
# print(model(input).shape) # torch.Size([1, 3, 32, 32])
print(input)

Image = torchvision.transforms.ToPILImage()(input.squeeze(0)) # print(Image)
Image.show() # 显示图片