#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :nn_torch_relu.py
# @Time :2024/7/26 20:28
# @Author :张文军
import torch
from torch.nn import Sigmoid, ReLU

relu = ReLU()
input = torch.randn(2, 2)

print(input)
print(relu(input))

output = torch.nn.functional.relu(input)
print(output)

# torch.nn.functional.relu(input) 和 ReLU() 的区别在于前者是函数，后者是类
# 函数可以接受任何输入，而类只能接受特定类型的输入
# 函数可以返回任何输出，而类只能返回特定类型的输出

# 所以，在大多数情况下，我们使用函数而不是类
# 但是，在某些情况下，我们可能需要使用类，例如当我们需要使用多个层时
# 在这种情况下，我们可以使用类来创建一个包含多个层的模型
# 例如，我们可以使用 nn.Sequential() 类来创建一个包含多个层的模型

import torch.nn.functional as F
output = F.sigmoid(input)
print(output)




