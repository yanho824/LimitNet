"""
临时LimitNet类定义，仅用于加载预训练模型的权重
这个文件解决了"No module named 'models'"的问题
"""

import torch
import torch.nn as nn


class LimitNet(nn.Module):
    """
    临时LimitNet类，用于加载原始预训练模型
    这个类只需要有基本结构，我们只关心权重
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        # 创建一些基本的占位符组件
        # 具体的层会在加载权重时自动创建
        pass
    
    def forward(self, x):
        # 这个方法不会被调用，只是为了满足nn.Module的要求
        return x


# 可能还需要的其他类
class Encoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def forward(self, x):
        return x


class Decoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def forward(self, x):
        return x


class SaliencyDecoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def forward(self, x):
        return x


class Classifier(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def forward(self, x):
        return x 