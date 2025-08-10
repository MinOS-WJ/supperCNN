# -*- coding: utf-8 -*-
"""
Model Creation Module
模型创建模块

This module provides functions for creating convolutional neural network models.
该模块提供创建卷积神经网络模型的函数。
"""

import torch
import torch.nn as nn
import warnings
import math

def conv_block(in_channels, out_channels, use_pool=True):
    """
    Create a convolutional block with conv + batchnorm + relu (+ optional pooling)
    创建包含卷积+批归一化+ReLU(+可选池化)的卷积块
    
    Args:
        in_channels (int): Number of input channels 输入通道数
        out_channels (int): Number of output channels 输出通道数
        use_pool (bool): Whether to include max pooling 是否包含最大池化
        
    Returns:
        nn.Sequential: Convolutional block 卷积块
    """
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if use_pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

def create_model(conv_layers: int, num_classes: int, image_size: int, device: torch.device) -> nn.Module:
    """
    Create a CNN model with specified number of convolutional layers
    创建具有指定卷积层数的CNN模型
    
    Args:
        conv_layers (int): Number of convolutional layers 卷积层数量
        num_classes (int): Number of output classes 输出类别数
        image_size (int): Input image size 输入图像尺寸
        device (torch.device): Device to put the model on 模型所在设备
        
    Returns:
        nn.Module: Created CNN model 创建的CNN模型
    """
    channels = [3]
    for i in range(conv_layers):
        channels.append(min(512, 32 * (2 ** i)))
    
    layers = []
    current_size = image_size
    
    for i in range(conv_layers):
        use_pool = (current_size > 4)
        layers.append(conv_block(channels[i], channels[i+1], use_pool))
        
        if use_pool:
            current_size //= 2
            if current_size < 4:
                warnings.warn(f"Feature map size reduced to {current_size}x{current_size} after {i+1} conv layers")

    layers.extend([
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(channels[-1], 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    ])
    
    return nn.Sequential(*layers).to(device)
