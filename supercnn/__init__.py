# -*- coding: utf-8 -*-
"""
SupperCNN Package Initialization
SupperCNN 包初始化文件

This file initializes the SupperCNN package and exports all public APIs.
该文件初始化 SupperCNN 包并导出所有公共 API。
"""

import torch
import torch.nn as nn
from .dataset import ImageDataset
from .model import create_model
from .trainer import Trainer
from .predictor import Predictor
from .utils import preprocess_dataset


# 创建统一对外调用接口
class SupperCNN:
    """
    SupperCNN统一接口类
    这个类提供了一个统一的接口来使用supercnn包的所有功能
    """
    
    def __init__(self):
        pass
    
    @staticmethod
    def create_model(conv_layers: int, num_classes: int, image_size: int, device: torch.device) -> nn.Module:
        return create_model(conv_layers, num_classes, image_size, device)
        
    @staticmethod
    def create_dataset(data_dir: str, transform=None, target_size=(256, 256)):
        return ImageDataset(data_dir, transform, target_size)
        
    @staticmethod
    def create_trainer(model, optimizer, criterion, device, use_amp, scaler, progress_callback=None):
        return Trainer(model, optimizer, criterion, device, use_amp, scaler, progress_callback)
        
    @staticmethod
    def create_predictor(model, class_names, device, image_size, use_amp):
        return Predictor(model, class_names, device, image_size, use_amp)
        
    @staticmethod
    def preprocess_dataset(source_dir: str, target_dir: str, image_size: int):
        return preprocess_dataset(source_dir, target_dir, image_size)


# 添加GUI控制器到导出列表
from .gui import TrainingControllerGUI, run_gui

__all__ = ['ImageDataset', 'create_model', 'Trainer', 'Predictor', 'preprocess_dataset', 
           'SupperCNN', 'TrainingControllerGUI', 'run_gui']