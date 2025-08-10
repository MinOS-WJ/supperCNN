# -*- coding: utf-8 -*-
"""
SupperCNN Package Initialization
SupperCNN 包初始化文件

This file initializes the SupperCNN package and exports all public APIs.
该文件初始化 SupperCNN 包并导出所有公共 API。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os
import csv
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

    # 新增: 集成训练方法，包含数据加载、训练和保存模型的完整流程
    @staticmethod
    def train_model(
        data_dir: str,
        conv_layers: int = 3,
        num_classes: int = 2,
        image_size: int = 64,
        batch_size: int = 32,
        epochs: int = 10,
        lr: float = 0.001,
        val_split: float = 0.2,
        device: torch.device = None,
        use_amp: bool = None,
        save_path: str = None,
        class_names: list = None,
        auto_preprocess: bool = True,
        preprocessed_dir: str = './processed_data'
    ):
        """
        简化版训练方法，集成了完整的训练流程
        Simplified training method with integrated training workflow
        
        Args:
            data_dir: 数据集目录路径
            conv_layers: 卷积层数量
            num_classes: 类别数量
            image_size: 图像尺寸
            batch_size: 批处理大小
            epochs: 训练轮数
            lr: 学习率
            val_split: 验证集比例
            device: 训练设备
            use_amp: 是否使用自动混合精度
            save_path: 模型保存路径
            class_names: 类别名称列表
            auto_preprocess: 是否自动预处理数据
            preprocessed_dir: 预处理后数据保存目录
            
        Returns:
            dict: 训练历史记录
        """
        # 设置默认设备
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        # 自动设置AMP
        if use_amp is None:
            use_amp = device.type == 'cuda'
            
        # 自动预处理数据
        if auto_preprocess:
            data_dir = SupperCNN.preprocess_dataset(data_dir, preprocessed_dir, image_size)
            
        # 创建数据集
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset = SupperCNN.create_dataset(data_dir, transform=transform, target_size=(image_size, image_size))
        
        # 划分训练集和验证集
        train_size = int((1 - val_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 创建模型
        model = SupperCNN.create_model(conv_layers, num_classes, image_size, device)
        
        # 创建训练器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scaler = torch.amp.GradScaler() if use_amp and device.type == 'cuda' else None
        
        trainer = SupperCNN.create_trainer(model, optimizer, criterion, device, use_amp, scaler)
        
        # 训练模型
        history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
        for epoch in range(epochs):
            train_loss, train_acc = trainer.train_epoch(train_loader, epoch, epochs)
            val_loss, val_acc = trainer.evaluate(val_loader, "Validation")
            
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        # 保存模型
        if save_path:
            torch.save({
                'model_state_dict': model.state_dict(),
                'conv_layers': conv_layers,
                'num_classes': num_classes,
                'image_size': image_size,
                'class_names': class_names
            }, save_path)
            print(f"Model saved to {save_path}")
            
        return history

    # 新增: 简化的单张图片预测方法
    @staticmethod
    def predict_single(
        image_path: str,
        model_path: str,
        device: torch.device = None
    ):
        """
        简化的单张图片预测方法
        Simplified single image prediction method
        
        Args:
            image_path: 图片路径
            model_path: 模型路径
            device: 预测设备
            
        Returns:
            tuple: (预测类别, 置信度)
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        # 加载模型
        checkpoint = torch.load(model_path, map_location=device)
        model = SupperCNN.create_model(
            checkpoint['conv_layers'],
            checkpoint['num_classes'],
            checkpoint['image_size'],
            device
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 创建预测器
        class_names = checkpoint.get('class_names', [f"Class {i}" for i in range(checkpoint['num_classes'])])
        predictor = SupperCNN.create_predictor(
            model, 
            class_names, 
            device, 
            checkpoint['image_size'], 
            device.type == 'cuda'
        )
        
        # 预测
        return predictor.predict(image_path)

    # 新增: 简化的批量预测方法
    @staticmethod
    def predict_batch(
        image_dir: str,
        model_path: str,
        batch_size: int = 32,
        device: torch.device = None
    ):
        """
        简化的批量预测方法
        Simplified batch prediction method
        
        Args:
            image_dir: 图片目录
            model_path: 模型路径
            batch_size: 批处理大小
            device: 预测设备
            
        Returns:
            list: 预测结果列表
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        # 加载模型
        checkpoint = torch.load(model_path, map_location=device)
        model = SupperCNN.create_model(
            checkpoint['conv_layers'],
            checkpoint['num_classes'],
            checkpoint['image_size'],
            device
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 创建预测器
        class_names = checkpoint.get('class_names', [f"Class {i}" for i in range(checkpoint['num_classes'])])
        predictor = SupperCNN.create_predictor(
            model, 
            class_names, 
            device, 
            checkpoint['image_size'], 
            device.type == 'cuda'
        )
        
        # 批量预测
        return predictor.predict_batch(image_dir, batch_size)

    # 新增: 将训练历史保存到CSV的方法
    @staticmethod
    def save_history_to_csv(history: dict, csv_path: str):
        """
        将训练历史保存到CSV文件
        Save training history to CSV file
        
        Args:
            history: 训练历史字典
            csv_path: CSV文件路径
        """
        if not history or 'train_acc' not in history or 'val_acc' not in history:
            raise ValueError("Invalid history data for saving to CSV")
        
        fieldnames = ['epoch', 'train_accuracy', 'val_accuracy', 'train_loss', 'val_loss']
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            num_epochs = len(history['train_acc'])
            for i in range(num_epochs):
                row = {
                    'epoch': i + 1,
                    'train_accuracy': history['train_acc'][i],
                    'val_accuracy': history['val_acc'][i]
                }
                
                # 如果有损失值也保存
                if 'train_loss' in history and i < len(history['train_loss']):
                    row['train_loss'] = history['train_loss'][i]
                if 'val_loss' in history and i < len(history['val_loss']):
                    row['val_loss'] = history['val_loss'][i]
                    
                writer.writerow(row)


__all__ = ['ImageDataset', 'create_model', 'Trainer', 'Predictor', 'preprocess_dataset', 'SupperCNN']