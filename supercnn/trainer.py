# -*- coding: utf-8 -*-
"""
Model Training Module
模型训练模块

This module provides functionality for training neural network models.
该模块提供训练神经网络模型的功能。
"""

import torch
from tqdm import tqdm
import time
import csv

class Trainer:
    """
    Model Trainer Class
    模型训练器类
    
    This class handles the training and evaluation of neural network models.
    该类处理神经网络模型的训练和评估。
    """
    
    def __init__(self, model, optimizer, criterion, device, use_amp, scaler, progress_callback=None):
        """
        Initialize the Trainer
        初始化训练器
        
        Args:
            model: Neural network model 神经网络模型
            optimizer: Optimization algorithm 优化算法
            criterion: Loss function 损失函数
            device: Device to train on 训练设备
            use_amp (bool): Whether to use automatic mixed precision 是否使用自动混合精度
            scaler: Gradient scaler for AMP AMP的梯度缩放器
            progress_callback: Callback function for progress updates 进度更新的回调函数
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.use_amp = use_amp
        self.scaler = scaler
        self.progress_callback = progress_callback

    def train_epoch(self, train_loader, epoch, total_epochs):
        """
        Train the model for one epoch
        训练模型一个周期
        
        Args:
            train_loader: DataLoader for training data 训练数据的DataLoader
            epoch (int): Current epoch number 当前周期数
            total_epochs (int): Total number of epochs 总周期数
            
        Returns:
            tuple: (train_loss, train_acc) 训练损失和准确率
        """
        self.model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        start_time = time.time()
        
        train_iter = tqdm(train_loader, desc=f'Epoch {epoch+1}/{total_epochs} [Train]')
        for batch_idx, (inputs, labels) in enumerate(train_iter):
            inputs = inputs.to(self.device, non_blocking=self.device.type == 'cuda')
            labels = labels.to(self.device, non_blocking=self.device.type == 'cuda')
            
            if self.use_amp:
                # 修改: 使用新的 autocast API
                # Fix: Using the new autocast API
                with torch.amp.autocast(device_type=self.device.type):
                    self.optimizer.zero_grad(set_to_none=True)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.zero_grad(set_to_none=True)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()
            
            train_iter.set_postfix(loss=loss.item())
            
            if callable(self.progress_callback):
                self.progress_callback(
                    phase='train',
                    epoch=epoch+1,
                    total_epochs=total_epochs,
                    batch_loss=loss.item(),
                    batch_acc=100.0 * predicted.eq(labels).sum().item() / labels.size(0),
                    batch_idx=batch_idx,
                    total_batches=len(train_loader)
                )
        
        epoch_time = time.time() - start_time
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct_train / total_train
        
        return train_loss, train_acc

    def evaluate(self, loader, desc: str = 'Evaluate'):
        """
        Evaluate the model on validation/test data
        在验证/测试数据上评估模型
        
        Args:
            loader: DataLoader for evaluation data 评估数据的DataLoader
            desc (str): Description for progress bar 进度条描述
            
        Returns:
            tuple: (eval_loss, eval_acc) 评估损失和准确率
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        with torch.no_grad():
            eval_iter = tqdm(loader, desc=desc)
            for inputs, labels in eval_iter:
                inputs = inputs.to(self.device, non_blocking=self.device.type == 'cuda')
                labels = labels.to(self.device, non_blocking=self.device.type == 'cuda')
                
                if self.use_amp:
                    # 修改: 使用新的 autocast API
                    # Fix: Using the new autocast API
                    with torch.amp.autocast(device_type=self.device.type):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                eval_iter.set_postfix(loss=loss.item(), acc=100.*correct/total)
        
        eval_time = time.time() - start_time
        print(f"{desc} time: {eval_time:.2f} seconds")
                
        return running_loss / len(loader), 100.0 * correct / total

    def save_history_to_csv(self, history, csv_path):
        """
        Save training history to CSV file
        将训练历史保存到CSV文件
        
        Args:
            history (dict): Training history dictionary 训练历史字典
            csv_path (str): Path to save CSV file CSV文件保存路径
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