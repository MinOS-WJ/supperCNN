# -*- coding: utf-8 -*-
"""
Model Prediction Module
模型预测模块

This module provides functionality for making predictions with trained models.
该模块提供使用训练好的模型进行预测的功能。
"""

import torch
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
from typing import List, Tuple

class Predictor:
    """
    Model Predictor Class
    模型预测器类
    
    This class handles making predictions on single images or batches of images.
    该类处理对单个图像或批量图像进行预测。
    """
    
    def __init__(self, model, class_names, device, image_size, use_amp):
        """
        Initialize the Predictor
        初始化预测器
        
        Args:
            model: Trained model 训练好的模型
            class_names (list): List of class names 类别名称列表
            device: Device to run predictions on 运行预测的设备
            image_size (int): Image size for resizing 图像尺寸
            use_amp (bool): Whether to use automatic mixed precision 是否使用自动混合精度
        """
        self.model = model
        self.class_names = class_names
        self.device = device
        self.image_size = image_size
        self.use_amp = use_amp

    def predict(self, image_path: str) -> Tuple[str, float]:
        """
        Predict the class of a single image
        预测单个图像的类别
        
        Args:
            image_path (str): Path to the image to predict 图像路径
            
        Returns:
            tuple: (predicted_class, confidence) 预测类别和置信度
        """
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(self.device, non_blocking=self.device.type == 'cuda')
        
        self.model.eval()
        with torch.no_grad():
            if self.use_amp:
                # 修改: 使用新的 autocast API
                # Fix: Using the new autocast API
                with torch.amp.autocast(device_type=self.device.type):
                    output = self.model(image)
                    probabilities = torch.nn.functional.softmax(output, dim=1)
            else:
                output = self.model(image)
                probabilities = torch.nn.functional.softmax(output, dim=1)
            
            confidence, predicted_idx = torch.max(probabilities, 1)
            
        return self.class_names[predicted_idx.item()], confidence.item()

    def predict_batch(self, image_dir: str, batch_size: int) -> List[Tuple[str, str, float]]:
        """
        Predict classes for a batch of images in a directory
        预测目录中批量图像的类别
        
        Args:
            image_dir (str): Directory containing images to predict 包含待预测图像的目录
            batch_size (int): Batch size for processing 批处理大小
            
        Returns:
            list: List of tuples (image_name, predicted_class, confidence) 图像名称、预测类别和置信度的元组列表
        """
        results = []
        image_files = [f for f in os.listdir(image_dir) 
                      if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
        batch_size = min(32, batch_size * 2)
        batches = [image_files[i:i + batch_size] 
                  for i in range(0, len(image_files), batch_size)]
        
        self.model.eval()
        with torch.no_grad():
            for batch_files in tqdm(batches, desc="批量预测"):
                images = []
                valid_files = []
                
                for img_name in batch_files:
                    img_path = os.path.join(image_dir, img_name)
                    try:
                        image = Image.open(img_path).convert('RGB')
                        image = transform(image)
                        images.append(image)
                        valid_files.append(img_name)
                    except Exception as e:
                        print(f"⚠️ 处理图像 {img_name} 时出错: {str(e)}")
                        print(f"⚠️ Error processing image {img_name}: {str(e)}")
                        results.append((img_name, "ERROR", 0.0))
                
                if not images:
                    continue
                    
                image_batch = torch.stack(images).to(self.device, non_blocking=self.device.type == 'cuda')
                
                if self.use_amp:
                    # 修改: 使用新的 autocast API
                    # Fix: Using the new autocast API
                    with torch.amp.autocast(device_type=self.device.type):
                        outputs = self.model(image_batch)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                else:
                    outputs = self.model(image_batch)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                confidences, predicted_idxs = torch.max(probabilities, 1)
                
                for i in range(len(valid_files)):
                    results.append((
                        valid_files[i],
                        self.class_names[predicted_idxs[i].item()],
                        confidences[i].item()
                    ))
        
        return results