# -*- coding: utf-8 -*-
"""
Image Dataset Module
图像数据集模块

This module provides functionality for loading and preprocessing image datasets.
该模块提供加载和预处理图像数据集的功能。
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ImageDataset(Dataset):
    """
    Custom Dataset for Loading Images
    自定义图像数据集类
    
    This class handles loading images from directories, applying transformations,
    and preparing data for training.
    该类处理从目录加载图像，应用变换，并为训练准备数据。
    """
    
    def __init__(self, data_dir: str, transform=None, target_size=(256, 256)):
        """
        Initialize the ImageDataset
        初始化图像数据集
        
        Args:
            data_dir (str): Path to the dataset directory 数据集目录路径
            transform: Transformations to apply to images 要应用到图像的变换
            target_size (tuple): Target size for resizing images 调整图像大小的目标尺寸
        """
        self.data_dir = data_dir
        self.transform = transform
        self.target_size = target_size
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.images = self._load_images()
        
    def _load_images(self):
        """
        Load image paths and labels from directory structure
        从目录结构加载图像路径和标签
        
        Returns:
            list: List of tuples (image_path, label) 图像路径和标签的元组列表
        """
        images = []
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('png', 'jpg', 'jpeg')):
                        img_path = os.path.join(class_dir, img_name)
                        images.append((img_path, self.class_to_idx[class_name]))
        return images

    def __len__(self):
        """
        Get the total number of images in the dataset
        获取数据集中图像的总数
        
        Returns:
            int: Number of images in the dataset 数据集中的图像数量
        """
        return len(self.images)

    def __getitem__(self, idx: int):
        """
        Get a single image and its label by index
        通过索引获取单个图像及其标签
        
        Args:
            idx (int): Index of the image to retrieve 要获取的图像索引
            
        Returns:
            tuple: (image_tensor, label) 图像张量和标签的元组
        """
        img_path, label = self.images[idx]
        try:
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            if img.size != self.target_size:
                img = img.resize(self.target_size)
            
            if self.transform:
                img = self.transform(img)
            else:
                img = transforms.ToTensor()(img)
            
            return img, label
        except Exception as e:
            print(f"⚠️ 加载图像 {img_path} 时出错: {str(e)}")
            print(f"⚠️ Error loading image {img_path}: {str(e)}")
            return torch.zeros(3, *self.target_size), -1