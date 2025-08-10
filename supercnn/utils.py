# -*- coding: utf-8 -*-
"""
Utility Functions Module
工具函数模块

This module provides various utility functions for data preprocessing and system configuration.
该模块提供各种用于数据预处理和系统配置的工具函数。
"""

import os
from PIL import Image
import platform

def preprocess_dataset(source_dir: str, target_dir: str, image_size: int):
    """
    Preprocess dataset by resizing images and organizing directory structure
    通过调整图像大小和组织目录结构来预处理数据集
    
    Args:
        source_dir (str): Source directory containing raw images 包含原始图像的源目录
        target_dir (str): Target directory for processed images 处理后图像的目标目录
        image_size (int): Target size for resizing images 调整图像大小的目标尺寸
    """
    print(f"🔄 开始预处理数据集: {source_dir} -> {target_dir}")
    print(f"🔄 Starting dataset preprocessing: {source_dir} -> {target_dir}")
    print(f"🖼️ 目标尺寸: {image_size}x{image_size}")
    print(f"🖼️ Target size: {image_size}x{image_size}")
    
    os.makedirs(target_dir, exist_ok=True)
    image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    
    for class_name in os.listdir(source_dir):
        class_dir = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        target_class_dir = os.path.join(target_dir, class_name)
        os.makedirs(target_class_dir, exist_ok=True)
        
        image_count = 0
        for i, filename in enumerate(os.listdir(class_dir), 1):
            if not filename.lower().endswith(image_exts):
                continue
                
            src_path = os.path.join(class_dir, filename)
            ext = os.path.splitext(filename)[1].lower()
            dst_path = os.path.join(target_class_dir, f"{i}{ext}")
            
            try:
                with Image.open(src_path) as img:
                    if img.mode in ('RGBA', 'LA'):
                        bg = Image.new('RGB', img.size, (255, 255, 255))
                        bg.paste(img, mask=img.split()[-1])
                        img = bg
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
                    img.save(dst_path)
                
                image_count += 1
                if image_count % 100 == 0:
                    print(f"📸 已处理 {class_name}: {image_count}张", end='\r')
                    print(f"📸 Processed {class_name}: {image_count} images", end='\r')
                    
            except Exception as e:
                print(f"⚠️ 处理失败: {src_path} - {str(e)}")
                print(f"⚠️ Processing failed: {src_path} - {str(e)}")
                continue
    
        print(f"✅ {class_name} 完成: {image_count}张图片")
        print(f"✅ {class_name} completed: {image_count} images")
    
    print(f"🎉 预处理完成! 所有图片已保存到: {os.path.abspath(target_dir)}")
    print(f"🎉 Preprocessing completed! All images saved to: {os.path.abspath(target_dir)}")
    return target_dir

def get_num_workers(num_workers: int) -> int:
    """
    Determine optimal number of worker processes for data loading
    确定数据加载的最佳工作进程数
    
    Args:
        num_workers (int): Desired number of workers 期望的工作进程数
        
    Returns:
        int: Optimal number of workers 最佳工作进程数
    """
    if num_workers >= 0:
        return num_workers
        
    if platform.system() == 'Windows':
        return min(4, os.cpu_count() // 2) if os.cpu_count() else 0
    else:
        return min(8, os.cpu_count() - 1) if os.cpu_count() else 4