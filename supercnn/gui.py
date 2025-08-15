# -*- coding: utf-8 -*-
"""
GUI Training Controller Module
图形化训练控制器模块

This module provides a graphical interface for controlling and visualizing 
the training process of SupperCNN models.
该模块提供图形界面来控制和可视化SupperCNN模型的训练过程。
"""

import sys
import os
import threading
import time
from typing import Optional, Dict, List
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

# 从项目模块导入所需组件
from . import SupperCNN
from .trainer import Trainer
from .predictor import Predictor


class TrainingControllerGUI:
    """
    Graphical Training Controller for SupperCNN
    SupperCNN图形化训练控制器
    
    This class provides a GUI interface for training SupperCNN models with 
    real-time visualization of training metrics.
    该类为训练SupperCNN模型提供图形界面，并实时可视化训练指标。
    """
    
    def __init__(self, root: tk.Tk):
        """
        Initialize the Training Controller GUI
        初始化训练控制器GUI
        
        Args:
            root (tk.Tk): Root tkinter window 根tkinter窗口
        """
        self.root = root
        self.root.title("SupperCNN Training Controller - SupperCNN训练控制器")
        self.root.geometry("1200x800")
        
        # Training state variables
        # 训练状态变量
        self.is_training = False
        self.training_thread: Optional[threading.Thread] = None
        self.model: Optional[nn.Module] = None
        self.trainer: Optional[Trainer] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.device = torch.device('cpu')
        self.history: Dict[str, List[float]] = {
            'train_acc': [], 
            'val_acc': [], 
            'train_loss': [], 
            'val_loss': []
        }
        
        # Best model tracking
        # 最佳模型跟踪
        self.best_val_acc = 0.0
        self.best_model_state = None
        
        # GUI variables
        # GUI变量
        self.dataset_path = tk.StringVar()
        self.model_save_path = tk.StringVar()
        self.conv_layers = tk.IntVar(value=3)
        self.num_classes = tk.IntVar(value=2)
        self.image_size = tk.IntVar(value=64)
        self.batch_size = tk.IntVar(value=32)
        self.epochs = tk.IntVar(value=10)
        self.learning_rate = tk.DoubleVar(value=0.001)
        self.use_amp = tk.BooleanVar(value=False)
        self.use_gpu = tk.BooleanVar(value=torch.cuda.is_available())
        
        # Training strategy variables
        # 训练策略变量
        self.optimizer_var = tk.StringVar(value="Adam")
        self.scheduler_var = tk.StringVar(value="None")
        self.dropout_rate = tk.DoubleVar(value=0.5)
        self.weight_decay = tk.DoubleVar(value=0.0)
        self.momentum = tk.DoubleVar(value=0.9)
        self.step_size = tk.IntVar(value=30)
        self.gamma = tk.DoubleVar(value=0.1)
        self.patience = tk.IntVar(value=5)
        
        # Prediction variables
        # 预测变量
        self.model_load_path = tk.StringVar()
        self.image_predict_path = tk.StringVar()
        self.predict_result = tk.StringVar(value="预测结果将显示在这里")
        
        # Create GUI
        # 创建GUI
        self._create_widgets()
        
        # Initialize plot
        # 初始化图表
        self._init_plot()
        
    def _create_widgets(self):
        """Create all GUI widgets 创建所有GUI控件"""
        # Main frame
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Configuration notebook for better organization
        # 配置笔记本以更好地组织界面
        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Basic configuration frame
        # 基本配置框架
        basic_frame = ttk.Frame(notebook, padding="10")
        notebook.add(basic_frame, text="Basic Configuration - 基本配置")
        
        # Dataset path
        # 数据集路径
        ttk.Label(basic_frame, text="Dataset Path - 数据集路径:").grid(row=0, column=0, sticky=tk.W, pady=2)
        dataset_frame = ttk.Frame(basic_frame)
        dataset_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2)
        dataset_frame.columnconfigure(0, weight=1)
        ttk.Entry(dataset_frame, textvariable=self.dataset_path).grid(row=0, column=0, sticky=(tk.W, tk.E))
        ttk.Button(dataset_frame, text="Browse - 浏览", command=self._browse_dataset).grid(row=0, column=1, padx=(5, 0))
        
        # Model save path
        # 模型保存路径
        ttk.Label(basic_frame, text="Model Save Path - 模型保存路径:").grid(row=1, column=0, sticky=tk.W, pady=2)
        save_frame = ttk.Frame(basic_frame)
        save_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=2)
        save_frame.columnconfigure(0, weight=1)
        ttk.Entry(save_frame, textvariable=self.model_save_path).grid(row=0, column=0, sticky=(tk.W, tk.E))
        ttk.Button(save_frame, text="Browse - 浏览", command=self._browse_save_path).grid(row=0, column=1, padx=(5, 0))
        
        # Model parameters
        # 模型参数
        params_frame = ttk.LabelFrame(basic_frame, text="Model Parameters - 模型参数", padding="10")
        params_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        params_frame.columnconfigure(1, weight=1)
        params_frame.columnconfigure(3, weight=1)
        params_frame.columnconfigure(5, weight=1)
        
        # Conv layers
        # 卷积层数
        ttk.Label(params_frame, text="Conv Layers - 卷积层数:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Spinbox(params_frame, from_=1, to=10, textvariable=self.conv_layers, width=5).grid(row=0, column=1, sticky=tk.W, padx=(0, 10))
        
        # Number of classes
        # 类别数
        ttk.Label(params_frame, text="Classes - 类别数:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        ttk.Spinbox(params_frame, from_=2, to=1000, textvariable=self.num_classes, width=5).grid(row=0, column=3, sticky=tk.W, padx=(0, 10))
        
        # Image size
        # 图像尺寸
        ttk.Label(params_frame, text="Image Size - 图像尺寸:").grid(row=0, column=4, sticky=tk.W, padx=(0, 5))
        ttk.Spinbox(params_frame, from_=32, to=512, increment=32, textvariable=self.image_size, width=5).grid(row=0, column=5, sticky=tk.W, padx=(0, 10))
        
        # Batch size
        # 批处理大小
        ttk.Label(params_frame, text="Batch Size - 批处理大小:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0), padx=(0, 5))
        ttk.Spinbox(params_frame, from_=1, to=512, textvariable=self.batch_size, width=5).grid(row=1, column=1, sticky=tk.W, pady=(5, 0), padx=(0, 10))
        
        # Epochs
        # 训练轮数
        ttk.Label(params_frame, text="Epochs - 训练轮数:").grid(row=1, column=2, sticky=tk.W, pady=(5, 0), padx=(0, 5))
        ttk.Spinbox(params_frame, from_=1, to=1000, textvariable=self.epochs, width=5).grid(row=1, column=3, sticky=tk.W, pady=(5, 0), padx=(0, 10))
        
        # Learning rate
        # 学习率
        ttk.Label(params_frame, text="Learning Rate - 学习率:").grid(row=1, column=4, sticky=tk.W, pady=(5, 0), padx=(0, 5))
        ttk.Spinbox(params_frame, from_=0.00001, to=1.0, increment=0.0001, textvariable=self.learning_rate, width=8).grid(row=1, column=5, sticky=tk.W, pady=(5, 0))
        
        # Options
        # 选项
        options_frame = ttk.LabelFrame(basic_frame, text="Options - 选项", padding="10")
        options_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        ttk.Checkbutton(options_frame, text="Use GPU - 使用GPU", variable=self.use_gpu).grid(row=0, column=0, sticky=tk.W)
        ttk.Checkbutton(options_frame, text="Use AMP - 使用自动混合精度", variable=self.use_amp).grid(row=0, column=1, sticky=tk.W, padx=(20, 0))
        
        # Training strategies frame
        # 训练策略框架
        strategy_frame = ttk.Frame(notebook, padding="10")
        notebook.add(strategy_frame, text="Training Strategies - 训练策略")
        
        # Optimizer settings
        # 优化器设置
        optimizer_frame = ttk.LabelFrame(strategy_frame, text="Optimizer Settings - 优化器设置", padding="10")
        optimizer_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        optimizer_frame.columnconfigure(1, weight=1)
        
        ttk.Label(optimizer_frame, text="Optimizer - 优化器:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        optimizer_combo = ttk.Combobox(optimizer_frame, textvariable=self.optimizer_var, 
                                      values=["Adam", "SGD", "AdamW"], width=10, state="readonly")
        optimizer_combo.grid(row=0, column=1, sticky=tk.W, padx=(0, 10))
        
        ttk.Label(optimizer_frame, text="Weight Decay - 权重衰减:").grid(row=0, column=2, sticky=tk.W, padx=(10, 5))
        ttk.Spinbox(optimizer_frame, from_=0.0, to=0.1, increment=0.0001, textvariable=self.weight_decay, width=8).grid(
            row=0, column=3, sticky=tk.W)
            
        # SGD specific settings
        # SGD特定设置
        ttk.Label(optimizer_frame, text="Momentum (for SGD) - 动量(用于SGD):").grid(row=1, column=0, sticky=tk.W, pady=(5, 0), padx=(0, 5))
        ttk.Spinbox(optimizer_frame, from_=0.0, to=1.0, increment=0.01, textvariable=self.momentum, width=8).grid(
            row=1, column=1, sticky=tk.W, pady=(5, 0))
        
        # Scheduler settings
        # 调度器设置
        scheduler_frame = ttk.LabelFrame(strategy_frame, text="Scheduler Settings - 调度器设置", padding="10")
        scheduler_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        scheduler_frame.columnconfigure(1, weight=1)
        
        ttk.Label(scheduler_frame, text="Scheduler - 调度器:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        scheduler_combo = ttk.Combobox(scheduler_frame, textvariable=self.scheduler_var,
                                      values=["None", "StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"], 
                                      width=15, state="readonly")
        scheduler_combo.grid(row=0, column=1, sticky=tk.W, padx=(0, 10))
        
        # StepLR parameters
        # StepLR参数
        ttk.Label(scheduler_frame, text="Step Size (for StepLR) - 步长(用于StepLR):").grid(row=1, column=0, sticky=tk.W, pady=(5, 0), padx=(0, 5))
        ttk.Spinbox(scheduler_frame, from_=1, to=100, textvariable=self.step_size, width=5).grid(
            row=1, column=1, sticky=tk.W, pady=(5, 0), padx=(0, 10))
            
        ttk.Label(scheduler_frame, text="Gamma (for StepLR) - 衰减因子(用于StepLR):").grid(row=1, column=2, sticky=tk.W, pady=(5, 0), padx=(0, 5))
        ttk.Spinbox(scheduler_frame, from_=0.01, to=1.0, increment=0.01, textvariable=self.gamma, width=5).grid(
            row=1, column=3, sticky=tk.W, pady=(5, 0))
            
        # ReduceLROnPlateau parameters
        # ReduceLROnPlateau参数
        ttk.Label(scheduler_frame, text="Patience (for ReduceLROnPlateau) - 耐心值(用于ReduceLROnPlateau):").grid(
            row=2, column=0, sticky=tk.W, pady=(5, 0), padx=(0, 5))
        ttk.Spinbox(scheduler_frame, from_=1, to=50, textvariable=self.patience, width=5).grid(
            row=2, column=1, sticky=tk.W, pady=(5, 0))
        
        # Regularization settings
        # 正则化设置
        regularization_frame = ttk.LabelFrame(strategy_frame, text="Regularization Settings - 正则化设置", padding="10")
        regularization_frame.grid(row=2, column=0, sticky=(tk.W, tk.E))
        regularization_frame.columnconfigure(1, weight=1)
        
        ttk.Label(regularization_frame, text="Dropout Rate - Dropout率:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Spinbox(regularization_frame, from_=0.0, to=0.9, increment=0.1, textvariable=self.dropout_rate, width=5).grid(
            row=0, column=1, sticky=tk.W)
        
        # Prediction frame
        # 预测框架
        predict_frame = ttk.Frame(notebook, padding="10")
        notebook.add(predict_frame, text="Prediction - 预测")
        
        # Model load path
        # 模型加载路径
        ttk.Label(predict_frame, text="Model Path - 模型路径:").grid(row=0, column=0, sticky=tk.W, pady=2)
        model_frame = ttk.Frame(predict_frame)
        model_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2)
        model_frame.columnconfigure(0, weight=1)
        ttk.Entry(model_frame, textvariable=self.model_load_path).grid(row=0, column=0, sticky=(tk.W, tk.E))
        ttk.Button(model_frame, text="Browse - 浏览", command=self._browse_model_path).grid(row=0, column=1, padx=(5, 0))
        
        # Image predict path
        # 图像预测路径
        ttk.Label(predict_frame, text="Image Path - 图像路径:").grid(row=1, column=0, sticky=tk.W, pady=2)
        image_frame = ttk.Frame(predict_frame)
        image_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=2)
        image_frame.columnconfigure(0, weight=1)
        ttk.Entry(image_frame, textvariable=self.image_predict_path).grid(row=0, column=0, sticky=(tk.W, tk.E))
        ttk.Button(image_frame, text="Browse - 浏览", command=self._browse_image_path).grid(row=0, column=1, padx=(5, 0))
        
        # Predict button
        # 预测按钮
        ttk.Button(predict_frame, text="Predict - 预测", command=self._predict_image).grid(row=2, column=0, columnspan=2, pady=10)
        
        # Predict result
        # 预测结果
        result_frame = ttk.LabelFrame(predict_frame, text="Prediction Result - 预测结果", padding="10")
        result_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        result_frame.columnconfigure(0, weight=1)
        ttk.Label(result_frame, textvariable=self.predict_result, font=("Arial", 12)).grid(row=0, column=0)
        
        # Control buttons
        # 控制按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, columnspan=2, pady=(0, 10))
        
        self.start_button = ttk.Button(button_frame, text="Start Training - 开始训练", command=self._start_training)
        self.start_button.grid(row=0, column=0, padx=(0, 10))
        
        self.stop_button = ttk.Button(button_frame, text="Stop Training - 停止训练", command=self._stop_training, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=(0, 10))
        
        self.save_button = ttk.Button(button_frame, text="Save Model - 保存模型", command=self._save_model, state=tk.DISABLED)
        self.save_button.grid(row=0, column=2, padx=(0, 10))
        
        # Progress info
        # 进度信息
        self.progress_var = tk.StringVar(value="Ready - 就绪")
        self.progress_label = ttk.Label(button_frame, textvariable=self.progress_var)
        self.progress_label.grid(row=0, column=3, padx=(20, 0))
        
        # Visualization frame
        # 可视化框架
        viz_frame = ttk.LabelFrame(main_frame, text="Training Visualization - 训练可视化", padding="10")
        viz_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        viz_frame.columnconfigure(0, weight=1)
        viz_frame.rowconfigure(0, weight=1)
        
        # Create matplotlib figure
        # 创建matplotlib图表
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 4))
        self.fig.tight_layout(pad=3.0)
        
        # Embed figure in tkinter
        # 在tkinter中嵌入图表
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Console output
        # 控制台输出
        console_frame = ttk.LabelFrame(main_frame, text="Training Log - 训练日志", padding="10")
        console_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        console_frame.columnconfigure(0, weight=1)
        console_frame.rowconfigure(0, weight=1)
        
        self.console_text = tk.Text(console_frame, height=8, state=tk.DISABLED)
        console_scroll = ttk.Scrollbar(console_frame, orient=tk.VERTICAL, command=self.console_text.yview)
        self.console_text.configure(yscrollcommand=console_scroll.set)
        
        self.console_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        console_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
    def _init_plot(self):
        """Initialize the training visualization plots 初始化训练可视化图表"""
        self.ax1.clear()
        self.ax1.set_title("Accuracy - 准确率")
        self.ax1.set_xlabel("Epoch - 训练轮数")
        self.ax1.set_ylabel("Accuracy (%) - 准确率 (%)")
        self.ax1.grid(True)
        
        self.ax2.clear()
        self.ax2.set_title("Loss - 损失")
        self.ax2.set_xlabel("Epoch - 训练轮数")
        self.ax2.set_ylabel("Loss - 损失值")
        self.ax2.grid(True)
        
        self.canvas.draw()
        
    def _browse_dataset(self):
        """Open file dialog to select dataset directory 打开文件对话框选择数据集目录"""
        path = filedialog.askdirectory()
        if path:
            self.dataset_path.set(path)
            
    def _browse_save_path(self):
        """Open file dialog to select model save path 打开文件对话框选择模型保存路径"""
        path = filedialog.askdirectory()
        if path:
            self.model_save_path.set(path)
            
    def _browse_model_path(self):
        """Open file dialog to select model file 打开文件对话框选择模型文件"""
        path = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pth")])
        if path:
            self.model_load_path.set(path)
            
    def _browse_image_path(self):
        """Open file dialog to select image file 打开文件对话框选择图像文件"""
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")])
        if path:
            self.image_predict_path.set(path)
            
    def _log_message(self, message: str):
        """Add message to console log 向控制台日志添加消息"""
        self.console_text.config(state=tk.NORMAL)
        self.console_text.insert(tk.END, f"{message}\n")
        self.console_text.see(tk.END)
        self.console_text.config(state=tk.DISABLED)
        self.root.update_idletasks()
        
    def _update_plot(self):
        """Update training visualization plots 更新训练可视化图表"""
        epochs = list(range(1, len(self.history['train_acc']) + 1))
        
        # Update accuracy plot
        # 更新准确率图表
        self.ax1.clear()
        self.ax1.plot(epochs, self.history['train_acc'], 'b-', label='Train Accuracy - 训练准确率')
        self.ax1.plot(epochs, self.history['val_acc'], 'r-', label='Validation Accuracy - 验证准确率')
        self.ax1.set_title("Accuracy - 准确率")
        self.ax1.set_xlabel("Epoch - 训练轮数")
        self.ax1.set_ylabel("Accuracy (%) - 准确率 (%)")
        self.ax1.legend()
        self.ax1.grid(True)
        
        # 添加数值标注
        if self.history['train_acc']:
            for i, acc in enumerate(self.history['train_acc']):
                self.ax1.annotate(f'{acc:.1f}', (epochs[i], acc), textcoords="offset points", 
                                 xytext=(0,10), ha='center', fontsize=8, color='blue')
            for i, acc in enumerate(self.history['val_acc']):
                self.ax1.annotate(f'{acc:.1f}', (epochs[i], acc), textcoords="offset points", 
                                 xytext=(0,-15), ha='center', fontsize=8, color='red')
        
        # Update loss plot
        # 更新损失图表
        self.ax2.clear()
        self.ax2.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss - 训练损失')
        self.ax2.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss - 验证损失')
        self.ax2.set_title("Loss - 损失")
        self.ax2.set_xlabel("Epoch - 训练轮数")
        self.ax2.set_ylabel("Loss - 损失值")
        self.ax2.legend()
        self.ax2.grid(True)
        
        # 添加数值标注
        if self.history['train_loss']:
            for i, loss in enumerate(self.history['train_loss']):
                self.ax2.annotate(f'{loss:.2f}', (epochs[i], loss), textcoords="offset points", 
                                 xytext=(0,10), ha='center', fontsize=8, color='blue')
            for i, loss in enumerate(self.history['val_loss']):
                self.ax2.annotate(f'{loss:.2f}', (epochs[i], loss), textcoords="offset points", 
                                 xytext=(0,-15), ha='center', fontsize=8, color='red')
        
        self.canvas.draw()
        
    def _progress_callback(self, phase: str, epoch: int, total_epochs: int, 
                          batch_loss: float, batch_acc: float, 
                          batch_idx: int, total_batches: int):
        """Callback function for training progress updates 训练进度更新的回调函数"""
        if phase == 'train':
            self.progress_var.set(f"Epoch {epoch}/{total_epochs} - Batch {batch_idx+1}/{total_batches} - "
                                 f"Loss: {batch_loss:.4f} - Acc: {batch_acc:.2f}%")
            
    def _setup_training(self):
        """Setup training environment and data loaders 设置训练环境和数据加载器"""
        # Check if dataset path is provided
        # 检查是否提供了数据集路径
        if not self.dataset_path.get():
            messagebox.showerror("Error - 错误", "Please select a dataset path - 请选择数据集路径")
            return False
            
        if not os.path.exists(self.dataset_path.get()):
            messagebox.showerror("Error - 错误", "Dataset path does not exist - 数据集路径不存在")
            return False
            
        # Set device
        # 设置设备
        if self.use_gpu.get() and torch.cuda.is_available():
            self.device = torch.device('cuda')
            self._log_message("Using GPU for training - 使用GPU进行训练")
        else:
            self.device = torch.device('cpu')
            self._log_message("Using CPU for training - 使用CPU进行训练")
            
        # Auto disable AMP on CPU
        # 在CPU上自动禁用AMP
        if self.use_amp.get() and self.device.type == 'cpu':
            self.use_amp.set(False)
            self._log_message("AMP disabled on CPU - 在CPU上禁用AMP")
            
        try:
            # Create dataset and data loaders
            # 创建数据集和数据加载器
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            dataset = SupperCNN.create_dataset(
                self.dataset_path.get(), 
                transform=transform, 
                target_size=(self.image_size.get(), self.image_size.get())
            )
            
            # Split dataset
            # 分割数据集
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            
            self.train_loader = DataLoader(
                train_dataset, 
                batch_size=self.batch_size.get(), 
                shuffle=True
            )
            self.val_loader = DataLoader(
                val_dataset, 
                batch_size=self.batch_size.get(), 
                shuffle=False
            )
            
            # Create model
            # 创建模型
            self.model = SupperCNN.create_model(
                self.conv_layers.get(),
                self.num_classes.get(),
                self.image_size.get(),
                self.device
            )
            
            # Create optimizer based on selection
            # 根据选择创建优化器
            if self.optimizer_var.get() == "Adam":
                optimizer = optim.Adam(self.model.parameters(), 
                                     lr=self.learning_rate.get(),
                                     weight_decay=self.weight_decay.get())
            elif self.optimizer_var.get() == "SGD":
                optimizer = optim.SGD(self.model.parameters(), 
                                    lr=self.learning_rate.get(),
                                    momentum=self.momentum.get(),
                                    weight_decay=self.weight_decay.get())
            elif self.optimizer_var.get() == "AdamW":
                optimizer = optim.AdamW(self.model.parameters(), 
                                      lr=self.learning_rate.get(),
                                      weight_decay=self.weight_decay.get())
            
            # Create scheduler if selected
            # 如果选择了调度器则创建
            scheduler = None
            if self.scheduler_var.get() == "StepLR":
                scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                                    step_size=self.step_size.get(), 
                                                    gamma=self.gamma.get())
            elif self.scheduler_var.get() == "CosineAnnealingLR":
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                               T_max=self.epochs.get())
            elif self.scheduler_var.get() == "ReduceLROnPlateau":
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                               mode='min', 
                                                               patience=self.patience.get())
            
            # Create trainer
            # 创建训练器
            criterion = nn.CrossEntropyLoss()
            scaler = torch.amp.GradScaler() if self.use_amp.get() and self.device.type == 'cuda' else None
            
            self.trainer = SupperCNN.create_trainer(
                self.model, 
                optimizer, 
                criterion, 
                self.device, 
                self.use_amp.get(), 
                scaler,
                progress_callback=self._progress_callback
            )
            
            # Store scheduler for use during training
            # 保存调度器以在训练中使用
            self.scheduler = scheduler
            
            # Initialize history and best model tracking
            # 初始化历史记录和最佳模型跟踪
            self.history = {
                'train_acc': [], 
                'val_acc': [], 
                'train_loss': [], 
                'val_loss': []
            }
            self.best_val_acc = 0.0
            self.best_model_state = None
            
            self._log_message("Training setup completed - 训练设置完成")
            return True
            
        except Exception as e:
            messagebox.showerror("Error - 错误", f"Failed to setup training: {str(e)}")
            self._log_message(f"Training setup failed: {str(e)}")
            return False
            
    def _training_loop(self):
        """Main training loop 主训练循环"""
        try:
            total_epochs = self.epochs.get()
            
            for epoch in range(total_epochs):
                if not self.is_training:
                    break
                    
                try:
                    # Train one epoch
                    # 训练一个周期
                    train_loss, train_acc = self.trainer.train_epoch(
                        self.train_loader, 
                        epoch, 
                        total_epochs
                    )
                    
                    # Evaluate on validation set
                    # 在验证集上评估
                    val_loss, val_acc = self.trainer.evaluate(
                        self.val_loader, 
                        "Validation - 验证"
                    )
                    
                    # Update scheduler if available
                    # 如果有调度器则更新
                    if self.scheduler:
                        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                            self.scheduler.step(val_loss)
                        else:
                            self.scheduler.step()
                    
                    # Track best model
                    # 跟踪最佳模型
                    if val_acc > self.best_val_acc:
                        self.best_val_acc = val_acc
                        self.best_model_state = self.model.state_dict().copy()
                        self.root.after(0, lambda: self._log_message(
                            f"🎉 New best model saved with validation accuracy: {val_acc:.2f}%"
                        ))
                    
                    # Update history
                    # 更新历史记录
                    self.history['train_acc'].append(train_acc)
                    self.history['val_acc'].append(val_acc)
                    self.history['train_loss'].append(train_loss)
                    self.history['val_loss'].append(val_loss)
                    
                    # Update plot
                    # 更新图表
                    self.root.after(0, self._update_plot)
                    
                    # Log epoch results
                    # 记录周期结果
                    self.root.after(0, lambda: self._log_message(
                        f"Epoch {epoch+1}/{total_epochs} - "
                        f"Train Acc: {train_acc:.2f}% Loss: {train_loss:.4f} - "
                        f"Val Acc: {val_acc:.2f}% Loss: {val_loss:.4f}"
                    ))
                except Exception as e:
                    self.root.after(0, lambda: self._log_message(
                        f"⚠️ Error in epoch {epoch+1}: {str(e)}"
                    ))
                    continue
                
            # Training completed
            # 训练完成
            if self.is_training:
                self.root.after(0, lambda: self._training_completed())
                
        except Exception as e:
            self.root.after(0, lambda: self._training_error(str(e)))
            
    def _training_completed(self):
        """Handle training completion 处理训练完成"""
        self.is_training = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.NORMAL)
        self.progress_var.set("Training completed - 训练完成")
        self._log_message("Training completed successfully - 训练成功完成")
        self._log_message(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        
    def _training_error(self, error_msg: str):
        """Handle training error 处理训练错误"""
        self.is_training = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
        self.progress_var.set("Training error - 训练错误")
        self._log_message(f"Training error: {error_msg}")
        messagebox.showerror("Training Error - 训练错误", f"Training failed: {error_msg}")
        
    def _start_training(self):
        """Start the training process 开始训练过程"""
        if self.is_training:
            return
            
        # Setup training
        # 设置训练
        if not self._setup_training():
            return
            
        # Update UI state
        # 更新UI状态
        self.is_training = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.DISABLED)
        self.progress_var.set("Starting training - 开始训练...")
        
        # Start training in separate thread
        # 在单独线程中开始训练
        self.training_thread = threading.Thread(target=self._training_loop)
        self.training_thread.daemon = True
        self.training_thread.start()
        
    def _stop_training(self):
        """Stop the training process 停止训练过程"""
        if self.is_training:
            self.is_training = False
            self.progress_var.set("Stopping training - 停止训练...")
            self._log_message("Training stop requested - 请求停止训练")
            
    def _save_model(self):
        """Save the trained model 保存训练好的模型"""
        if not self.model:
            messagebox.showerror("Error - 错误", "No model to save - 没有模型可保存")
            return
            
        save_path = self.model_save_path.get()
        if not save_path:
            messagebox.showerror("Error - 错误", "Please specify a save path - 请指定保存路径")
            return
            
        try:
            # 创建完整的模型文件路径
            model_file_path = os.path.join(save_path, "supercnn_model.pth")
            
            # Ensure the save directory exists
            os.makedirs(save_path, exist_ok=True)
            
            # Use best model state if available, otherwise use current model state
            # 如果有最佳模型则使用，否则使用当前模型状态
            model_state = self.best_model_state if self.best_model_state is not None else self.model.state_dict()
            
            torch.save({
                'model_state_dict': model_state,
                'conv_layers': self.conv_layers.get(),
                'num_classes': self.num_classes.get(),
                'image_size': self.image_size.get(),
                'history': self.history,
                'optimizer': self.optimizer_var.get(),
                'learning_rate': self.learning_rate.get(),
                'dropout_rate': self.dropout_rate.get(),
                'best_val_acc': self.best_val_acc
            }, model_file_path)
            
            # 保存完整的模型结构和参数，便于在其他应用程序中直接使用
            # Save complete model for easy usage in other applications
            full_model_path = os.path.join(save_path, "supercnn_model_full.pth")
            torch.save(self.model, full_model_path)
            
            # Save training history to CSV
            csv_path = os.path.join(save_path, "training_history.csv")
            try:
                self.trainer.save_history_to_csv(self.history, csv_path)
                self._log_message(f"Training history saved to {csv_path} - 训练历史已保存到 {csv_path}")
            except Exception as e:
                self._log_message(f"⚠️ Failed to save training history: {str(e)}")
            
            self._log_message(f"Model saved to {model_file_path} - 模型已保存到 {model_file_path}")
            self._log_message(f"Full model saved to {full_model_path} - 完整模型已保存到 {full_model_path}")
            messagebox.showinfo("Success - 成功", f"Model saved successfully to:\n{model_file_path}\n\nFull model saved to:\n{full_model_path}")
            
        except Exception as e:
            error_msg = f"Failed to save model: {str(e)}"
            self._log_message(error_msg)
            messagebox.showerror("Error - 错误", error_msg)
            
    def _predict_image(self):
        """Predict an image using the loaded model 使用加载的模型进行图像预测"""
        model_path = self.model_load_path.get()
        image_path = self.image_predict_path.get()
        
        if not model_path:
            messagebox.showerror("Error - 错误", "Please select a model file - 请选择模型文件")
            return
            
        if not image_path:
            messagebox.showerror("Error - 错误", "Please select an image file - 请选择图像文件")
            return
            
        if not os.path.exists(model_path):
            messagebox.showerror("Error - 错误", "Model file does not exist - 模型文件不存在")
            return
            
        if not os.path.exists(image_path):
            messagebox.showerror("Error - 错误", "Image file does not exist - 图像文件不存在")
            return
            
        try:
            # Load model
            # 加载模型
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Create model with parameters from checkpoint
            # 使用检查点中的参数创建模型
            model = SupperCNN.create_model(
                checkpoint.get('conv_layers', 3),
                checkpoint.get('num_classes', 2),
                checkpoint.get('image_size', 64),
                torch.device('cpu')  # Use CPU for prediction to ensure compatibility
            )
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Create predictor
            # 创建预测器
            # Note: We don't have class names here, so we'll use generic names
            class_names = [f"Class {i}" for i in range(checkpoint.get('num_classes', 2))]
            
            predictor = Predictor(
                model=model,
                class_names=class_names,
                device=torch.device('cpu'),
                image_size=checkpoint.get('image_size', 64),
                use_amp=False  # Disable AMP for prediction
            )
            
            # Make prediction
            # 进行预测
            predicted_class, confidence = predictor.predict(image_path)
            
            # Update result
            # 更新结果
            self.predict_result.set(f"Predicted Class: {predicted_class}\nConfidence: {confidence:.2%}")
            self._log_message(f"Prediction: {predicted_class} (Confidence: {confidence:.2%})")
            
        except Exception as e:
            error_msg = f"Failed to make prediction: {str(e)}"
            self._log_message(error_msg)
            messagebox.showerror("Error - 错误", error_msg)
            self.predict_result.set("预测失败")


def run_gui():
    """Run the GUI application 运行GUI应用程序"""
    root = tk.Tk()
    app = TrainingControllerGUI(root)
    root.mainloop()