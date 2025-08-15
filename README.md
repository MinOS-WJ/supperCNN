# SupperCNN: 高效图像分类CNN框架

SupperCNN is a flexible and efficient convolutional neural network framework designed for image classification tasks. It provides a unified interface for dataset preprocessing, model creation, training, and prediction, along with a user-friendly GUI for easier operation.

SupperCNN是一个灵活高效的卷积神经网络框架，专为图像分类任务设计。它提供了统一的接口用于数据预处理、模型创建、训练和预测，并配有用户友好的GUI界面，便于操作。


## 功能特点 | Features

- 简洁统一的API接口，易于集成到现有项目 | Clean and unified API, easy to integrate into existing projects
- 支持CPU和GPU训练，自动适配硬件环境 | Supports CPU and GPU training, automatically adapts to hardware environment
- 支持自动混合精度(AMP)训练，加速训练过程 | Supports Automatic Mixed Precision (AMP) training to accelerate training
- 内置数据预处理工具，自动整理图像数据集 | Built-in data preprocessing tools to automatically organize image datasets
- 可视化训练过程，实时监控损失和准确率 | Visual training process with real-time monitoring of loss and accuracy
- 提供图形用户界面(GUI)，简化操作流程 | Provides Graphical User Interface (GUI) to simplify operation
- 支持模型保存与加载，方便后续预测 | Supports model saving and loading for subsequent predictions


## 安装要求 | Installation Requirements

### 所需Python包 | Required Python Packages

```
torch>=1.10.0
torchvision>=0.11.1
Pillow>=8.4.0
matplotlib>=3.5.0
tkinter>=8.6  # 通常Python已内置 | Usually pre-installed with Python
tqdm>=4.62.3
numpy>=1.21.4
```

### 安装方法 | Installation Method

```bash
pip install torch torchvision pillow matplotlib tqdm numpy
```


## 推荐设备配置 | Recommended Hardware Configuration

### 最低配置 | Minimum Configuration
- CPU: Intel Core i5 或同等处理器 | Intel Core i5 or equivalent processor
- 内存: 8GB RAM
- 硬盘: 至少10GB可用空间 | At least 10GB free space
- 操作系统: Windows 10/11, macOS 10.15+, Linux

### 推荐配置 | Recommended Configuration
- CPU: Intel Core i7/i9 或 AMD Ryzen 7/9
- 内存: 16GB RAM 或更高 | 16GB RAM or higher
- GPU: NVIDIA GeForce RTX 2060 或更高(支持CUDA) | NVIDIA GeForce RTX 2060 or higher (with CUDA support)
- 硬盘: 50GB SSD可用空间 | 50GB SSD free space
- 操作系统: Windows 10/11, Ubuntu 20.04+


## 快速开始 | Quick Start

### 1. 使用GUI界面 (推荐新手) | Using GUI Interface (Recommended for Beginners)

```python
from supercnn import run_gui

if __name__ == "__main__":
    run_gui()  # 启动图形界面 | Launch graphical interface
```

### 2. 使用Python API | Using Python API

```python
from supercnn import SupperCNN
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

# 1. 数据预处理 | Data preprocessing
source_dir = "path/to/raw/images"
target_dir = "path/to/processed/images"
image_size = 64
SupperCNN.preprocess_dataset(source_dir, target_dir, image_size)

# 2. 创建数据集 | Create dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataset = SupperCNN.create_dataset(target_dir, transform=transform, target_size=(image_size, image_size))

# 3. 划分训练集和验证集 | Split into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 4. 创建模型 | Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SupperCNN.create_model(conv_layers=3, num_classes=2, image_size=image_size, device=device)

# 5. 创建训练器 | Create trainer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
use_amp = device.type == 'cuda'  # 仅在GPU上使用AMP | Use AMP only on GPU
scaler = torch.cuda.amp.GradScaler() if use_amp else None
trainer = SupperCNN.create_trainer(model, optimizer, criterion, device, use_amp, scaler)

# 6. 训练模型 | Train model
epochs = 10
history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}

for epoch in range(epochs):
    train_loss, train_acc = trainer.train_epoch(train_loader, epoch, epochs)
    val_loss, val_acc = trainer.evaluate(val_loader, "Validation")
    
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    
    print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

# 7. 保存模型 | Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'conv_layers': 3,
    'num_classes': 2,
    'image_size': image_size
}, 'supper_cnn_model.pth')

# 8. 加载模型并预测 | Load model and predict
checkpoint = torch.load('supper_cnn_model.pth', map_location=device)
loaded_model = SupperCNN.create_model(
    checkpoint['conv_layers'],
    checkpoint['num_classes'],
    checkpoint['image_size'],
    device
)
loaded_model.load_state_dict(checkpoint['model_state_dict'])

# 9. 创建预测器 | Create predictor
predictor = SupperCNN.create_predictor(loaded_model, ['class1', 'class2'], device, image_size, use_amp=False)

# 10. 单张图像预测 | Predict single image
class_name, confidence = predictor.predict("test_image.jpg")
print(f"Predicted: {class_name}, Confidence: {confidence:.4f}")
```


## 接口详细说明 | API Detailed Explanation

### SupperCNN 主类 | Main Class

SupperCNN类提供了所有功能的统一入口，包含以下静态方法：

The SupperCNN class provides a unified entry point for all functions, with the following static methods:

#### 1. create_model
创建卷积神经网络模型 | Create a convolutional neural network model

```python
SupperCNN.create_model(conv_layers: int, num_classes: int, image_size: int, device: torch.device) -> nn.Module
```

参数说明 | Parameters:
- `conv_layers`: 卷积层数量，范围1-10 | Number of convolutional layers, range 1-10
- `num_classes`: 分类类别数量 | Number of classification categories
- `image_size`: 输入图像尺寸(边长) | Input image size (side length)
- `device`: 模型运行设备 (torch.device) | Device for model operation (torch.device)


#### 2. create_dataset
创建图像数据集 | Create image dataset

```python
SupperCNN.create_dataset(data_dir: str, transform=None, target_size=(256, 256)) -> ImageDataset
```

参数说明 | Parameters:
- `data_dir`: 数据集目录，应包含子目录作为类别 | Dataset directory, should contain subdirectories as categories
- `transform`: 图像变换函数 | Image transformation function
- `target_size`: 图像目标尺寸 | Target size for images


#### 3. create_trainer
创建模型训练器 | Create model trainer

```python
SupperCNN.create_trainer(model, optimizer, criterion, device, use_amp, scaler, progress_callback=None) -> Trainer
```

参数说明 | Parameters:
- `model`: 待训练的模型 | Model to be trained
- `optimizer`: 优化器 | Optimizer
- `criterion`: 损失函数 | Loss function
- `device`: 训练设备 | Training device
- `use_amp`: 是否使用自动混合精度 | Whether to use automatic mixed precision
- `scaler`: AMP梯度缩放器 | AMP gradient scaler
- `progress_callback`: 进度回调函数 | Progress callback function


#### 4. create_predictor
创建预测器 | Create predictor

```python
SupperCNN.create_predictor(model, class_names, device, image_size, use_amp) -> Predictor
```

参数说明 | Parameters:
- `model`: 训练好的模型 | Trained model
- `class_names`: 类别名称列表 | List of class names
- `device`: 预测设备 | Prediction device
- `image_size`: 图像尺寸 | Image size
- `use_amp`: 是否使用自动混合精度 | Whether to use automatic mixed precision


#### 5. preprocess_dataset
预处理数据集 | Preprocess dataset

```python
SupperCNN.preprocess_dataset(source_dir: str, target_dir: str, image_size: int) -> str
```

参数说明 | Parameters:
- `source_dir`: 原始图像目录 | Raw image directory
- `target_dir`: 处理后图像保存目录 | Directory for processed images
- `image_size`: 图像目标尺寸 | Target size for images


### Trainer 类方法 | Trainer Class Methods

#### train_epoch
训练一个epoch | Train for one epoch

```python
trainer.train_epoch(train_loader, epoch, total_epochs) -> (train_loss, train_acc)
```

#### evaluate
评估模型性能 | Evaluate model performance

```python
trainer.evaluate(loader, desc='Evaluate') -> (eval_loss, eval_acc)
```

#### save_history_to_csv
保存训练历史到CSV文件 | Save training history to CSV file

```python
trainer.save_history_to_csv(history, csv_path)
```


### Predictor 类方法 | Predictor Class Methods

#### predict
预测单张图像 | Predict single image

```python
predictor.predict(image_path: str) -> (class_name, confidence)
```

#### predict_batch
批量预测图像 | Batch predict images

```python
predictor.predict_batch(image_dir: str, batch_size: int) -> List[(image_name, class_name, confidence)]
```


## 目录结构 | Directory Structure

```

supercnn/
├── __init__.py        # 包初始化和API导出
├── dataset.py         # 数据集处理
├── model.py           # 模型创建
├── trainer.py         # 训练器
├── predictor.py       # 预测器
├── utils.py           # 工具函数
└── gui.py             # 图形用户界面

```


## 注意事项 | Notes

1. 数据集应按照类别组织在子目录中 | Datasets should be organized by category in subdirectories
2. 使用GPU训练需要安装对应版本的CUDA | GPU training requires installation of the corresponding CUDA version
3. 自动混合精度(AMP)仅在GPU上有效 | Automatic Mixed Precision (AMP) only works on GPU
4. 图像预处理会将所有图像转换为RGB格式 | Image preprocessing converts all images to RGB format
5. 建议在训练前进行数据预处理，以提高训练效率 | It is recommended to preprocess data before training to improve efficiency


通过SupperCNN，您可以快速构建和训练卷积神经网络模型，无需深入了解复杂的深度学习细节。无论是初学者还是专业人士，都能轻松上手使用。

With SupperCNN, you can quickly build and train convolutional neural network models without deep knowledge of complex deep learning details. Both beginners and professionals can easily get started.
