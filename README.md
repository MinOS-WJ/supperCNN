# SupperCNN 接口使用说明与项目依赖 (SupperCNN Interface Guide & Dependencies)

## 概述 (Overview)

SupperCNN 是一个基于 PyTorch 的卷积神经网络工具包，提供简洁易用的接口用于图像分类任务。该工具包整合了数据预处理、模型构建、训练和预测等全流程功能，使开发者无需深入了解神经网络细节即可快速实现图像分类应用。

SupperCNN is a PyTorch-based convolutional neural network toolkit that provides simple interfaces for image classification tasks. It integrates end-to-end functionalities including data preprocessing, model building, training, and prediction, enabling developers to quickly implement image classification applications without deep knowledge of neural networks.

## 核心接口说明 (Core Interface Description)

### SupperCNN 类 (SupperCNN Class)

该类是所有功能的统一入口，提供以下静态方法：

This class serves as the unified entry point for all functionalities, with the following static methods:

#### 1. 模型训练 (Model Training)

```python
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
) -> dict
```

**参数说明 (Parameters):**
- `data_dir`: 原始数据集目录路径 (Path to raw dataset directory)
- `conv_layers`: 卷积层数量 (Number of convolutional layers, default: 3)
- `num_classes`: 分类类别数量 (Number of classes, default: 2)
- `image_size`: 图像尺寸（长和宽）(Image size for resizing, default: 64)
- `batch_size`: 批处理大小 (Batch size for training, default: 32)
- `epochs`: 训练轮数 (Number of training epochs, default: 10)
- `lr`: 学习率 (Learning rate, default: 0.001)
- `val_split`: 验证集占比 (Validation set ratio, default: 0.2)
- `device`: 训练设备（如未指定自动选择）(Training device, auto-selected if None)
- `use_amp`: 是否启用自动混合精度训练（默认GPU自动启用）(Whether to use AMP, auto-enabled for GPU)
- `save_path`: 模型保存路径（可选）(Path to save model, optional)
- `class_names`: 类别名称列表（可选）(List of class names, optional)
- `auto_preprocess`: 是否自动预处理数据 (Whether to auto-preprocess data, default: True)
- `preprocessed_dir`: 预处理数据保存目录 (Directory for preprocessed data, default: './processed_data')

**返回值 (Return Value):**
- 训练历史字典，包含训练/验证的准确率和损失值 (Training history dict with train/val accuracy and loss)

#### 2. 单张图像预测 (Single Image Prediction)

```python
@staticmethod
def predict_single(
    image_path: str,
    model_path: str,
    device: torch.device = None
) -> tuple
```

**参数说明 (Parameters):**
- `image_path`: 待预测图像路径 (Path to image for prediction)
- `model_path`: 训练好的模型文件路径 (Path to trained model file)
- `device`: 预测设备（如未指定自动选择）(Prediction device, auto-selected if None)

**返回值 (Return Value):**
- 元组 `(预测类别, 置信度)` (Tuple of `(predicted_class, confidence)`)

#### 3. 批量图像预测 (Batch Image Prediction)

```python
@staticmethod
def predict_batch(
    image_dir: str,
    model_path: str,
    batch_size: int = 32,
    device: torch.device = None
) -> list
```

**参数说明 (Parameters):**
- `image_dir`: 包含待预测图像的目录 (Directory containing images for prediction)
- `model_path`: 训练好的模型文件路径 (Path to trained model file)
- `batch_size`: 批处理大小 (Batch size for prediction, default: 32)
- `device`: 预测设备（如未指定自动选择）(Prediction device, auto-selected if None)

**返回值 (Return Value):**
- 预测结果列表，每个元素为 `(图像名称, 预测类别, 置信度)` (List of tuples `(image_name, predicted_class, confidence)`)

#### 4. 训练历史保存 (Save Training History)

```python
@staticmethod
def save_history_to_csv(history: dict, csv_path: str)
```

**参数说明 (Parameters):**
- `history`: 训练历史字典（`train_model` 方法的返回值）(Training history dict from `train_model`)
- `csv_path`: 保存CSV文件的路径 (Path to save CSV file)

#### 5. 数据集预处理 (Dataset Preprocessing)

```python
@staticmethod
def preprocess_dataset(source_dir: str, target_dir: str, image_size: int) -> str
```

**参数说明 (Parameters):**
- `source_dir`: 原始数据集目录 (Source directory with raw images)
- `target_dir`: 预处理后数据保存目录 (Target directory for processed images)
- `image_size`: 目标图像尺寸 (Target size for resizing images)

**返回值 (Return Value):**
- 预处理后数据目录路径 (Path to preprocessed data directory)

## 使用示例 (Usage Examples)

### 示例 1: 训练模型 (Training a Model)

```python
from supercnn import SupperCNN
import torch

# 训练模型
history = SupperCNN.train_model(
    data_dir="./dataset",          # 原始数据集目录
    conv_layers=4,                 # 4个卷积层
    num_classes=3,                 # 3个类别
    image_size=128,                # 图像大小调整为128x128
    batch_size=16,                 # 批处理大小为16
    epochs=20,                     # 训练20个epoch
    lr=0.0005,                     # 学习率0.0005
    val_split=0.2,                 # 20%数据作为验证集
    device=torch.device('cuda'),   # 使用GPU训练
    save_path="./models/model.pth",# 模型保存路径
    class_names=["cat", "dog", "bird"],  # 类别名称
    auto_preprocess=True,          # 自动预处理数据
    preprocessed_dir="./processed_data"  # 预处理数据保存目录
)

# 保存训练历史到CSV
SupperCNN.save_history_to_csv(history, "./training_history.csv")
```

### 示例 2: 单张图像预测 (Single Image Prediction)

```python
from supercnn import SupperCNN

# 预测单张图片
image_path = "./test_image.jpg"
model_path = "./models/model.pth"

predicted_class, confidence = SupperCNN.predict_single(image_path, model_path)

print(f"预测类别: {predicted_class}")
print(f"置信度: {confidence:.2f}")
# 输出:
# 预测类别: cat
# 置信度: 0.98
```

### 示例 3: 批量图像预测 (Batch Image Prediction)

```python
from supercnn import SupperCNN

# 批量预测图片
image_dir = "./test_images"
model_path = "./models/model.pth"

results = SupperCNN.predict_batch(image_dir, model_path, batch_size=32)

# 打印结果
for image_name, pred_class, conf in results:
    print(f"{image_name}: {pred_class} ({conf:.2f})")
# 输出:
# image1.jpg: cat (0.97)
# image2.jpg: dog (0.99)
# image3.jpg: bird (0.85)
```

### 示例 4: 手动预处理数据 (Manual Data Preprocessing)

```python
from supercnn import SupperCNN

# 手动预处理数据
source_dir = "./raw_data"
target_dir = "./processed_data"
image_size = 128

processed_dir = SupperCNN.preprocess_dataset(source_dir, target_dir, image_size)
print(f"预处理完成，数据保存到: {processed_dir}")

# 使用预处理后的数据训练模型
history = SupperCNN.train_model(
    data_dir=processed_dir,
    auto_preprocess=False,  # 已手动预处理，设为False
    # 其他参数...
)
```

## 硬件要求 (Hardware Requirements)

### 最低配置 (Minimum Requirements)
- CPU: 支持64位运算的处理器 (64-bit processor)
- 内存: 4GB RAM
- 存储: 至少1GB可用空间（用于安装依赖和存储数据/模型）(At least 1GB free space)
- 显卡: 可选（推荐NVIDIA GPU以加速训练）(Optional, NVIDIA GPU recommended for faster training)

### 推荐配置 (Recommended Requirements)
- CPU: 多核处理器（如Intel i5/i7/i9或AMD Ryzen系列）(Multi-core processor)
- 内存: 8GB RAM或更高 (8GB RAM or higher)
- 存储: 10GB可用空间 (10GB free space)
- 显卡: 支持CUDA的NVIDIA GPU（计算能力3.5+），用于加速训练 (NVIDIA GPU with CUDA support, Compute Capability 3.5+)

## Python库依赖 (Python Library Dependencies)

| 库名称 (Library) | 版本要求 (Version Requirement) | 用途 (Purpose) |
|------------------|-------------------------------|----------------|
| `torch`          | ≥ 1.10.0                      | 深度学习框架 (Deep learning framework) |
| `torchvision`    | ≥ 0.11.0                      | 计算机视觉工具 (Computer vision tools) |
| `Pillow`         | ≥ 8.0.0                       | 图像处理 (Image processing) |
| `tqdm`           | ≥ 4.50.0                      | 进度条显示 (Progress bar display) |
| `python`         | ≥ 3.7                         | 编程语言 (Programming language) |

## 安装命令 (Installation Commands)

```bash
# 基础安装 (Basic installation)
pip install torch>=1.10.0 torchvision>=0.11.0 pillow>=8.0.0 tqdm>=4.50.0

# CUDA加速安装（需根据CUDA版本调整）
# CUDA acceleration (adjust based on your CUDA version)
# 例如 (For example):
# pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/cu117/torch_stable.html
```

## 注意事项 (Notes)

1. 数据集目录结构需按类别组织，每个类别一个子目录 (Dataset should be organized by class with one subdirectory per class)
2. 支持的图像格式: .jpg, .jpeg, .png, .bmp, .gif, .tiff (Supported image formats)
3. 自动混合精度(AMP)训练仅在NVIDIA GPU且安装CUDA的系统上可用 (AMP training is only available on systems with NVIDIA GPUs and CUDA)
4. 模型保存文件包含模型参数、架构信息和类别名称 (Model save files include parameters, architecture, and class names)
5. Windows系统上的数据加载器工作进程数会自动限制，以避免多进程问题 (Data loader worker processes are limited on Windows to avoid multi-processing issues)
