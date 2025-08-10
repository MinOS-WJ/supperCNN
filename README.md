# SupperCNN 接口使用说明 (Interface Usage Guide)

## 概述 (Overview)

SupperCNN 是一个基于PyTorch的卷积神经网络工具包，提供了简单易用的接口用于图像分类任务。该工具包封装了模型创建、数据处理、模型训练和预测等功能，使开发者能够快速构建和部署图像分类模型。

SupperCNN is a PyTorch-based convolutional neural network toolkit that provides easy-to-use interfaces for image classification tasks. It encapsulates functionalities such as model creation, data processing, model training, and prediction, enabling developers to quickly build and deploy image classification models.

## 主要功能 (Main Features)

- 自动数据预处理 (Automatic data preprocessing)
- 灵活的模型创建 (Flexible model creation)
- 简化的训练流程 (Simplified training process)
- 支持单张和批量图像预测 (Support for single and batch image prediction)
- 训练历史记录与保存 (Training history recording and saving)

## 接口说明 (Interface Description)

### 1. SupperCNN 类 (SupperCNN Class)

该类提供了所有功能的统一入口，包含以下主要静态方法：

This class provides a unified entry point for all functionalities, including the following main static methods:

#### 1.1 训练模型 (Train Model)

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
- `data_dir`: 数据集目录路径 (Dataset directory path)
- `conv_layers`: 卷积层数量 (Number of convolutional layers)
- `num_classes`: 类别数量 (Number of classes)
- `image_size`: 图像尺寸 (Image size)
- `batch_size`: 批处理大小 (Batch size)
- `epochs`: 训练轮数 (Number of training epochs)
- `lr`: 学习率 (Learning rate)
- `val_split`: 验证集比例 (Validation set split ratio)
- `device`: 训练设备 (Training device)
- `use_amp`: 是否使用自动混合精度 (Whether to use automatic mixed precision)
- `save_path`: 模型保存路径 (Model save path)
- `class_names`: 类别名称列表 (List of class names)
- `auto_preprocess`: 是否自动预处理数据 (Whether to automatically preprocess data)
- `preprocessed_dir`: 预处理后数据保存目录 (Directory for preprocessed data)

**返回值 (Return Value):**
- 训练历史字典，包含训练和验证的准确率及损失 (Training history dictionary containing training and validation accuracy and loss)

#### 1.2 单张图片预测 (Single Image Prediction)

```python
@staticmethod
def predict_single(
    image_path: str,
    model_path: str,
    device: torch.device = None
) -> tuple
```

**参数说明 (Parameters):**
- `image_path`: 图片路径 (Image path)
- `model_path`: 模型路径 (Model path)
- `device`: 预测设备 (Prediction device)

**返回值 (Return Value):**
- 元组 (预测类别, 置信度) (Tuple of (predicted class, confidence))

#### 1.3 批量图片预测 (Batch Image Prediction)

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
- `image_dir`: 图片目录 (Image directory)
- `model_path`: 模型路径 (Model path)
- `batch_size`: 批处理大小 (Batch size)
- `device`: 预测设备 (Prediction device)

**返回值 (Return Value):**
- 预测结果列表，每个元素为(图像名称, 预测类别, 置信度) (List of prediction results, each element is (image name, predicted class, confidence))

#### 1.4 保存训练历史到CSV (Save Training History to CSV)

```python
@staticmethod
def save_history_to_csv(history: dict, csv_path: str)
```

**参数说明 (Parameters):**
- `history`: 训练历史字典 (Training history dictionary)
- `csv_path`: CSV文件路径 (CSV file path)

## 使用示例 (Usage Examples)

### 示例1: 训练模型 (Training a Model)

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

### 示例2: 单张图片预测 (Single Image Prediction)

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

### 示例3: 批量图片预测 (Batch Image Prediction)

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

### 示例4: 手动预处理数据 (Manual Data Preprocessing)

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

## 注意事项 (Notes)

1. 数据集目录结构应按照类别组织，每个类别一个子目录 (Dataset directory structure should be organized by class, one subdirectory per class)
2. 支持的图像格式: .jpg, .jpeg, .png, .bmp, .gif, .tiff (Supported image formats)
3. 自动混合精度训练(AMP)仅在CUDA设备上有效 (Automatic Mixed Precision training (AMP) is only effective on CUDA devices)
4. 模型保存文件包含模型参数、架构信息和类别名称 (Model save files include model parameters, architecture information, and class names)

通过以上接口，您可以轻松完成从数据预处理到模型训练和预测的全流程，无需深入了解复杂的神经网络实现细节。

With the above interfaces, you can easily complete the entire process from data preprocessing to model training and prediction without needing to understand complex neural network implementation details.
