# 🧠 CNN-MNIST 手写数字识别项目

## 📌 项目简介

本项目基于卷积神经网络（CNN）实现对 MNIST 手写数字数据集的分类任务。通过构建多层卷积结构，模型能够自动提取图像特征，并完成 0–9 数字的识别。

该项目涵盖了数据预处理、模型设计、训练优化、模型评估等完整深度学习流程，适合作为深度学习入门与实践项目。

---

## 🚀 项目特点

* ✅ 使用 PyTorch 搭建 CNN 模型
* ✅ 三层卷积结构（Conv + ReLU + MaxPool）
* ✅ 支持 GPU 加速训练
* ✅ 包含训练、验证、测试完整流程
* ✅ 可视化训练损失与准确率变化
* ✅ 使用混淆矩阵评估分类效果

---

## 🏗️ 模型结构

输入图像尺寸：28 × 28（灰度图）

网络结构如下：

Conv1 (1 → 32, 3×3) → ReLU → MaxPool
Conv2 (32 → 64, 3×3) → ReLU → MaxPool
Conv3 (64 → 128, 3×3) → ReLU → MaxPool
Flatten
FC1 → ReLU
FC2 → Softmax（隐含在 CrossEntropyLoss 中）

---

## 📊 数据集说明

本项目使用 MNIST 数据集（CSV格式）：

* 训练集：包含标签（label + 784像素）
* 测试集：仅包含像素数据

⚠️ **注意：由于 GitHub 文件大小限制（100MB），本仓库不包含原始数据集（train部分进行了删减）。**

👉 请从以下链接下载数据：

Kaggle：
https://www.kaggle.com/competitions/digit-recognizer/data

下载后放入目录：

```
datas/
├── mnist_train.csv
└── mnist_test.csv
```

---

## ⚙️ 环境依赖

```bash
pip install torch numpy pandas matplotlib seaborn scikit-learn tqdm
```

---

## ▶️ 使用方法

### 1️⃣ 克隆项目

```bash
git clone https://github.com/Xianwuchuan/CNN-Mnist.git
cd CNN-Mnist
```

### 2️⃣ 准备数据集

将 MNIST CSV 文件放入 `datas/` 目录

### 3️⃣ 运行训练

```bash
python cnn-mnist.py
```

---

## 📈 结果展示

训练过程中会输出：

* Training Loss 曲线
* Validation Loss 曲线
* Validation Accuracy 曲线
* Confusion Matrix（混淆矩阵）

---

## 🧠 核心技术点

* 卷积神经网络（CNN）
* 图像特征提取
* 批量训练（DataLoader）
* 交叉熵损失函数（CrossEntropyLoss）
* Adam 优化器
* GPU 加速（CUDA）
* 模型评估（Accuracy / F1-score / Confusion Matrix）

---

## ❓ 常见问题

### Q1：为什么不上传数据集？

由于 GitHub 单文件限制为 100MB，而 MNIST CSV 超出限制，因此采用外部下载方式。

---

### Q2：为什么输入需要 reshape？

原始输入为 784 维向量，需要转换为 1×28×28 才能输入 CNN。

---

### Q3：如何提升模型性能？

* 增加 Dropout 防止过拟合
* 使用数据增强
* 调整学习率
* 使用更深层网络（如 ResNet）

---

## 📌 项目总结

本项目实现了一个完整的 CNN 图像分类流程，从数据预处理到模型评估，帮助理解深度学习在计算机视觉中的基本应用。

---

## 👤 作者

GitHub: https://github.com/Xianwuchuan

---

## ⭐ 如果这个项目对你有帮助，欢迎 Star！
