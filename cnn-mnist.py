import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset,random_split
from tqdm import tqdm

train_data = pd.read_csv('./datas/mnist_train_new.csv')
test_data = pd.read_csv('./datas/mnist_test.csv')

"""
train_data.describe()
print(train_data['label'].value_counts())
"""

y = train_data['label'].values
X = train_data.drop('label',axis=1).values

"""
plt.figure(figsize=(15,5))

for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(X[i].reshape(28,28),cmap='gray')
    plt.title(f"Label: {y[i]}")
    plt.axis('off')

plt.show()
"""

X = X/X.max()

X_tensor = torch.tensor(X,dtype=torch.float32)
y_tensor = torch.tensor(y,dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)

train_size = int(len(dataset)*0.8)
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # 第一层卷积：输入通道1（灰度图像），输出通道32，卷积核大小3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # 输出：32 x 28 x 28
        # 第二层卷积：输入通道32，输出通道64，卷积核大小3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 输出：64 x 28 x 28
        # 第三层卷积：输入通道64，输出通道128，卷积核大小3x3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # 输出：128 x 28 x 28

        # 池化层：2x2最大池化，减少空间维度
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 输出：128 x 14 x 14

        # ReLU激活函数（将用于卷积层输出后的激活）
        self.relu = nn.ReLU()

        # 全连接层
        self.fc1 = nn.Linear(128 * 3 * 3, 512)  # 将卷积输出展平后接全连接层
        self.fc2 = nn.Linear(512, 10)  # 输出10类

    def forward(self, x):
        x = x.view(-1,1, 28, 28)  # 保证输入是28x28的图像


        # 第一层卷积，激活函数
        x = self.relu(self.conv1(x))  # 输入：1x28x28 -> 输出：32x28x28

        x = self.pool(x)  # 池化层，输出：32x14x14

        # 第二层卷积，激活函数
        x = self.relu(self.conv2(x))  # 输入：32x14x14 -> 输出：64x14x14
        x = self.pool(x)  # 池化层，输出：64x7x7

        # 第三层卷积，激活函数
        x = self.relu(self.conv3(x))  # 输入：64x7x7 -> 输出：128x7x7
        x = self.pool(x)  # 池化层，输出：128x3x3

        # 展平多维输入数据为一维
        x = x.view(x.size(0), -1)  # 自动计算展平后的大小

        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # 输出层

        return x

if torch.cuda.is_available():
    device = torch.device('cuda')  # 如果有 GPU，使用 GPU
    print("CUDA is available! Using GPU.")
else:
    device = torch.device('cpu')  # 如果没有 GPU，使用 CPU
    print("CUDA is not available. Using CPU.")


# 初始化模型、损失函数和优化器
model = CNN()
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 用于记录每个 epoch 的训练损失和验证损失
train_losses = []
val_losses = []
val_accuracies = []

# 用于记录每一步的训练损失
train_losses_per_step = []


model = model.to(device)  # 将模型移动到 GPU 或 CPU 上
# 训练过程
num_epochs = 10  # 假设训练 10 个 epoch
for epoch in range(num_epochs):
    # 1. 训练阶段
    model.train()
    running_train_loss = 0.0
    for i, (inputs, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
        # 将数据移到设备
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # 清空梯度
        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        running_train_loss += loss.item()

        # 记录每一步的训练损失
        train_losses_per_step.append(loss.item())

        # 每 100 步打印一次训练损失
        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    avg_train_loss = running_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)  # 保存每个 epoch 的训练损失

    # 2. 验证阶段
    model.eval()  # 设置为评估模式，禁用 Dropout 等
    running_val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():  # 不计算梯度
        for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
            # 将数据移到设备
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_val_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = running_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)  # 保存每个 epoch 的验证损失
    val_accuracy = correct / total  # 计算验证准确率
    val_accuracies.append(val_accuracy)  # 保存每个 epoch 的验证准确率

    print(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # 绘制每一步的训练损失曲线
plt.figure(figsize=(12, 6))
plt.plot(train_losses_per_step, label="Training Loss", color='blue', marker='o', markersize=2)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss per Step')
plt.legend()
plt.grid(True)
plt.show()

# 绘制损失曲线
plt.figure(figsize=(12, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss", color='blue', marker='o')
plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss", color='red', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# 绘制验证准确率曲线
plt.figure(figsize=(12, 6))
plt.plot(range(1, num_epochs + 1), val_accuracies, label="Validation Accuracy", color='green', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()


X_test = test_data.drop('label', axis=1).values  # 这里的 X_test 是 (n_samples, 784) 的数组
X_test = X_test / 255.0  # 归一化到 [0, 1]

# 创建 TensorDataset 和 DataLoader
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
test_dataset = TensorDataset(X_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 设置模型为评估模式
model.eval()

# 用于存储所有的图像和预测标签（没有真实标签，因为 test_data 没有标签）
all_images = []
all_predictions = []

# 计算所有测试集的预测结果
with torch.no_grad():  # 禁用梯度计算，推理时不需要计算梯度
    for (inputs,) in test_loader:
        inputs = inputs.to(device)  # 直接用
        outputs = model(inputs) # 推理
        _, predicted = torch.max(outputs, 1)  # 获取预测的标签

        # 将当前 batch 的图像和预测标签保存
        all_images.extend(inputs.view(-1, 28, 28).cpu().numpy())  # 转换为 28x28 图片
        all_predictions.extend(predicted.cpu().numpy())

image_ids = range(1,len(all_predictions)+1)

# 创建结果的 DataFrame
results_df = pd.DataFrame({
    'ImageId': image_ids,
    'Label': all_predictions
})

# 保存为 CSV 文件
results_df.to_csv('./outfile/submissions.csv', index=False)

print("Predictions saved to 'submissions.csv'")


y_true = []
y_pred = []

# 计算所有验证集的预测结果
with torch.no_grad():  # 禁用梯度计算，推理时不需要计算梯度
    for inputs, labels in val_loader:
        inputs = inputs.view(-1, 28,28).to(device)  # 将每张图片展平64x28x28
        outputs = model(inputs)  # 推理
        _, predicted = torch.max(outputs, 1)  # 获取预测的标签

        # 保存真实标签和预测标签
        y_true.extend(labels.cpu().numpy())  # 真实标签
        y_pred.extend(predicted.cpu().numpy())  # 预测标签

from sklearn.metrics import confusion_matrix,f1_score
# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred)
# 计算每个类别的 F1 分数
f1_scores = f1_score(y_true, y_pred,average=None)  # 每个类别的 F1 分数

# 绘制混淆矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()