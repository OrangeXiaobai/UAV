import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

# 检查是否有 CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using {device} device.')

# 加载数据
train_joint = np.load('data/train_joint.npy')  # (16432, 3, 300, 17, 2)
train_label = np.load('data/train_label.npy')  # (16432,)
test_joint = np.load('data/test_joint_B.npy')  # (2000, 3, 300, 17, 2) 没有label

# 在数据加载后调整维度
train_joint = train_joint.reshape(-1, 3, 300, 34)  # 将17和2合并
test_joint = test_joint.reshape(-1, 3, 300, 34)    # 将17和2合并

# 数据归一化
train_joint = train_joint.astype(np.float32) / 255.0
test_joint = test_joint.astype(np.float32) / 255.0

# 转换为张量
train_tensor = torch.tensor(train_joint)
train_labels = torch.tensor(train_label).long()

# 创建数据集和数据加载器
train_dataset = TensorDataset(train_tensor, train_labels)
train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, 
    num_workers=8, pin_memory=True
)

# 定义模型
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 75 * 8, 128)  # 根据输入数据调整
        self.fc2 = nn.Linear(128, 155)  # 假设输出类别为155类

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model = CNNModel().to(device)

# 设置损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 400
with open('trainB/train_log.txt', 'w') as train_log:
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            for inputs, labels in pbar:
                inputs = inputs.to(device).float()
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                pbar.set_postfix(loss=total_loss / len(train_loader), accuracy=correct / total)

            # 记录训练日志
            train_log.write(f'Epoch: {epoch+1}, Loss: {total_loss/len(train_loader)}, Accuracy: {correct/total}\n')

# 推理：对测试数据进行预测（无标签）
test_dataset = TensorDataset(torch.tensor(test_joint))
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model.eval()
predictions_list = []

with torch.no_grad():
    for inputs in tqdm(test_loader, desc="Inference"):
        inputs = inputs[0].to(device).float()
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1)  # 计算类别置信度
        predictions_list.append(probs.cpu().numpy())

# 合并预测结果为(2000, 155)数组
predictions = np.vstack(predictions_list)

# 保存预测结果和置信度文件
np.save('trainB/pred7.npy', predictions)

# 记录推理日志
with open('trainB/inference_log.txt', 'w') as inference_log:
    for i, pred in enumerate(predictions):
        inference_log.write(f'Sample {i}, Predictions: {pred}\n')

print("Inference complete. Predictions saved to 'trainB/pred7.npy'.")
