import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")

train_joint = np.load('data/train_joint.npy')  # (16432, 3, 300, 17, 2)
train_label = np.load('data/train_label.npy')  # (16432,)
test_joint = np.load('data/test_joint_A.npy')  # (2000, 3, 300, 17, 2)
test_label = np.load('data/test_label_A.npy')  # (2000,)

# 整维度
train_joint = train_joint.reshape(-1, 3, 300, 34)  # 将17和2合并
test_joint = test_joint.reshape(-1, 3, 300, 34)

# 数据归一化
train_joint = train_joint.astype(np.float32) / 255.0
test_joint = test_joint.astype(np.float32) / 255.0

# 转张量
train_tensor = torch.tensor(train_joint)
train_labels = torch.tensor(train_label)

# 创建数据集和数据加载器
train_dataset = TensorDataset(train_tensor, train_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 定义模型
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 75 * 8, 128)
        self.fc2 = nn.Linear(128, len(np.unique(train_label)))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNNModel().to(device)

# 设置损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 300
with open('train_log.txt', 'w') as train_log:
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            for inputs, labels in pbar:
                inputs = inputs.to(device).float()
                labels = labels.to(device).long()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                pbar.set_postfix(loss=total_loss/len(train_loader), accuracy=correct/total)

            # 记录训练日志
            train_log.write(f'Epoch: {epoch+1}, Loss: {total_loss/len(train_loader)}, Accuracy: {correct/total}\n')

# 测试模型
model.eval()
test_tensor = torch.tensor(test_joint).float().to(device)
predictions_list = []

with torch.no_grad():
    predictions = model(test_tensor)
    predictions = F.softmax(predictions, dim=1)
    predicted_classes = torch.argmax(predictions, dim=1)

# 计算测试准确率
correct = (predicted_classes.cpu() == torch.tensor(test_label)).sum().item()
accuracy = correct / len(test_label)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# 保存置信度文件
np.save('eval/pred.npy', predictions.cpu().numpy())

# 记录推理日志
with open('inference_log.txt', 'w') as inference_log:  # 打开推理日志文件
    for i, pred in enumerate(predictions):
        inference_log.write(f'Sample {i}, Predictions: {pred.cpu().numpy()}\n')
