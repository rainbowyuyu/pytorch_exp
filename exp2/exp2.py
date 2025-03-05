from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 超参
batch_size = 64
epochs = 20
learning_rate = 0.01

# 导入训练数据
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data = datasets.FashionMNIST(root='./data', train=False, transform=transform)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


# 神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 512)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


# 模型初始化
net = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# 存储损失
train_losses = []


# 训练模型
def train():
    net.train()
    for epoch in range(epochs):
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # 数据迁移到 GPU

            # 前向传播
            outputs = net(images)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))  # 记录每个 epoch 的平均损失
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {train_loss/len(train_loader):.4f}')


# 绘制损失函数曲线
def plot_loss():
    plt.plot(range(1, epochs + 1), train_losses, marker='o')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('training_loss_curve.png')
    plt.close()


# 模型测试
def test():
    net.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():  # 关闭梯度计算，提高测试效率
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  # 数据迁移到 GPU
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f} %')

    # 绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=train_data.classes, yticklabels=train_data.classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    plt.close()


# 运行
if __name__ == '__main__':
    train()
    test()
    plot_loss()
