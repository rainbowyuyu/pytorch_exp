# rainbow_yu exp4 🐋✨
# 猫狗分类

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from time import time
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import pandas as pd

from pretrained_models import (
    My_CNN,
    Better_CNN,
    CustomVGGNet,
    CustomResNet,
    CustomAlexNet,
)

# 数据集路径
file_path = r"E:\python_project\datasets\cats_and_dogs\PetImages"

# 数据增强
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据
train_data = datasets.ImageFolder(os.path.join(file_path, "train"), transform)
test_data = datasets.ImageFolder(os.path.join(file_path, "test"), transform)

batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

# 训练参数
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomAlexNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()
model_name = model.__class__.__name__

# 训练函数
def train(model, device, train_loader, optimizer, epoch, losses, accuracies):
    model.train()
    total_loss = 0.0
    correct = 0
    num_samples = len(train_loader.dataset)

    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")

    for idx, (t_data, t_target) in progress_bar:
        t_data, t_target = t_data.to(device), t_target.to(device)
        optimizer.zero_grad()
        pred = model(t_data)
        loss = loss_fn(pred, t_target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * t_data.size(0)
        pred_class = pred.argmax(dim=1)
        correct += pred_class.eq(t_target).sum().item()

    acc = correct / num_samples
    avg_loss = total_loss / num_samples
    losses.append(avg_loss)
    accuracies.append(acc)  # 新增准确率记录
    print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, Train Accuracy = {acc:.4f}")


# 测试函数，包含混淆矩阵
def test(model, device, test_loader):
    model.eval()
    correct = 0
    total_loss = 0.0
    num_samples = len(test_loader.dataset)
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for t_data, t_target in tqdm(test_loader, desc="Testing"):
            t_data, t_target = t_data.to(device), t_target.to(device)
            pred = model(t_data)
            loss = loss_fn(pred, t_target)
            total_loss += loss.item() * t_data.size(0)
            pred_class = pred.argmax(dim=1)
            correct += pred_class.eq(t_target).sum().item()

            all_preds.extend(pred_class.cpu().numpy())
            all_targets.extend(t_target.cpu().numpy())

    acc = correct / num_samples
    avg_loss = total_loss / num_samples
    print(f"Test Accuracy: {acc:.4f}, Test Loss: {avg_loss:.4f}")

    # 绘制混淆矩阵
    cm = confusion_matrix(all_targets, all_preds)
    df_cm = pd.DataFrame(cm, index=['Cat', 'Dog'], columns=['Cat', 'Dog'])
    plt.figure(figsize=(6, 5))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(f'output/{model_name}_confusion_matrix.png')
    plt.close()


# 训练和测试
num_epochs = 10

if __name__ == "__main__":
    start_time = time()
    losses = []
    accuracies = []  # 新增准确率列表

    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch, losses, accuracies)

    test(model, device, test_loader)
    end_time = time()
    print(f"Total Training Time: {end_time - start_time:.2f} seconds")

    # 绘制损失曲线
    plt.plot(range(1, num_epochs + 1), losses, marker='o', linestyle='-', color='b')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid()
    plt.savefig(f'output/{model_name}_training_loss_curve.png')
    plt.close()
    # 绘制准确率曲线
    plt.plot(range(1, num_epochs + 1), accuracies, marker='s', linestyle='-', color='g')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Over Epochs')
    plt.grid()
    plt.savefig(f'output/{model_name}_training_accuracy_curve.png')
    plt.close()
