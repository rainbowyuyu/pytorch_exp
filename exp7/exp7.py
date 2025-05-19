# rainbow_yu exp7 🐋✨
# 基于多头注意力机制的文本分类 + 可视化

from models import TransformerClassifier
from dataloader import train_iter, test_iter, collate_batch, vocab
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

# 数据加载
train_dataloader = DataLoader(list(train_iter)[:2000], batch_size=32, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(list(test_iter)[:1000], batch_size=32, collate_fn=collate_batch)

# 模型定义
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerClassifier(len(vocab), embed_dim=64, num_heads=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 存储每轮数据
train_losses = []
test_accuracies = []

# 训练
for epoch in range(5):
    model.train()
    total_loss = 0
    for text, label in train_dataloader:
        text, label = text.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(text)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    train_losses.append(total_loss)
    print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    # 每轮测试准确率
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for text, label in test_dataloader:
            text, label = text.to(device), label.to(device)
            output = model(text)
            preds = torch.argmax(output, dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)
    accuracy = correct / total
    test_accuracies.append(accuracy)
    print(f"Test Accuracy: {accuracy:.4f}")

# 可视化训练损失
plt.figure()
plt.plot(range(1, 6), train_losses, marker='o', label='Training Loss')
plt.title('Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.savefig("output/loss.png")

# 可视化测试准确率
plt.figure()
plt.plot(range(1, 6), test_accuracies, marker='o', color='green', label='Test Accuracy')
plt.title('Test Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.savefig("output/accuracy.png")
