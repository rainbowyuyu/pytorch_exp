# rainbow_yu exp7 🐋✨
# 基于多头注意力机制的文本分类 + 可视化

from models import TransformerClassifier, MHAClassifier
from dataloader_no_torchtext import train_iter, test_iter, collate_batch, vocab
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import time

start_time = time.time()


# 数据加载
train_dataloader = DataLoader(list(train_iter)[:2000], batch_size=16, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(list(test_iter)[:1000], batch_size=16, collate_fn=collate_batch)

# 模型定义
num_heads = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = TransformerClassifier(len(vocab), embed_dim=64, num_heads=num_heads).to(device)
model = MHAClassifier(len(vocab), embed_dim=64, num_heads=num_heads).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 存储每轮数据
train_losses = []
test_accuracies = []

epochs = 5

# 训练
for epoch in range(epochs):
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
plt.plot(range(1, epochs+1), train_losses, marker='o', label='Training Loss')
plt.title('Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.savefig(f"output/loss_{epochs}_{num_heads}_nn.png")

# 可视化测试准确率
plt.figure()
plt.plot(range(1, epochs+1), test_accuracies, marker='o', color='green', label='Test Accuracy')
plt.title('Test Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.savefig(f"output/accuracy_{epochs}_{num_heads}_nn.png")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"总运行时间: {elapsed_time:.2f} 秒")

with torch.no_grad():
    for text_batch, label_batch in test_dataloader:
        text_sample = text_batch[0].unsqueeze(0).to(device)  # shape: [1, T]
        label_sample = label_batch[0].unsqueeze(0).to(device)
        output = model(text_sample)
        attn_weights = model.attn_weights  # shape: [1, T, T] or [1, heads, T, T]

        if attn_weights.dim() == 4:
            attn_weights = attn_weights[0, 0]  # 取第一个样本、第一头
        elif attn_weights.dim() == 3:
            attn_weights = attn_weights[0]  # 取第一个样本
        else:
            raise ValueError(f"unexpected attention shape: {attn_weights.shape}")

        tokens = [vocab.lookup_token(idx.item()) for idx in text_sample[0]]

        plt.figure(figsize=(8, 6))
        plt.imshow(attn_weights.cpu(), cmap='viridis')
        plt.colorbar()
        plt.xticks(range(len(tokens)), tokens, rotation=90)
        plt.yticks(range(len(tokens)), tokens)
        plt.title("Attention Map (Head 0)")
        plt.tight_layout()
        plt.savefig("output/attention_sample.png")
        plt.show()
        break  # 只画一个样本
