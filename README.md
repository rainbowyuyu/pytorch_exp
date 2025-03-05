# pytorch实验
## rainbow鱼 🐋✨

---

### exp2
全连接神经网络
```python
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
```

> 完成FashionMnist的训练，epcho:20，lr:0.01，batch:64

![confusion_matrix.png](exp2%2Fconfusion_matrix.png)

![training_loss_curve.png](exp2%2Ftraining_loss_curve.png)