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

<p align="center">
    <img src="exp2%2Fconfusion_matrix.png" width="45%"/>
    <img src="exp2%2Ftraining_loss_curve.png" width="45%"/>
</p>

---

### exp3
LeNet5
```python
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1, 2)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, 1)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```
> 完成Mnist的训练，epcho:10，lr:0.001，batch:64

<p align="center">
    <img src="exp3%2Fconfusion_matrix.png" width="45%"/>
    <img src="exp3%2Ftraining_loss_curve.png" width="45%"/>
</p>