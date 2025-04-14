### [exp4](exp4)
猫狗实验 Cats and Dogs
 
> 选取了两个模型进行实验
- My_CNN(课堂实验资料)
```python
class My_CNN(nn.Module):
    def __init__(self):
        super(My_CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(10, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```
> 完成My_CNN的训练，epcho:20，lr:1e-4，batch:64

- Better_CNN(自己搭建的模型)
```python
class Better_CNN(nn.Module):
    def __init__(self):
        super(Better_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # [32, 128, 128]
        x = F.max_pool2d(x, 2, 2)            # [32, 64, 64]

        x = F.relu(self.bn2(self.conv2(x)))  # [64, 64, 64]
        x = F.max_pool2d(x, 2, 2)            # [64, 32, 32]

        x = F.relu(self.bn3(self.conv3(x)))  # [128, 32, 32]
        x = F.max_pool2d(x, 2, 2)            # [128, 16, 16]

        x = x.view(x.size(0), -1)            # [batch_size, 128, 16, 16]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # 直接输出 logits（不用 softmax）
```
> 完成Better_CNN的训练，epcho:10，lr:1e-4，batch:64

<p align="center">
    <img src="exp4%2FBetter_CNN_confusion_matrix.png" width="45%"/>
    <img src="exp4%2FBetter_CNN_training_loss_curve.png" width="45%"/>
</p>