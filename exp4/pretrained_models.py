import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import alexnet, AlexNet_Weights
import torch.nn.functional as F

# 定义 CNN 网络
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
        x = F.max_pool2d(x, 2, 2)  # [32, 64, 64]

        x = F.relu(self.bn2(self.conv2(x)))  # [64, 64, 64]
        x = F.max_pool2d(x, 2, 2)  # [64, 32, 32]

        x = F.relu(self.bn3(self.conv3(x)))  # [128, 32, 32]
        x = F.max_pool2d(x, 2, 2)  # [128, 16, 16]

        x = x.view(x.size(0), -1)  # 展平成 [batch_size, 128, 16, 16]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # 直接输出 logits（不用 softmax）

class CustomVGGNet(nn.Module):
    def __init__(self):
        super(CustomVGGNet, self).__init__()

        weights = VGG16_Weights.DEFAULT
        vgg = vgg16(weights=weights)
        for param in vgg.features.parameters():
            param.requires_grad = False

        self.vgg = vgg.features
        self.classifier = nn.Sequential(
            nn.Linear(8192, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, 2)  # 移除Softmax
        )

    def forward(self, x):
        x = self.vgg(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class CustomResNet(nn.Module):
    def __init__(self):
        super(CustomResNet, self).__init__()

        # 加载预训练的 ResNet18
        weights = ResNet18_Weights.DEFAULT
        resnet = resnet18(weights=weights)
        # 冻结除最后一层外的参数
        for param in resnet.parameters():
            param.requires_grad = False
        # 解冻最后一层（layer4）和 fc 层用于微调
        for param in resnet.layer4.parameters():
            param.requires_grad = True
        for param in resnet.fc.parameters():
            param.requires_grad = True
        # 替换分类器
        in_features = resnet.fc.in_features
        resnet.fc = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)  # 输出为2类：猫和狗
        )
        self.resnet = resnet

    def forward(self, x):
        return self.resnet(x)


class CustomAlexNet(nn.Module):
    def __init__(self):
        super(CustomAlexNet, self).__init__()

        # 加载预训练权重
        weights = AlexNet_Weights.DEFAULT
        alex = alexnet(weights=weights)

        # 冻结特征提取部分参数
        for param in alex.features.parameters():
            param.requires_grad = False

        # 替换分类器部分（原来是输出1000类）
        alex.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)  # 输出2类：猫和狗
        )

        self.alexnet = alex

    def forward(self, x):
        return self.alexnet(x)