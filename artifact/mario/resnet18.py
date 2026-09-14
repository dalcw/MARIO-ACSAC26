import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock18(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ClientResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_planes = 64

        # CIFAR-10 스타일: 3x3 conv, stride=1, no maxpool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64,  2, stride=1)  # 32x32
        self.layer2 = self._make_layer(128, 2, stride=2)  # 16x16

    def _make_layer(self, planes, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock18(self.in_planes, planes, stride=s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # (B, 64, 32, 32)
        x = self.layer1(x)                   # (B, 64, 32, 32)
        x = self.layer2(x)                   # (B, 128, 16, 16)
        return x

class ServerResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 128  # cut 이후 채널 수와 맞춤

        self.layer3 = self._make_layer(256, 2, stride=2)  # 16x16 -> 8x8
        self.layer4 = self._make_layer(512, 2, stride=2)  #  8x8 -> 4x4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(512, num_classes)

    def _make_layer(self, planes, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers  = []
        for s in strides:
            layers.append(BasicBlock18(self.in_planes, planes, stride=s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, 128, 16, 16)
        x = self.layer3(x)              # (B, 256, 8, 8)
        x = self.layer4(x)              # (B, 512, 4, 4)
        x = self.avgpool(x)             # (B, 512, 1, 1)
        x = torch.flatten(x, 1)         # (B, 512)
        x = self.fc(x)                  # (B, num_classes)
        return x