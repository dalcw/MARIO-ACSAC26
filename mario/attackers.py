import torch
import torch.nn as nn


class SmashedReconstructionAttacker(nn.Module):
    """
    Reconstruction attacker for CIFAR-10-style smashed data.

    Input : (B, 128, 16, 16)
    Output: (B, 3, 32, 32)
    """
    def __init__(self, in_channels=128, img_channels=3):
        super().__init__()

        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.dec = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, img_channels, kernel_size=3, padding=1),
        )

    def forward(self, smashed_like):
        x = self.enc(smashed_like)
        x = self.dec(x)
        return x


class SmashedReconstructionAttacker224(nn.Module):
    """
    Reconstruction attacker for 224x224 image tasks such as X-ray/CelebA.

    Input : (B, 128, 28, 28)
    Output: (B, 3, 224, 224)
    """
    def __init__(self, in_channels=128, img_channels=3, out_activation=None):
        super().__init__()

        def conv_bn_relu(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        def up_block(cin, c_mid, cout):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                conv_bn_relu(cin, c_mid),
                conv_bn_relu(c_mid, cout),
            )

        self.enc = nn.Sequential(
            conv_bn_relu(in_channels, 256),
            conv_bn_relu(256, 128),
        )

        self.dec = nn.Sequential(
            up_block(128, 128, 64),
            up_block(64, 64, 32),
            up_block(32, 32, 16),
            nn.Conv2d(16, img_channels, kernel_size=3, padding=1),
        )

        if out_activation is None:
            self.out_act = nn.Identity()
        elif out_activation.lower() == "sigmoid":
            self.out_act = nn.Sigmoid()
        elif out_activation.lower() == "tanh":
            self.out_act = nn.Tanh()
        else:
            raise ValueError("out_activation must be one of {None, 'sigmoid', 'tanh'}")

    def forward(self, smashed_like):
        x = self.enc(smashed_like)
        x = self.dec(x)
        return self.out_act(x)


class SmashedBinaryClassifier(nn.Module):
    """
    Property inference attacker over smashed data.

    Input : (B, 128, H, W), commonly 16x16 or 28x28 in the experiments.
    Output: (B, 2)
    """
    def __init__(self, in_ch=128, num_classes=2, dropout=0.2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        logits = self.head(x)
        return logits

