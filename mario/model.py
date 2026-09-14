import torch.nn as nn

from .adapter import Adapter
from .resnet18 import ClientResNet18, ServerResNet18


class MarioCIFAR10(nn.Module):
    def __init__(self, num_classes=10, embed_dim=128, dropout=0.1):
        super().__init__()
        self.client_model = ClientResNet18()
        self.adapter_model = Adapter(channel=128, width=16, height=16, embed_dim=embed_dim, dropout=dropout)
        self.server_model = ServerResNet18(num_classes=num_classes)

    def forward(self, x):
        smashed_data = self.client_model(x)
        x_pub, x_priv_rec, z_mu, z_logvar, z_pub, z_priv = self.adapter_model(smashed_data)
        preds = self.server_model(x_pub)
        return {
            "preds": preds,
            "smashed_data": smashed_data,
            "x_pub": x_pub,
            "x_priv_rec": x_priv_rec,
            "z_mu": z_mu,
            "z_logvar": z_logvar,
            "z_pub": z_pub,
            "z_priv": z_priv,
        }


def build_mario_cifar10(num_classes=10, embed_dim=128, dropout=0.1):
    return MarioCIFAR10(num_classes=num_classes, embed_dim=embed_dim, dropout=dropout)
