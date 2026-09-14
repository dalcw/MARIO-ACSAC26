import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock18(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class ClientResNet18(nn.Module):
    """
    Copied split client used by Vanilla/Noise/NoPeek/R3eLU/Our.
    input_size=32 gives CIFAR-style stem, input_size=224 gives ImageNet-style stem.
    """
    def __init__(self, input_size=32):
        super().__init__()
        assert input_size in (32, 224), "input_size must be 32 or 224"
        self.in_planes = 64

        if input_size == 224:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.maxpool = None

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)

    def _make_layer(self, planes, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock18(self.in_planes, planes, stride=s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        if self.maxpool is not None:
            x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class DiscoClientResNet18(nn.Module):
    """
    Copied DISCO split client. DISCO preprocessing emits d^2=64 channels.
    """
    def __init__(self):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)

    def _make_layer(self, planes, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock18(self.in_planes, planes, stride=s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class ServerResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 128
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock18(self.in_planes, planes, stride=s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ChannelSelfAttn(nn.Module):
    def __init__(self, width, height, embed_dim, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(width * height, embed_dim)
        self.linear_q = nn.Linear(embed_dim, embed_dim)
        self.linear_k = nn.Linear(embed_dim, embed_dim)
        self.linear_v = nn.Linear(embed_dim, embed_dim)
        self.linear_out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim

    def forward(self, smashed_data):
        smashed_data = torch.flatten(smashed_data, start_dim=2)
        smashed_embed = self.embedding(smashed_data)
        query = self.linear_q(smashed_embed)
        key = self.linear_k(smashed_embed)
        value = self.linear_v(smashed_embed)
        weight = F.softmax(torch.matmul(query, key.transpose(-1, -2)) / (self.embed_dim ** 0.5), dim=-1)
        weight = self.dropout(weight)
        context = torch.matmul(weight, value)
        return self.linear_out(context)


class ChannelDecoder(nn.Module):
    def __init__(self, width, height, embed_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, embed_dim * 2)
        self.linear2 = nn.Linear(embed_dim * 2, embed_dim * 4)
        self.linear3 = nn.Linear(embed_dim * 4, width * height)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.width = width
        self.height = height

    def forward(self, x):
        b, c, _ = x.shape
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.relu(self.linear2(x))
        x = self.dropout(x)
        x = self.linear3(x)
        return x.view(b, c, self.width, self.height)


class Adapter(nn.Module):
    """
    Copied MARIO/Our adapter.
    """
    def __init__(self, channel, width, height, embed_dim, dropout=0.1):
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even."
        self.channel = channel
        self.width = width
        self.height = height
        self.embed_dim = embed_dim
        self.encoder = ChannelSelfAttn(width, height, embed_dim, dropout)
        self.pub_decoder = ChannelDecoder(width, height, embed_dim, dropout)
        self.priv_decoder = ChannelDecoder(width, height, embed_dim, dropout)
        self.info_bottleneck = nn.Sequential(nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(inplace=True))
        self.info_expand_pub = nn.Sequential(nn.Linear(embed_dim // 2, embed_dim), nn.ReLU(inplace=True))
        self.info_expand_priv = nn.Sequential(nn.Linear(embed_dim // 2, embed_dim), nn.ReLU(inplace=True))

        z_dim = channel * (embed_dim // 2)
        self.z_dim = z_dim
        self.linear_z = nn.Sequential(nn.Linear(z_dim, z_dim), nn.ReLU(inplace=True))
        self.linear_z_pub = nn.Sequential(nn.Linear(z_dim, z_dim), nn.ReLU(inplace=True))
        self.linear_mu = nn.Linear(z_dim, z_dim)
        self.linear_logvar = nn.Linear(z_dim, z_dim)

    def forward(self, smashed_data):
        b = smashed_data.shape[0]
        encoded = self.encoder(smashed_data)
        bottleneck = self.info_bottleneck(encoded)
        flat = torch.flatten(bottleneck, start_dim=1)
        z = self.linear_z(flat)
        z_pub = self.linear_z_pub(z)
        z_priv = z - z_pub

        z_mu = self.linear_mu(z_pub)
        z_logvar = self.linear_logvar(z_pub)
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        z_pub_sample = z_mu + std * eps

        z_pub_sample = z_pub_sample.reshape(b, self.channel, -1)
        pub_feat = self.info_expand_pub(z_pub_sample)
        x_pub = self.pub_decoder(pub_feat)

        z_priv = z_priv.reshape(b, self.channel, -1)
        priv_feat = self.info_expand_priv(z_priv)
        x_priv_rec = self.priv_decoder(priv_feat)
        return x_pub, x_priv_rec, z_mu, z_logvar, z_pub, z_priv


class AddNoise(nn.Module):
    def __init__(self, mu=0, sigma=50):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, z):
        noise = torch.rand_like(z) * self.sigma + self.mu
        return z + noise


class R3eLU(nn.Module):
    def __init__(self, K_ratio=0.1, C=1.0, epsilon_p=0.5, noise_scale=0.5):
        super().__init__()
        self.K_ratio = K_ratio
        self.C = C
        self.epsilon_p = epsilon_p
        self.noise_scale = noise_scale

    def forward(self, x):
        b, c, h, w = x.shape
        n = c * h * w
        k = max(1, int(self.K_ratio * n))
        v = x.view(b, -1)
        _, topk_idx = v.abs().topk(k, dim=1)
        mask = torch.zeros_like(v)
        mask.scatter_(1, topk_idx, 1.0)
        v_hat = torch.clamp(v * mask, -self.C, self.C)

        norm_inf = v_hat.abs().max(dim=1, keepdim=True)[0] + 1e-6
        v_normed = v_hat / norm_inf
        alpha = (math.exp(self.epsilon_p) - 1.0) / (math.exp(self.epsilon_p) + 1.0)
        p = torch.clamp(0.5 + 0.5 * alpha * v_normed, 0.0, 1.0)
        active = (torch.rand_like(v_hat) < p).float()

        if self.noise_scale > 0:
            lap = torch.distributions.Laplace(loc=0.0, scale=self.noise_scale)
            noise = lap.sample(v_hat.shape).to(v_hat.device)
        else:
            noise = torch.zeros_like(v_hat)

        out = F.relu(v_hat + noise) * active
        return out.view(b, c, h, w)


class DistCorrelation(nn.Module):
    def pairwise_distances(self, x):
        x_norm = (x ** 2).sum(1).view(-1, 1)
        dist = x_norm + x_norm.view(1, -1) - 2.0 * torch.mm(x, x.transpose(0, 1))
        return torch.clamp(dist, 0.0, float("inf"))

    def forward(self, x, z):
        x_flat = x.view(x.shape[0], -1)
        z_flat = z.view(z.shape[0], -1)
        a = self.pairwise_distances(x_flat)
        b = self.pairwise_distances(z_flat)
        a_centered = a - a.mean(dim=0).unsqueeze(1) - a.mean(dim=1) + a.mean()
        b_centered = b - b.mean(dim=0).unsqueeze(1) - b.mean(dim=1) + b.mean()
        dcov = torch.sqrt(torch.sum(a_centered * b_centered) / a.shape[1] ** 2)
        var_aa = torch.sqrt(torch.sum(a_centered * a_centered) / a.shape[1] ** 2)
        var_bb = torch.sqrt(torch.sum(b_centered * b_centered) / a.shape[1] ** 2)
        return dcov / torch.sqrt(var_aa * var_bb + 1e-8)


class NoPeekClient(nn.Module):
    def __init__(self, client_model, alpha=0.8):
        super().__init__()
        self.client_model = client_model
        self.alpha = alpha
        self.dcor_criterion = DistCorrelation()
        self.last_dcor = 0.0

    def forward(self, x):
        z = self.client_model(x)
        if self.alpha > 0:
            dcor_loss = self.dcor_criterion(x, z)
            self.last_dcor = dcor_loss.item()
            return z, dcor_loss
        return z, torch.tensor(0.0, device=x.device)


class PreProcessingModule(nn.Module):
    def __init__(self, d=8, toggle=True, kernel_size=3):
        super().__init__()
        self.d = d
        self.toggle = toggle
        self.F = d * d
        self.conv = nn.Conv2d(3, self.F, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        b, c, h, w = x.shape
        if not self.toggle:
            return self.conv(x)

        assert h % self.d == 0 and w % self.d == 0, "H,W must be divisible by d"
        ph, pw = h // self.d, w // self.d
        patches = F.unfold(x, kernel_size=(ph, pw), stride=(ph, pw))
        patches = patches.transpose(1, 2).contiguous()
        patches = patches.view(b * (self.d * self.d), c, ph, pw)
        patches = F.interpolate(patches, size=(h, w), mode="bilinear", align_corners=False)
        feat = self.conv(patches)
        feat = feat.mean(dim=1, keepdim=True)
        return feat.view(b, self.d * self.d, h, w)


class PruningNetwork(nn.Module):
    def __init__(self, channels=128, pruning_ratio=0.6, pruning_style="learnable", temp=1 / 30):
        super().__init__()
        self.channels = channels
        self.pruning_ratio = pruning_ratio
        self.pruning_style = pruning_style
        self.temp = temp
        if pruning_style == "learnable":
            self.score_net = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(channels, channels))
        elif pruning_style == "random":
            self.decoy = nn.Linear(1, 1)

    @staticmethod
    def get_random_prune_idx(c, ratio, device):
        k = int(c * ratio)
        return torch.randperm(c, device=device)[:k]

    def forward(self, x):
        if self.pruning_ratio <= 0:
            return x
        if self.pruning_style == "random":
            _, c, _, _ = x.shape
            idx = self.get_random_prune_idx(c, self.pruning_ratio, x.device)
            y = x.clone()
            y[:, idx] = 0.0
            return y

        score = self.score_net(x)
        _, c = score.shape
        k = min(max(int(c * self.pruning_ratio), 0), c - 1)
        sorted_score, _ = torch.sort(score, dim=1)
        thr = sorted_score[:, k].unsqueeze(1)
        soft = torch.sigmoid((score - thr) / self.temp)
        hard = (soft > 0.5).float()
        b = hard + soft - soft.detach()
        return x * b.unsqueeze(-1).unsqueeze(-1)


class RepresentationExposureModel(nn.Module):
    """
    Wraps each method and exposes exactly the representation observed by the attacker.
    """
    def __init__(self, method, input_size):
        super().__init__()
        self.method = method
        self.input_size = input_size

        if method == "disco":
            self.preprocessing_module = PreProcessingModule(d=8, toggle=True, kernel_size=3)
            self.client_model = DiscoClientResNet18()
            self.pruning_network = PruningNetwork(channels=128, pruning_ratio=0.6, pruning_style="learnable", temp=1 / 30)
            self.server_model = ServerResNet18(num_classes=10)
        else:
            self.client_model = ClientResNet18(input_size=input_size)
            self.server_model = ServerResNet18(num_classes=10)
            if method == "our":
                width = 16 if input_size == 32 else 28
                embed_dim = 128 if input_size == 32 else 256
                self.adapter_model = Adapter(channel=128, width=width, height=width, embed_dim=embed_dim)
            elif method == "noise":
                self.adapter_model = AddNoise(mu=0, sigma=50)
            elif method == "r3elu":
                self.adapter_model = R3eLU(K_ratio=0.1, C=1.0, epsilon_p=0.5, noise_scale=0.5)
            elif method == "nopeek":
                self.nopeek_client = NoPeekClient(self.client_model, alpha=0.8)
            elif method != "vanilla":
                raise ValueError(f"Unknown method: {method}")

    def load_checkpoint(self, checkpoint):
        if self.method == "disco":
            if "preprocessing_module" in checkpoint:
                self.preprocessing_module.load_state_dict(checkpoint["preprocessing_module"])
            self.client_model.load_state_dict(checkpoint["client_state_dict"])
            self.pruning_network.load_state_dict(checkpoint["pruning_network"])
            self.server_model.load_state_dict(checkpoint["server_state_dict"])
            return

        if self.method == "nopeek":
            key = "nopeed_client_state_dict" if "nopeed_client_state_dict" in checkpoint else "nopeek_client_state_dict"
            self.nopeek_client.load_state_dict(checkpoint[key])
            self.server_model.load_state_dict(checkpoint["server_state_dict"])
            return

        self.client_model.load_state_dict(checkpoint["client_state_dict"])
        self.server_model.load_state_dict(checkpoint["server_state_dict"])
        if hasattr(self, "adapter_model") and "adapter_state_dict" in checkpoint:
            self.adapter_model.load_state_dict(checkpoint["adapter_state_dict"])

    def forward(self, x):
        if self.method == "disco":
            a = self.preprocessing_module(x)
            z_hat = self.client_model(a)
            return self.pruning_network(z_hat)
        if self.method == "our":
            smashed = self.client_model(x)
            x_pub, _, _, _, _, _ = self.adapter_model(smashed)
            return x_pub
        if self.method in {"noise", "r3elu"}:
            z = self.client_model(x)
            return self.adapter_model(z)
        if self.method == "nopeek":
            return self.nopeek_client.client_model(x)
        return self.client_model(x)
