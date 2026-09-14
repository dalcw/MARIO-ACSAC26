import torch
import torch.nn as nn
import torch.nn.functional as F

# channel self attention
class ChannelSelfAttn(nn.Module):
    def __init__(self, width, height, embed_dim, dropout=0.1):
        super().__init__()

        self.embedding = nn.Linear(width*height, embed_dim)

        self.linear_q = nn.Linear(embed_dim, embed_dim)
        self.linear_k = nn.Linear(embed_dim, embed_dim)
        self.linear_v = nn.Linear(embed_dim, embed_dim)

        self.linear_out = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        self.embed_dim = embed_dim
        
    def forward(self, smashed_data):

        # data flatten: 2차원 형태의 dim을 깬다
        smashed_data = torch.flatten(smashed_data, start_dim=2)
        smashed_embed = self.embedding(smashed_data)

        smashed_query = self.linear_q(smashed_embed)
        smashed_key = self.linear_k(smashed_embed)
        smashed_value = self.linear_v(smashed_embed)

        # attn score
        attn_weight = F.softmax(torch.matmul(smashed_query, smashed_key.transpose(-1, -2)) / (self.embed_dim ** 0.5), dim=-1)
        attn_weight = self.dropout(attn_weight)
        
        context_matrix = torch.matmul(attn_weight, smashed_value)
        context_matrix = self.linear_out(context_matrix)
        
        return context_matrix


# reverse decoder
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
        """
        x: (B, C, E)
        return: (B, C, H, W)
        """
        B, C, E = x.shape

        x = self.relu(self.linear1(x))      # (B, C, 2E)
        x = self.dropout(x)
        x = self.relu(self.linear2(x))      # (B, C, 4E)
        x = self.dropout(x)
        x = self.linear3(x)                 # (B, C, H*W)

        x = x.view(B, C, self.width, self.height)
        return x


class Adapter(nn.Module):
    def __init__(self, channel, width, height, embed_dim, dropout=0.1):
        super().__init__()

        assert embed_dim % 2 == 0, "embed_dim must be even (for //2 bottleneck)."

        self.channel = channel
        self.width = width
        self.height = height
        self.embed_dim = embed_dim

        # encoder / decoder (pub/priv)
        self.encoder = ChannelSelfAttn(
            width=width,
            height=height,
            embed_dim=embed_dim,
            dropout=dropout,
        )
        self.pub_decoder = ChannelDecoder(
            width=width,
            height=height,
            embed_dim=embed_dim,
            dropout=dropout,
        )
        self.priv_decoder = ChannelDecoder(
            width=width,
            height=height,
            embed_dim=embed_dim,
            dropout=dropout,
        )

        # bottleneck: E -> E/2
        self.info_bottleneck = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
        )

        # expand back: E/2 -> E (pub/priv 따로 둠)
        self.info_expand_pub = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim),
            nn.ReLU(inplace=True),
        )
        self.info_expand_priv = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim),
            nn.ReLU(inplace=True),
        )

        # flattened bottleneck dimension
        z_dim = channel * (embed_dim // 2)
        self.z_dim = z_dim

        self.linear_z = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.ReLU(inplace=True),
        )
        self.linear_z_pub = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.ReLU(inplace=True),
        )

        # VAE parameters
        self.linear_mu = nn.Linear(z_dim, z_dim)
        self.linear_logvar = nn.Linear(z_dim, z_dim)

    def forward(self, smashed_data):
        """
        smashed_data : (B, C, H, W)
        return:
            x_pub      : (B, C, H, W)  # 서버로 갈 public representation (noisy)
            x_priv_rec : (B, C, H, W)  # priv branch reconstruction (attacker)
            z_mu       : (B, z_dim)
            z_logvar   : (B, z_dim)
            z_pub      : (B, z_dim)
            z_priv     : (B, z_dim)
        """
        B = smashed_data.shape[0]

        # 1) encoding: (B, C, H, W) -> (B, C, E)
        smashed_encoded = self.encoder(smashed_data)  # (B, C, E)

        # 2) bottleneck: (B, C, E) -> (B, C, E/2) -> flatten to (B, z_dim)
        smashed_bottleneck = self.info_bottleneck(smashed_encoded)              # (B, C, E/2)
        smashed_bottleneck_flat = torch.flatten(smashed_bottleneck, start_dim=1)  # (B, z_dim)

        # 3) information separation in flattened space
        z = self.linear_z(smashed_bottleneck_flat)      # (B, z_dim)
        z_pub = self.linear_z_pub(z)                    # (B, z_dim)
        z_priv = z - z_pub                              # (B, z_dim)

        # 4) VAE-style public branch: mu, logvar, reparameterization
        z_mu = self.linear_mu(z_pub)                    # (B, z_dim)
        z_logvar = self.linear_logvar(z_pub)            # (B, z_dim)

        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        z_pub_sample = z_mu + std * eps                 # (B, z_dim)

        # 5) decode public: (B, z_dim) -> (B, C, E/2) -> (B, C, E) -> (B, C, H, W)
        z_pub_sample_reshaped = z_pub_sample.reshape(B, self.channel, -1)   # (B, C, E/2)
        pub_feat = self.info_expand_pub(z_pub_sample_reshaped)             # (B, C, E)
        x_pub = self.pub_decoder(pub_feat)                                 # (B, C, H, W)

        # 6) decode private (attacker): (B, z_dim) -> (B, C, E/2) -> (B, C, E) -> (B, C, H, W)
        z_priv_reshaped = z_priv.reshape(B, self.channel, -1)              # (B, C, E/2)
        priv_feat = self.info_expand_priv(z_priv_reshaped)                 # (B, C, E)
        x_priv_rec = self.priv_decoder(priv_feat)                          # (B, C, H, W)

        return x_pub, x_priv_rec, z_mu, z_logvar, z_pub, z_priv