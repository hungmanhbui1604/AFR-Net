import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from timm.models.vision_transformer import Block

class NormalizeModule(nn.Module):
    def __init__(self, m0=0.0, var0=1.0, eps=1e-6):
        super(NormalizeModule, self).__init__()
        self.m0 = m0
        self.var0 = var0
        self.eps = eps

    def forward(self, x):
        x_m = x.mean(dim=(1, 2, 3), keepdim=True)
        x_var = x.var(dim=(1, 2, 3), keepdim=True)
        y = (self.var0 * (x - x_m)**2 / x_var.clamp_min(self.eps)).sqrt()
        y = torch.where(x > x_m, self.m0 + y, self.m0 - y)
        return y


class SpatialTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # Matches JIPNet but uses 32 hidden units as in paper Table 1
        self.localization = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 24, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(24),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(24, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(start_dim=1),
            nn.Linear(64 * 7 * 7, 32), # Table 1 specifies Linear 32
            nn.ReLU(True),
            nn.Linear(32, 4),
        )

    def forward(self, x):
        B, C, H, W = x.size()
        z = self.localization(x.detach())
        s, z = z[:, 0], z[:, 1:]
        s.clamp_(0.8, 1.2)  # scale limitation
        pose = z.clamp(-1, 1)  # tx, ty, theta limitation
        theta, tx, ty = pose[:, 0] * 60, pose[:, 1], pose[:, 2]
        
        cos_theta = torch.deg2rad(theta).cos()
        sin_theta = torch.deg2rad(theta).sin()

        T = torch.stack(
            (
                torch.stack([s * cos_theta, s * sin_theta, tx], dim=1),
                torch.stack([s * -sin_theta, s * cos_theta, ty], dim=1),
            ),
            dim=1,
        )
        grid = F.affine_grid(T, torch.Size((B, C, H, W)), align_corners=False)
        y = F.grid_sample(x, grid, mode="bilinear", align_corners=False, padding_mode="border")
        return y


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class AttentionClassificationHead(nn.Module):
    def __init__(self, input_size=224):
        super(AttentionClassificationHead, self).__init__()
        inner_size = input_size // 16
        self.num_patches = inner_size * inner_size
        
        self.att_mlp = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(True),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 384))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, 384), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(384, 6, 4, qkv_bias=True, norm_layer=nn.LayerNorm)
            for i in range(12)
        ])
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.vit_feature = nn.Linear(384, 384)
        self.initialize_weights()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        torch.nn.init.normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def forward(self, inner_feature):
        B, C, H, W = inner_feature.shape
        inner_feature = inner_feature.flatten(2).transpose(1, 2)
        
        inner_feature = self.att_mlp(inner_feature.reshape(-1, C)).reshape(B, H * W, -1)
        inner_feature = inner_feature + self.pos_embed[:, 1:, :]
        
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        inner_feature = torch.cat((cls_tokens, inner_feature), dim=1)
        
        for blk in self.blocks:
            inner_feature = blk(inner_feature)
            
        inner_feature = inner_feature[:, 1:, :]
        inner_feature = inner_feature.reshape(B, H * W, -1).transpose(1, 2)
        inner_feature = self.avgpool(inner_feature).squeeze(-1)
        vit_feature = self.vit_feature(inner_feature)
        
        return vit_feature


class AFRNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.input_norm = NormalizeModule()
        self.stn = SpatialTransformer()
        
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Feature Extraction Branch
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1 # Conv2
        self.layer2 = resnet.layer2 # Conv3
        self.layer3 = resnet.layer3 # Conv4 (outputs 1024x14x14)
        
        # CNN Classification Head
        self.layer4 = resnet.layer4 # Conv5 (outputs 2048x7x7)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.zc_linear = nn.Linear(2048, 384)
        
        # Attention Classification Head
        self.attention_head = AttentionClassificationHead(input_size=224)

    def forward(self, x):
        x = self.input_norm(x)
        # 1. Spatial Alignment
        x = self.stn(x)
        
        # 2. Shared Feature Extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        shared_features = self.layer3(x) # [B, 1024, 14, 14]
        
        # 3. CNN Head
        c_feat = self.layer4(shared_features) # [B, 2048, 7, 7]
        c_feat = self.global_pool(c_feat).flatten(1) # [B, 2048]
        zc = self.zc_linear(c_feat) # [B, 384]
            
        # 4. Attention Head
        za = self.attention_head(shared_features) # [B, 384]
            
        return zc, za, shared_features


def get_model(model_name, model_cfg):
    if model_name == "afrnet":
        return AFRNet(
            pretrained=model_cfg.get("pretrained", True)
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
