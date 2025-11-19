import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
import pywt
from einops import rearrange
from torchvision.ops.deform_conv import DeformConv2d
import math
from torchvision.models.mobilenetv2 import _make_divisible
import einops
from einops import rearrange


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)





class SpatialGate(nn.Module):
    # """轻量化空间注意力（参数量<1K）"""
    def __init__(self, channels):
        super().__init__()
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=5,
                                 padding=2, groups=channels)
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x):
        mask = self.dw_conv(x)
        mask = self.conv(mask)
        return x * torch.sigmoid(mask)

    ##########################################################################


## Gated-Dconv Feed-Forward Network (GDFN)
class GDFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(GDFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

    ##########################################################################


## Divide and Multiply Feed-Forward Network (DMFN)
class DMFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(DMFN, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.patch_size = 8

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                      groups=hidden_features * 2, bias=bias),
            nn.GELU(),
            nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=1, bias=bias)
        )

        self.fft = nn.Parameter(torch.ones((hidden_features * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        # 添加局部特征增强（可选）
        self.local_conv = nn.Sequential(
            nn.Conv2d(hidden_features * 2, dim, kernel_size=3, padding=1, groups=dim, bias=bias),  # 深度可分离卷积
            nn.GELU()
        )

    def forward(self, x):
        x = self.project_in(x)
        x_patch = rearrange(x, 'b c (h p1) (w p2) -> b c h w p1 p2', p1=self.patch_size, p2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w p1 p2 -> b c (h p1) (w p2)', p1=self.patch_size, p2=self.patch_size)

        # 加入局部卷积增强
        x_local = self.local_conv(x)

        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2+F.gelu(x2)*x1+x2*x1
        x = self.project_out(x + x_local)  # 融合局部信息
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias,m):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class TransBlock(nn.Module):
    def __init__(self, dim, num_heads, kernel, dilation, ffn_expansion_factor, bias, sa=True):
        super(TransBlock, self).__init__()

        self.attn = Attention(dim, num_heads, bias, dilation)
        self.norm1 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.norm2 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.ffn = DMFN(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias)
        self.sa = sa
        # 引入局部卷积模块
        self.local_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=bias),  # 深度可分离
            nn.GELU()
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        # 局部细节增强
        local_feat = self.local_conv(x)
        x = x + local_feat
        x = x + self.ffn(self.norm2(x))

        return x


# 引入 SE 模块 (Squeeze and Excitation)
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = torch.mean(x, dim=[2, 3], keepdim=True)  # Global average pooling
        y = self.fc1(y)
        y = self.fc2(y)
        return x * self.sigmoid(y) + x


# CBAM 模块 (Convolutional Block Attention Module)
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = SEBlock(channels, reduction)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_ca = self.channel_attention(x)
        avg_pool = torch.mean(x_ca, dim=1, keepdim=True)
        max_pool, _ = torch.max(x_ca, dim=1, keepdim=True)
        spatial_concat = torch.cat([avg_pool, max_pool], dim=1)
        x_sa = self.spatial_attention(spatial_concat)
        return x_ca * x_sa + x






class Embeddings_output(nn.Module):
    def __init__(self, dim, num_blocks, kernel, heads, bias):
        super(Embeddings_output, self).__init__()
        self.activation = nn.LeakyReLU(0.2, True)
        self.de_trans_level3 = nn.Sequential(*[

            TransBlock(dim * 2 ** 2, heads[3], kernel, 9, 1, bias=bias) for i in range(num_blocks[3])

        ])

        self.up3_2 = nn.Sequential(
            nn.ConvTranspose2d(dim * 2 ** 2, dim * 2, kernel_size=4, stride=2, padding=1, bias=bias),
            self.activation,
        )

        self.fusion_level2 = LGFF(dim * 4, dim * 2, 1, bias)

        self.de_trans_level2 = nn.Sequential(*[
            TransBlock(dim * 2, heads[3], kernel, 9, 1, bias=bias) for i in range(num_blocks[3])
        ])

        self.up2_1 = nn.Sequential(
            nn.ConvTranspose2d(dim * 2, dim, kernel_size=4, stride=2, padding=1, bias=bias),
            self.activation,
        )

        self.fusion_level1 = LGFF(dim * 2, dim * 1, 1, bias)

        self.de_trans_level1 = nn.Sequential(*[
            TransBlock(dim, heads[1], kernel, 9, 1, bias=bias) for i in range(num_blocks[1])
        ])

        self.refinement = nn.Sequential(*[
            TransBlock(dim, heads[0], kernel, 9, 1, bias=bias) for i in range(num_blocks[0])
        ])
        self.de_layer1_2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim, dim, kernel_size=3, padding=1))
        self.output = nn.Sequential(
            nn.Conv2d(dim, 3, kernel_size=3, padding=1, bias=bias),
            self.activation,
        )

    def forward(self, x, residual_1, residual_2):
        hx = self.de_trans_level3(x)
        hx = self.up3_2(hx)
        hx = self.fusion_level2(torch.cat((hx, residual_2), dim=1))
        hx = self.de_trans_level2(hx)
        hx = self.up2_1(hx)
        hx = self.fusion_level1(torch.cat((hx, residual_1), dim=1))
        hx = self.activation(self.de_layer1_2(hx) + hx)
        hx = self.output(hx)
        return hx

class Restormer(nn.Module):
    def __init__(self,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_heads=[1, 2, 4, 8],
                 kernel=7,
                 ffn_expansion_factor=2.66,
                 bias=False):
        super(Restormer, self).__init__()

        self.encoder = Embeddings(dim)

        self.multi_scale_fusion_level1 = LGFF(dim * 7, dim * 1, ffn_expansion_factor, bias)
        self.multi_scale_fusion_level2 = LGFF(dim * 7, dim * 2, ffn_expansion_factor, bias)

        self.decoder = Embeddings_output(dim, num_blocks, kernel,
                                         num_heads, bias)

    def forward(self, x):


        hx, res1, res2 = self.encoder(x)



        res2_1 = F.interpolate(res2, scale_factor=2)
        res1_2 = F.interpolate(res1, scale_factor=0.5)
        hx_2 = F.interpolate(hx, scale_factor=2)
        hx_1 = F.interpolate(hx_2, scale_factor=2)


        res1 = F.relu(self.multi_scale_fusion_level1(torch.cat((res1, res2_1, hx_1), dim=1)))
        res2 = F.relu(self.multi_scale_fusion_level2(torch.cat((res1_2, res2, hx_2), dim=1)))

        hx = self.decoder(hx, res1, res2)

        return hx + x


