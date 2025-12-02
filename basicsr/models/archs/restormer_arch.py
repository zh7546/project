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



