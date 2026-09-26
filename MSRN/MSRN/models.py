"""Reconstruction and loss-prediction networks from ReFusion_spec.py.

Layer names and checkpoint keys are preserved. Fast weights are created only
by the differentiable inner update, on the model's own device.
"""

import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaConv2d(nn.Module):
    """Keep legacy checkpoint aliases; use temporary weights only in meta mode."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(*args, **kwargs)
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.stride = self.conv.stride
        self.padding = self.conv.padding
        self.dilation = self.conv.dilation
        self.groups = self.conv.groups
        self.kernel_size = self.conv.kernel_size
        self.weight = self.conv.weight
        self.bias = self.conv.bias
        self.weight_meta = None
        self.bias_meta = None

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

    def forward(self, x, meta=False):
        if meta:
            weight = self.weight if self.weight_meta is None else self.weight_meta
            bias = self.bias if self.bias_meta is None else self.bias_meta
            return F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            return self.conv(x)

def to_3d(x):
    return x.flatten(2).transpose(1, 2)

def to_4d(x, h, w):
    return x.transpose(1, 2).reshape(x.shape[0], -1, h, w)

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
        self.weight_meta = None
        self.bias_meta = None

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

    def forward(self, x, meta=False):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        if meta:
            weight = self.weight if self.weight_meta is None else self.weight_meta
            bias = self.bias if self.bias_meta is None else self.bias_meta
            return (x - mu) / torch.sqrt(sigma + 1e-05) * weight + bias
        else:
            return (x - mu) / torch.sqrt(sigma + 1e-05) * self.weight + self.bias

class LayerNorm(nn.Module):

    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = WithBias_LayerNorm(dim)

    def forward(self, x, meta=False):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x), meta=meta), h, w)

class FeedForward(nn.Module):

    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = MetaConv2d(in_channels=dim, out_channels=hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = MetaConv2d(in_channels=hidden_features * 2, out_channels=hidden_features * 2, stride=1, padding=1, kernel_size=3, groups=hidden_features * 2, bias=bias)
        self.project_out = MetaConv2d(in_channels=hidden_features, out_channels=dim, kernel_size=1, bias=bias)

    def forward(self, x, meta=False):
        x = self.project_in(x, meta=meta)
        x1, x2 = self.dwconv(x, meta=meta).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x, meta=meta)
        return x

class Attention(nn.Module):

    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature_meta = None
        self.qkv = MetaConv2d(in_channels=dim, out_channels=dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = MetaConv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = MetaConv2d(dim, dim, kernel_size=1, bias=bias)

    def named_leaves(self):
        return [('temperature', self.temperature)]

    def forward(self, x, meta=False):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x, meta=meta), meta=meta)
        q, k, v = qkv.chunk(3, dim=1)
        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        if meta:
            attn = q @ k.transpose(-2, -1) * (self.temperature if self.temperature_meta is None else self.temperature_meta)
        else:
            attn = q @ k.transpose(-2, -1) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.reshape(b, c, h, w)
        out = self.project_out(out, meta=meta)
        return out

class TransformerBlock(nn.Module):

    def __init__(self, dim, num_heads, ffn_expansion_factor, bias):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, meta=False):
        x = x + self.attn(self.norm1(x, meta=meta), meta=meta)
        x = x + self.ffn(self.norm2(x, meta=meta), meta=meta)
        return x

class MetaPReLU(nn.Module):

    def __init__(self, num_parameters: int=1, init: float=0.25) -> None:
        super(MetaPReLU, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_parameters).fill_(init))
        self.weight_meta = None

    def named_leaves(self):
        return [('weight', self.weight)]

    def forward(self, x, meta=False):
        if meta:
            return F.prelu(x, self.weight if self.weight_meta is None else self.weight_meta)
        else:
            return F.prelu(x, self.weight)

class MetaResidualBlock(nn.Module):

    def __init__(self, in_ch, out_ch, bias=True):
        super().__init__()
        self.conv1 = MetaConv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=bias)
        self.act1 = MetaPReLU(out_ch)
        self.conv2 = MetaConv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=bias)
        self.act2 = MetaPReLU(out_ch)
        if in_ch != out_ch:
            self.shortcut = MetaConv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=bias)
        else:
            self.shortcut = None

    def forward(self, x, meta=False):
        identity = x
        out = self.conv1(x, meta=meta)
        out = self.act1(out, meta=meta)
        out = self.conv2(out, meta=meta)
        if self.shortcut is not None:
            identity = self.shortcut(identity, meta=meta)
        out = out + identity
        out = self.act2(out, meta=meta)
        return out

class MetaDown(nn.Module):

    def __init__(self, in_ch, out_ch, bias=True):
        super().__init__()
        self.pool = nn.AvgPool2d(2)
        self.block = MetaResidualBlock(in_ch, out_ch, bias=bias)

    def forward(self, x, meta=False):
        x = self.pool(x)
        x = self.block(x, meta=meta)
        return x

class MetaUp(nn.Module):

    def __init__(self, in_ch, skip_ch, out_ch, bias=True):
        super().__init__()
        self.block = MetaResidualBlock(in_ch + skip_ch, out_ch, bias=bias)

    def forward(self, x, skip, meta=False):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.block(x, meta=meta)
        return x

class Reconstruction(nn.Module):
    """Residual encoder-decoder with spectral and learned RGB output heads."""

    def __init__(self, inc=32, base_ch=128, bias=True, rgb_band=(13, 7, 5)):
        super().__init__()
        self.inc = inc
        self.rgb_band = rgb_band
        self.enc1 = MetaResidualBlock(inc, base_ch, bias=bias)
        self.down1 = MetaDown(base_ch, base_ch * 2, bias=bias)
        self.down2 = MetaDown(base_ch * 2, base_ch * 4, bias=bias)
        self.down3 = MetaDown(base_ch * 4, base_ch * 8, bias=bias)
        self.bottleneck = MetaResidualBlock(base_ch * 8, base_ch * 8, bias=bias)
        self.up3 = MetaUp(base_ch * 8, base_ch * 4, base_ch * 4, bias=bias)
        self.up2 = MetaUp(base_ch * 4, base_ch * 2, base_ch * 2, bias=bias)
        self.up1 = MetaUp(base_ch * 2, base_ch, base_ch, bias=bias)
        self.spec_head = MetaConv2d(base_ch, inc, kernel_size=3, stride=1, padding=1, bias=bias)
        self.rgb_head = MetaConv2d(inc, 3, kernel_size=1, stride=1, padding=0, bias=bias)
        self.act = nn.Sigmoid()

    def forward(self, x, rgb_band=None, meta=False):
        if rgb_band is not None:
            self.rgb_band = rgb_band
        e1 = self.enc1(x, meta=meta)
        e2 = self.down1(e1, meta=meta)
        e3 = self.down2(e2, meta=meta)
        e4 = self.down3(e3, meta=meta)
        b = self.bottleneck(e4, meta=meta)
        d3 = self.up3(b, e3, meta=meta)
        d2 = self.up2(d3, e2, meta=meta)
        d1 = self.up1(d2, e1, meta=meta)
        X_rec = self.spec_head(d1, meta=meta)
        X_rec = self.act(X_rec)
        interpretable_image = self.rgb_head(X_rec, meta=meta)
        interpretable_image = self.act(interpretable_image)
        return (interpretable_image, X_rec)

class LPN(nn.Module):
    """Predict a shared spatial weight map for intensity and spectral losses."""

    def __init__(self, inc=32, dim=64, bias=False, se_ratio=1):
        super().__init__()
        self.embed = MetaConv2d(inc, dim, 3, 1, 1, bias=bias)
        self.block1 = MetaResidualBlock(dim, dim * 2, bias=bias)
        self.block2 = MetaResidualBlock(dim * 2, dim, bias=bias)
        self.trans1 = TransformerBlock(dim, num_heads=8, ffn_expansion_factor=2, bias=bias)
        self.trans2 = TransformerBlock(dim, num_heads=8, ffn_expansion_factor=2, bias=bias)
        se_hidden = max(dim // se_ratio, 8)
        self.se_reduce = MetaConv2d(dim, se_hidden * 2, 1, 1, 0, bias=True)
        self.se_act = MetaPReLU(se_hidden * 2)
        self.se_expand = MetaConv2d(se_hidden * 2, dim, 1, 1, 0, bias=True)
        self.spec_conv1 = MetaConv2d(dim, dim * 2, 3, 1, 1, bias=bias)
        self.spec_act1 = MetaPReLU(dim * 2)
        self.spec_conv2 = MetaConv2d(dim * 2, 1, 3, 1, 1, bias=bias)

    def forward(self, x, meta=False):
        feat = self.embed(x, meta=meta)
        feat = self.block1(feat, meta=meta)
        feat = self.block2(feat, meta=meta)
        feat = self.trans1(feat, meta=meta)
        feat = self.trans2(feat, meta=meta)
        se = F.adaptive_avg_pool2d(feat, 1)
        se = self.se_reduce(se, meta=meta)
        se = self.se_act(se, meta=meta)
        se = torch.sigmoid(self.se_expand(se, meta=meta))
        feat_spec = feat * se
        w = self.spec_conv1(feat_spec, meta=meta)
        w = self.spec_act1(w, meta=meta)
        w = torch.sigmoid(self.spec_conv2(w, meta=meta))
        return w
