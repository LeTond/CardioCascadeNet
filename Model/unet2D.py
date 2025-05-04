 # -*- coding: utf-8 -*-
"""
Name: Anatoliy Levchuk
Version: 1.4
Date: 04-05-2025
Email: feuerlag999@yandex.ru
GitHub: https://github.com/LeTond
"""


import math
import torch

import torch.nn.functional as F

from torch import nn
from torchvision import models
from collections import OrderedDict


import CardioCascadeNet


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows

def window_reverse(windows, window_size, H, W):
    C = windows.shape[-1]
    x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    return x

def get_relative_position_index(win_h: int, win_w: int):
    # get pair-wise relative position index for each token inside the window
    coords = torch.stack(torch.meshgrid(torch.arange(win_h), torch.arange(win_w),indexing = 'ij'))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += win_h - 1  # shift to start from 0
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)  # Wh*Ww, Wh*Ww


class WindowAttention(nn.Module):
    def __init__(
            self,
            dim,
            window_size,
    ):
        super().__init__()
        self.window_size = window_size
        self.window_area = self.window_size[0]*self.window_size[1]
        self.num_heads = 4
        head_dim =  dim // self.num_heads
        # attn_dim = head_dim * self.num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) **2, self.num_heads))

        # get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(self.window_size[0], self.window_size[1]), persistent=False)

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        torch.nn.init.trunc_normal_(self.relative_position_bias_table, std = .02)
        self.softmax = nn.Softmax(dim = -1)

    def _get_rel_pos_bias(self):
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        return relative_position_bias.unsqueeze(0)

    def forward(self, x, mask = None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)


        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn + self._get_rel_pos_bias()
        if mask is not None:
            num_win = mask.shape[0]
            attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B_, N, -1)
        x = self.proj(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, window_size = 6, shift_size = 0):
        super().__init__()

        self.input_resolution = input_resolution
        window_size = (window_size, window_size)
        shift_size = (shift_size, shift_size)
        self.window_size = window_size
        self.shift_size = shift_size
        self.window_area = self.window_size[0] * self.window_size[1]

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim,
            window_size = self.window_size,
        )

        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.LayerNorm(4 * dim),
            nn.Linear( 4 * dim, dim)
        )

        if self.shift_size:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            H = math.ceil(H / self.window_size[0]) * self.window_size[0]
            W = math.ceil(W / self.window_size[1]) * self.window_size[1]
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            cnt = 0
            for h in (
                    slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None)):
                for w in (
                        slice(0, -self.window_size[1]),
                        slice(-self.window_size[1], -self.shift_size[1]),
                        slice(-self.shift_size[1], None)):
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_area)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask, persistent = False)

    def _attn(self, x):
        B, H, W, C = x.shape

        # cyclic shift
        if self.shift_size:
            shifted_x = torch.roll(x, shifts = (-self.shift_size[0], -self.shift_size[1]), dims = (1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask = self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        shifted_x = shifted_x[:, :H, :W, :].contiguous()

        # reverse cyclic shift
        if self.shift_size:
            x = torch.roll(shifted_x, shifts=self.shift_size, dims = (1, 2))
        else:
            x = shifted_x
        return x

    def forward(self, x):
        B, H, W, C = x.shape
        B, H, W, C = x.shape
        x = x + self._attn(self.norm1(x))
        x = x.reshape(B, -1, C)
        x = x + self.mlp(self.norm2(x))
        x = x.reshape(B, H, W, C)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, in_ch, num_feat, patch_size):
        super().__init__()
        self.conv = nn.Conv2d(in_ch,num_feat, kernel_size=patch_size,
                                  stride=patch_size)

    def forward(self, X):
        # Output shape: (batch size, no. of patches, no. of channels)
        return self.conv(X).permute(0,2,3,1)


class PatchMerging(nn.Module):

    def __init__(
            self,
            dim
    ):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4*dim, 2*dim, bias=False)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchExpansion(nn.Module):

    def __init__(
            self,
            dim
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim//2)
        self.expand = nn.Linear(dim, 2*dim, bias=False)

    def forward(self, x):

        x = self.expand(x)
        B, H, W, C = x.shape

        x = x.view(B, H , W, 2, 2, C//4)
        x = x.permute(0,1,3,2,4,5)

        x = x.reshape(B,H*2, W*2 , C//4)

        x = self.norm(x)
        return x


class FinalPatchExpansion(nn.Module):

    def __init__(
            self,
            dim
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.expand = nn.Linear(dim, 16*dim, bias=False)

    def forward(self, x):

        x = self.expand(x)
        B, H, W, C = x.shape

        x = x.view(B, H , W, 4, 4, C//16)
        x = x.permute(0,1,3,2,4,5)

        x = x.reshape(B,H*4, W*4 , C//16)

        x = self.norm(x)
        return x


class SwinBlock(nn.Module):
    def __init__(self, dims, ip_res, ss_size = 3):
        super().__init__()
        self.swtb1 = SwinTransformerBlock(dim=dims, input_resolution=ip_res)
        self.swtb2 = SwinTransformerBlock(dim=dims, input_resolution=ip_res, shift_size=ss_size)

    def forward(self, x):
        return self.swtb2(self.swtb1(x))


class Encoder(nn.Module):
    def __init__(self, C, partioned_ip_res, num_blocks=3):
        super().__init__()
        H,W = partioned_ip_res[0], partioned_ip_res[1]
        self.enc_swin_blocks = nn.ModuleList([
            SwinBlock(C, (H, W)),
            SwinBlock(2 * C, (H // 2, W // 2)),
            SwinBlock(4 * C, (H // 4, W // 4))
        ])
        self.enc_patch_merge_blocks = nn.ModuleList([
            PatchMerging(C),
            PatchMerging(2 * C),
            PatchMerging(4 * C)
        ])

    def forward(self, x):
        skip_conn_ftrs = []
        for swin_block,patch_merger in zip(self.enc_swin_blocks, self.enc_patch_merge_blocks):
            x = swin_block(x)
            skip_conn_ftrs.append(x)
            x = patch_merger(x)
        return x, skip_conn_ftrs


class Decoder(nn.Module):
    def __init__(self, C, partioned_ip_res, num_blocks=3):
        super().__init__()
        H,W = partioned_ip_res[0], partioned_ip_res[1]
        self.dec_swin_blocks = nn.ModuleList([
            SwinBlock(4*C, (H//4, W//4)),
            SwinBlock(2*C, (H//2, W//2)),
            SwinBlock(C, (H, W))
        ])
        self.dec_patch_expand_blocks = nn.ModuleList([
            PatchExpansion(8*C),
            PatchExpansion(4*C),
            PatchExpansion(2*C)
        ])
        self.skip_conn_concat = nn.ModuleList([
            nn.Linear(8*C, 4*C),
            nn.Linear(4*C, 2*C),
            nn.Linear(2*C, 1*C)
        ])

    def forward(self, x, encoder_features):
        for patch_expand,swin_block, enc_ftr, linear_concatter in zip(self.dec_patch_expand_blocks, self.dec_swin_blocks, encoder_features,self.skip_conn_concat):
            x = patch_expand(x)
            x = torch.cat([x, enc_ftr], dim=-1)
            x = linear_concatter(x)
            x = swin_block(x)
        return x


class SwinUNet(nn.Module, CardioCascadeNet.MetaParameters):
    def __init__(self, num_blocks = 3, patch_size = 4):
        super().__init__()
        super(CardioCascadeNet.MetaParameters, self).__init__()

        in_channels = self.CHANNELS
        out_channels = self.NUM_CLASS
        C = self.BT_SZ
        # H = self.CROPP_KERNEL
        H = self.KERNEL
        # W = self.CROPP_KERNEL
        W = self.KERNEL

        self.patch_embed = PatchEmbedding(in_channels, C, patch_size)
        self.encoder = Encoder(C, (H // patch_size, W // patch_size), num_blocks)
        self.bottleneck = \
                    SwinBlock(C * (2 ** num_blocks), (H // (patch_size * (2 ** num_blocks)), W // (patch_size * (2 ** num_blocks))))
        self.decoder = Decoder(C, (H // patch_size, W // patch_size), num_blocks)
        self.final_expansion = FinalPatchExpansion(C)
        self.head = nn.Conv2d(C, out_channels, 1, padding='same')

    def forward(self, x):
        x = self.patch_embed(x)

        x,skip_ftrs = self.encoder(x)

        x = self.bottleneck(x)
        x = self.decoder(x, skip_ftrs[::-1])
        x = self.final_expansion(x)
        x = self.head(x.permute(0, 3, 1, 2))

        return x


class UNet_2D(nn.Module, CardioCascadeNet.MetaParameters):

    def __init__(self):
        super(UNet_2D, self).__init__()
        super(CardioCascadeNet.MetaParameters, self).__init__()

        features = self.FEATURES
        in_channels = self.CHANNELS
        out_channels = self.NUM_CLASS
        dropout = self.DROPOUT

        self.dropout = nn.Dropout2d(dropout)
        self.encoder1 = UNet_2D.Conv2x2(in_channels, features, name = "enc1")
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.encoder2 = UNet_2D.Conv2x2(features, features * 2, name = "enc2")
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.encoder3 = UNet_2D.Conv2x2(features * 2, features * 4, name = "enc3")
        self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.encoder4 = UNet_2D.Conv2x2(features * 4, features * 8, name = "enc4")
        self.pool4 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.bottleneck = UNet_2D.Conv2x2(features * 8, features * 16, name = "bottleneck")
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size = 2, stride = 2)
        self.decoder4 = UNet_2D.Conv2x2((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size = 2, stride = 2)
        self.decoder3 = UNet_2D.Conv2x2((features * 4) * 2, features * 4, name = "dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size = 2, stride = 2)
        self.decoder2 = UNet_2D.Conv2x2((features * 2) * 2, features * 2, name = "dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size = 2, stride = 2)
        self.decoder1 = UNet_2D.Conv2x2(features * 2, features, name = "dec1")
        self.conv = nn.Conv2d(in_channels = features, out_channels = out_channels, kernel_size = 1)

    def forward(self, x):

        enc1 = self.encoder1(x)
        enc1 = self.dropout(enc1)

        enc2 = self.encoder2(self.pool1(enc1))
        enc2 = self.dropout(enc2)

        enc3 = self.encoder3(self.pool2(enc2))
        enc3 = self.dropout(enc3)

        enc4 = self.encoder4(self.pool3(enc3))
        enc4 = self.dropout(enc4)

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim = 1)
        dec4 = self.dropout(dec4)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim = 1)
        dec3 = self.dropout(dec3)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim = 1)
        dec2 = self.dropout(dec2)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim = 1)
        dec1 = self.dropout(dec1)
        dec1 = self.decoder1(dec1)

        # return torch.softmax(self.conv(dec1), dim=1)
        # return torch.sigmoid(self.conv(dec1))
        return self.conv(dec1)

    @staticmethod
    def Conv2x2(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels = in_channels,
                            out_channels = features,
                            kernel_size = 3,
                            stride = 1,
                            padding = 1,
                            bias = False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features = features, affine = True)),   #, eps=1e-05, momentum=0.5, affine=True, track_running_stats=True
                    # (name + "relu1", nn.LeakyReLU(negative_slope = 0.1, inplace = True)),
                    (name + "relu1", nn.ReLU()),

                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels = features,
                            out_channels = features,
                            kernel_size = 3,
                            stride = 1,
                            padding = 1,
                            bias = False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features = features, affine = True)),
                    # (name + "relu2", nn.LeakyReLU(negative_slope = 0.1, inplace = True)),
                    (name + "relu2", nn.ReLU()),

                ]
            )
        )


class UNet_2D_AttantionLayer(nn.Module, CardioCascadeNet.MetaParameters):
    def __init__(self):
        super(UNet_2D_AttantionLayer, self).__init__()
        super(CardioCascadeNet.MetaParameters, self).__init__()

        features = self.FEATURES
        in_channels = self.CHANNELS
        out_channels = self.NUM_CLASS
        dropout = self.DROPOUT
        freeze_bn = self.FREEZE_BN

        self.dropout = nn.Dropout2d(dropout)
        self.encoder1 = UNet_2D_AttantionLayer.Conv2x2(in_channels, features, name = "enc1")
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        # self.pool1 = nn.AvgPool2d(kernel_size = 2, stride = 2)

        self.encoder2 = UNet_2D_AttantionLayer.Conv2x2(features, features * 2, name = "enc2")
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        # self.pool2 = nn.AvgPool2d(kernel_size = 2, stride = 2)
        
        self.encoder3 = UNet_2D_AttantionLayer.Conv2x2(features * 2, features * 4, name = "enc3")
        self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        # self.pool3 = nn.AvgPool2d(kernel_size = 2, stride = 2)
        
        self.encoder4 = UNet_2D_AttantionLayer.Conv2x2(features * 4, features * 8, name = "enc4")
        self.pool4 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        # self.pool4 = nn.AvgPool2d(kernel_size = 2, stride = 2)

        self.bottleneck = UNet_2D_AttantionLayer.Conv2x2(features * 8, features * 16, name = "bottleneck")
        
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size = 2, stride = 2)
        self.Att4 = Attention_2D(features * 8,features * 8,features * 4)
        self.decoder4 = UNet_2D_AttantionLayer.Conv2x2((features * 8) * 2, features * 8, name="dec4")
        
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size = 2, stride = 2)
        self.Att3 = Attention_2D(features * 4,features * 4,features * 2)
        self.decoder3 = UNet_2D_AttantionLayer.Conv2x2((features * 4) * 2, features * 4, name = "dec3")
        
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size = 2, stride = 2)
        self.Att2 = Attention_2D(features * 2,features * 2,features * 1)
        self.decoder2 = UNet_2D_AttantionLayer.Conv2x2((features * 2) * 2, features * 2, name = "dec2")
        
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size = 2, stride = 2)
        self.Att1 = Attention_2D(features, features, features // 2)
        self.decoder1 = UNet_2D_AttantionLayer.Conv2x2(features * 2, features, name = "dec1")
        
        self.conv = nn.Conv2d(in_channels = features, out_channels = out_channels, kernel_size = 1)

        # initialize_weights(self)

        # if freeze_bn:
        #     self.freeze_bn()

    def forward(self, x):

        enc1 = self.encoder1(x)
        enc1 = self.dropout(enc1)

        enc2 = self.encoder2(self.pool1(enc1))
        enc2 = self.dropout(enc2)

        enc3 = self.encoder3(self.pool2(enc2))
        enc3 = self.dropout(enc3)

        enc4 = self.encoder4(self.pool3(enc3))
        enc4 = self.dropout(enc4)

        bottleneck = self.bottleneck(self.pool4(enc4))
        bottleneck = self.dropout(bottleneck)

        dec4 = self.upconv4(bottleneck)
        enc4 = self.Att4(dec4,enc4)
        dec4 = torch.cat((dec4, enc4), dim = 1)
        dec4 = self.dropout(dec4)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        enc3 = self.Att3(dec3,enc3)
        dec3 = torch.cat((dec3, enc3), dim = 1)
        dec3 = self.dropout(dec3)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        enc2 = self.Att2(dec2,enc2)
        dec2 = torch.cat((dec2, enc2), dim = 1)
        dec2 = self.dropout(dec2)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        enc1 = self.Att1(dec1,enc1)
        dec1 = torch.cat((dec1, enc1), dim = 1)
        dec1 = self.dropout(dec1)
        dec1 = self.decoder1(dec1)

        return torch.softmax(self.conv(dec1), dim=1)
        # return self.conv(dec1)

    @staticmethod
    def Conv2x2(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels = in_channels,
                            out_channels = features,
                            kernel_size = 3,
                            stride = 1,
                            padding = 1,
                            bias = False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features = features)),   #, eps=1e-05, momentum=0.5, affine=True, track_running_stats=True
                    # (name + "norm1", nn.InstanceNorm2d(features, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = False)),
                    (name + "relu1", nn.LeakyReLU(negative_slope = 0.1, inplace = True)),
                    # (name + "relu1", nn.ReLU()),

                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels = features,
                            out_channels = features,
                            kernel_size = 3,
                            stride = 1,
                            padding = 1,
                            bias = False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features = features)),
                    # (name + "norm1", nn.InstanceNorm2d(features, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = False)),
                    (name + "relu2", nn.LeakyReLU(negative_slope = 0.1, inplace = True)),
                    # (name + "relu1", nn.ReLU()),

                ]
            )
        )

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()


class Attention_2D(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_2D,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size = 1, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size = 1, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size = 1, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(1),
            nn.Softmax(dim = 1)
        )
        
        self.relu = nn.ReLU(inplace = True)
        # self.relu = nn.LeakyReLU(negative_slope = 0.1, inplace = True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class UNet_2D_mini(nn.Module, CardioCascadeNet.MetaParameters):

    def __init__(self):
        super(UNet_2D_mini, self).__init__()
        super(CardioCascadeNet.MetaParameters, self).__init__()

        features = self.FEATURES
        in_channels = self.CHANNELS
        out_channels = self.NUM_CLASS
        dropout = self.DROPOUT

        self.dropout = nn.Dropout2d(dropout)
        self.encoder1 = UNet_2D_mini.Conv2x2(in_channels, features, name = "enc1")
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.encoder2 = UNet_2D_mini.Conv2x2(features, features * 2, name = "enc2")
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.bottleneck = UNet_2D_mini.Conv2x2(features * 2, features * 4, name = "bottleneck")

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size = 2, stride = 2)
        self.decoder2 = UNet_2D_mini.Conv2x2((features * 2) * 2, features * 2, name = "dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size = 2, stride = 2)
        self.decoder1 = UNet_2D_mini.Conv2x2(features * 2, features, name = "dec1")
        self.conv = nn.Conv2d(in_channels = features, out_channels = out_channels, kernel_size = 1)

    def forward(self, x):

        enc1 = self.encoder1(x)
        enc1 = self.dropout(enc1)

        enc2 = self.encoder2(self.pool1(enc1))
        enc2 = self.dropout(enc2)

        bottleneck = self.bottleneck(self.pool2(enc2))

        dec2 = self.upconv2(bottleneck)
        dec2 = torch.cat((dec2, enc2), dim = 1)
        dec2 = self.dropout(dec2)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim = 1)
        dec1 = self.dropout(dec1)
        dec1 = self.decoder1(dec1)

        return torch.softmax(self.conv(dec1), dim=1)

    @staticmethod
    def Conv2x2(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels = in_channels,
                            out_channels = features,
                            kernel_size = 3,
                            # stride=1,
                            padding = 1,
                            bias = False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features = features, affine=False)),   #, eps=1e-05, momentum=0.5, affine=True, track_running_stats=True
                    # (name + "norm1", nn.InstanceNorm2d(32, eps = 1e-5, momentum = 0.1, affine = True, num_features = features)),
                    # (name + "relu1", nn.LeakyReLU(negative_slope = 0.01, inplace = True)),
                    (name + "relu1", nn.ReLU()),

                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels = features,
                            out_channels = features,
                            kernel_size = 3,
                            # stride=1,
                            padding = 1,
                            bias = False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features = features, affine=False)),
                    # (name + "relu2", nn.LeakyReLU(negative_slope = 0.01, inplace = True)),
                    (name + "relu2", nn.ReLU()),

                ]
            )
        )


class U_Net(nn.Module):
    def __init__(self, img_ch=1, num_classes=4):
        super(U_Net, self).__init__()
 
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
 
        self.Conv1 = convConv2x2(ch_in=img_ch, ch_out=64)
        self.Conv2 = convConv2x2(ch_in=64, ch_out=128)
        self.Conv3 = convConv2x2(ch_in=128, ch_out=256)
        self.Conv4 = convConv2x2(ch_in=256, ch_out=512)
        self.Conv5 = convConv2x2(ch_in=512, ch_out=1024)
 
        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = convConv2x2(ch_in=1024, ch_out=512)
 
        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = convConv2x2(ch_in=512, ch_out=256)
 
        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = convConv2x2(ch_in=256, ch_out=128)
 
        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = convConv2x2(ch_in=128, ch_out=64)
 
        self.Conv_1x1 = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)
        initialize_weights(self)
 
    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
 
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
 
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
 
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
 
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
 
        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
 
        d5 = self.Up_conv5(d5)
 
        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)
 
        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)
 
        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
 
        d1 = self.Conv_1x1(d2)
 
        return d1


def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        # self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self):
        raise NotImplementedError

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info(f'Nbr of trainable parameters: {nbr_params}')

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        return super(BaseModel, self).__str__() + f'\nNbr of trainable parameters: {nbr_params}'
        #return summary(self, input_shape=(2, 3, 224, 224))


class UNetResnet(BaseModel, CardioCascadeNet.MetaParameters):
    def __init__(self, backbone='resnet50', pretrained=False, freeze_bn=False, freeze_backbone=False, **_):
        super(UNetResnet, self).__init__()
        super(CardioCascadeNet.MetaParameters, self).__init__()

        features = self.FEATURES
        in_channels = self.CHANNELS
        num_classes = self.NUM_CLASS
        dropout = self.DROPOUT
        freeze_bn = self.FREEZE_BN

        model = getattr(CardioCascadeNet.resnet, backbone)(pretrained, norm_layer=nn.BatchNorm2d)

        self.initial = list(model.children())[:4]
        if in_channels != 3:
            self.initial[0] = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.initial = nn.Sequential(*self.initial)

        # encoder
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        # decoder
        self.conv1 = nn.Conv2d(2048, 192, kernel_size=3, stride=1, padding=1)
        self.upconv1 =  nn.ConvTranspose2d(192, 128, 4, 2, 1, bias=False)

        self.conv2 = nn.Conv2d(1152, 128, kernel_size=3, stride=1, padding=1)
        self.upconv2 = nn.ConvTranspose2d(128, 96, 4, 2, 1, bias=False)

        self.conv3 = nn.Conv2d(608, 96, kernel_size=3, stride=1, padding=1)
        self.upconv3 = nn.ConvTranspose2d(96, 64, 4, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(320, 64, kernel_size=3, stride=1, padding=1)
        self.upconv4 = nn.ConvTranspose2d(64, 48, 4, 2, 1, bias=False)
        
        self.conv5 = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1)
        self.upconv5 = nn.ConvTranspose2d(48, 32, 4, 2, 1, bias=False)

        self.conv6 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(32, num_classes, kernel_size=1, bias=False)

        initialize_weights(self)

        if freeze_bn:
            self.freeze_bn()
        if freeze_backbone: 
            set_trainable([self.initial, self.layer1, self.layer2, self.layer3, self.layer4], False)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        x1 = self.layer1(self.initial(x))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        x = self.upconv1(self.conv1(x4))
        x = F.interpolate(x, size=(x3.size(2), x3.size(3)), mode="bilinear", align_corners=True)
        x = torch.cat([x, x3], dim=1)
        x = self.upconv2(self.conv2(x))

        x = F.interpolate(x, size=(x2.size(2), x2.size(3)), mode="bilinear", align_corners=True)
        x = torch.cat([x, x2], dim=1)
        x = self.upconv3(self.conv3(x))

        x = F.interpolate(x, size=(x1.size(2), x1.size(3)), mode="bilinear", align_corners=True)
        x = torch.cat([x, x1], dim=1)

        x = self.upconv4(self.conv4(x))

        x = self.upconv5(self.conv5(x))

        # if the input is not divisible by the output stride
        if x.size(2) != H or x.size(3) != W:
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=True)

        x = self.conv7(self.conv6(x))
        
        return x

    def get_backbone_params(self):
        return chain(self.initial.parameters(), self.layer1.parameters(), self.layer2.parameters(), 
                    self.layer3.parameters(), self.layer4.parameters())

    def get_decoder_params(self):
        return chain(self.conv1.parameters(), self.upconv1.parameters(), self.conv2.parameters(), self.upconv2.parameters(),
                    self.conv3.parameters(), self.upconv3.parameters(), self.conv4.parameters(), self.upconv4.parameters(),
                    self.conv5.parameters(), self.upconv5.parameters(), self.conv6.parameters(), self.conv7.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()


class SegNet(BaseModel, CardioCascadeNet.MetaParameters):
    def __init__(self, pretrained=False, freeze_bn=False, **_):
        super(SegNet, self).__init__()
        super(CardioCascadeNet.MetaParameters, self).__init__()

        features = self.FEATURES
        in_channels = self.CHANNELS
        num_classes = self.NUM_CLASS
        dropout = self.DROPOUT
        freeze_bn = self.FREEZE_BN

        vgg_bn = models.vgg16_bn(pretrained= pretrained)
        encoder = list(vgg_bn.features.children())

        # Adjust the input size
        if in_channels != 3:
            encoder[0] = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)

        # Encoder, VGG without any maxpooling
        self.stage1_encoder = nn.Sequential(*encoder[:6])
        self.stage2_encoder = nn.Sequential(*encoder[7:13])
        self.stage3_encoder = nn.Sequential(*encoder[14:23])
        self.stage4_encoder = nn.Sequential(*encoder[24:33])
        self.stage5_encoder = nn.Sequential(*encoder[34:-1])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # Decoder, same as the encoder but reversed, maxpool will not be used
        decoder = encoder
        decoder = [i for i in list(reversed(decoder)) if not isinstance(i, nn.MaxPool2d)]
        # Replace the last conv layer
        decoder[-1] = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # When reversing, we also reversed conv->batchN->relu, correct it
        decoder = [item for i in range(0, len(decoder), 3) for item in decoder[i:i+3][::-1]]
        # Replace some conv layers & batchN after them
        for i, module in enumerate(decoder):
            if isinstance(module, nn.Conv2d):
                if module.in_channels != module.out_channels:
                    decoder[i+1] = nn.BatchNorm2d(module.in_channels)
                    decoder[i] = nn.Conv2d(module.out_channels, module.in_channels, kernel_size=3, stride=1, padding=1)

        self.stage1_decoder = nn.Sequential(*decoder[0:9])
        self.stage2_decoder = nn.Sequential(*decoder[9:18])
        self.stage3_decoder = nn.Sequential(*decoder[18:27])
        self.stage4_decoder = nn.Sequential(*decoder[27:33])
        self.stage5_decoder = nn.Sequential(*decoder[33:],
                nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1)
        )
    
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # self.unpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self._initialize_weights(self.stage1_decoder, self.stage2_decoder, self.stage3_decoder,
                                    self.stage4_decoder, self.stage5_decoder)
        if freeze_bn: 
            self.freeze_bn()
        # else: 
        #     set_trainable([self.stage1_encoder, self.stage2_encoder, self.stage3_encoder, self.stage4_encoder, self.stage5_encoder], False)

    def _initialize_weights(self, *stages):
        for modules in stages:
            for module in modules.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()

    def forward(self, x):
        # Encoder
        x = self.stage1_encoder(x)
        x1_size = x.size()
        x, indices1 = self.pool(x)

        x = self.stage2_encoder(x)
        x2_size = x.size()
        x, indices2 = self.pool(x)

        x = self.stage3_encoder(x)
        x3_size = x.size()
        x, indices3 = self.pool(x)

        x = self.stage4_encoder(x)
        x4_size = x.size()
        x, indices4 = self.pool(x)

        x = self.stage5_encoder(x)
        x5_size = x.size()
        x, indices5 = self.pool(x)

        # Decoder
        x = self.unpool(x, indices=indices5, output_size=x5_size)
        x = self.stage1_decoder(x)

        x = self.unpool(x, indices=indices4, output_size=x4_size)
        x = self.stage2_decoder(x)

        x = self.unpool(x, indices=indices3, output_size=x3_size)
        x = self.stage3_decoder(x)

        x = self.unpool(x, indices=indices2, output_size=x2_size)
        x = self.stage4_decoder(x)

        x = self.unpool(x, indices=indices1, output_size=x1_size)
        x = self.stage5_decoder(x)

        # return x
        return torch.softmax(self.conv(x), dim=1)
        

    def get_backbone_params(self):
        return []

    def get_decoder_params(self):
        return self.parameters()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()
