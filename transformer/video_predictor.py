import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys




class SwinTransformerVideoPredictor(SwinTransformerSys):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dim=96, depths=[2, 2, 2, 2],
                 depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 ape=False, patch_norm=True, use_checkpoint=False, final_upsample="expand_first", sequence_length=5):
        super().__init__(img_size, patch_size, in_chans, num_classes, embed_dim, depths, depths_decoder, num_heads,
                         window_size, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate,
                         norm_layer, ape, patch_norm, use_checkpoint, final_upsample)

        self.sequence_length = sequence_length

        # Temporal modeling using a transformer
        self.temporal_transformer = nn.Transformer(
            d_model=self.num_features, nhead=8, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=2048
        )

        # Optional: Temporal modeling using a 3D CNN
        self.temporal_cnn = nn.Conv3d(
            in_channels=self.num_features, out_channels=self.num_features, kernel_size=(3, 3, 3), padding=(1, 1, 1)
        )

    def forward(self, x):
        # x: (B, T, C, H, W) where T is the sequence length
        B, T, C, H, W = x.shape
        assert T == self.sequence_length, f"Expected sequence length {self.sequence_length}, but got {T}"

        # Process each frame through the encoder
        x = x.view(B * T, C, H, W)  # Merge batch and sequence dimensions
        x, x_downsample = self.forward_features(x)  # Encoder
        x = x.view(B, T, -1)  # Reshape for temporal modeling

        # Temporal modeling using transformer
        x = x.permute(1, 0, 2)  # (T, B, C) for transformer
        x = self.temporal_transformer(x, x)
        x = x.permute(1, 0, 2)  # Back to (B, T, C)

        # Optional: Temporal modeling using 3D CNN
        # x = x.view(B, self.num_features, T, H // 4, W // 4)  # Reshape for 3D CNN
        # x = self.temporal_cnn(x)
        # x = x.view(B, T, -1)  # Flatten back

        # Use the last frame's features for decoding
        x = x[:, -1, :]  # Take the last frame's features
        x = self.forward_up_features(x, x_downsample[-T:])  # Decoder
        x = self.up_x4(x)  # Upsample to original resolution

        return x