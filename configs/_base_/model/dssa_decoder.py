# Copyright (c) OpenMMLab. All rights reserved.
from traceback import print_tb

from mmcv.cnn import build_norm_layer
from mmengine.model import ModuleList
from torch import Tensor
import torch
import torch.nn.functional as F
import torch.nn as nn
from mmdet.models.layers import DeformableDetrTransformerEncoder
from mmdet.models.layers import DetrTransformerDecoder, DetrTransformerDecoderLayer
from timm.models.layers import DropPath
from typing import Optional


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Mask2FormerTransformerEncoder(DeformableDetrTransformerEncoder):
    """Encoder in PixelDecoder of Mask2Former."""

    def forward(self, query: Tensor, query_pos: Tensor,
                key_padding_mask: Tensor, spatial_shapes: Tensor,
                level_start_index: Tensor, valid_ratios: Tensor,
                reference_points: Tensor, **kwargs) -> Tensor:
        """Forward function of Transformer encoder.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            query_pos (Tensor): The positional encoding for query, has shape
                (bs, num_queries, dim). If not None, it will be added to the
                `query` before forward function. Defaults to None.
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (bs, num_queries).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 2) with the last dimension arranged
                as (cx, cy).

        Returns:
            Tensor: Output queries of Transformer encoder, which is also
            called 'encoder output embeddings' or 'memory', has shape
            (bs, num_queries, dim)
        """
        for layer in self.layers:
            query = layer(
                query=query,
                query_pos=query_pos,
                key_padding_mask=key_padding_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points,
                **kwargs)
        return query


class Mask2FormerTransformerDecoder(DetrTransformerDecoder):
    """Decoder of Mask2Former."""

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            Mask2FormerTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        self.post_norm = build_norm_layer(self.post_norm_cfg,
                                          self.embed_dims)[1]





class DQGSA(nn.Module):
    def __init__(self, kernel_size=7, layer_scale_init_value=1e-6, drop_path=0.):
        super(DQGSA, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.d_model = 256
        self.conv2 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()
        self.transconv = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1,
                                            output_padding=1)

        self.norm1 = LayerNorm(self.d_model, eps=1e-6)
        self.pwconv1_1 = nn.Linear(self.d_model, 4 * self.d_model)
        self.act1 = nn.GELU()
        self.pwconv1_2 = nn.Linear(4 * self.d_model, self.d_model)
        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((self.d_model)),
                                   requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = LayerNorm(self.d_model, eps=1e-6)
        self.qH = 10
        self.qW = 10
        self.smooth_factor = 0.5
        self.epsilon = 1e-8
    def forward(self, x1, x2=None):
        assert x2 is not None, 'x2 must be exist'
        self.bs = x1.shape[0]
        x1 = x1.permute(0, 2, 1).view(self.bs, self.d_model, self.qH, self.qW)
        x2 = x2.permute(0, 2, 1).view(self.bs, self.d_model, self.qH, self.qW)
        #
        x1 = self.conv2(x1)
        distance_map = torch.norm(x2 - x1, dim=1, keepdim=True) + self.epsilon
        distance_map = self.conv3(distance_map)
        weights = self.sigmoid2(distance_map * self.smooth_factor)
        x2_i = (1 - weights) * x2
        x1_i = weights * x1
        mix_out = x1_i + x2_i
        avg_out = torch.mean(mix_out, dim=1, keepdim=True)
        max_out, _ = torch.max(mix_out, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x)

        x = x * x1
        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x)
        x = self.pwconv1_1(x)
        x = self.act1(x)
        x = self.pwconv1_2(x)
        if self.gamma1 is not None:
            x = self.gamma1 * x
        x = x.permute(0, 3, 1, 2)
        x = self.drop_path1(x) + x2
        x = x.permute(0, 2, 3, 1).flatten(1, 2)
        return x


class Mask2FormerTransformerDecoderLayer(DetrTransformerDecoderLayer):
    """Implements decoder layer in Mask2Former transformer."""

    def __init__(self, self_attn_cfg, cross_attn_cfg, ffn_cfg):
        super().__init__(self_attn_cfg, cross_attn_cfg, ffn_cfg)
        self.DQGSA = DQGSA(kernel_size=7)

    def forward(self,
                query: Tensor,
                key: Tensor = None,
                value: Tensor = None,
                query_pos: Tensor = None,
                key_pos: Tensor = None,
                self_attn_mask: Tensor = None,
                cross_attn_mask: Tensor = None,
                key_padding_mask: Tensor = None,
                **kwargs) -> Tensor:
        """
        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            key (Tensor, optional): The input key, has shape (bs, num_keys,
                dim). If `None`, the `query` will be used. Defaults to `None`.
            value (Tensor, optional): The input value, has the same shape as
                `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
                `key` will be used. Defaults to `None`.
            query_pos (Tensor, optional): The positional encoding for `query`,
                has the same shape as `query`. If not `None`, it will be added
                to `query` before forward function. Defaults to `None`.
            key_pos (Tensor, optional): The positional encoding for `key`, has
                the same shape as `key`. If not `None`, it will be added to
                `key` before forward function. If None, and `query_pos` has the
                same shape as `key`, then `query_pos` will be used for
                `key_pos`. Defaults to None.
            self_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            cross_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor, optional): The `key_padding_mask` of
                `self_attn` input. ByteTensor, has shape (bs, num_value).
                Defaults to None.

        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """

        query_0 = self.cross_attn(
            query=query,
            key=key,
            value=value,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=cross_attn_mask,
            key_padding_mask=key_padding_mask,
            **kwargs)
        query_1 = self.norms[0](query_0)
        query_2 = self.DQGSA(query, query_1)
        query_2 = self.norms[1](query_2)
        query_3 = self.ffn(query_2)
        query_3 = self.norms[2](query_3)

        return query_3
