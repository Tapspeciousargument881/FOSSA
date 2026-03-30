# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_
import numpy as np
from easydict import EasyDict

from .backbone_layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block
from .motion_module.motion_module import TemporalModule


logger = logging.getLogger("dinov2")


def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x


class FocusStackTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,

        # Parameters for motion module and temporal collapse
        num_layers_until_collapse=None,
        temporal_fuse_method=None,
        num_frames=32,
        pe=None,
        turn_off_motion_module=False,

    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings
            interpolate_offset: (float) work-around offset to apply when interpolating positional embeddings
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()


        # Motion Module Parameters

        assert num_layers_until_collapse <= depth, f"num_layers_until_collapse {num_layers_until_collapse} should be <= depth {depth}"
        assert num_layers_until_collapse >= 1, f"num_layers_until_collapse {num_layers_until_collapse} should be >= 1"
        assert num_frames > 0, f"max number of frames should be > 0"
        assert temporal_fuse_method is not None, f"temporal_fuse_method should be specified"
        assert pe in ['ape', 'fde_add', 'fde_film'], f"pe should be either 'ape', 'fde_add' or 'fde_film'"
        if temporal_fuse_method not in ['mean_in_encoder', 'max_in_encoder']:
            raise ValueError(f"temporal_fuse_method should be either 'mean_in_encoder' or 'max_in_encoder'")
        self.temporal_fuse_method = temporal_fuse_method
        self.num_layers_until_collapse = num_layers_until_collapse

        motion_module_kwargs = EasyDict(num_attention_heads        = 8,
                                num_transformer_block              = 1,
                                num_attention_blocks               = 2,
                                temporal_max_len                   = num_frames,
                                zero_initialize                    = True,
                                pos_embedding_type                 = pe)

        if not turn_off_motion_module:
            self.motion_modules = nn.ModuleList([
                TemporalModule(in_channels=embed_dim, **motion_module_kwargs)
            for _ in range(num_layers_until_collapse)])

        self.turn_off_motion_module = turn_off_motion_module

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        # DINOv2 with register modify the interpolate_offset from 0.1 to 0.0
        w0, h0 = w0 + self.interpolate_offset, h0 + self.interpolate_offset
        # w0, h0 = w0 + 0.1, h0 + 0.1

        sqrt_N = math.sqrt(N)
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(0, 3, 1, 2),
            scale_factor=(sx, sy),
            # (int(w0), int(h0)), # to solve the upsampling shape issue
            mode="bicubic",
            antialias=self.interpolate_antialias
        )

        assert int(w0) == patch_pos_embed.shape[-2]
        assert int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        """
        x: (B'=BxN, C, H, W) --> (B', 1+patched_h*patched_w, embed_dim)
        """


        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = x + self.interpolate_pos_encoding(x, w, h)

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        return x

    def forward_features_list(self, x_list, masks_list):
        x = [self.prepare_tokens_with_masks(x, masks) for x, masks in zip(x_list, masks_list)]

        for blk in self.blocks:
            x = blk(x)

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x, masks=None):

        if isinstance(x, list):
            return self.forward_features_list(x, masks)


        x = self.prepare_tokens_with_masks(x, masks)


        for blk in self.blocks:
            x = blk(x)

        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }

    def _get_intermediate_layers_not_chunked(self, x, n=1, fd_list=None, actual_batch_size=None, frame_length=None, temporal_fuse_method=None):
        assert fd_list is not None, "fd_list is required in FOSSA"
        assert actual_batch_size is not None, "actual_batch_size is required in FOSSA"
        assert frame_length is not None, "frame_length is required in FOSSA"
        assert temporal_fuse_method is not None, "temporal_fuse_method is required in FOSSA"
        _, _, w, h = x.shape
        patched_h, patched_w = w // self.patch_size, h // self.patch_size


        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n

        for i, blk in enumerate(self.blocks):

            x = blk(x)

            if i < self.num_layers_until_collapse:
                cls_token, x = x[:, 0:1], x[:, 1:]

                # x shape = B'(B x T), patch_h*patch_w, embed_dim
                x = x.unflatten(0, (actual_batch_size, frame_length)) #  B'(B x N), patch_h*patch_w, embed_dim --> B, N, patch_h*patch_w, embed_dim

                x = x.unflatten(2, (patched_h, patched_w))  # B, N, patch_h*patch_w, embed_dim -> B, N, patched_h, patched_w, embed_dim

                x = x.permute(0, 4, 1, 2, 3)  # B, embed_dim, N, patched_h, patched_w

                if not self.turn_off_motion_module:
                    # Applying motion module
                    x = self.motion_modules[i](x, fd_list=fd_list, encoder_hidden_states=None, attention_mask=None)
                else:
                    # Skipping motion module if turn_off_motion_module is set to True
                    pass

                x = x.permute(0, 2, 3, 4, 1).flatten(0, 1) # B, embed_dim, N, patched_h, patched_w -> B, N, patched_h, patched_w, embed_dim -> B'(B x N), patched_h, patched_w, embed_dim

                x = x.flatten(1, 2)  # B'(B x N), patched_h, patched_w, embed_dim -> B'(B x N), patched_h*patched_w, embed_dim


                # Concatenate the cls token back
                x = torch.cat((cls_token, x), dim=1) # B'(B x N), 1+patched_h*patched_w, embed_dim

                if i in blocks_to_take:

                    x_output = x.unflatten(0, (actual_batch_size, frame_length)) #  B'(B x N), 1+patch_h*patch_w, embed_dim --> B, N, 1+patch_h*patch_w, embed_dim

                    # Max pooling/avg pooling
                    if self.temporal_fuse_method == "max_in_encoder":
                        x_output, _ = torch.max(x_output, dim=1, keepdim=False) # B, N, 1+patch_h*patch_w, embed_dim -> B, 1+patch_h*patch_w, embed_dim

                    elif self.temporal_fuse_method == "mean_in_encoder":
                        x_output = torch.mean(x_output, dim=1, keepdim=False) # B, N, 1+patch_h*patch_w, embed_dim -> B, 1+patch_h*patch_w, embed_dim


                    output.append(x_output)

            elif i == self.num_layers_until_collapse:
                x = x.unflatten(0, (actual_batch_size, frame_length)) #  B'(B x N), 1+patch_h*patch_w, embed_dim --> B, N, 1+patch_h*patch_w, embed_dim

                # Max pooling/avg pooling
                if self.temporal_fuse_method == "max_in_encoder":
                    x, _ = torch.max(x, dim=1, keepdim=False) # B, N, 1+patch_h*patch_w, embed_dim -> B, 1+patch_h*patch_w, embed_dim
                elif self.temporal_fuse_method == "mean_in_encoder":
                    x = torch.mean(x, dim=1, keepdim=False) # B, N, 1+patch_h*patch_w, embed_dim -> B, 1+patch_h*patch_w, embed_dim

                if i in blocks_to_take:
                    output.append(x)


            elif i > self.num_layers_until_collapse:
                if i in blocks_to_take:
                    output.append(x)

        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for i, blk in enumerate(block_chunk[i:]):  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
        fd_list=None,
        actual_batch_size=None,
        frame_length=None,
        temporal_fuse_method=None
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:

        if fd_list is None:
            raise ValueError("fd_list is required in FOSSA")

        if actual_batch_size is None:
            raise ValueError("actual_batch_size is required in FOSSA")

        if frame_length is None:
            raise ValueError("frame_length is required in FOSSA")

        if temporal_fuse_method is None:
            raise ValueError("temporal_fuse_method is required in FOSSA")

        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n, fd_list, actual_batch_size, frame_length, temporal_fuse_method)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens:] for out in outputs]
        # B=B'=5(focal stack size)x2
        # B = B
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def vit_small(patch_size=16, num_register_tokens=0, **kwargs):
    model = FocusStackTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_base(patch_size=16, num_register_tokens=0, **kwargs):
    model = FocusStackTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, num_register_tokens=0, **kwargs):
    model = FocusStackTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, num_register_tokens=0, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = FocusStackTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def FST(model_name, **kwargs):
    model_zoo = {
        "vits": vit_small,
        "vitb": vit_base,
        "vitl": vit_large,
        "vitg": vit_giant2
    }

    return model_zoo[model_name](
        img_size=518,
        patch_size=14,
        init_values=1.0,
        ffn_layer="mlp" if model_name != "vitg" else "swiglufused",
        block_chunks=0,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        **kwargs
    )
