from functools import partial
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from mamba_ssm.models.mixer_seq_simple import RMSNorm, _init_weights, create_block

try:
    from mamba_ssm.ops.triton.layer_norm import layer_norm_fn, rms_norm_fn
except ImportError:
    layer_norm_fn, rms_norm_fn = None, None



class Mamba2backbone(nn.Module):
    """
    Stack of Mamba2 blocks with pre-norm architecture.
    Final layer norm is moved to the main model (transformers pattern).
    """

    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_intermediate: int = 0,
        ssm_cfg: Optional[dict] = None,
        rms_norm: bool = True,
        fused_add_norm: bool = True,
        norm_epsilon: float = 1e-5,
        residual_in_fp32: bool = True,
        initializer_cfg: Optional[dict] = None,
        drop_out: float = 0.0,
    ):
        super().__init__()
        
        if ssm_cfg is None:
            ssm_cfg = {"layer": "Mamba2", "d_state": 128, "d_conv": 4, "expand": 2}

        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=None,
                    attn_cfg=None,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model,
            eps=norm_epsilon,
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1
                if d_intermediate == 0
                else 2,  # 2 if we have MLP
            )
        )

        if drop_out > 0:
            self.dropout = nn.Dropout(drop_out)
        else:
            self.dropout = None

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(
                batch_size, max_seqlen, dtype=dtype, **kwargs
            )
            for i, layer in enumerate(self.layers)
        }

    def forward(
        self,
        hidden_states,
        inference_params=None,
        **mixer_kwargs,
    ):
        hidden_states_list = []
        residual = None
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states,
                residual,
                inference_params=inference_params,
                **mixer_kwargs,
            )

            if self.dropout is not None:
                hidden_states = self.dropout(hidden_states)

        if not self.fused_add_norm:
            residual = (
                (hidden_states + residual) if residual is not None else hidden_states
            )
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm),
            )
        return hidden_states


class MambaEye(nn.Module):
    """
    MambaEye model that concatenates an image sequence and a move embedding,
    then processes them through a Mamba2 backbone for classification.
    """

    def __init__(
        self,
        num_classes: int,
        input_dim: int,
        dim: int,
        depth: int,
        d_state: int,
        d_conv: int,
        expand: int,
        d_intermediate_multiple: int = 0,
        layer_norm_eps: float = 1e-5,
        residual_in_fp32: bool = True,
        drop_out: float = 0.0,
    ):
        super().__init__()
        # Initial projection
        self.init_fc = nn.Sequential(nn.Linear(input_dim, dim), nn.SiLU())

        # Mamba2 backbone with pre-norm architecture
        self.mamba2_net = Mamba2backbone(
            d_model=dim,
            n_layer=depth,
            d_intermediate=d_intermediate_multiple * dim,
            ssm_cfg={
                "layer": "Mamba2",
                "d_state": d_state,
                "d_conv": d_conv,
                "expand": expand,
            },
            rms_norm=True,
            fused_add_norm=True,
            norm_epsilon=layer_norm_eps,
            residual_in_fp32=residual_in_fp32,
            drop_out=drop_out,
        )
        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(dim, num_classes),
        )

        if drop_out > 0:
            self.dropout = nn.Dropout(drop_out)
        else:
            self.dropout = None

    def forward(
        self,
        img_sequence,
        move_embedding,
        inference_params=None,
    ):
        # Concatenate inputs
        x = torch.cat((img_sequence, move_embedding), dim=-1)  # (B, L, D)

        # Initial projection
        x = self.init_fc(x)
        if self.dropout is not None:
            x = self.dropout(x)

        # Forward through Mamba2 backbone
        x = self.mamba2_net(
            x, inference_params=inference_params
        )

        # Classification head
        classification_sequence = self.classification_head(x)

        return classification_sequence
