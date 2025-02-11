import os, sys
import argparse
import math, random
import torch
import torch.nn as nn
from models.custom_layers import FMoE
from models.custom_layers import FMoELinear
from models.custom_layers_opt import FMoEOpt


class _Expert(nn.Module):
    r"""
    An expert using 2 FMoELinear modules to speed up the computation of experts
    within one worker.
    """

    def __init__(self, num_expert, d_model, d_hidden, activation, rank=0):
        super().__init__()
        self.htoh4 = FMoELinear(num_expert, d_model, d_hidden, bias=True, rank=rank)
        self.h4toh = FMoELinear(num_expert, d_hidden, d_model, bias=True, rank=rank)
        self.activation = activation

    def forward(self, inp, fwd_expert_count):
        r"""
        First expand input to 4h (the hidden size is variable, but is called h4
        for convenience). Then perform activation. Finally shirink back to h.
        """
        #import pdb;pdb.set_trace()
        x = self.htoh4(inp, fwd_expert_count)
        x = self.activation(x)
        x = self.h4toh(x, fwd_expert_count)
        return x


class FMoESSMMLP(FMoE):
    r"""
    A complete MoE MLP module in a SSM block.
    * `activation` is the activation function to be used in MLP in each expert.
    * `d_hidden` is the dimension of the MLP layer.
    """

    def __init__(
        self,
        num_expert=32,
        d_model=1024,
        d_hidden=4096,
        activation=torch.nn.GELU(),
        expert_dp_comm="none",
        expert_rank=0,
        moe_top_k=2,
        reorder = False,
        **kwargs
    ):
        super().__init__(
            num_expert=num_expert, d_model=d_model, moe_top_k=moe_top_k, reorder = reorder, **kwargs
        )
        self.experts = _Expert(
            num_expert, d_model, d_hidden, activation, rank=expert_rank
        )
        self.reorder = reorder
        self.mark_parallel_comm(expert_dp_comm)
        self.reorder = reorder

    def forward(self, inp: torch.Tensor):
        r"""
        This module wraps up the FMoE module with reshape, residual and layer
        normalization.
        """
        original_shape = inp.shape
        # print("original_shape", original_shape)
        inp = inp.reshape(-1, self.d_model)
        if self.reorder:
            output, gate_top_k_idx, gate_score = super().forward(inp)
            return output.reshape(original_shape), gate_top_k_idx, gate_score
        else:
            output = super().forward(inp)
            return output.reshape(original_shape)

class FMoESSMMLPOpt(FMoEOpt):
    r"""
    A complete MoE MLP module in a SSM block.
    * `activation` is the activation function to be used in MLP in each expert.
    * `d_hidden` is the dimension of the MLP layer.
    """

    def __init__(
        self,
        num_expert=32,
        d_model=1024,
        d_hidden=4096,
        activation=torch.nn.GELU(),
        expert_dp_comm="none",
        expert_rank=0,
        moe_top_k=2,
        freq=0.0,
        alpha=0.0,
        act_experts="shuffle",
        g_blance=False,
        opt_blance=False,
        combine_gate=False,
        opt_loss="mse",
        **kwargs
    ):
        super().__init__(
            num_expert=num_expert,
            d_model=d_model,
            moe_top_k=moe_top_k,
            freq=freq,
            alpha=alpha,
            act_experts=act_experts,
            g_blance=g_blance,
            opt_blance=opt_blance,
            combine_gate=combine_gate,
            opt_loss=opt_loss,
            **kwargs
        )
        self.experts = _Expert(
            num_expert, d_model, d_hidden, activation, rank=expert_rank
        )
        self.mark_parallel_comm(expert_dp_comm)

    def forward(self, inp: torch.Tensor):
        r"""
        This module wraps up the FMoE module with reshape, residual and layer
        normalization.
        """
        original_shape = inp.shape
        inp = inp.reshape(-1, self.d_model)
        output = super().forward(inp)
        return output.reshape(original_shape)
