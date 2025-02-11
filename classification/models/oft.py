"""
This script utilizes code from lora available at: 
https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py

Original Author: Simo Ryu
License: Apache License 2.0
"""


import json
import math
from itertools import groupby
from typing import Callable, Dict, List, Optional, Set, Tuple, Type, Union

import pickle

import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from safetensors.torch import safe_open
    from safetensors.torch import save_file as safe_save

    safetensors_available = True
except ImportError:
    from .safe_open import safe_open

    def safe_save(
        tensors: Dict[str, torch.Tensor],
        filename: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        raise EnvironmentError(
            "Saving safetensors requires the safetensors library. Please install with pip or similar."
        )

    safetensors_available = False


def project(R, eps):
    I = torch.zeros((R.size(0), R.size(0)), dtype=R.dtype, device=R.device)
    diff = R - I
    norm_diff = torch.norm(diff)
    if norm_diff <= eps:
        return R
    else:
        return I + eps * (diff / norm_diff)

def project_batch(R, eps=1e-5):
    # scaling factor for each of the smaller block matrix
    eps = eps * 1 / torch.sqrt(torch.tensor(R.shape[0]))
    I = torch.zeros((R.size(1), R.size(1)), device=R.device, dtype=R.dtype).unsqueeze(0).expand_as(R)
    diff = R - I
    norm_diff = torch.norm(R - I, dim=(1, 2), keepdim=True)
    mask = (norm_diff <= eps).bool()
    out = torch.where(mask, R, I + eps * (diff / norm_diff))
    return out

def cayley(data):
    r, c = list(data.shape)
    # Ensure the input matrix is skew-symmetric
    skew = 0.5 * (data - data.t())
    I = torch.eye(r, device=data.device)
    
    # Perform the Cayley parametrization
    Q = torch.mm(I + skew, torch.inverse(I - skew))
    return Q

def cayley_batch( data):
    b, r, c = data.shape
    # Ensure the input matrix is skew-symmetric
    skew = 0.5 * (data - data.transpose(1, 2))
    I = torch.eye(r, device=data.device).unsqueeze(0).expand(b, r, c)

    # Perform the Cayley parametrization
    Q = torch.bmm(I - skew, torch.inverse(I + skew))
    return Q

def block_diagonal(R, r = 4, block_share = False):
    if block_share:
        # Create a list of R repeated block_count times
        blocks = [R] * r
    else:
        # Create a list of R slices along the third dimension
        blocks = [R[i, ...] for i in range(r)]

    # Use torch.block_diag to create the block diagonal matrix
    A = torch.block_diag(*blocks)

    return A

def is_orthogonal(R, eps=1e-5):
    with torch.no_grad():
        RtR = torch.matmul(R.t(), R)
        diff = torch.abs(RtR - torch.eye(R.shape[1], dtype=R.dtype, device=R.device))
        return torch.all(diff < eps)

def is_identity_matrix(tensor):
    if not torch.is_tensor(tensor):
        raise TypeError("Input must be a PyTorch tensor.")
    if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
        return False
    identity = torch.eye(tensor.shape[0], device=tensor.device)
    return torch.all(torch.eq(tensor, identity))

def get_weight_bias(in_features, OFT_weight, OFT_bias, r=4, eps=1e-5, is_coft=True, block_share=False):
    if block_share:
        # Initialized as an identity matrix
        R_shape = [in_features //r, in_features // r]
        R = nn.Parameter(torch.zeros(R_shape[0], R_shape[0]), requires_grad=True)

        eps = eps * R_shape[0] * R_shape[0]
    else:
        # Initialized as an identity matrix
        R_shape = [r, in_features // r, in_features // r]
        R = torch.zeros(R_shape[1], R_shape[1])
        R = torch.stack([R] * r)
        R = nn.Parameter(R, requires_grad=True)
        eps = eps * R_shape[1] * R_shape[1]
    R_shape = [in_features // r, in_features // r]
    if block_share:
        if is_coft:
            with torch.no_grad():
                R.copy_(project(R, eps=eps))
        orth_rotate = cayley(R)
    else:
        if is_coft:
            with torch.no_grad():
                R.copy_(project_batch(R, eps=eps))
        orth_rotate = cayley_batch(R)

    # Block-diagonal parametrization
    block_diagonal_matrix = block_diagonal(orth_rotate)
    
    # fix filter
    fix_filt = OFT_weight.data
    fix_filt = torch.transpose(fix_filt, 0, 1)
    #print(fix_filt.shape)
    filt = torch.mm(block_diagonal_matrix.to('cuda'), fix_filt.to('cuda'))
    filt = torch.transpose(filt, 0, 1)

    # Apply the trainable identity matrix
    bias_term = OFT_bias.data if OFT_bias.data is not None else None
    return filt, bias_term
