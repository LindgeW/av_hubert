"""
Utility functions for AV-HuBERT model.
"""

import math
import random
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def compute_mask_indices(
    shape: Tuple[int, int],
    padding_mask: Optional[Tensor],
    mask_prob: float,
    mask_length: int,
    mask_type: str = "static",
    mask_other: float = 0.0,
    min_masks: int = 0,
    no_overlap: bool = False,
    min_space: int = 0,
    require_same_masks: bool = True,
    mask_dropout: float = 0.0,
) -> Tensor:
    """Computes random mask spans for a given shape.
    
    Args:
        shape: The shape for which to compute masks.
        padding_mask: The padding mask of the same shape, containing 1 at padding positions.
        mask_prob: Probability for each token to be chosen as start of the span to be masked.
        mask_length: Size of the span to be masked.
        mask_type: How to compute mask lengths. Options: static, uniform, normal, poisson.
        mask_other: Secondary mask argument (used for more complex distributions).
        min_masks: Minimum number of masked spans.
        no_overlap: If true, masked spans cannot overlap.
        min_space: Minimum space between spans (if no_overlap is enabled).
        require_same_masks: If true, will ensure that each seq has same number of masks.
        mask_dropout: Probability of dropping a mask.
    
    Returns:
        A boolean tensor of shape `shape`, containing True at masked positions.
    """
    
    bsz, all_sz = shape
    mask = torch.full((bsz, all_sz), False, dtype=torch.bool, device=padding_mask.device)
    
    all_num_mask = int(
        mask_prob * all_sz / float(mask_length)
        + random.random()
    )
    
    all_num_mask = max(min_masks, all_num_mask)
    
    mask_idcs = []
    for i in range(bsz):
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            num_mask = int(
                mask_prob * sz / float(mask_length)
                + random.random()
            )
            num_mask = max(min_masks, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask

        lengths = torch.full((num_mask,), mask_length, dtype=torch.long)
        
        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)

        if no_overlap:
            mask_idc = []
            
            def arrange(s, e, length, keep_length):
                span_start = torch.randint(s, e - length, (1,))
                mask_idc.extend(span_start + i for i in range(length))
                
                new_s = span_start + length + min_space
                if new_s < e and keep_length > 0:
                    arrange(new_s, e, length, keep_length - 1)
            
            arrange(0, sz, mask_length, num_mask - 1)
        else:
            min_len = min(lengths)
            if sz - min_len <= num_mask:
                min_len = sz - num_mask - 1
            
            mask_idc = torch.multinomial(torch.ones(sz - min_len), num_mask, replacement=False)
            
            mask_idc = torch.minimum(
                mask_idc + torch.arange(num_mask) * min_len, torch.arange(num_mask) * min_len + sz - min_len
            )
            mask_idc = torch.cat([torch.arange(mask_idc[i], mask_idc[i] + lengths[i]) for i in range(num_mask)])
            mask_idc = mask_idc[mask_idc < sz]

        if len(mask_idc) >= sz:
            mask_idc = mask_idc[:sz]
        
        mask_idcs.append(torch.unique(mask_idc))
    
    if require_same_masks:
        min_len = min(len(m) for m in mask_idcs)
        for i, mask_idc in enumerate(mask_idcs):
            mask_idcs[i] = mask_idc[:min_len]
    
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > 0:
            mask[i, mask_idc] = True
    
    if mask_dropout > 0:
        mask = torch.where(
            torch.rand(mask.shape) < mask_dropout,
            torch.zeros_like(mask),
            mask,
        )
    
    return mask


def get_activation_fn(activation: str) -> nn.Module:
    """Returns the activation function corresponding to `activation`."""
    
    def _gelu_linear(x):
        return x * 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    
    def _relu_squared(x):
        return torch.pow(F.relu(x), 2)
    
    def _swish(x):
        return x * torch.sigmoid(x)
    
    def _mish(x):
        return x * torch.tanh(F.softplus(x))
    
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "gelu_fast":
        return _gelu_linear
    elif activation == "gelu_accurate":
        return F.gelu
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    elif activation == "swish":
        return _swish
    elif activation == "mish":
        return _mish
    elif activation == "relu_squared":
        return _relu_squared
    else:
        raise RuntimeError(f"activation {activation} not supported")


def make_conv_pos(e, k, g):
    """Creates a convolutional positional embedding layer."""
    pos_conv = nn.Conv1d(
        e,
        e,
        kernel_size=k,
        padding=k // 2,
        groups=g,
    )
    dropout = 0
    std = math.sqrt(4.0 / (e * k))
    nn.init.normal_(pos_conv.weight, mean=0.0, std=std)
    nn.init.constant_(pos_conv.bias, 0)
    
    pos_conv = nn.utils.weight_norm(pos_conv, name="weight", dim=2)
    pos_conv = nn.Sequential(pos_conv, SamePad(k), nn.Dropout(dropout))
    
    return pos_conv


class SamePad(nn.Module):
    """Module to ensure same padding for convolutional layers."""
    
    def __init__(self, kernel_size):
        super().__init__()
        self.remove = kernel_size % 2 == 0
    
    def forward(self, x):
        if self.remove:
            x = x[:, :, :-1]
        return x


class GradMultiply(torch.autograd.Function):
    """Custom autograd function for gradient multiplication."""
    
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res
    
    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


def index_put(tensor, indices, value):
    """Helper function to put values at specific indices."""
    if tensor.device.type == "cpu":
        tensor[indices] = value
    else:
        tensor.index_put_(indices, value)
    return tensor


def pad_to_multiple(x, multiple, dim=-1, value=0):
    """Pads tensor to be a multiple of `multiple` along the specified dimension."""
    tsz = x.size(dim)
    m = tsz / multiple
    remainder = math.ceil(m) * multiple - tsz
    if remainder > 0:
        pad = [0] * (2 * len(x.size()))
        pad[2 * dim + 1] = remainder
        x = F.pad(x, pad, value=value)
    return x


def compute_mask_indices_hierarchical(
    shape: Tuple[int, int],
    padding_mask: Optional[Tensor],
    mask_prob: float,
    mask_length: int,
    mask_type: str = "static",
    mask_other: float = 0.0,
    min_masks: int = 0,
    no_overlap: bool = False,
    min_space: int = 0,
    require_same_masks: bool = True,
    mask_dropout: float = 0.0,
) -> Tensor:
    """Computes hierarchical mask indices for audio-visual data."""
    # This is a simplified version - in practice, you might want more sophisticated
    # hierarchical masking for audio-visual data
    return compute_mask_indices(
        shape, padding_mask, mask_prob, mask_length, mask_type, mask_other,
        min_masks, no_overlap, min_space, require_same_masks, mask_dropout
    )