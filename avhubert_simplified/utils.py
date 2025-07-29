"""
Utility functions for AVHuBERT.
Simplified version without fairseq dependencies.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List
import math


def compute_mask_indices(
    shape: Tuple[int, int],
    padding_mask: Optional[torch.Tensor],
    mask_prob: float,
    mask_length: int,
    mask_type: str = "static",
    mask_other: float = 0.0,
    min_masks: int = 0,
    no_overlap: bool = False,
    min_space: int = 0,
) -> torch.Tensor:
    """
    Computes random mask spans for a given shape
    
    Args:
        shape: (batch_size, timesteps)
        padding_mask: (batch_size, timesteps) padding mask 
        mask_prob: probability of each token being masked
        mask_length: mask length
        mask_type: mask type ("static", "uniform", "normal", "poisson")
        mask_other: secondary mask probability
        min_masks: minimum number of masks
        no_overlap: if false, a mask can be applied to a masked token
        min_space: minimum space between masks
    """
    batch_size, sequence_length = shape
    mask = torch.full((batch_size, sequence_length), False, dtype=torch.bool)
    
    for i in range(batch_size):
        # Get valid indices (exclude padding)
        if padding_mask is not None:
            valid_length = (~padding_mask[i]).sum().item()
        else:
            valid_length = sequence_length
            
        if valid_length == 0:
            continue
            
        # Calculate number of masks
        num_mask = int(
            # add a random number for probabilistic rounding
            mask_prob * valid_length / float(mask_length)
            + torch.rand(1).item()
        )
        num_mask = max(min_masks, num_mask)
        
        # Generate mask lengths based on type
        if mask_type == "static":
            mask_lengths = [mask_length] * num_mask
        elif mask_type == "uniform":
            mask_lengths = torch.randint(1, mask_length + 1, (num_mask,)).tolist()
        elif mask_type == "normal":
            mask_lengths = torch.normal(mask_length, mask_length * 0.1, (num_mask,))
            mask_lengths = torch.clamp(mask_lengths, 1, mask_length * 2).int().tolist()
        elif mask_type == "poisson":
            mask_lengths = torch.poisson(torch.tensor(mask_length).float().expand(num_mask)).int().tolist()
        else:
            raise ValueError(f"Unknown mask type: {mask_type}")
        
        # Remove lengths that are too long
        mask_lengths = [l for l in mask_lengths if l <= valid_length]
        if not mask_lengths:
            continue
            
        # Generate mask starts
        mask_starts = []
        for mask_len in mask_lengths:
            if no_overlap:
                # Find positions that don't overlap with existing masks
                available_starts = []
                for start in range(valid_length - mask_len + 1):
                    if not any(mask[i, start:start + mask_len]):
                        # Check minimum space constraint
                        if min_space > 0:
                            valid_start = True
                            for existing_start in mask_starts:
                                if abs(start - existing_start) < min_space:
                                    valid_start = False
                                    break
                            if valid_start:
                                available_starts.append(start)
                        else:
                            available_starts.append(start)
                
                if available_starts:
                    start = torch.randint(0, len(available_starts), (1,)).item()
                    start = available_starts[start]
                    mask_starts.append(start)
                    mask[i, start:start + mask_len] = True
            else:
                # Allow overlapping masks
                if valid_length > mask_len:
                    start = torch.randint(0, valid_length - mask_len + 1, (1,)).item()
                    mask[i, start:start + mask_len] = True
    
    return mask


class GradMultiply(nn.Module):
    """Gradient multiplication layer"""
    
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        if self.training and self.scale != 1.0:
            return GradMultiplyFunction.apply(x, self.scale)
        return x


class GradMultiplyFunction(torch.autograd.Function):
    """Gradient multiplication function"""
    
    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        return input

    @staticmethod  
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


def apply_mask(x: torch.Tensor, mask_indices: torch.Tensor, mask_value: torch.Tensor) -> torch.Tensor:
    """Apply mask to input tensor"""
    if mask_indices.any():
        x = x.clone()
        if mask_value.dim() == 1:
            # Broadcast mask value to match input dimensions
            mask_value = mask_value.view(1, 1, -1).expand(x.size(0), 1, -1)
        x[mask_indices] = mask_value.to(x.device)
    return x


def get_conv_output_lengths(input_lengths: torch.Tensor, conv_layers: List[Tuple[int, int, int]]) -> torch.Tensor:
    """
    Compute output lengths after a series of 1D convolutions
    
    Args:
        input_lengths: input sequence lengths
        conv_layers: list of (out_channels, kernel_size, stride) tuples
    """
    for _, kernel_size, stride in conv_layers:
        input_lengths = torch.floor((input_lengths + 2 * 0 - kernel_size) / stride + 1).long()
        input_lengths = torch.clamp(input_lengths, min=0)
    return input_lengths


def sample_negatives(
    features: torch.Tensor,
    num_negatives: int,
    mask_indices: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Sample negative examples for contrastive learning
    
    Args:
        features: (batch_size, seq_len, feature_dim)
        num_negatives: number of negatives to sample
        mask_indices: (batch_size, seq_len) boolean mask
        
    Returns:
        Negative samples of shape (batch_size, seq_len, num_negatives, feature_dim)
    """
    batch_size, seq_len, feature_dim = features.shape
    
    # Create indices for sampling
    if mask_indices is not None:
        # Only sample from masked positions
        valid_indices = mask_indices.nonzero(as_tuple=False)  # (num_valid, 2)
        if valid_indices.size(0) == 0:
            return torch.zeros(batch_size, seq_len, num_negatives, feature_dim, device=features.device)
    else:
        # Sample from all positions
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, seq_len)
        seq_indices = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        valid_indices = torch.stack([batch_indices.flatten(), seq_indices.flatten()], dim=1)
    
    # Sample negative indices
    num_valid = valid_indices.size(0)
    neg_indices = torch.randint(0, num_valid, (batch_size, seq_len, num_negatives), device=features.device)
    
    # Get negative features
    negatives = torch.zeros(batch_size, seq_len, num_negatives, feature_dim, device=features.device)
    
    for b in range(batch_size):
        for t in range(seq_len):
            for n in range(num_negatives):
                neg_idx = neg_indices[b, t, n]
                if neg_idx < num_valid:
                    neg_b, neg_t = valid_indices[neg_idx]
                    negatives[b, t, n] = features[neg_b, neg_t]
    
    return negatives


def compute_cosine_similarity(x: torch.Tensor, y: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute cosine similarity between tensors"""
    return F.cosine_similarity(x, y, dim=dim)


def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """Compute accuracy between predictions and targets"""
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=-1)
    
    correct = (predictions == targets).float()
    
    if mask is not None:
        correct = correct * mask.float()
        total = mask.sum().float()
    else:
        total = correct.numel()
    
    if total == 0:
        return 0.0
    
    return (correct.sum() / total).item()


def buffered_arange(max_val: int, device: torch.device = None) -> torch.Tensor:
    """Create a buffered arange tensor"""
    if device is None:
        device = torch.device('cpu')
    return torch.arange(max_val, device=device)


def parse_conv_feature_layers(conv_feature_layers: str) -> List[Tuple[int, int, int]]:
    """Parse conv feature layers string into list of tuples"""
    return eval(conv_feature_layers)


class Fp32GroupNorm(nn.GroupNorm):
    """GroupNorm in FP32 for stable training"""
    
    def forward(self, input):
        output = F.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


class Fp32LayerNorm(nn.LayerNorm):
    """LayerNorm in FP32 for stable training"""
    
    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


def get_annealed_rate(start: float, end: float, curr_step: int, total_steps: int) -> float:
    """Get annealed rate for curriculum learning"""
    r = curr_step / total_steps
    return start + (end - start) * r


def get_activation_fn(activation: str):
    """Get activation function by name"""
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu": 
        return nn.GELU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "swish":
        return nn.SiLU()
    else:
        raise RuntimeError(f"activation function {activation} not supported")


# Import F for functions that need it
import torch.nn.functional as F