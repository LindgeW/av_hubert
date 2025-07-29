"""
Transformer encoder for AVHuBERT.
Simplified version without fairseq dependencies.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


def get_activation_fn(activation: str):
    """Return activation function"""
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    else:
        raise RuntimeError(f"Unsupported activation: {activation}")


class LayerNorm(nn.Module):
    """Layer normalization module"""
    
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


class MultiheadAttention(nn.Module):
    """Multi-head attention module"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.scaling = self.head_dim ** -0.5
        
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
            
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
    
    def forward(self, query, key=None, value=None, key_padding_mask=None, attn_mask=None):
        """
        Args:
            query: (seq_len, batch_size, embed_dim)
            key: (seq_len, batch_size, embed_dim) 
            value: (seq_len, batch_size, embed_dim)
            key_padding_mask: (batch_size, seq_len)
            attn_mask: (seq_len, seq_len)
        """
        if key is None:
            key = query
        if value is None:
            value = query
            
        seq_len, batch_size, embed_dim = query.size()
        
        # Linear projections
        q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(seq_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(seq_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1) 
        v = v.view(seq_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        
        # Scale query
        q = q * self.scaling
        
        # Attention scores
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        
        # Apply attention mask
        if attn_mask is not None:
            attn_weights += attn_mask.unsqueeze(0)
            
        # Apply key padding mask  
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(batch_size, self.num_heads, seq_len, seq_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )
            attn_weights = attn_weights.view(batch_size * self.num_heads, seq_len, seq_len)
        
        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # Apply attention to values
        attn_output = torch.bmm(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(0, 1).contiguous().view(seq_len, batch_size, embed_dim)
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attn_weights


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer"""
    
    def __init__(self, embed_dim, num_heads, ffn_embed_dim, dropout=0.1, 
                 attention_dropout=0.1, activation_dropout=0.0, activation_fn="relu",
                 layer_norm_first=False):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.self_attn = MultiheadAttention(embed_dim, num_heads, dropout=attention_dropout)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(activation_dropout) 
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation_fn = get_activation_fn(activation_fn)
        
        self.self_attn_layer_norm = LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        self.final_layer_norm = LayerNorm(embed_dim)
        
        self.layer_norm_first = layer_norm_first
        
    def forward(self, x, self_attn_mask=None, self_attn_padding_mask=None):
        """
        Args:
            x: (seq_len, batch_size, embed_dim)
            self_attn_mask: (seq_len, seq_len)
            self_attn_padding_mask: (batch_size, seq_len)
        """
        residual = x
        
        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, _ = self.self_attn(x, attn_mask=self_attn_mask, key_padding_mask=self_attn_padding_mask)
            x = self.dropout1(x)
            x = residual + x
            
            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            x, _ = self.self_attn(x, attn_mask=self_attn_mask, key_padding_mask=self_attn_padding_mask)
            x = self.dropout1(x)
            x = residual + x
            x = self.self_attn_layer_norm(x)
            
            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)
            
        return x


class TransformerEncoder(nn.Module):
    """Transformer encoder stack"""
    
    def __init__(self, config):
        super().__init__()
        
        self.num_layers = config.encoder_layers
        self.embed_dim = config.encoder_embed_dim
        self.layerdrop = getattr(config, 'encoder_layerdrop', 0.0)
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=config.encoder_embed_dim,
                num_heads=config.encoder_attention_heads,
                ffn_embed_dim=config.encoder_ffn_embed_dim,
                dropout=config.dropout,
                attention_dropout=config.attention_dropout,
                activation_dropout=config.activation_dropout,
                activation_fn=config.activation_fn,
                layer_norm_first=config.layer_norm_first,
            )
            for _ in range(config.encoder_layers)
        ])
        
        if config.layer_norm_first:
            self.layer_norm = LayerNorm(config.encoder_embed_dim)
        else:
            self.layer_norm = None
            
    def forward(self, x, padding_mask=None):
        """
        Args:
            x: (batch_size, seq_len, embed_dim) - note different from layers
            padding_mask: (batch_size, seq_len)
        """
        # Convert to (seq_len, batch_size, embed_dim) for transformer layers
        x = x.transpose(0, 1)
        
        # Pass through transformer layers
        for layer in self.layers:
            # Apply layer dropout during training
            if self.training and self.layerdrop > 0:
                dropout_probability = torch.empty(1).uniform_()
                if dropout_probability < self.layerdrop:
                    continue
                    
            x = layer(x, self_attn_padding_mask=padding_mask)
        
        if self.layer_norm is not None:
            x = self.layer_norm(x)
            
        # Convert back to (batch_size, seq_len, embed_dim)
        x = x.transpose(0, 1)
        
        return x


class ConvFeatureExtractionModel(nn.Module):
    """Convolutional feature extraction model"""
    
    def __init__(self, conv_layers, dropout=0.0, mode="default", conv_bias=False):
        super().__init__()
        
        def block(n_in, n_out, k, stride, is_layer_norm=False, is_group_norm=False, conv_bias=False):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (is_layer_norm and is_group_norm) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def forward(self, x):
        # BTC -> BCT
        x = x.transpose(1, 2)

        for conv in self.conv_layers:
            x = conv(x)

        # BCT -> BTC
        x = x.transpose(1, 2)
        return x


class TransposeLast(nn.Module):
    """Transpose last two dimensions"""
    
    def __init__(self, deconstruct_idx=None):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(-2, -1)


class SubsampleLayer(nn.Module):
    """Subsampling layer for reducing sequence length"""
    
    def __init__(self, input_dim, output_dim, conv_layers, dropout=0.1):
        super().__init__()
        
        self.conv_layers = ConvFeatureExtractionModel(
            conv_layers=conv_layers,
            dropout=dropout,
            mode="default",
            conv_bias=False
        )
        
        self.proj = nn.Linear(input_dim, output_dim)
        
    def forward(self, x, padding_mask=None):
        x = self.conv_layers(x)
        x = self.proj(x)
        
        # Update padding mask if provided
        if padding_mask is not None:
            padding_mask = self._get_conv_output_lengths(padding_mask)
            
        return x, padding_mask
    
    def _get_conv_output_lengths(self, input_lengths):
        """Compute output lengths after convolution"""
        # This is a simplified version - in practice you'd need to compute
        # the exact output lengths based on the conv layer parameters
        return input_lengths