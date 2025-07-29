"""
Core neural network modules for AVHuBERT.
Replaces fairseq modules with clean PyTorch implementations.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LayerNorm(nn.Module):
    """Layer normalization module."""
    
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


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


def grad_multiply(x, scale):
    """Multiply gradients by a scale factor."""
    return GradMultiply.apply(x, scale)


class PositionalEmbedding(nn.Module):
    """Positional embedding for transformer models."""
    
    def __init__(self, num_embeddings, embedding_dim, padding_idx, learned=False):
        super().__init__()
        self.learned = learned
        if self.learned:
            self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
            nn.init.normal_(self.weight, mean=0, std=embedding_dim ** -0.5)
            nn.init.constant_(self.weight[padding_idx], 0)
        else:
            self.register_buffer("weight", self._get_sinusoidal_embedding(num_embeddings, embedding_dim, padding_idx))
        self.padding_idx = padding_idx

    def _get_sinusoidal_embedding(self, num_embeddings, embedding_dim, padding_idx):
        """Create sinusoidal positional embeddings."""
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input):
        return F.embedding(input, self.weight, self.padding_idx)


class ConvFeatureExtractionModel(nn.Module):
    """Convolutional feature extraction model for audio."""
    
    def __init__(self, conv_layers, dropout=0.0, mode="default"):
        super().__init__()
        
        def block(n_in, n_out, k, stride, is_layer_norm=False, is_group_norm=False, conv_bias=False):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv
            
            assert (is_layer_norm and is_group_norm) == False, "layer norm and group norm are mutually exclusive"
            
            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())
        
        self.conv_layers = nn.ModuleList()
        in_d = 1
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
        # x: [B, T]
        x = x.unsqueeze(1)  # [B, 1, T]
        for conv in self.conv_layers:
            x = conv(x)
        return x


class TransposeLast(nn.Module):
    """Transpose the last two dimensions."""
    
    def forward(self, x):
        return x.transpose(-2, -1)


class Fp32LayerNorm(LayerNorm):
    """LayerNorm with fp32 precision."""
    
    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float(),
            self.bias.float(),
            self.eps,
        )
        return output.type_as(input)


class Fp32GroupNorm(nn.GroupNorm):
    """GroupNorm with fp32 precision."""
    
    def forward(self, input):
        output = F.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float(),
            self.bias.float(),
            self.eps,
        )
        return output.type_as(input)


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer."""
    
    def __init__(
        self,
        embed_dim,
        ffn_embed_dim,
        attention_heads,
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.1,
        activation_fn="relu",
        layer_norm_first=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_fn = activation_fn
        self.layer_norm_first = layer_norm_first
        
        self.self_attn = self.build_self_attention(self.embed_dim, attention_heads, attention_dropout)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(self.embed_dim, self.ffn_embed_dim)
        self.fc2 = nn.Linear(self.ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(activation_dropout)

    def build_self_attention(self, embed_dim, attention_heads, attention_dropout):
        return MultiheadAttention(embed_dim, attention_heads, dropout=attention_dropout)

    def forward(self, x, self_attn_mask=None, self_attn_padding_mask=None):
        residual = x
        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                attn_mask=self_attn_mask,
            )
            x = self.dropout1(x)
            x = residual + x
            
            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout3(x)
            x = self.fc2(x)
            x = self.dropout2(x)
            x = residual + x
        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                attn_mask=self_attn_mask,
            )
            x = self.dropout1(x)
            x = residual + x
            x = self.self_attn_layer_norm(x)
            
            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout3(x)
            x = self.fc2(x)
            x = self.dropout2(x)
            x = residual + x
            x = self.final_layer_norm(x)
        
        return x, attn


class MultiheadAttention(nn.Module):
    """Multi-head attention module."""
    
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        self.scaling = self.head_dim ** -0.5
        
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.k_proj.bias is not None:
            nn.init.constant_(self.k_proj.bias, 0.0)
        if self.v_proj.bias is not None:
            nn.init.constant_(self.v_proj.bias, 0.0)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.0)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        attn_mask=None,
        need_weights=True,
        is_training=True,
    ):
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        
        q = self.q_proj(query) * self.scaling
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        
        if attn_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attn_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        if is_training:
            attn_weights = F.dropout(attn_weights, p=self.dropout)
        
        attn_output = torch.bmm(attn_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        
        if need_weights:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            return attn_output, attn_weights
        else:
            return attn_output, None


def get_activation_fn(activation: str):
    """Get activation function by name."""
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "swish":
        return F.silu
    else:
        raise ValueError(f"Unknown activation function: {activation}")