"""
Transformer encoder and decoder components for AV-HuBERT.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .utils import get_activation_fn


class TransformerEncoder(nn.Module):
    """Transformer encoder for AV-HuBERT."""
    
    def __init__(
        self,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        layers: int = 12,
        attention_heads: int = 12,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        layerdrop: float = 0.0,
        layer_norm_first: bool = False,
        activation_fn: str = "gelu",
    ):
        super().__init__()
        
        self.dropout = dropout
        self.layerdrop = layerdrop
        self.embed_dim = embed_dim
        self.layers = layers
        
        self.layers = nn.ModuleList([])
        for layer_idx in range(layers):
            layer = TransformerEncoderLayer(
                embed_dim=embed_dim,
                ffn_embed_dim=ffn_embed_dim,
                attention_heads=attention_heads,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation_dropout=activation_dropout,
                layer_norm_first=layer_norm_first,
                activation_fn=activation_fn,
            )
            self.layers.append(layer)
        
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        x: Tensor,
        padding_mask: Optional[Tensor] = None,
        layer_to_past_key_value: Optional[Dict[int, Tuple[Tensor, Tensor]]] = None,
        return_all_hiddens: bool = False,
    ) -> Tuple[Tensor, Optional[Dict[int, Tuple[Tensor, Tensor]]], List[Tensor]]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            padding_mask: Boolean tensor of shape (batch_size, seq_len) where True indicates padding
            layer_to_past_key_value: Dictionary mapping layer indices to cached key-value pairs
            return_all_hiddens: Whether to return hidden states from all layers
        
        Returns:
            Tuple of (output, cached_key_values, all_hidden_states)
        """
        all_hidden_states = []
        
        for layer_idx, layer in enumerate(self.layers):
            if self.training and self.layerdrop > 0.0:
                dropout_probability = torch.empty(1).uniform_()
                if dropout_probability < self.layerdrop:
                    continue
            
            past_key_value = layer_to_past_key_value.get(layer_idx, None) if layer_to_past_key_value is not None else None
            
            x, past_key_value = layer(
                x,
                padding_mask=padding_mask,
                past_key_value=past_key_value,
            )
            
            if layer_to_past_key_value is not None:
                layer_to_past_key_value[layer_idx] = past_key_value
            
            if return_all_hiddens:
                all_hidden_states.append(x)
        
        x = self.layer_norm(x)
        
        return x, layer_to_past_key_value, all_hidden_states


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer."""
    
    def __init__(
        self,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        attention_heads: int = 12,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        layer_norm_first: bool = False,
        activation_fn: str = "gelu",
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.layer_norm_first = layer_norm_first
        
        self.self_attn = MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=attention_heads,
            dropout=attention_dropout,
            self_attention=True,
        )
        
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation_fn = get_activation_fn(activation_fn)
        
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        
        self.final_layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        x: Tensor,
        padding_mask: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            padding_mask: Boolean tensor of shape (batch_size, seq_len) where True indicates padding
            past_key_value: Cached key-value pairs from previous forward pass
        
        Returns:
            Tuple of (output, cached_key_value)
        """
        residual = x
        
        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
        
        x, attn_weights, cached_key_value = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=padding_mask,
            need_weights=False,
            attn_mask=None,
            past_key_value=past_key_value,
        )
        
        x = self.dropout1(x)
        x = residual + x
        
        if not self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
        
        residual = x
        
        if self.layer_norm_first:
            x = self.final_layer_norm(x)
        
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = residual + x
        
        if not self.layer_norm_first:
            x = self.final_layer_norm(x)
        
        return x, cached_key_value


class TransformerDecoder(nn.Module):
    """Transformer decoder for sequence-to-sequence tasks."""
    
    def __init__(
        self,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        layers: int = 6,
        attention_heads: int = 4,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        layerdrop: float = 0.0,
        layer_norm_first: bool = False,
        activation_fn: str = "gelu",
        normalize_before: bool = False,
        learned_pos: bool = False,
        no_token_positional_embeddings: bool = False,
        max_target_positions: int = 2048,
    ):
        super().__init__()
        
        self.dropout = dropout
        self.layerdrop = layerdrop
        self.embed_dim = embed_dim
        self.layers = layers
        self.normalize_before = normalize_before
        
        self.layers = nn.ModuleList([])
        for layer_idx in range(layers):
            layer = TransformerDecoderLayer(
                embed_dim=embed_dim,
                ffn_embed_dim=ffn_embed_dim,
                attention_heads=attention_heads,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation_dropout=activation_dropout,
                layer_norm_first=layer_norm_first,
                activation_fn=activation_fn,
                normalize_before=normalize_before,
            )
            self.layers.append(layer)
        
        if normalize_before:
            self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        x: Tensor,
        encoder_out: Optional[Tensor] = None,
        encoder_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        return_all_hiddens: bool = False,
    ) -> Tuple[Tensor, List[Tensor]]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            encoder_out: Output from the encoder
            encoder_padding_mask: Padding mask from the encoder
            incremental_state: State for incremental decoding
            return_all_hiddens: Whether to return hidden states from all layers
        
        Returns:
            Tuple of (output, all_hidden_states)
        """
        all_hidden_states = []
        
        for layer_idx, layer in enumerate(self.layers):
            if self.training and self.layerdrop > 0.0:
                dropout_probability = torch.empty(1).uniform_()
                if dropout_probability < self.layerdrop:
                    continue
            
            x = layer(
                x,
                encoder_out=encoder_out,
                encoder_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
            )
            
            if return_all_hiddens:
                all_hidden_states.append(x)
        
        if self.normalize_before:
            x = self.layer_norm(x)
        
        return x, all_hidden_states


class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer."""
    
    def __init__(
        self,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        attention_heads: int = 4,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        layer_norm_first: bool = False,
        activation_fn: str = "gelu",
        normalize_before: bool = False,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.layer_norm_first = layer_norm_first
        self.normalize_before = normalize_before
        
        self.self_attn = MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=attention_heads,
            dropout=attention_dropout,
            self_attention=True,
        )
        
        self.encoder_attn = MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=attention_heads,
            dropout=attention_dropout,
            self_attention=False,
        )
        
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.encoder_attn_layer_norm = nn.LayerNorm(embed_dim)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation_fn = get_activation_fn(activation_fn)
        
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        
        self.final_layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        x: Tensor,
        encoder_out: Optional[Tensor] = None,
        encoder_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            encoder_out: Output from the encoder
            encoder_padding_mask: Padding mask from the encoder
            incremental_state: State for incremental decoding
        
        Returns:
            Output tensor
        """
        residual = x
        
        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
        
        x, attn_weights, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=None,
        )
        
        x = self.dropout1(x)
        x = residual + x
        
        if not self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
        
        if encoder_out is not None:
            residual = x
            
            if self.layer_norm_first:
                x = self.encoder_attn_layer_norm(x)
            
            x, attn_weights, _ = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                need_weights=False,
                attn_mask=None,
            )
            
            x = self.dropout2(x)
            x = residual + x
            
            if not self.layer_norm_first:
                x = self.encoder_attn_layer_norm(x)
        
        residual = x
        
        if self.layer_norm_first:
            x = self.final_layer_norm(x)
        
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = self.dropout3(x)
        x = residual + x
        
        if not self.layer_norm_first:
            x = self.final_layer_norm(x)
        
        return x


class MultiheadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        self_attention: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        self.scaling = self.head_dim ** -0.5
        self.self_attention = self_attention
        
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
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tuple[Tensor, Tensor]]]:
        """
        Args:
            query: Query tensor of shape (batch_size, seq_len, embed_dim)
            key: Key tensor of shape (batch_size, seq_len, embed_dim)
            value: Value tensor of shape (batch_size, seq_len, embed_dim)
            key_padding_mask: Boolean tensor of shape (batch_size, seq_len) where True indicates padding
            incremental_state: State for incremental decoding
            need_weights: Whether to return attention weights
            attn_mask: Attention mask
            past_key_value: Cached key-value pairs from previous forward pass
        
        Returns:
            Tuple of (output, attention_weights, cached_key_value)
        """
        batch_size, tgt_len, embed_dim = query.size()
        src_len = key.size(1)
        
        q = self.q_proj(query) * self.scaling
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        if past_key_value is not None:
            # Reuse k, v from previous forward pass
            past_key, past_value = past_key_value
            k = torch.cat([past_key, k], dim=2)
            v = torch.cat([past_value, v], dim=2)
        
        cached_key_value = (k, v) if self.self_attention else None
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        
        if attn_mask is not None:
            attn_weights = attn_weights.masked_fill(attn_mask == 0, float("-inf"))
        
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, tgt_len, embed_dim
        )
        
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attn_weights if need_weights else None, cached_key_value