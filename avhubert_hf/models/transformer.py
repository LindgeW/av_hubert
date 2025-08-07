import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple

from ..utils.activations import get_activation_fn
from ..utils.positional_encoding import SinusoidalPositionalEmbedding


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer."""
    
    def __init__(
        self,
        embed_dim: int,
        ffn_embed_dim: int,
        attention_heads: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
        has_relative_attention_bias: bool = False,
        max_relative_position: int = 32,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_fn = get_activation_fn(activation_fn)
        self.layer_norm_first = layer_norm_first
        
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.attention_heads,
            dropout=self.attention_dropout,
            self_attention=True,
            has_relative_attention_bias=has_relative_attention_bias,
            max_relative_position=max_relative_position,
        )
        
        self.dropout1 = nn.Dropout(self.dropout)
        self.activation_dropout1 = nn.Dropout(self.activation_dropout)
        
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        
        self.fc1 = nn.Linear(self.embed_dim, self.ffn_embed_dim)
        self.fc2 = nn.Linear(self.ffn_embed_dim, self.embed_dim)
        
        self.dropout2 = nn.Dropout(self.dropout)
        self.activation_dropout2 = nn.Dropout(self.activation_dropout)
        
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        
        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
        
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
            position_bias=position_bias,
        )
        
        x = self.dropout1(x)
        x = residual + x
        
        if not self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
        
        residual = x
        
        if self.layer_norm_first:
            x = self.final_layer_norm(x)
        
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = residual + x
        
        if not self.layer_norm_first:
            x = self.final_layer_norm(x)
        
        return x


class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer."""
    
    def __init__(
        self,
        embed_dim: int,
        ffn_embed_dim: int,
        attention_heads: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
        normalize_before: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_fn = get_activation_fn(activation_fn)
        self.layer_norm_first = layer_norm_first
        self.normalize_before = normalize_before
        
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.attention_heads,
            dropout=self.attention_dropout,
            self_attention=True,
        )
        
        self.dropout1 = nn.Dropout(self.dropout)
        self.activation_dropout1 = nn.Dropout(self.activation_dropout)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        
        self.encoder_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.attention_heads,
            dropout=self.attention_dropout,
            self_attention=False,
        )
        
        self.dropout2 = nn.Dropout(self.dropout)
        self.activation_dropout2 = nn.Dropout(self.activation_dropout)
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        
        self.fc1 = nn.Linear(self.embed_dim, self.ffn_embed_dim)
        self.fc2 = nn.Linear(self.ffn_embed_dim, self.embed_dim)
        
        self.dropout3 = nn.Dropout(self.dropout)
        self.activation_dropout3 = nn.Dropout(self.activation_dropout)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = x
        
        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
        
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[torch.Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 4:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
                saved_state["prev_value_padding_mask"] = prev_self_attn_state[3]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        
        x = self.dropout1(x)
        x = residual + x
        
        if not self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
        
        residual = x
        
        if self.layer_norm_first:
            x = self.encoder_attn_layer_norm(x)
        
        if prev_attn_state is not None:
            prev_key, prev_value = prev_attn_state[:2]
            saved_state: Dict[str, Optional[torch.Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_attn_state) >= 4:
                saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                saved_state["prev_value_padding_mask"] = prev_attn_state[3]
            assert incremental_state is not None
            self.encoder_attn._set_input_buffer(incremental_state, saved_state)
        
        x, attn = self.encoder_attn(
            query=x,
            key=encoder_out,
            value=encoder_out,
            key_padding_mask=encoder_padding_mask,
            incremental_state=incremental_state,
            static_kv=True,
            need_weights=need_attn or (not self.training and self.need_attn),
            need_head_weights=need_head_weights,
        )
        
        x = self.dropout2(x)
        x = residual + x
        
        if not self.layer_norm_first:
            x = self.encoder_attn_layer_norm(x)
        
        residual = x
        
        if self.layer_norm_first:
            x = self.final_layer_norm(x)
        
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout1(x)
        x = self.fc2(x)
        x = self.dropout3(x)
        x = residual + x
        
        if not self.layer_norm_first:
            x = self.final_layer_norm(x)
        
        return x, attn


class MultiheadAttention(nn.Module):
    """Multi-headed attention."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        self_attention: bool = False,
        encoder_decoder_attention: bool = False,
        has_relative_attention_bias: bool = False,
        max_relative_position: int = 32,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        self.scaling = self.head_dim ** -0.5
        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None
        
        self.add_zero_attn = add_zero_attn
        self.reset_parameters()
        
        self.onnx_trace = False
        self.tpu = False
    
    def reset_parameters(self):
        if self.bias_k is not None:
            nn.init.xavier_uniform_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_uniform_(self.bias_v)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
        position_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Time x Batch x Channel
        
        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
            position_bias (Tensor, optional): position bias for relative
                positional encoding.
        """
        if need_head_weights:
            need_weights = True
        
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None
        
        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(value)
        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        
        q *= self.scaling
        
        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat([key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)
        
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        
        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            self._set_input_buffer(incremental_state, saved_state)
        
        src_len = k.size(1) if k is not None else 0
        
        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None
        
        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).expand(-1, self.num_heads, -1, -1).contiguous().view(bsz * self.num_heads, 1, src_len)
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask.expand(bsz, self.num_heads, tgt_len, src_len).contiguous().view(bsz * self.num_heads, tgt_len, src_len) + key_padding_mask
        
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, bsz * self.num_heads, tgt_len, src_len]:
                raise RuntimeError(f"The shape of the attn_mask is {list(attn_mask.size())}, but should be {[1, bsz * self.num_heads, tgt_len, src_len]}")
            attn_mask = attn_mask.expand(bsz * self.num_heads, -1, -1, -1)
        
        if position_bias is not None:
            if attn_mask is not None:
                attn_mask = attn_mask + position_bias
            else:
                attn_mask = position_bias
        
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)
        
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
        
        if attn_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(attn_mask, float("-inf"))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        
        if before_softmax:
            return attn_weights, v
        
        attn_weights_float = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        
        if need_weights:
            attn_weights = attn_weights_float.view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 2)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=1)
        else:
            attn_weights = None
        
        return attn, attn_weights
    
    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights
    
    def _set_input_buffer(self, incremental_state: Dict[str, Dict[str, Optional[torch.Tensor]]], buffer: Dict[str, Optional[torch.Tensor]]):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)
    
    def get_incremental_state(self, incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]], key: str) -> Optional[Dict[str, Optional[torch.Tensor]]]:
        """Helper for getting incremental state for an nn.Module."""
        if incremental_state is None or key not in incremental_state:
            return None
        return incremental_state[key]
    
    def set_incremental_state(self, incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]], key: str, value: Dict[str, Optional[torch.Tensor]]) -> Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]]:
        """Helper for setting incremental state for an nn.Module."""
        if incremental_state is not None:
            incremental_state[key] = value
        return incremental_state
    
    def _get_input_buffer(self, incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]]) -> Optional[Dict[str, Optional[torch.Tensor]]]:
        return self.get_incremental_state(incremental_state, "attn_state")


class TransformerEncoder(nn.Module):
    """Transformer encoder."""
    
    def __init__(
        self,
        embed_dim: int,
        ffn_embed_dim: int,
        layers: int,
        attention_heads: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
        has_relative_attention_bias: bool = False,
        max_relative_position: int = 32,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.layers = layers
        self.attention_heads = attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_fn = activation_fn
        self.layer_norm_first = layer_norm_first
        
        self.layers = nn.ModuleList([])
        for _ in range(layers):
            self.layers.append(
                TransformerEncoderLayer(
                    embed_dim=embed_dim,
                    ffn_embed_dim=ffn_embed_dim,
                    attention_heads=attention_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    layer_norm_first=layer_norm_first,
                    has_relative_attention_bias=has_relative_attention_bias,
                    max_relative_position=max_relative_position,
                )
            )
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(
                x,
                encoder_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
                position_bias=position_bias,
            )
        return x


class TransformerDecoder(nn.Module):
    """Transformer decoder."""
    
    def __init__(
        self,
        embed_dim: int,
        ffn_embed_dim: int,
        layers: int,
        attention_heads: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
        normalize_before: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.layers = layers
        self.attention_heads = attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_fn = activation_fn
        self.layer_norm_first = layer_norm_first
        self.normalize_before = normalize_before
        
        self.layers = nn.ModuleList([])
        for _ in range(layers):
            self.layers.append(
                TransformerDecoderLayer(
                    embed_dim=embed_dim,
                    ffn_embed_dim=ffn_embed_dim,
                    attention_heads=attention_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    layer_norm_first=layer_norm_first,
                    normalize_before=normalize_before,
                )
            )
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn = None
        for layer in self.layers:
            x, attn = layer(
                x,
                encoder_out=encoder_out,
                encoder_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                prev_self_attn_state=prev_self_attn_state,
                prev_attn_state=prev_attn_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=need_attn,
                need_head_weights=need_head_weights,
            )
        return x, attn