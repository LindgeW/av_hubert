import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class LearnedPositionalEmbedding(nn.Module):
    """Learned positional embedding for transformer models."""
    
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int = 0):
        super().__init__()
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.max_positions = num_embeddings
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        positions = self.make_positions(input)
        return self.embedding(positions)
    
    def make_positions(self, input: torch.Tensor) -> torch.Tensor:
        mask = input.ne(self.padding_idx)
        return torch.cumsum(mask.long(), dim=1) * mask.long() + self.padding_idx


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding for transformer models."""
    
    def __init__(self, embedding_dim: int, padding_idx: int = 0, init_size: int = 1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size, embedding_dim, padding_idx
        )
        self.register_buffer('_float_tensor', torch.FloatTensor(1))
    
    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        """Build sinusoidal embeddings.
        
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb
    
    def forward(self, input: torch.Tensor, incremental_state: Optional[dict] = None, timestep: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos, self.embedding_dim, self.padding_idx
            )
        self.weights = self.weights.to(self._float_tensor)
        
        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            pos = timestep.view(-1)[0] if timestep is not None else seq_len - 1
            return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)
        
        positions = self.make_positions(input)
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()
    
    def make_positions(self, input: torch.Tensor) -> torch.Tensor:
        mask = input.ne(self.padding_idx)
        return torch.cumsum(mask.long(), dim=1) * mask.long() + self.padding_idx