from __future__ import annotations

"""
Simplified, self-contained AVHubert implementation built on top of the HuggingFace
`transformers` library.  This file purposely *does not* depend on fairseq and
exposes an interface similar to other HuggingFace models so that end-users can
interact with it using a familiar, lightweight API:

```python
from avhubert import AVHubertModel, AVHubertConfig

config = AVHubertConfig()
model = AVHubertModel(config)          # randomly initialised
# OR load a checkpoint (either HF-style or a plain torch .pt file containing
# the model state-dict)
model = AVHubertModel.from_pretrained("path/to/checkpoint.pt")

audio   = torch.randn(2, 16000)                 # (batch, audio_len)
video   = torch.randn(2, 1, 25, 112, 112)       # (batch, C, T, H, W)
outputs = model(audio, video)
```

The goal is *not* to faithfully reproduce the original Facebook research code
in its entirety but to provide a pragmatic, easy-to-use wrapper that is good
enough for fine-tuning and inference workflows.
"""

import math

# --------------------------------------------------------------------------------------
# Positional encoding helpers
# --------------------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Standard transformer sinusoidal positional encoding."""

    def __init__(self, dim: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, dim)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encodings to *x* (B, T, D)."""
        return x + self.pe[:, : x.size(1)]

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any

import torch
import torch.nn as nn
from torch import Tensor

try:
    from transformers import (  # type: ignore
        HubertModel,
        PreTrainedModel,
        PretrainedConfig,
    )
except ImportError as e:  # pragma: no cover – help users with an actionable err
    raise ImportError(
        "`transformers` is required for `avhubert.modeling_avhubert`.\n"
        "Install with `pip install transformers`"
    ) from e

from .resnet import ResEncoder

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


class AVHubertConfig(PretrainedConfig):
    """Configuration object holding hyper-parameters for :class:`AVHubertModel`."""

    model_type = "avhubert"

    def __init__(
        self,
        vision_relu_type: str = "prelu",
        vision_weights: Optional[str] = None,
        fusion_hidden_size: int = 768,
        modality_fuse: str = "add",  # "add"|"concat" – fusion *before* transformer
        dropout: float = 0.1,
        audio_feat_dim: int = 80,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        **kwargs,
    ) -> None:
        self.vision_relu_type = vision_relu_type
        self.vision_weights = vision_weights
        self.fusion_hidden_size = fusion_hidden_size
        self.modality_fuse = modality_fuse
        self.dropout = dropout

        self.audio_feat_dim = audio_feat_dim
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size

        super().__init__(**kwargs)


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------


class AVHubertModel(PreTrainedModel):
    """A light-weight audio-visual HUBERT implementation.

    The model is composed of two sub-networks:

    1. *Audio branch*  – any HuggingFace `HubertModel`.
    2. *Vision branch* – a 3D-2D ResNet frontend (`ResEncoder`).

    The two modal features are fused (concatenated or added) and optionally
    projected down to *fusion_hidden_size*.
    """

    config_class = AVHubertConfig
    base_model_prefix = "avhubert"

    def __init__(self, config: AVHubertConfig):
        super().__init__(config)

        # AUDIO -------------------------------------------------------------------------------
        self.audio_proj = nn.Linear(config.audio_feat_dim, config.hidden_size)

        # VISION ------------------------------------------------------------------------------
        self.vision_model = ResEncoder(config.vision_relu_type, config.vision_weights)
        self.vision_proj = nn.Linear(self.vision_model.backend_out, config.hidden_size)

        # Positional encoding (shared)
        self.pos_enc = SinusoidalPositionalEncoding(config.hidden_size)

        # Transformer encoder for fusion ------------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_hidden_layers)

        # Fusion method specific layer when concatenating
        if config.modality_fuse == "concat":
            self.concat_proj = nn.Linear(config.hidden_size * 2, config.hidden_size)

        # Classifier / projection head --------------------------------------------------------
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, config.fusion_hidden_size)

        # weight init for new layers ------------------------------------------------------------
        self.post_init()

    # ---------------------------------------------------------------------------------- utils
    def _align_modalities(self, audio_feats: Tensor, vision_feats: Tensor) -> Tensor:
        """Align visual features to the audio sequence length (nearest neighbour)."""

        b, t_audio, _ = audio_feats.shape  # (B, T_a, D_a)
        b_v, d_v, t_vid = vision_feats.shape  # (B, D_v, T_v) coming from ResEncoder
        assert b == b_v, "Batch sizes of audio and video inputs must match."

        # Reshape video to (B, T_v, D_v)
        vision_feats = vision_feats.permute(0, 2, 1)

        if t_vid != t_audio:
            # interpolate over the *temporal* dimension
            vision_feats = torch.nn.functional.interpolate(
                vision_feats.transpose(1, 2),  # (B, D_v, T_v)
                size=t_audio,
                mode="nearest",
            ).transpose(1, 2)

        return vision_feats  # (B, T_a, D_v)

    # -------------------------------------------------------------------------------- forward
    def forward(
        self,
        audio_features: Tensor,  # (B, T, feature_dim)
        video_frames: Tensor,  # (B, 1, T, H, W)
        attention_mask: Optional[Tensor] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **unused,
    ) -> Union[Dict[str, Any], Tuple[Tensor, ...]]:
        """Runs a forward pass.

        Parameters
        ----------
        input_values: torch.FloatTensor
            Raw audio waveforms
        video_frames: torch.FloatTensor
            5-D tensor with shape *(batch, 1, time, height, width)*
        """

        # Assumes *audio_features* are already extracted (e.g., log-Mel)
        audio_feats = self.audio_proj(audio_features)  # (B, T, H)


        vision_feats: Tensor = self.vision_model(video_frames)  # (B, D_v, T_v)
        vision_feats = self._align_modalities(audio_feats, vision_feats)
        vision_feats = self.vision_proj(vision_feats)

        # Fuse modalities (add or concat before transformer)
        if self.config.modality_fuse == "concat":
            fused = torch.cat([audio_feats, vision_feats], dim=-1)
            fused = self.concat_proj(fused)
        else:  # default 'add'
            fused = audio_feats + vision_feats

        # Positional encoding + Transformer
        fused = self.pos_enc(fused)
        fused = self.transformer(fused)  # (B, T, H)

        fused = self.dropout(fused)
        logits = self.classifier(fused)

        if not return_dict:
            return logits,

        return {
            "logits": logits,
            "hidden_states": fused,
        }

    # --------------------------------------------------------------------------- loading utils
    @classmethod
    def _load_pretrained_state_dict(
        cls, checkpoint: Union[str, Path, Dict[str, Tensor]], device: torch.device
    ) -> Dict[str, Tensor]:
        """Light wrapper around `torch.load` that also tolerates HF checkpoints."""

        if isinstance(checkpoint, (str, Path)):
            checkpoint = str(checkpoint)
            if checkpoint.endswith(".bin"):
                # HuggingFace state-dict – load using from_pretrained helper
                return torch.load(checkpoint, map_location=device)
            else:
                obj = torch.load(checkpoint, map_location=device)
                # Fairseq style checkpoints wrap state-dict in extra keys – be resilient
                if "state_dict" in obj:
                    obj = obj["state_dict"]
                return obj
        elif isinstance(checkpoint, dict):
            return checkpoint
        else:  # pragma: no cover
            raise TypeError("`checkpoint` must be a path or a `dict`.")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: Union[str, Path],
        config: Optional[AVHubertConfig] = None,
        map_location: Union[str, torch.device] = "cpu",
        **kwargs,
    ) -> "AVHubertModel":
        """Load a pre-trained checkpoint **without** relying on fairseq."""

        device = torch.device(map_location)

        # 1) Load config if missing -----------------------------------------------------------
        if config is None:
            config = AVHubertConfig()

        model = cls(config)

        # 2) Resolve checkpoint --------------------------------------------------------------
        state_dict = cls._load_pretrained_state_dict(pretrained_path, device)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print("[AVHubert] ‑ WARNING: missing keys:", missing)
        if unexpected:
            print("[AVHubert] ‑ WARNING: unexpected keys:", unexpected)

        return model