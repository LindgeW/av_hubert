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
        audio_model_name: str = "facebook/hubert-base-ls960",
        vision_relu_type: str = "prelu",
        vision_weights: Optional[str] = None,
        fusion_hidden_size: int = 768,
        modality_fuse: str = "concat",  # "concat" | "add"
        dropout: float = 0.1,
        mask_audio: bool = False,
        mask_image: bool = False,
        mask_prob_audio: float = 0.65,
        mask_length_audio: int = 10,
        mask_prob_image: float = 0.65,
        mask_length_image: int = 10,
        **kwargs,
    ) -> None:
        self.audio_model_name = audio_model_name
        self.vision_relu_type = vision_relu_type
        self.vision_weights = vision_weights
        self.fusion_hidden_size = fusion_hidden_size
        self.modality_fuse = modality_fuse
        self.dropout = dropout

        self.mask_audio = mask_audio
        self.mask_image = mask_image
        self.mask_prob_audio = mask_prob_audio
        self.mask_length_audio = mask_length_audio
        self.mask_prob_image = mask_prob_image
        self.mask_length_image = mask_length_image

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

        # AUDIO ---------------------------------------------------------------------------------
        self.audio_model = HubertModel.from_pretrained(
            config.audio_model_name
        ) if isinstance(config.audio_model_name, str) else HubertModel(config.audio_model_name)  # type: ignore[arg-type]

        audio_hidden = self.audio_model.config.hidden_size

        # VISION --------------------------------------------------------------------------------
        self.vision_model = ResEncoder(config.vision_relu_type, config.vision_weights)
        vision_hidden = self.vision_model.backend_out  # 512 for ResEncoder

        # FUSION -------------------------------------------------------------------------------
        if config.modality_fuse not in {"concat", "add"}:
            raise ValueError("`modality_fuse` must be 'concat' or 'add'.")

        if config.modality_fuse == "concat":
            fusion_in = audio_hidden + vision_hidden
        else:  # add – dimensions *must* match
            if audio_hidden != vision_hidden:
                # Project vision to audio dimension so shapes match for addition.
                self.vision_proj = nn.Linear(vision_hidden, audio_hidden)
                vision_hidden = audio_hidden
            fusion_in = audio_hidden  # unchanged

        self.fusion_proj = nn.Linear(fusion_in, config.fusion_hidden_size)
        self.layer_norm = nn.LayerNorm(config.fusion_hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        # A very small classification head – replace/extend as needed.
        self.classifier = nn.Linear(config.fusion_hidden_size, config.fusion_hidden_size)

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
        input_values: Tensor,  # Audio waveform (B, L)
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

        audio_out = self.audio_model(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        audio_feats: Tensor = audio_out.last_hidden_state  # (B, T_a, D_a)

        vision_feats: Tensor = self.vision_model(video_frames)  # (B, D_v, T_v)
        vision_feats = self._align_modalities(audio_feats, vision_feats)

        if hasattr(self, "vision_proj"):
            vision_feats = self.vision_proj(vision_feats)

        if self.config.modality_fuse == "concat":
            fused = torch.cat([audio_feats, vision_feats], dim=-1)
        else:  # add
            fused = audio_feats + vision_feats

        fused = self.fusion_proj(fused)
        fused = self.layer_norm(fused)
        fused = self.dropout(fused)

        logits = self.classifier(fused)

        if not return_dict:
            return logits,

        return {
            "logits": logits,
            "last_hidden_state": fused,
            "audio_hidden_states": audio_out.hidden_states if output_hidden_states else None,
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