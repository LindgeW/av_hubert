import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from ..configs.model_config import AVHubertConfig
from ..utils.masking import compute_mask_indices
from ..utils.activations import get_activation_fn
from .resnet import ResEncoder
from .transformer import TransformerEncoder, TransformerDecoder


logger = logging.getLogger(__name__)


class ConvFeatureExtractionModel(nn.Module):
    """Convolutional feature extraction model."""
    
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        in_d: int = 1,
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
        conv_type: str = "default",
    ):
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
                        nn.LayerNorm(dim, elementwise_affine=True),
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
        
        self.in_d = in_d
        self.conv_layers = nn.ModuleList()
        n_in = in_d
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl
            
            self.conv_layers.append(
                block(
                    n_in,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )
            n_in = dim
    
    def forward(self, x):
        # BxT -> BxCxT
        x = x.unsqueeze(1)
        for conv in self.conv_layers:
            x = conv(x)
        return x


class TransposeLast(nn.Module):
    def __init__(self, deconstruct_idx=None):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx
    
    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(-2, -1)


class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res
    
    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


class AVHubertModel(nn.Module):
    """AV-HuBERT model for audio-visual speech representation learning."""
    
    def __init__(
        self,
        config: AVHubertConfig,
        dictionaries: List = None,
    ):
        super().__init__()
        self.config = config
        self.dictionaries = dictionaries or []
        
        # Feature extractors
        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=eval(config.conv_feature_layers),
            in_d=1,
            dropout=config.dropout_input,
            mode=config.extractor_mode.value,
            conv_bias=config.conv_bias,
        )
        
        # Video encoder (ResNet)
        self.video_encoder = ResEncoder(
            relu_type=config.resnet_relu_type,
            weights=config.resnet_weights,
        )
        
        # Positional encoding
        self.pos_conv = nn.Conv1d(
            config.encoder_embed_dim,
            config.encoder_embed_dim,
            kernel_size=config.conv_pos,
            padding=config.conv_pos // 2,
            groups=config.conv_pos_groups,
        )
        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (config.conv_pos * config.encoder_embed_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)
        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(config.conv_pos), nn.GELU())
        
        # Transformer encoder
        self.encoder = TransformerEncoder(
            embed_dim=config.encoder_embed_dim,
            ffn_embed_dim=config.encoder_ffn_embed_dim,
            layers=config.encoder_layers,
            attention_heads=config.encoder_attention_heads,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            activation_dropout=config.activation_dropout,
            activation_fn=config.activation_fn.value,
            layer_norm_first=config.layer_norm_first,
        )
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(config.encoder_embed_dim)
        
        # Projection layers
        self.final_proj = nn.Linear(config.encoder_embed_dim, config.final_dim)
        
        # Target projection
        if config.target_glu:
            self.target_glu = nn.Linear(config.final_dim, config.final_dim * 2)
        
        # Modality fusion
        self.modality_fuse = config.modality_fuse
        self.modality_dropout = config.modality_dropout
        self.audio_dropout = config.audio_dropout
        
        # Sub-encoder for single modality
        if config.sub_encoder_layers > 0:
            self.sub_encoder = TransformerEncoder(
                embed_dim=config.encoder_embed_dim,
                ffn_embed_dim=config.encoder_ffn_embed_dim,
                layers=config.sub_encoder_layers,
                attention_heads=config.encoder_attention_heads,
                dropout=config.dropout,
                attention_dropout=config.attention_dropout,
                activation_dropout=config.activation_dropout,
                activation_fn=config.activation_fn.value,
                layer_norm_first=config.layer_norm_first,
            )
        else:
            self.sub_encoder = None
        
        # Decoder (for seq2seq tasks)
        if hasattr(config, 'decoder_layers') and config.decoder_layers > 0:
            self.decoder = TransformerDecoder(
                embed_dim=config.decoder_embed_dim,
                ffn_embed_dim=config.decoder_ffn_embed_dim,
                layers=config.decoder_layers,
                attention_heads=config.decoder_attention_heads,
                dropout=config.decoder_dropout,
                attention_dropout=config.decoder_attention_dropout,
                activation_dropout=config.decoder_activation_dropout,
                activation_fn=config.activation_fn.value,
                layer_norm_first=config.layer_norm_first,
                normalize_before=config.decoder_normalize_before,
            )
            self.decoder_embed_tokens = nn.Embedding(
                len(dictionaries[0]) if dictionaries else 1000,
                config.decoder_embed_dim,
                padding_idx=0,
            )
            self.decoder_embed_positions = SinusoidalPositionalEmbedding(
                config.max_target_positions,
                config.decoder_embed_dim,
                padding_idx=0,
            )
        else:
            self.decoder = None
    
    def forward(
        self,
        source: torch.Tensor,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None,
        modality: str = "audio",
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the AV-HuBERT model.
        
        Args:
            source: Input tensor of shape (batch_size, sequence_length, channels)
            target_list: List of target tensors for supervised learning
            padding_mask: Boolean mask indicating padded positions
            mask: Whether to apply masking during training
            features_only: Whether to return only features without computing loss
            output_layer: Specific layer to output features from
            modality: Input modality ("audio", "video", or "both")
        
        Returns:
            Dictionary containing model outputs and loss information
        """
        features, padding_mask = self.extract_features(
            source, padding_mask, mask, output_layer, modality
        )
        
        if features_only:
            return {"features": features, "padding_mask": padding_mask}
        
        if target_list is not None:
            x, target = self.forward_targets(features, padding_mask, target_list)
            logits = self.compute_logits(x, target)
            result = {
                "logits": logits,
                "target": target,
                "features": features,
                "padding_mask": padding_mask,
            }
        else:
            result = {
                "features": features,
                "padding_mask": padding_mask,
            }
        
        return result
    
    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
        output_layer: Optional[int] = None,
        modality: str = "audio",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features from the input."""
        if modality == "audio":
            features = self.extract_audio_features(source, padding_mask, mask)
        elif modality == "video":
            features = self.extract_video_features(source, padding_mask, mask)
        elif modality == "both":
            features = self.extract_multimodal_features(source, padding_mask, mask)
        else:
            raise ValueError(f"Unknown modality: {modality}")
        
        if output_layer is not None:
            features = features[:, :output_layer]
        
        return features, padding_mask
    
    def extract_audio_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
    ) -> torch.Tensor:
        """Extract audio features."""
        if self.feature_extractor is not None:
            features = self.feature_extractor(source)
            if padding_mask is not None:
                input_lengths = (1 - padding_mask.long()).sum(-1)
                # pad input is the same as the original input
                pad_amp = (input_lengths.max() - input_lengths).to(torch.long)
                pad_amp = pad_amp.unsqueeze(-1).unsqueeze(-1)
                features = torch.nn.functional.pad(features, (0, 0, 0, pad_amp))
        else:
            features = source
        
        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        
        if mask:
            x, mask_indices = self.apply_input_mask(features, padding_mask)
        else:
            x = features
            mask_indices = None
        
        # Apply positional encoding
        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv
        
        if self.sub_encoder is not None:
            x = self.sub_encoder(x, padding_mask)
        
        x = self.encoder(x, padding_mask)
        
        return x
    
    def extract_video_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
    ) -> torch.Tensor:
        """Extract video features."""
        # source shape: (batch_size, channels, time, height, width)
        features = self.video_encoder(source)
        
        if mask:
            x, mask_indices = self.apply_input_mask(features, padding_mask)
        else:
            x = features
            mask_indices = None
        
        if self.sub_encoder is not None:
            x = self.sub_encoder(x, padding_mask)
        
        x = self.encoder(x, padding_mask)
        
        return x
    
    def extract_multimodal_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
    ) -> torch.Tensor:
        """Extract multimodal features from both audio and video."""
        # Assuming source contains both audio and video
        # This is a simplified version - actual implementation would depend on data format
        audio_features = self.extract_audio_features(source, padding_mask, mask)
        video_features = self.extract_video_features(source, padding_mask, mask)
        
        # Fuse modalities
        if self.modality_fuse == "concat":
            features = torch.cat([audio_features, video_features], dim=-1)
        elif self.modality_fuse == "add":
            features = audio_features + video_features
        else:
            raise ValueError(f"Unknown fusion method: {self.modality_fuse}")
        
        return features
    
    def apply_input_mask(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply input masking for masked training."""
        B, T, C = x.shape
        
        if self.config.masking_type == "input":
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.config.mask_prob_audio,
                self.config.mask_length_audio,
                self.config.mask_selection.value,
                self.config.mask_other,
                min_masks=2,
                no_overlap=self.config.no_mask_overlap,
                min_space=self.config.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None
        
        return x, mask_indices
    
    def forward_targets(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
        target_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for target prediction."""
        # This is a simplified version - actual implementation would handle
        # different target types and alignments
        x = self.final_proj(features)
        
        if self.config.target_glu:
            x = self.target_glu(x)
            x = F.glu(x, dim=-1)
        
        # Align targets with features
        target = target_list[0] if target_list else None
        
        return x, target
    
    def compute_logits(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute logits for target prediction."""
        # This is a simplified version - actual implementation would compute
        # logits based on the specific task (e.g., clustering, classification)
        if target is not None:
            # For clustering-based tasks
            logits = F.linear(x, target)
        else:
            # For other tasks
            logits = x
        
        return logits
    
    def get_extra_losses(self, net_output):
        """Get extra losses if any."""
        return {}
    
    def remove_pretraining_modules(self):
        """Remove pretraining-specific modules."""
        self.target_glu = None
        self.final_proj = None


class SamePad(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.remove = kernel_size % 2 == 0
    
    def forward(self, x):
        if self.remove:
            x = x[:, :, :-1]
        return x