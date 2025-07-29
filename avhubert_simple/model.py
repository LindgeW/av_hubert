"""
Main AVHuBERT model implementation.
Simplified version without fairseq dependencies.
"""

import ast
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from .config import AVHuBERTConfig
from .modules import (
    ConvFeatureExtractionModel,
    TransformerEncoderLayer,
    LayerNorm,
    grad_multiply,
    get_activation_fn,
)
from .resnet import ResEncoder
from .utils import compute_mask_indices


class SubModel(nn.Module):
    """Sub-model for single modality processing."""
    
    def __init__(self, resnet=None, input_dim=None, config=None):
        super().__init__()
        self.resnet = resnet
        self.input_dim = input_dim
        self.config = config
        
        if config.sub_encoder_layers > 0:
            self.sub_encoder = nn.ModuleList([
                TransformerEncoderLayer(
                    embed_dim=config.encoder_embed_dim,
                    ffn_embed_dim=config.encoder_ffn_embed_dim,
                    attention_heads=config.encoder_attention_heads,
                    dropout=config.dropout,
                    attention_dropout=config.attention_dropout,
                    activation_dropout=config.activation_dropout,
                    activation_fn=config.activation_fn,
                    layer_norm_first=config.layer_norm_first,
                ) for _ in range(config.sub_encoder_layers)
            ])
        else:
            self.sub_encoder = None

    def forward(self, x):
        if self.resnet is not None:
            x = self.resnet(x)
        
        if self.sub_encoder is not None:
            x = x.transpose(0, 1)  # [T, B, D]
            for layer in self.sub_encoder:
                x, _ = layer(x)
            x = x.transpose(0, 1)  # [B, T, D]
        
        return x


class TransformerEncoder(nn.Module):
    """Transformer encoder for multimodal fusion."""
    
    def __init__(self, config: AVHuBERTConfig):
        super().__init__()
        self.config = config
        
        self.dropout = config.dropout
        self.embedding_dim = config.encoder_embed_dim
        self.required_seq_len_multiple = 1
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=config.encoder_embed_dim,
                ffn_embed_dim=config.encoder_ffn_embed_dim,
                attention_heads=config.encoder_attention_heads,
                dropout=config.dropout,
                attention_dropout=config.attention_dropout,
                activation_dropout=config.activation_dropout,
                activation_fn=config.activation_fn,
                layer_norm_first=config.layer_norm_first,
            ) for _ in range(config.encoder_layers)
        ])
        
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = config.encoder_layerdrop

    def forward(self, x, padding_mask=None, layer=None):
        x = x.transpose(0, 1)  # [T, B, D]
        
        if not self.training or self.layerdrop == 0:
            for layer in self.layers:
                x, _ = layer(x, self_attn_padding_mask=padding_mask)
        else:
            for layer in self.layers:
                if torch.rand(1).item() > self.layerdrop:
                    x, _ = layer(x, self_attn_padding_mask=padding_mask)
        
        x = self.layer_norm(x)
        x = x.transpose(0, 1)  # [B, T, D]
        
        return x


class AVHuBERTModel(nn.Module):
    """
    AVHuBERT model for audio-visual representation learning.
    
    This is a simplified implementation that removes fairseq dependencies
    and provides a cleaner, more user-friendly interface.
    """
    
    def __init__(self, config: AVHuBERTConfig):
        super().__init__()
        self.config = config
        
        # Parse convolutional feature layers
        conv_layers = eval(config.conv_feature_layers)
        
        # Audio feature extractor
        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=conv_layers,
            dropout=config.dropout_input,
            mode=config.extractor_mode,
        )
        
        # Visual feature extractor (ResNet)
        self.visual_encoder = ResEncoder(
            relu_type=config.resnet_relu_type,
            weights=config.resnet_weights,
        )
        
        # Sub-models for single modality processing
        self.audio_sub_model = SubModel(
            resnet=None,
            input_dim=config.audio_feat_dim,
            config=config,
        )
        
        self.visual_sub_model = SubModel(
            resnet=self.visual_encoder,
            input_dim=512,  # ResNet output dimension
            config=config,
        )
        
        # Main transformer encoder
        self.encoder = TransformerEncoder(config)
        
        # Final projection layers
        self.final_proj = nn.Linear(config.encoder_embed_dim, config.final_dim)
        
        # Mask embedding
        self.mask_emb = nn.Parameter(torch.FloatTensor(config.encoder_embed_dim).uniform_())
        
        # Positional embeddings
        self.pos_conv = nn.Conv1d(
            config.encoder_embed_dim,
            config.encoder_embed_dim,
            kernel_size=config.conv_pos,
            padding=config.conv_pos // 2,
            groups=config.conv_pos_groups,
        )
        
        # Layer normalization
        self.layer_norm = LayerNorm(config.encoder_embed_dim)
        
        # Dropout layers
        self.dropout_input = nn.Dropout(config.dropout_input)
        self.dropout_features = nn.Dropout(config.dropout_features)
        
        # Feature gradient multiplier
        self.feature_grad_mult = config.feature_grad_mult
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def extract_features(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
        output_layer: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features from audio and video inputs.
        
        Args:
            audio: Audio tensor of shape [batch_size, time_steps]
            video: Video tensor of shape [batch_size, time_steps, channels, height, width]
            padding_mask: Optional padding mask
            mask: Whether to apply masking
            output_layer: Specific layer to output features from
            
        Returns:
            Tuple of (features, padding_mask)
        """
        # Extract audio features
        if self.feature_grad_mult != 1.0:
            audio = grad_multiply(audio, self.feature_grad_mult)
        
        audio_features = self.feature_extractor(audio)  # [B, C, T]
        audio_features = audio_features.transpose(1, 2)  # [B, T, C]
        
        # Extract visual features
        visual_features = self.visual_sub_model(video)  # [B, T, 512]
        
        # Process audio features through sub-encoder if needed
        if self.config.sub_encoder_layers > 0:
            audio_features = self.audio_sub_model(audio_features)
        
        # Fuse modalities
        if self.config.modality_fuse == "concat":
            # Project visual features to match audio dimension
            if visual_features.size(-1) != audio_features.size(-1):
                visual_proj = nn.Linear(visual_features.size(-1), audio_features.size(-1)).to(visual_features.device)
                visual_features = visual_proj(visual_features)
            
            # Concatenate along feature dimension
            features = torch.cat([audio_features, visual_features], dim=-1)
            
            # Project to encoder dimension
            if features.size(-1) != self.config.encoder_embed_dim:
                proj = nn.Linear(features.size(-1), self.config.encoder_embed_dim).to(features.device)
                features = proj(features)
        else:  # "add"
            # Project visual features to match audio dimension
            if visual_features.size(-1) != audio_features.size(-1):
                visual_proj = nn.Linear(visual_features.size(-1), audio_features.size(-1)).to(visual_features.device)
                visual_features = visual_proj(visual_features)
            
            # Add modalities
            features = audio_features + visual_features
            
            # Project to encoder dimension
            if features.size(-1) != self.config.encoder_embed_dim:
                proj = nn.Linear(features.size(-1), self.config.encoder_embed_dim).to(features.device)
                features = proj(features)
        
        # Apply masking if requested
        if mask:
            features = self.apply_masking(features, padding_mask)
        
        # Apply positional embeddings
        features = self.apply_positional_embeddings(features)
        
        # Apply dropouts
        features = self.dropout_input(features)
        features = self.dropout_features(features)
        
        # Pass through transformer encoder
        features = self.encoder(features, padding_mask)
        
        # Apply final projection
        features = self.final_proj(features)
        
        return features, padding_mask

    def apply_masking(self, features: torch.Tensor, padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Apply masking to features."""
        if self.config.masking_type == "input":
            # Apply input masking
            mask_indices = compute_mask_indices(
                shape=features.size()[:2],
                padding_mask=padding_mask,
                mask_prob=self.config.mask_prob_audio,
                mask_length=self.config.mask_length_audio,
                mask_type=self.config.mask_selection,
                mask_other=self.config.mask_other,
                no_overlap=self.config.no_mask_overlap,
                min_space=self.config.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(features.device)
            features[mask_indices] = self.mask_emb.to(features.dtype)
        
        return features

    def apply_positional_embeddings(self, features: torch.Tensor) -> torch.Tensor:
        """Apply positional embeddings to features."""
        features = features.transpose(1, 2)  # [B, D, T]
        features = self.pos_conv(features)
        features = features.transpose(1, 2)  # [B, T, D]
        return features

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            audio: Audio tensor of shape [batch_size, time_steps]
            video: Video tensor of shape [batch_size, time_steps, channels, height, width]
            padding_mask: Optional padding mask
            mask: Whether to apply masking
            features_only: Whether to return only features
            output_layer: Specific layer to output features from
            
        Returns:
            Dictionary containing model outputs
        """
        features, padding_mask = self.extract_features(
            audio=audio,
            video=video,
            padding_mask=padding_mask,
            mask=mask,
            output_layer=output_layer,
        )
        
        if features_only:
            return {"features": features, "padding_mask": padding_mask}
        
        # For pretraining, we would compute losses here
        # For now, just return features
        return {
            "features": features,
            "padding_mask": padding_mask,
            "mask": mask,
        }

    def get_extra_losses(self, net_output):
        """Get extra losses if any."""
        return {}

    def remove_pretraining_modules(self):
        """Remove pretraining-specific modules."""
        pass