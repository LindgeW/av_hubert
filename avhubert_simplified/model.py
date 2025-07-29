"""
Simplified AVHuBERT model implementation.
HuggingFace-style interface without fairseq dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import logging
from copy import deepcopy

from .config import AVHubertConfig
from .resnet import ResEncoder
from .transformer import TransformerEncoder, ConvFeatureExtractionModel, LayerNorm
from .utils import (
    compute_mask_indices, 
    apply_mask, 
    GradMultiply, 
    parse_conv_feature_layers,
    get_conv_output_lengths
)

logger = logging.getLogger(__name__)


class SubModel(nn.Module):
    """Sub-model for feature extraction (audio or video)"""
    
    def __init__(self, resnet=None, input_dim=80, cfg=None):
        super().__init__()
        
        self.resnet = resnet
        self.encoder_embed_dim = cfg.encoder_embed_dim
        
        # Feature extraction layers
        if resnet is not None:
            # Video features
            self.post_extract_proj = nn.Linear(input_dim, cfg.encoder_embed_dim)
        else:
            # Audio features - use conv layers for feature extraction  
            conv_layers = parse_conv_feature_layers(cfg.conv_feature_layers)
            self.feature_extractor = ConvFeatureExtractionModel(
                conv_layers=conv_layers,
                dropout=0.0,
                mode=cfg.extractor_mode,
                conv_bias=cfg.conv_bias,
            )
            
            # Calculate final conv output dimension
            n_in = 1
            for dim, k, stride in conv_layers:
                n_in = dim
            self.post_extract_proj = nn.Linear(n_in, cfg.encoder_embed_dim)
        
        # Sub-encoder (lightweight transformer)
        if cfg.sub_encoder_layers > 0:
            sub_config = deepcopy(cfg)
            sub_config.encoder_layers = cfg.sub_encoder_layers
            self.encoder = TransformerEncoder(sub_config)
        else:
            self.encoder = None
    
    def forward(self, x, padding_mask=None):
        """
        Args:
            x: Input features
            padding_mask: Padding mask
        """
        if self.resnet is not None:
            # Video features through ResNet
            x = self.resnet(x)  # (B, T, D)
        else:
            # Audio features through conv layers
            x = self.feature_extractor(x)  # (B, T, D)
        
        # Project to embedding dimension
        x = self.post_extract_proj(x)
        
        # Apply sub-encoder if present
        if self.encoder is not None:
            x = self.encoder(x, padding_mask=padding_mask)
        
        return x


class AVHubertModel(nn.Module):
    """
    Simplified AVHuBERT model with HuggingFace-style interface.
    
    This model supports:
    - Audio-only processing (input_modality="audio")
    - Video-only processing (input_modality="video") 
    - Audio-visual processing (input_modality="audio_video")
    """
    
    def __init__(self, config: AVHubertConfig):
        super().__init__()
        
        self.config = config
        logger.info(f"AVHuBERT Config: {config}")
        
        # Feature extractors
        self.setup_feature_extractors()
        
        # Modality fusion
        self.setup_modality_fusion()
        
        # Main transformer encoder
        self.encoder = TransformerEncoder(config)
        
        # Masking parameters
        self.setup_masking()
        
        # Output projections
        self.setup_output_projections()
        
        # Dropouts
        self.dropout_input = nn.Dropout(config.dropout_input)
        self.dropout_features = nn.Dropout(config.dropout_features)
        
    def setup_feature_extractors(self):
        """Setup audio and video feature extractors"""
        config = self.config
        
        # Create sub-config for feature extractors
        sub_cfg = deepcopy(config)
        sub_cfg.encoder_layers = config.sub_encoder_layers
        
        # Video feature extractor (ResNet)
        resnet = ResEncoder(
            relu_type=config.resnet_relu_type, 
            weights=config.resnet_weights
        )
        self.feature_extractor_video = SubModel(
            resnet=resnet, 
            input_dim=resnet.backend_out, 
            cfg=sub_cfg
        )
        
        # Audio feature extractor (Conv layers)
        self.feature_extractor_audio = SubModel(
            resnet=None, 
            input_dim=config.audio_feat_dim, 
            cfg=sub_cfg
        )
        
    def setup_modality_fusion(self):
        """Setup modality fusion layer"""
        config = self.config
        
        self.modality_fuse = config.modality_fuse
        self.modality_dropout = config.modality_dropout
        self.audio_dropout = config.audio_dropout
        
        if self.modality_fuse == 'concat':
            self.embed_dim = config.encoder_embed_dim * 2
        elif self.modality_fuse == 'add':
            self.embed_dim = config.encoder_embed_dim
        else:
            raise ValueError(f"Unknown modality fusion: {self.modality_fuse}")
        
        # Projection to encoder dimension if needed
        self.post_extract_proj = (
            nn.Linear(self.embed_dim, config.encoder_embed_dim)
            if self.embed_dim != config.encoder_embed_dim
            else None
        )
        
        # Layer norm for fused features
        self.layer_norm = LayerNorm(self.embed_dim)
        
    def setup_masking(self):
        """Setup masking parameters"""
        config = self.config
        
        self.mask_prob_audio = config.mask_prob_audio
        self.mask_prob_image = config.mask_prob_image
        self.mask_length_audio = config.mask_length_audio
        self.mask_length_image = config.mask_length_image
        self.mask_selection = config.mask_selection
        self.mask_other = config.mask_other
        self.no_mask_overlap = config.no_mask_overlap
        self.mask_min_space = config.mask_min_space
        self.masking_type = config.masking_type
        
        # Mask embedding
        if self.masking_type == 'input':
            mask_embed_dim = config.audio_feat_dim
        else:
            mask_embed_dim = config.encoder_embed_dim
            
        self.mask_emb = nn.Parameter(
            torch.FloatTensor(mask_embed_dim).uniform_()
        )
        
    def setup_output_projections(self):
        """Setup output projection layers"""
        config = self.config
        
        final_dim = config.final_dim if config.final_dim > 0 else config.encoder_embed_dim
        
        # Target GLU if specified
        self.target_glu = None
        if config.target_glu:
            self.target_glu = nn.Sequential(
                nn.Linear(final_dim, final_dim * 2), 
                nn.GLU()
            )
        
        # Final projection
        self.final_proj = nn.Linear(config.encoder_embed_dim, final_dim)
        
        # Gradient scaling
        self.feature_grad_mult = config.feature_grad_mult
        if self.feature_grad_mult != 1.0:
            self.grad_multiply = GradMultiply(self.feature_grad_mult)
        else:
            self.grad_multiply = None
    
    def extract_features(
        self,
        audio: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None, 
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Extract and fuse audio-visual features
        
        Args:
            audio: Audio features (B, T, audio_feat_dim)
            video: Video features (B, T, C, H, W) 
            padding_mask: Padding mask (B, T)
            mask: Whether to apply masking
            
        Returns:
            Tuple of (features, mask_indices)
        """
        features_list = []
        mask_indices = None
        
        # Process audio features
        if audio is not None:
            audio_features = self.feature_extractor_audio(audio, padding_mask)
            
            # Apply audio dropout
            if self.training and self.audio_dropout > 0:
                keep_prob = 1 - self.audio_dropout
                keep_mask = torch.rand_like(audio_features[:, :, 0]) < keep_prob
                audio_features = audio_features * keep_mask.unsqueeze(-1)
            
            features_list.append(audio_features)
            
            # Generate audio mask
            if mask and self.mask_prob_audio > 0:
                audio_mask = compute_mask_indices(
                    shape=audio_features.shape[:2],
                    padding_mask=padding_mask,
                    mask_prob=self.mask_prob_audio,
                    mask_length=self.mask_length_audio,
                    mask_type=self.mask_selection,
                    mask_other=self.mask_other,
                    no_overlap=self.no_mask_overlap,
                    min_space=self.mask_min_space,
                )
                mask_indices = audio_mask
        
        # Process video features  
        if video is not None:
            video_features = self.feature_extractor_video(video, padding_mask)
            features_list.append(video_features)
            
            # Generate video mask
            if mask and self.mask_prob_image > 0:
                video_mask = compute_mask_indices(
                    shape=video_features.shape[:2],
                    padding_mask=padding_mask,
                    mask_prob=self.mask_prob_image,
                    mask_length=self.mask_length_image,
                    mask_type=self.mask_selection,
                    mask_other=self.mask_other,
                    no_overlap=self.no_mask_overlap,
                    min_space=self.mask_min_space,
                )
                
                # Combine masks
                if mask_indices is not None:
                    mask_indices = mask_indices | video_mask
                else:
                    mask_indices = video_mask
        
        if not features_list:
            raise ValueError("At least one modality (audio or video) must be provided")
        
        # Fuse modalities
        if len(features_list) == 1:
            # Single modality
            features = features_list[0]
        else:
            # Multi-modal fusion
            if self.modality_fuse == 'concat':
                features = torch.cat(features_list, dim=-1)
            elif self.modality_fuse == 'add':
                features = sum(features_list)
            else:
                raise ValueError(f"Unknown fusion method: {self.modality_fuse}")
        
        # Apply modality dropout
        if self.training and self.modality_dropout > 0:
            keep_prob = 1 - self.modality_dropout  
            keep_mask = torch.rand(features.shape[0], device=features.device) < keep_prob
            features = features * keep_mask.unsqueeze(1).unsqueeze(2)
        
        return features, mask_indices
    
    def apply_masking(self, features: torch.Tensor, mask_indices: torch.Tensor) -> torch.Tensor:
        """Apply masking to features"""
        if mask_indices is not None and mask_indices.any():
            if self.masking_type == 'input':
                # Mask at input level (before projection)
                features = apply_mask(features, mask_indices, self.mask_emb)
            else:
                # Mask at feature level (after projection)
                features = apply_mask(features, mask_indices, self.mask_emb)
        return features
    
    def forward(
        self,
        audio: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of AVHuBERT model
        
        Args:
            audio: Audio features (B, T, audio_feat_dim)
            video: Video features (B, T, C, H, W)
            padding_mask: Padding mask (B, T) 
            mask: Whether to apply masking during training
            output_hidden_states: Whether to return hidden states
            output_attentions: Whether to return attention weights
            
        Returns:
            Dictionary containing:
                - last_hidden_state: Final hidden states (B, T, D)
                - projected_states: Projected states for prediction (B, T, final_dim)
                - mask_indices: Applied mask indices (B, T) if masking was used
                - hidden_states: All hidden states if output_hidden_states=True
                - attentions: Attention weights if output_attentions=True
        """
        # Extract and fuse features
        features, mask_indices = self.extract_features(
            audio=audio,
            video=video, 
            padding_mask=padding_mask,
            mask=mask and self.training,
        )
        
        # Apply input dropout
        features = self.dropout_input(features)
        
        # Apply masking 
        if mask_indices is not None:
            features = self.apply_masking(features, mask_indices)
        
        # Apply feature dropout
        features = self.dropout_features(features)
        
        # Layer normalization
        features = self.layer_norm(features)
        
        # Project to encoder dimension if needed
        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)
        
        # Apply gradient multiplication
        if self.grad_multiply is not None:
            features = self.grad_multiply(features)
        
        # Main transformer encoder
        encoded = self.encoder(features, padding_mask=padding_mask)
        
        # Final projection
        projected = self.final_proj(encoded)
        
        # Apply target GLU if specified
        if self.target_glu is not None:
            projected = self.target_glu(projected)
        
        # Prepare output
        output = {
            'last_hidden_state': encoded,
            'projected_states': projected,
        }
        
        if mask_indices is not None:
            output['mask_indices'] = mask_indices
            
        # TODO: Add hidden_states and attentions if requested
        # This would require modifying the transformer to return these
        
        return output
    
    def get_targets(self, projected_states: torch.Tensor, mask_indices: torch.Tensor) -> torch.Tensor:
        """Get target representations for masked positions"""
        if mask_indices is None or not mask_indices.any():
            return projected_states
        
        # Only return features at masked positions
        return projected_states[mask_indices]
    
    def compute_mask_loss(
        self, 
        projected_states: torch.Tensor,
        targets: torch.Tensor,
        mask_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute masked language modeling loss"""
        if mask_indices is None or not mask_indices.any():
            return torch.tensor(0.0, device=projected_states.device)
        
        # Get predictions and targets for masked positions
        predictions = projected_states[mask_indices]
        targets = targets[mask_indices]
        
        # Compute cosine similarity loss (as in original HuBERT)
        loss = F.cosine_embedding_loss(
            predictions, targets, torch.ones(predictions.size(0), device=predictions.device)
        )
        
        return loss
    
    @classmethod
    def from_pretrained(cls, model_path: str, config: Optional[AVHubertConfig] = None) -> 'AVHubertModel':
        """Load pretrained model from checkpoint"""
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if config is None:
            # Try to extract config from checkpoint
            if 'config' in checkpoint:
                config = checkpoint['config']
            else:
                config = AVHubertConfig()
                logger.warning("No config found in checkpoint, using default config")
        
        model = cls(config)
        
        # Load state dict
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Handle fairseq to simplified model key mapping
        model.load_state_dict(state_dict, strict=False)
        
        return model
    
    def save_pretrained(self, save_path: str):
        """Save model and config"""
        torch.save({
            'model': self.state_dict(),
            'config': self.config,
        }, save_path)
        
        logger.info(f"Model saved to {save_path}")


# Factory functions for easy model creation
def create_avhubert_base(config: Optional[AVHubertConfig] = None) -> AVHubertModel:
    """Create base AVHuBERT model"""
    if config is None:
        config = AVHubertConfig()
    return AVHubertModel(config)


def create_avhubert_large(config: Optional[AVHubertConfig] = None) -> AVHubertModel:
    """Create large AVHuBERT model"""
    if config is None:
        config = AVHubertConfig(
            encoder_layers=24,
            encoder_embed_dim=1024,
            encoder_ffn_embed_dim=4096,
            encoder_attention_heads=16,
        )
    return AVHubertModel(config)