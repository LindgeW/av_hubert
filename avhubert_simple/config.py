"""
Configuration class for AVHuBERT model.
Simplified version without fairseq dependencies.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union


@dataclass
class AVHuBERTConfig:
    """Configuration for AVHuBERT model."""
    
    # Model architecture
    encoder_layers: int = 12
    encoder_embed_dim: int = 768
    encoder_ffn_embed_dim: int = 3072
    encoder_attention_heads: int = 12
    activation_fn: str = "gelu"
    
    # Dropouts
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.0
    encoder_layerdrop: float = 0.0
    dropout_input: float = 0.0
    dropout_features: float = 0.0
    
    # Feature extraction
    conv_feature_layers: str = "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2"
    conv_bias: bool = False
    conv_pos: int = 128
    conv_pos_groups: int = 16
    
    # Final projection
    final_dim: int = 0  # 0 means use encoder_embed_dim
    untie_final_proj: bool = False
    layer_norm_first: bool = False
    
    # Masking
    mask_length_audio: int = 10
    mask_prob_audio: float = 0.65
    mask_length_image: int = 10
    mask_prob_image: float = 0.65
    mask_selection: str = "static"
    mask_other: float = 0.0
    no_mask_overlap: bool = False
    mask_min_space: int = 1
    
    # Channel masking
    mask_channel_length: int = 10
    mask_channel_prob: float = 0.0
    mask_channel_selection: str = "static"
    mask_channel_other: float = 0.0
    no_mask_channel_overlap: bool = False
    mask_channel_min_space: int = 1
    
    # Loss computation
    skip_masked: bool = False
    skip_nomask: bool = False
    logit_temp: float = 0.1
    target_glu: bool = False
    feature_grad_mult: float = 1.0
    
    # ResNet configuration
    resnet_relu_type: str = "prelu"
    resnet_weights: Optional[str] = None
    
    # Modality fusion
    modality_fuse: str = "concat"  # "add" or "concat"
    modality_dropout: float = 0.0
    audio_dropout: float = 0.0
    selection_type: str = "same_other_seq"
    masking_type: str = "input"
    
    # Sub-encoder
    sub_encoder_layers: int = 0
    audio_feat_dim: int = -1
    
    # Decoder (for ASR)
    decoder_embed_dim: int = 768
    decoder_ffn_embed_dim: int = 3072
    decoder_layers: int = 6
    decoder_layerdrop: float = 0.0
    decoder_attention_heads: int = 4
    decoder_learned_pos: bool = False
    decoder_normalize_before: bool = False
    decoder_dropout: float = 0.1
    decoder_attention_dropout: float = 0.1
    decoder_activation_dropout: float = 0.0
    max_target_positions: int = 2048
    share_decoder_input_output_embed: bool = False
    no_token_positional_embeddings: bool = False
    no_scale_embedding: bool = True
    
    # Similarity
    sim_type: str = "cosine"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.final_dim <= 0:
            self.final_dim = self.encoder_embed_dim
            
        if self.modality_fuse not in ["add", "concat"]:
            raise ValueError("modality_fuse must be 'add' or 'concat'")
            
        if self.mask_selection not in ["static", "uniform", "normal", "poisson"]:
            raise ValueError("mask_selection must be one of: static, uniform, normal, poisson")
            
        if self.activation_fn not in ["relu", "gelu", "swish"]:
            raise ValueError("activation_fn must be one of: relu, gelu, swish")