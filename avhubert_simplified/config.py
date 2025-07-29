"""
Configuration classes for AVHuBERT model.
Simplified version without fairseq dependencies.
"""

from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class AVHubertConfig:
    """Configuration class for AVHuBERT model."""
    
    # Model architecture
    encoder_layers: int = 12
    encoder_embed_dim: int = 768
    encoder_ffn_embed_dim: int = 3072
    encoder_attention_heads: int = 12
    
    # Sub-encoder (for audio/video feature extraction)
    sub_encoder_layers: int = 2
    
    # Audio features
    audio_feat_dim: int = 80  # MFCC features
    
    # Visual features (ResNet)
    resnet_relu_type: str = "prelu" 
    resnet_weights: Optional[str] = None
    
    # Modality fusion
    modality_fuse: str = "concat"  # "concat" or "add"
    modality_dropout: float = 0.0
    audio_dropout: float = 0.0
    
    # Masking parameters
    mask_prob_audio: float = 0.80
    mask_prob_image: float = 0.75
    mask_length_audio: int = 10
    mask_length_image: int = 10
    mask_selection: str = "static"
    mask_other: float = 0.0
    no_mask_overlap: bool = False
    mask_min_space: int = 1
    masking_type: str = "feature"  # "feature" or "input"
    
    # Channel masking
    mask_channel_prob: float = 0.0
    mask_channel_selection: str = "static"
    mask_channel_other: float = 0.0
    mask_channel_length: int = 10
    no_mask_channel_overlap: bool = False
    mask_channel_min_space: int = 1
    
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
    extractor_mode: str = "default"
    
    # Training parameters
    feature_grad_mult: float = 1.0
    logit_temp: float = 0.1
    
    # Output projection
    final_dim: int = 0  # If <= 0, uses encoder_embed_dim
    untie_final_proj: bool = False
    target_glu: bool = False
    
    # Misc
    layer_norm_first: bool = False
    activation_fn: str = "gelu"
    skip_masked: bool = False
    skip_nomask: bool = False
    sim_type: str = "cosine"
    selection_type: str = "cluster"
    
    # Task-specific (can be set during initialization)
    label_rate: int = 50
    input_modality: str = "video"  # "audio", "video", or "audio_video"
    
    def __post_init__(self):
        if self.final_dim <= 0:
            self.final_dim = self.encoder_embed_dim


@dataclass 
class AVHubertPretrainConfig:
    """Configuration for AVHuBERT pretraining task."""
    
    # Data parameters
    data_path: str = ""
    label_dir: str = ""
    labels: List[str] = None
    
    # Audio parameters
    sample_rate: int = 16000
    normalize: bool = False
    enable_padding: bool = False
    max_sample_size: Optional[int] = None
    min_sample_size: Optional[int] = None
    
    # Video parameters  
    image_mean: float = 0.421
    image_std: float = 0.165
    image_crop_size: int = 88
    image_aug: bool = False
    
    # Modality settings
    input_modality: str = "video"
    
    # Label rate for clustering
    label_rate: int = 50
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = ["km"]