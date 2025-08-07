"""
Configuration class for AV-HuBERT model.
"""

import math
from typing import List, Optional, Tuple, Union

from transformers import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class AVHubertConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of an [`AVHubertModel`]. It is used to instantiate an
    AV-HuBERT model according to the specified arguments, defining the model architecture.
    """

    model_type = "avhubert"

    def __init__(
        self,
        # Audio feature extraction
        conv_feature_layers: str = "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
        conv_bias: bool = False,
        extractor_mode: str = "default",
        
        # Encoder configuration
        encoder_layers: int = 12,
        encoder_embed_dim: int = 768,
        encoder_ffn_embed_dim: int = 3072,
        encoder_attention_heads: int = 12,
        encoder_layerdrop: float = 0.0,
        activation_fn: str = "gelu",
        layer_norm_first: bool = False,
        
        # Dropout settings
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        dropout_input: float = 0.0,
        dropout_features: float = 0.0,
        
        # Masking configuration
        mask_length_audio: int = 10,
        mask_prob_audio: float = 0.65,
        mask_length_image: int = 10,
        mask_prob_image: float = 0.65,
        mask_selection: str = "static",
        mask_other: float = 0.0,
        no_mask_overlap: bool = False,
        mask_min_space: int = 1,
        
        # Channel masking
        mask_channel_length: int = 10,
        mask_channel_prob: float = 0.0,
        mask_channel_selection: str = "static",
        mask_channel_other: float = 0.0,
        no_mask_channel_overlap: bool = False,
        mask_channel_min_space: int = 1,
        
        # Positional embeddings
        conv_pos: int = 128,
        conv_pos_groups: int = 16,
        
        # Loss computation
        skip_masked: bool = False,
        skip_nomask: bool = False,
        feature_grad_mult: float = 1.0,
        
        # Video processing
        resnet_relu_type: str = "prelu",
        resnet_weights: Optional[str] = None,
        sim_type: str = "cosine",
        
        # Modality fusion
        sub_encoder_layers: int = 0,
        audio_feat_dim: int = -1,
        modality_dropout: float = 0.0,
        audio_dropout: float = 0.0,
        modality_fuse: str = "concat",
        selection_type: str = "same_other_seq",
        masking_type: str = "input",
        
        # Decoder configuration (for seq2seq)
        decoder_embed_dim: int = 768,
        decoder_ffn_embed_dim: int = 3072,
        decoder_layers: int = 6,
        decoder_layerdrop: float = 0.0,
        decoder_attention_heads: int = 4,
        decoder_learned_pos: bool = False,
        decoder_normalize_before: bool = False,
        decoder_dropout: float = 0.1,
        decoder_attention_dropout: float = 0.1,
        decoder_activation_dropout: float = 0.0,
        max_target_positions: int = 2048,
        share_decoder_input_output_embed: bool = False,
        no_token_positional_embeddings: bool = False,
        no_scale_embedding: bool = True,
        
        # Training configuration
        label_rate: int = 100,
        sample_rate: int = 16000,
        normalize: bool = False,
        final_dim: int = 0,
        untie_final_proj: bool = False,
        logit_temp: float = 0.1,
        target_glu: bool = False,
        latent_temp: Tuple[float, float, float] = (2, 0.5, 0.999995),
        
        # Initialization
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        # Audio feature extraction
        self.conv_feature_layers = conv_feature_layers
        self.conv_bias = conv_bias
        self.extractor_mode = extractor_mode
        
        # Encoder configuration
        self.encoder_layers = encoder_layers
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_ffn_embed_dim = encoder_ffn_embed_dim
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_layerdrop = encoder_layerdrop
        self.activation_fn = activation_fn
        self.layer_norm_first = layer_norm_first
        
        # Dropout settings
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.dropout_input = dropout_input
        self.dropout_features = dropout_features
        
        # Masking configuration
        self.mask_length_audio = mask_length_audio
        self.mask_prob_audio = mask_prob_audio
        self.mask_length_image = mask_length_image
        self.mask_prob_image = mask_prob_image
        self.mask_selection = mask_selection
        self.mask_other = mask_other
        self.no_mask_overlap = no_mask_overlap
        self.mask_min_space = mask_min_space
        
        # Channel masking
        self.mask_channel_length = mask_channel_length
        self.mask_channel_prob = mask_channel_prob
        self.mask_channel_selection = mask_channel_selection
        self.mask_channel_other = mask_channel_other
        self.no_mask_channel_overlap = no_mask_channel_overlap
        self.mask_channel_min_space = mask_channel_min_space
        
        # Positional embeddings
        self.conv_pos = conv_pos
        self.conv_pos_groups = conv_pos_groups
        
        # Loss computation
        self.skip_masked = skip_masked
        self.skip_nomask = skip_nomask
        self.feature_grad_mult = feature_grad_mult
        
        # Video processing
        self.resnet_relu_type = resnet_relu_type
        self.resnet_weights = resnet_weights
        self.sim_type = sim_type
        
        # Modality fusion
        self.sub_encoder_layers = sub_encoder_layers
        self.audio_feat_dim = audio_feat_dim
        self.modality_dropout = modality_dropout
        self.audio_dropout = audio_dropout
        self.modality_fuse = modality_fuse
        self.selection_type = selection_type
        self.masking_type = masking_type
        
        # Decoder configuration
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_ffn_embed_dim = decoder_ffn_embed_dim
        self.decoder_layers = decoder_layers
        self.decoder_layerdrop = decoder_layerdrop
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_learned_pos = decoder_learned_pos
        self.decoder_normalize_before = decoder_normalize_before
        self.decoder_dropout = decoder_dropout
        self.decoder_attention_dropout = decoder_attention_dropout
        self.decoder_activation_dropout = decoder_activation_dropout
        self.max_target_positions = max_target_positions
        self.share_decoder_input_output_embed = share_decoder_input_output_embed
        self.no_token_positional_embeddings = no_token_positional_embeddings
        self.no_scale_embedding = no_scale_embedding
        
        # Training configuration
        self.label_rate = label_rate
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.final_dim = final_dim
        self.untie_final_proj = untie_final_proj
        self.logit_temp = logit_temp
        self.target_glu = target_glu
        self.latent_temp = latent_temp
        
        # Set final_dim to encoder_embed_dim if not specified
        if self.final_dim <= 0:
            self.final_dim = self.encoder_embed_dim