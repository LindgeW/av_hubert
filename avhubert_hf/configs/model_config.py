from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
from enum import Enum


class ExtractorMode(str, Enum):
    DEFAULT = "default"
    LAYER_NORM = "layer_norm"


class MaskingDistribution(str, Enum):
    STATIC = "static"
    UNIFORM = "uniform"
    NORMAL = "normal"
    POISSON = "poisson"


class ActivationFn(str, Enum):
    RELU = "relu"
    GELU = "gelu"
    SWISH = "swish"
    SILU = "silu"


@dataclass
class AVHubertConfig:
    """Configuration for AV-HuBERT model."""
    
    # Model architecture
    encoder_layers: int = field(default=12, metadata={"help": "num encoder layers in the transformer"})
    encoder_embed_dim: int = field(default=768, metadata={"help": "encoder embedding dimension"})
    encoder_ffn_embed_dim: int = field(default=3072, metadata={"help": "encoder embedding dimension for FFN"})
    encoder_attention_heads: int = field(default=12, metadata={"help": "num encoder attention heads"})
    activation_fn: ActivationFn = field(default=ActivationFn.GELU, metadata={"help": "activation function to use"})
    
    # Dropouts
    dropout: float = field(default=0.1, metadata={"help": "dropout probability for the transformer"})
    attention_dropout: float = field(default=0.1, metadata={"help": "dropout probability for attention weights"})
    activation_dropout: float = field(default=0.0, metadata={"help": "dropout probability after activation in FFN"})
    encoder_layerdrop: float = field(default=0.0, metadata={"help": "probability of dropping a transformer layer"})
    dropout_input: float = field(default=0.0, metadata={"help": "dropout to apply to the input (after feat extr)"})
    dropout_features: float = field(default=0.0, metadata={"help": "dropout to apply to the features (after feat extr)"})
    
    # Feature extraction
    final_dim: int = field(default=0, metadata={"help": "project final representations and targets to this many dimensions"})
    untie_final_proj: bool = field(default=False, metadata={"help": "use separate projection for each target"})
    layer_norm_first: bool = field(default=False, metadata={"help": "apply layernorm first in the transformer"})
    conv_feature_layers: str = field(default="[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2", metadata={"help": "convolutional feature extraction layers"})
    conv_bias: bool = field(default=False, metadata={"help": "include bias in conv encoder"})
    extractor_mode: ExtractorMode = field(default=ExtractorMode.DEFAULT, metadata={"help": "mode for feature extractor"})
    
    # Loss computation
    logit_temp: float = field(default=0.1, metadata={"help": "temperature to divide logits by"})
    target_glu: bool = field(default=False, metadata={"help": "adds projection + glu to targets"})
    feature_grad_mult: float = field(default=1.0, metadata={"help": "multiply feature extractor var grads by this"})
    skip_masked: bool = field(default=False, metadata={"help": "skip computing losses over masked frames"})
    skip_nomask: bool = field(default=False, metadata={"help": "skip computing losses over unmasked frames"})
    
    # Masking
    mask_length_audio: int = field(default=10, metadata={"help": "mask length for audio"})
    mask_prob_audio: float = field(default=0.65, metadata={"help": "probability of replacing a token with mask for audio"})
    mask_length_image: int = field(default=10, metadata={"help": "mask length for image"})
    mask_prob_image: float = field(default=0.65, metadata={"help": "probability of replacing a token with mask for image"})
    mask_selection: MaskingDistribution = field(default=MaskingDistribution.STATIC, metadata={"help": "how to choose mask length"})
    mask_other: float = field(default=0, metadata={"help": "secondary mask argument"})
    no_mask_overlap: bool = field(default=False, metadata={"help": "whether to allow masks to overlap"})
    mask_min_space: int = field(default=1, metadata={"help": "min space between spans"})
    
    # Channel masking
    mask_channel_length: int = field(default=10, metadata={"help": "length of the mask for features (channels)"})
    mask_channel_prob: float = field(default=0.0, metadata={"help": "probability of replacing a feature with 0"})
    mask_channel_selection: MaskingDistribution = field(default=MaskingDistribution.STATIC, metadata={"help": "how to choose mask length for channel masking"})
    mask_channel_other: float = field(default=0, metadata={"help": "secondary mask argument for channel masking"})
    no_mask_channel_overlap: bool = field(default=False, metadata={"help": "whether to allow channel masks to overlap"})
    mask_channel_min_space: int = field(default=1, metadata={"help": "min space between spans for channel masking"})
    
    # Positional embeddings
    conv_pos: int = field(default=128, metadata={"help": "number of filters for convolutional positional embeddings"})
    conv_pos_groups: int = field(default=16, metadata={"help": "number of groups for convolutional positional embedding"})
    
    # Legacy
    latent_temp: Tuple[float, float, float] = field(default=(2, 0.5, 0.999995), metadata={"help": "legacy (to be removed)"})
    
    # ResNet specific
    resnet_relu_type: str = field(default='prelu', metadata={"help": 'relu type for resnet'})
    resnet_weights: Optional[str] = field(default=None, metadata={"help": 'resnet weights'})
    sim_type: str = field(default='cosine', metadata={"help": 'similarity type'})
    
    # Modality specific
    sub_encoder_layers: int = field(default=0, metadata={'help': 'number of transformer layers for single modality'})
    audio_feat_dim: int = field(default=-1, metadata={'help': 'audio feature dimension'})
    modality_dropout: float = field(default=0, metadata={'help': 'drop one modality'})
    audio_dropout: float = field(default=0, metadata={'help': 'drop audio feature'})
    modality_fuse: str = field(default='concat', metadata={'help': 'fusing two modalities: add,concat'})
    selection_type: str = field(default='same_other_seq', metadata={'help': 'type of selecting images'})
    masking_type: str = field(default='input', metadata={'help': 'input or feature masking'})
    
    # Decoder specific
    decoder_embed_dim: int = field(default=768, metadata={"help": "decoder embedding dimension"})
    decoder_ffn_embed_dim: int = field(default=3072, metadata={"help": "decoder embedding dimension for FFN"})
    decoder_layers: int = field(default=6, metadata={"help": "num of decoder layers"})
    decoder_layerdrop: float = field(default=0.0, metadata={"help": "decoder layerdrop chance"})
    decoder_attention_heads: int = field(default=4, metadata={"help": "num decoder attention heads"})
    decoder_learned_pos: bool = field(default=False, metadata={"help": "use learned positional embeddings in the decoder"})
    decoder_normalize_before: bool = field(default=False, metadata={"help": "apply layernorm before each decoder block"})
    no_token_positional_embeddings: bool = field(default=False, metadata={"help": "if set, disables positional embeddings"})
    decoder_dropout: float = field(default=0.1, metadata={"help": "dropout probability in the decoder"})
    decoder_attention_dropout: float = field(default=0.1, metadata={"help": "dropout probability for attention weights inside the decoder"})
    decoder_activation_dropout: float = field(default=0.0, metadata={"help": "dropout probability after activation in FFN inside the decoder"})
    max_target_positions: int = field(default=2048, metadata={"help": "max target positions"})
    share_decoder_input_output_embed: bool = field(default=False, metadata={"help": "share decoder input and output embeddings"})
    no_scale_embedding: bool = field(default=True, metadata={'help': 'scale embedding'})
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.final_dim <= 0:
            self.final_dim = self.encoder_embed_dim