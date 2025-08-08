"""
AV-HuBERT configuration
"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from typing import List


logger = logging.get_logger(__name__)


class AVHubertConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`AVHubertModel`]. It is used to instantiate an
    AV-HuBERT model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 32):
            Vocabulary size of the AV-HuBERT model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`AVHubertModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        final_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for the final projection layer.
        layerdrop (`float`, *optional*, defaults to 0.1):
            The LayerDrop probability.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        feat_extract_norm (`str`, *optional*, defaults to `"group"`):
            The norm to be applied to 1D convolutional layers in feature encoder. One of `"group"` for group
            normalization of only the first 1D convolutional layer or `"layer"` for layer normalization of all 1D
            convolutional layers.
        feat_proj_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for output of the feature encoder.
        feat_extract_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for feature extractor.
        conv_feature_layers (`List[List[int]]`, *optional*):
            A list of lists to define the 1D convolutional layers of the feature extractor. Each element of the list
            corresponds to one convolutional layer and is defined by 3 integers: (output_channels, kernel_size, stride).
            The default value is for AV-HuBERT Base.
        conv_pos (`int`, *optional*, defaults to 128):
            Number of filters for convolutional positional embeddings.
        conv_pos_groups (`int`, *optional*, defaults to 16):
            Number of groups for convolutional positional embedding.
        mask_time_prob (`float`, *optional*, defaults to 0.05):
            Percentage (between 0 and 1) of all feature vectors along the time axis which will be masked. The masking
            procecure generates `mask_time_prob*len(time_axis)/mask_time_length` independent masks over the axis. If
            reasoning from the propability: `mask_time_prob*len(time_axis)/mask_time_length ~= mask_time_prob` the
            number of masks is `mask_time_prob*len(time_axis)/mask_time_length`.
        mask_time_length (`int`, *optional*, defaults to 10):
            Length of vector span to mask along the time axis.
        mask_feature_prob (`float`, *optional*, defaults to 0.0):
            Percentage (between 0 and 1) of all feature vectors along the feature axis which will be masked. The
            masking procecure generates `mask_feature_prob*len(feature_axis)/mask_time_length` independent masks over
            the axis. If reasoning from the propability:
            `mask_feature_prob*len(feature_axis)/mask_feature_length ~= mask_feature_prob` the number of masks is
            `mask_feature_prob*len(feature_axis)/mask_feature_length`.
        mask_feature_length (`int`, *optional*, defaults to 10):
            Length of vector span to mask along the feature axis.
        modality_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for modality selection.
        audio_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for audio features.
        modality_fuse (`str`, *optional*, defaults to `"concat"`):
            How to fuse audio and video modalities. One of `"concat"` or `"add"`.
        selection_type (`str`, *optional*, defaults to `"same_seq"`):
            Type of sequence selection for modality dropout.
        masking_type (`str`, *optional*, defaults to `"feature"`):
            Type of masking to apply. One of `"feature"` or `"input"`.
        ctc_loss_reduction (`str`, *optional*, defaults to `"sum"`):
            Specifies the reduction to apply to the output of `torch.nn.CTCLoss`. Only relevant when training an
            instance of [`AVHubertForCTC`].
        ctc_zero_infinity (`bool`, *optional*, defaults to `False`):
            Whether to zero infinite losses and the associated gradients of `torch.nn.CTCLoss`. Infinite losses mainly
            occur when the inputs are too short to be aligned to the targets. Only relevant when training an instance
            of [`AVHubertForCTC`].
        use_weighted_layer_sum (`bool`, *optional*, defaults to `False`):
            Whether to use a weighted average of layer outputs with learned weights. Only relevant when using an
            instance of [`AVHubertForSequenceClassification`].
        classifier_proj_size (`int`, *optional*, defaults to 256):
            Dimensionality of the projection before the classification head.
    """

    model_type = "avhubert"

    def __init__(
        self,
        vocab_size=32,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        final_dropout=0.1,
        layerdrop=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        feat_extract_norm="group",
        feat_proj_dropout=0.0,
        feat_extract_dropout=0.0,
        conv_feature_layers=None,
        conv_pos=128,
        conv_pos_groups=16,
        mask_time_prob=0.05,
        mask_time_length=10,
        mask_feature_prob=0.0,
        mask_feature_length=10,
        modality_dropout=0.0,
        audio_dropout=0.0,
        modality_fuse="concat",
        selection_type="same_seq",
        masking_type="feature",
        ctc_loss_reduction="sum",
        ctc_zero_infinity=False,
        use_weighted_layer_sum=False,
        classifier_proj_size=256,
        # AV-specific parameters
        audio_feat_dim=104,
        video_feat_dim=512,
        label_rate=100.0,
        skip_masked=False,
        skip_nomask=False,
        mask_prob_image=0.8,
        mask_length_image=10,
        mask_prob_audio=0.8,
        mask_length_audio=10,
        extractor_mode="default",
        final_dim=256,
        feature_grad_mult=0.1,
        untie_final_proj=True,
        layer_norm_first=True,
        wav_input=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.final_dropout = final_dropout
        self.layerdrop = layerdrop
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.feat_extract_norm = feat_extract_norm
        self.feat_proj_dropout = feat_proj_dropout
        self.feat_extract_dropout = feat_extract_dropout
        
        if conv_feature_layers is None:
            conv_feature_layers = [
                [512, 10, 5],
                [512, 3, 2],
                [512, 3, 2],
                [512, 3, 2],
                [512, 3, 2],
                [512, 2, 2],
                [512, 2, 2]
            ]
        self.conv_feature_layers = conv_feature_layers
        
        self.conv_pos = conv_pos
        self.conv_pos_groups = conv_pos_groups
        self.mask_time_prob = mask_time_prob
        self.mask_time_length = mask_time_length
        self.mask_feature_prob = mask_feature_prob
        self.mask_feature_length = mask_feature_length
        self.modality_dropout = modality_dropout
        self.audio_dropout = audio_dropout
        self.modality_fuse = modality_fuse
        self.selection_type = selection_type
        self.masking_type = masking_type
        self.ctc_loss_reduction = ctc_loss_reduction
        self.ctc_zero_infinity = ctc_zero_infinity
        self.use_weighted_layer_sum = use_weighted_layer_sum
        self.classifier_proj_size = classifier_proj_size
        
        # AV-specific parameters
        self.audio_feat_dim = audio_feat_dim
        self.video_feat_dim = video_feat_dim
        self.label_rate = label_rate
        self.skip_masked = skip_masked
        self.skip_nomask = skip_nomask
        self.mask_prob_image = mask_prob_image
        self.mask_length_image = mask_length_image
        self.mask_prob_audio = mask_prob_audio
        self.mask_length_audio = mask_length_audio
        self.extractor_mode = extractor_mode
        self.final_dim = final_dim
        self.feature_grad_mult = feature_grad_mult
        self.untie_final_proj = untie_final_proj
        self.layer_norm_first = layer_norm_first
        self.wav_input = wav_input