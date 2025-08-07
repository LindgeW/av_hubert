"""
AV-HuBERT model implementation following Hugging Face patterns.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutput, Seq2SeqLMOutput
from transformers.utils import logging

from .configuration_avhubert import AVHubertConfig
from .resnet import ResEncoder
from .transformer import TransformerEncoder, TransformerDecoder
from .utils import (
    compute_mask_indices,
    make_conv_pos,
    GradMultiply,
    get_activation_fn,
    pad_to_multiple,
)

logger = logging.get_logger(__name__)


class ConvFeatureExtractionModel(nn.Module):
    """Convolutional feature extraction model for audio."""
    
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
        in_d: int = 1,
    ):
        super().__init__()
        
        assert mode in {"default", "layer_norm"}
        
        def block(
            n_in,
            n_out,
            k,
            stride,
            is_layer_norm=False,
            is_group_norm=False,
            conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv
            
            assert (
                is_layer_norm is False or is_group_norm is False
            ), "layer norm and group norm are mutually exclusive"
            
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
        
        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl
            
            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim
    
    def forward(self, x):
        # BxT -> BxCxT
        x = x.unsqueeze(1)
        
        for conv in self.conv_layers:
            x = conv(x)
        
        return x


class TransposeLast(nn.Module):
    """Transpose the last two dimensions."""
    
    def __init__(self, deconstruct_idx=None):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx
    
    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(-2, -1)


class AVHubertModel(PreTrainedModel):
    """
    The bare AV-HuBERT Model transformer outputting raw hidden-states without any specific head on top.
    """
    
    config_class = AVHubertConfig
    base_model_prefix = "avhubert"
    supports_gradient_checkpointing = True
    
    def __init__(self, config: AVHubertConfig):
        super().__init__(config)
        
        self.config = config
        
        # Feature extraction
        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=eval(config.conv_feature_layers),
            dropout=config.dropout_input,
            mode=config.extractor_mode,
            conv_bias=config.conv_bias,
        )
        
        # Positional embeddings
        self.post_extract_proj = (
            nn.Linear(config.encoder_embed_dim, config.encoder_embed_dim)
            if config.encoder_embed_dim != config.encoder_embed_dim
            else None
        )
        
        self.mask_emb = nn.Parameter(
            torch.FloatTensor(config.encoder_embed_dim).uniform_()
        )
        
        # Video encoder
        self.video_encoder = ResEncoder(
            relu_type=config.resnet_relu_type,
            weights=config.resnet_weights,
        )
        
        # Transformer encoder
        self.encoder = TransformerEncoder(
            embed_dim=config.encoder_embed_dim,
            ffn_embed_dim=config.encoder_ffn_embed_dim,
            layers=config.encoder_layers,
            attention_heads=config.encoder_attention_heads,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            activation_dropout=config.activation_dropout,
            layerdrop=config.encoder_layerdrop,
            layer_norm_first=config.layer_norm_first,
            activation_fn=config.activation_fn,
        )
        
        # Positional embeddings
        self.pos_conv = make_conv_pos(
            config.encoder_embed_dim,
            config.conv_pos,
            config.conv_pos_groups,
        )
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(config.encoder_embed_dim)
        
        # Projection layers
        self.final_proj = nn.Linear(config.encoder_embed_dim, config.final_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def get_input_embeddings(self):
        return self.feature_extractor
    
    def set_input_embeddings(self, value):
        self.feature_extractor = value
    
    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """
        
        def _conv_out_length(input_length, kernel_size, stride):
            return torch.floor((input_length - kernel_size) / stride + 1)
        
        for i in range(len(self.config.conv_feature_layers)):
            input_lengths = _conv_out_length(
                input_lengths,
                self.config.conv_feature_layers[i][1],
                self.config.conv_feature_layers[i][2],
            )
        
        return input_lengths.to(torch.long)
    
    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        video_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.BoolTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        """
        Args:
            input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file
                into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`).
                To prepare the array into `input_values`, the [`AVHubertProcessor`] should be used for padding and conversion into a
                tensor of type `torch.FloatTensor`.
            video_values (`torch.FloatTensor` of shape `(batch_size, num_frames, height, width, channels)`):
                Float values of input video frames. Values can be obtained by loading video files and extracting frames.
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            mask_time_indices (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices to mask extracted features for contrastive loss. When in training mode, model learns to predict
                masked extracted features.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
                tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
                more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        
        Returns:
            [`~utils.ModelOutput`] or `tuple`:
                If `return_dict=False`, the function returns a tuple. If `return_dict=True`, the function returns a
                [`~utils.ModelOutput`] with the following fields:
                - **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) --
                  Sequence of hidden-states at the output of the last layer of the model.
                - **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`) --
                  Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
                  of shape `(batch_size, sequence_length, hidden_size)`.
                - **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True`) --
                  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
                  sequence_length)`.
        """
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Extract features
        if input_values is not None:
            hidden_states = self.feature_extractor(input_values)
            hidden_states = hidden_states.transpose(1, 2)
        else:
            hidden_states = None
        
        # Extract video features
        if video_values is not None:
            video_features = self.video_encoder(video_values)
        else:
            video_features = None
        
        # Combine audio and video features
        if hidden_states is not None and video_features is not None:
            # Align video features to audio features
            if video_features.size(1) != hidden_states.size(1):
                # Interpolate video features to match audio length
                video_features = F.interpolate(
                    video_features.transpose(1, 2),
                    size=hidden_states.size(1),
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            
            # Concatenate or add features based on config
            if self.config.modality_fuse == "concat":
                hidden_states = torch.cat([hidden_states, video_features], dim=-1)
                if self.post_extract_proj is not None:
                    hidden_states = self.post_extract_proj(hidden_states)
            elif self.config.modality_fuse == "add":
                hidden_states = hidden_states + video_features
        elif video_features is not None:
            hidden_states = video_features
        elif hidden_states is None:
            raise ValueError("Either input_values or video_values must be provided")
        
        # Apply masking if provided
        if mask_time_indices is not None:
            hidden_states[mask_time_indices] = self.mask_emb.to(hidden_states.dtype)
        
        # Apply positional embeddings
        if self.pos_conv is not None:
            hidden_states = hidden_states.transpose(1, 2)
            hidden_states = self.pos_conv(hidden_states)
            hidden_states = hidden_states.transpose(1, 2)
        
        # Apply layer norm
        hidden_states = self.layer_norm(hidden_states)
        
        # Create attention mask
        if attention_mask is not None:
            # Convert attention mask to padding mask
            padding_mask = attention_mask == 0
        else:
            padding_mask = None
        
        # Pass through transformer encoder
        encoder_outputs = self.encoder(
            hidden_states,
            padding_mask=padding_mask,
            return_all_hiddens=output_hidden_states,
        )
        
        hidden_states = encoder_outputs[0]
        
        # Apply final projection
        if self.final_proj is not None:
            hidden_states = self.final_proj(hidden_states)
        
        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]
        
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs[2] if output_hidden_states else None,
            attentions=encoder_outputs[1] if output_attentions else None,
        )


class AVHubertForPreTraining(AVHubertModel):
    """
    AV-HuBERT model for pre-training with masked prediction.
    """
    
    def __init__(self, config: AVHubertConfig):
        super().__init__(config)
        
        # Projection head for masked prediction
        self.projection_head = nn.Linear(config.final_dim, config.final_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        video_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.BoolTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        """
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked),
                the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = super().forward(
            input_values=input_values,
            video_values=video_values,
            attention_mask=attention_mask,
            mask_time_indices=mask_time_indices,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0]
        
        # Apply projection head
        projected_states = self.projection_head(hidden_states)
        
        # Compute loss if labels are provided
        loss = None
        if labels is not None and mask_time_indices is not None:
            # Only compute loss on masked positions
            masked_states = projected_states[mask_time_indices]
            masked_labels = labels[mask_time_indices]
            
            # Compute contrastive loss
            loss = F.cross_entropy(masked_states, masked_labels)
        
        if not return_dict:
            return (projected_states, loss) + outputs[1:]
        
        return CausalLMOutput(
            loss=loss,
            logits=projected_states,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class AVHubertForCTC(AVHubertModel):
    """
    AV-HuBERT model for CTC (Connectionist Temporal Classification) speech recognition.
    """
    
    def __init__(self, config: AVHubertConfig):
        super().__init__(config)
        
        # CTC head
        self.ctc_head = nn.Linear(config.final_dim, config.vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        video_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        """
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
                Labels for computing the CTC loss. Indices should be in `[0, ..., config.vocab_size - 1]`.
        """
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = super().forward(
            input_values=input_values,
            video_values=video_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0]
        
        # Apply CTC head
        logits = self.ctc_head(hidden_states)
        
        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            # CTC loss computation would go here
            # This is a simplified version - in practice, you'd use torch.nn.CTCLoss
            pass
        
        if not return_dict:
            return (logits, loss) + outputs[1:]
        
        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )