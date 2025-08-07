"""
Processor class for AV-HuBERT.
"""

from typing import List, Optional, Union

import numpy as np
import torch

from transformers import ProcessorMixin
from transformers.utils import BatchFeature

from .feature_extraction_avhubert import AVHubertFeatureExtractor
from .tokenization_avhubert import AVHubertTokenizer


class AVHubertProcessor(ProcessorMixin):
    """
    Constructs an AV-HuBERT processor which wraps an AV-HuBERT feature extractor and an AV-HuBERT tokenizer into a single
    processor.
    
    [`AVHubertProcessor`] offers all the functionalities of [`AVHubertFeatureExtractor`] and [`AVHubertTokenizer`].
    See the [`~AVHubertProcessor.__call__`] and [`~AVHubertProcessor.decode`] for more information.
    
    Args:
        feature_extractor (`AVHubertFeatureExtractor`):
            An instance of [`AVHubertFeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`AVHubertTokenizer`):
            An instance of [`AVHubertTokenizer`]. The tokenizer is a required input.
    """
    
    feature_extractor_class = "AVHubertFeatureExtractor"
    tokenizer_class = "AVHubertTokenizer"
    
    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)
        self.current_processor = self.feature_extractor
    
    def __call__(
        self,
        audio: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        text: Union[str, List[str]] = None,
        sampling_rate: Optional[int] = None,
        return_tensors: Optional[Union[str, "TensorType"]] = None,
        **kwargs,
    ):
        """
        Main method to prepare for the model one or several sequence(s) and corresponding text(s). This method forwards
        the `audio` and `kwargs` arguments to AVHubertFeatureExtractor's [`~AVHubertFeatureExtractor.__call__`] and the
        `text` and `kwargs` arguments to AVHubertTokenizer's [`~AVHubertTokenizer.__call__`]. Please refer to the
        docstrings of the above two methods for more information.
        
        Args:
            audio (`np.ndarray`, `List[float]`, `List[np.ndarray]`, `List[List[float]]`):
                The sequence or batch of sequences to be processed. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values.
            text (`str`, `List[str]`, *optional*):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            sampling_rate (`int`, *optional*):
                The sampling rate of the input audio. If not provided, will use the sampling rate used to load the audio.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.
        
        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:
            - **input_values** -- Audio input values to be fed to a model, of shape (batch_size, num_channels, height, width).
            - **attention_mask** -- Attention mask to be fed to a model, of shape (batch_size, num_channels, height, width).
            - **labels** -- Labels to be fed to a model, of shape (batch_size, sequence_length).
        """
        
        # Process audio
        audio_features = self.feature_extractor(
            audio,
            sampling_rate=sampling_rate,
            return_tensors=return_tensors,
            **kwargs,
        )
        
        # Process text if provided
        if text is not None:
            text_features = self.tokenizer(
                text,
                return_tensors=return_tensors,
                **kwargs,
            )
            audio_features.update(text_features)
        
        return audio_features
    
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to AVHubertTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)
    
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to AVHubertTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)
    
    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + feature_extractor_input_names))