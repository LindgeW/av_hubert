"""
Feature extraction classes for AV-HuBERT.
"""

import warnings
from typing import List, Optional, Union

import numpy as np
import torch

from transformers import PreTrainedFeatureExtractor
from transformers.utils import logging

logger = logging.get_logger(__name__)


class AVHubertFeatureExtractor(PreTrainedFeatureExtractor):
    """
    Constructs an AV-HuBERT feature extractor.
    
    This feature extractor inherits from [`PreTrainedFeatureExtractor`] which contains most of the main methods.
    Users should refer to this superclass for more information regarding those methods.
    
    Args:
        feature_size (`int`, defaults to 1):
            The feature dimension of the extracted features.
        sampling_rate (`int`, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in Hertz per second (Hz).
        padding_value (`float`, defaults to 0.0):
            The value that is used to fill the padding values / vectors.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the input. Normalization can help to improve model performance.
        return_attention_mask (`bool`, *optional*, defaults to `True`):
            Whether or not [`~AVHubertFeatureExtractor.__call__`] should return `attention_mask`.
    """
    
    model_input_names = ["input_values", "attention_mask"]
    
    def __init__(
        self,
        feature_size: int = 1,
        sampling_rate: int = 16000,
        padding_value: float = 0.0,
        do_normalize: bool = True,
        return_attention_mask: bool = True,
        **kwargs,
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        self.do_normalize = do_normalize
        self.return_attention_mask = return_attention_mask
    
    def normalize(self, input_values):
        """Normalize the input values."""
        if self.do_normalize:
            input_values = (input_values - input_values.mean()) / (input_values.std() + 1e-8)
        return input_values
    
    def __call__(
        self,
        audio: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        sampling_rate: Optional[int] = None,
        return_tensors: Optional[Union[str, "TensorType"]] = None,
        **kwargs,
    ):
        """
        Main method to featurize and prepare for the model one or several sequence(s).
        
        Args:
            audio (`np.ndarray`, `List[float]`, `List[np.ndarray]`, `List[List[float]]`):
                The sequence or batch of sequences to be processed. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values.
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
        """
        
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            warnings.warn(
                f"The sampling_rate you passed to {self.__class__.__name__} is {sampling_rate}, but the model was trained with {self.sampling_rate}. "
                f"Resampling the audio to {self.sampling_rate}."
            )
            # Resample audio if needed
            # This would require librosa or similar library for resampling
        
        is_batched = bool(
            isinstance(audio, (list, tuple))
            and (isinstance(audio[0], (list, tuple, np.ndarray)) or isinstance(audio[0], (int, float)))
        )
        
        if is_batched:
            audio = [np.asarray(a, dtype=np.float32) for a in audio]
        else:
            audio = np.asarray(audio, dtype=np.float32)
        
        # Normalize audio
        if self.do_normalize:
            if is_batched:
                audio = [self.normalize(a) for a in audio]
            else:
                audio = self.normalize(audio)
        
        # Convert to tensor if requested
        if return_tensors is not None:
            if return_tensors == "pt":
                if is_batched:
                    audio = [torch.tensor(a, dtype=torch.float32) for a in audio]
                else:
                    audio = torch.tensor(audio, dtype=torch.float32)
            elif return_tensors == "np":
                if not is_batched:
                    audio = np.expand_dims(audio, 0)
            else:
                raise ValueError(f"return_tensors {return_tensors} not supported")
        
        # Create attention mask
        attention_mask = None
        if self.return_attention_mask:
            if is_batched:
                attention_mask = [np.ones_like(a) for a in audio]
            else:
                attention_mask = np.ones_like(audio)
        
        return {"input_values": audio, "attention_mask": attention_mask}
    
    def pad(
        self,
        processed_features,
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, "TensorType"]] = None,
    ):
        """
        Pad input values / input vectors to a maximum length and to a multiple value.
        
        Args:
            processed_features:
                Dictionary of input values / input vectors / input tensors to be padded.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:
                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
                the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta), or on TPUs which
                benefit from having sequence lengths be a multiple of 128.
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific feature_extractor's default.
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
        """
        
        # This is a simplified implementation
        # In practice, you would implement proper padding logic here
        
        return processed_features