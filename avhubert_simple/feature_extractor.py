"""
Feature extractor for AVHuBERT.
Provides HuggingFace-style preprocessing for audio and video inputs.
"""

import cv2
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from .utils import load_video, Compose, Normalize, CenterCrop, RandomCrop, HorizontalFlip


class AVHuBERTFeatureExtractor:
    """
    Feature extractor for AVHuBERT model.
    
    This class provides preprocessing functionality for both audio and video inputs,
    following the HuggingFace transformers style interface.
    """
    
    def __init__(
        self,
        audio_sample_rate: int = 16000,
        video_size: Tuple[int, int] = (88, 88),
        video_mean: float = 0.421,
        video_std: float = 0.1652,
        do_normalize: bool = True,
        do_center_crop: bool = True,
        do_random_crop: bool = False,
        do_horizontal_flip: bool = False,
        flip_ratio: float = 0.5,
        **kwargs
    ):
        """
        Initialize the feature extractor.
        
        Args:
            audio_sample_rate: Target sample rate for audio
            video_size: Target size for video frames (height, width)
            video_mean: Mean for video normalization
            video_std: Standard deviation for video normalization
            do_normalize: Whether to normalize video frames
            do_center_crop: Whether to center crop video frames
            do_random_crop: Whether to randomly crop video frames
            do_horizontal_flip: Whether to apply horizontal flipping
            flip_ratio: Probability of horizontal flipping
        """
        self.audio_sample_rate = audio_sample_rate
        self.video_size = video_size
        self.video_mean = video_mean
        self.video_std = video_std
        self.do_normalize = do_normalize
        self.do_center_crop = do_center_crop
        self.do_random_crop = do_random_crop
        self.do_horizontal_flip = do_horizontal_flip
        self.flip_ratio = flip_ratio
        
        # Build video transforms
        self.video_transforms = self._build_video_transforms()

    def _build_video_transforms(self):
        """Build video preprocessing pipeline."""
        transforms = []
        
        if self.do_center_crop:
            transforms.append(CenterCrop(self.video_size))
        elif self.do_random_crop:
            transforms.append(RandomCrop(self.video_size))
        
        if self.do_horizontal_flip:
            transforms.append(HorizontalFlip(self.flip_ratio))
        
        if self.do_normalize:
            transforms.append(Normalize(self.video_mean, self.video_std))
        
        return Compose(transforms)

    def __call__(
        self,
        audio: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        video: Optional[Union[str, np.ndarray, List[np.ndarray]]] = None,
        audio_path: Optional[str] = None,
        video_path: Optional[str] = None,
        return_tensors: Optional[str] = "pt",
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess audio and video inputs.
        
        Args:
            audio: Audio array or list of audio arrays
            video: Video array, list of video arrays, or video file path
            audio_path: Path to audio file
            video_path: Path to video file
            return_tensors: Type of tensors to return ("pt", "np", or None)
            
        Returns:
            Dictionary containing processed audio and video tensors
        """
        # Load audio
        if audio_path is not None:
            audio = self._load_audio(audio_path)
        elif audio is None:
            raise ValueError("Either audio or audio_path must be provided")
        
        # Load video
        if video_path is not None:
            video = self._load_video(video_path)
        elif isinstance(video, str):
            video = self._load_video(video)
        elif video is None:
            raise ValueError("Either video or video_path must be provided")
        
        # Process audio
        processed_audio = self._process_audio(audio)
        
        # Process video
        processed_video = self._process_video(video)
        
        # Convert to tensors if requested
        if return_tensors == "pt":
            processed_audio = torch.tensor(processed_audio, dtype=torch.float32)
            processed_video = torch.tensor(processed_video, dtype=torch.float32)
        
        return {
            "audio": processed_audio,
            "video": processed_video,
        }

    def _load_audio(self, audio_path: str) -> np.ndarray:
        """Load audio from file."""
        # This is a placeholder - you would implement actual audio loading here
        # For now, we'll assume it's already loaded as numpy array
        raise NotImplementedError("Audio loading from file not implemented yet")

    def _load_video(self, video_path: str) -> np.ndarray:
        """Load video from file."""
        return load_video(video_path)

    def _process_audio(self, audio: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Process audio input."""
        if isinstance(audio, list):
            # Handle batch of audio inputs
            processed_audio = []
            for audio_item in audio:
                processed_audio.append(self._process_single_audio(audio_item))
            return np.stack(processed_audio)
        else:
            return self._process_single_audio(audio)

    def _process_single_audio(self, audio: np.ndarray) -> np.ndarray:
        """Process a single audio input."""
        # Ensure audio is 1D
        if audio.ndim > 1:
            audio = audio.squeeze()
        
        # Resample if needed (placeholder)
        # You would implement actual resampling here
        
        return audio

    def _process_video(self, video: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Process video input."""
        if isinstance(video, list):
            # Handle batch of video inputs
            processed_video = []
            for video_item in video:
                processed_video.append(self._process_single_video(video_item))
            return np.stack(processed_video)
        else:
            return self._process_single_video(video)

    def _process_single_video(self, video: np.ndarray) -> np.ndarray:
        """Process a single video input."""
        # Apply video transforms
        video = self.video_transforms(video)
        
        # Add channel dimension if needed
        if video.ndim == 3:  # [T, H, W]
            video = video[:, np.newaxis, :, :]  # [T, 1, H, W]
        
        return video

    def batch_decode(self, features: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        """
        Decode features back to original format (for visualization).
        
        Args:
            features: Dictionary containing processed features
            
        Returns:
            Dictionary containing decoded features
        """
        decoded = {}
        
        if "audio" in features:
            audio = features["audio"]
            if isinstance(audio, torch.Tensor):
                audio = audio.numpy()
            decoded["audio"] = audio
        
        if "video" in features:
            video = features["video"]
            if isinstance(video, torch.Tensor):
                video = video.numpy()
            
            # Reverse normalization
            if self.do_normalize:
                video = video * self.video_std + self.video_mean
            
            decoded["video"] = video
        
        return decoded

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        """
        Load feature extractor from pretrained model.
        
        Args:
            model_name_or_path: Path to pretrained model or model identifier
            **kwargs: Additional arguments
            
        Returns:
            Feature extractor instance
        """
        # This would load configuration from the pretrained model
        # For now, return default configuration
        return cls(**kwargs)