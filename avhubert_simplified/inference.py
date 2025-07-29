"""
Simplified inference interface for AVHuBERT.
Easy-to-use API for lip reading and audio-visual speech recognition.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, Dict, List, Tuple
import logging

from .model import AVHubertModel, create_avhubert_base
from .config import AVHubertConfig
from .preprocessors import AVHubertPreprocessor

logger = logging.getLogger(__name__)


class AVHubertInference:
    """
    High-level inference interface for AVHuBERT.
    
    This class provides an easy-to-use API for:
    - Lip reading (video-only)
    - Audio speech recognition (audio-only)  
    - Audio-visual speech recognition (both modalities)
    """
    
    def __init__(
        self,
        model: Optional[AVHubertModel] = None,
        config: Optional[AVHubertConfig] = None,
        device: str = "auto",
    ):
        """
        Initialize AVHuBERT inference
        
        Args:
            model: Pre-loaded AVHuBERT model. If None, creates a base model.
            config: Model configuration. If None, uses default config.
            device: Device to run inference on ("auto", "cpu", "cuda", etc.)
        """
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize model
        if model is None:
            if config is None:
                config = AVHubertConfig()
            model = create_avhubert_base(config)
        
        self.model = model.to(self.device)
        self.config = self.model.config
        
        # Initialize preprocessor
        self.preprocessor = AVHubertPreprocessor()
        
        # Set model to eval mode
        self.model.eval()
        
        logger.info(f"AVHuBERT inference initialized on {self.device}")
    
    @classmethod
    def from_pretrained(
        cls, 
        model_path: str,
        config: Optional[AVHubertConfig] = None,
        device: str = "auto",
    ) -> 'AVHubertInference':
        """
        Load pretrained model for inference
        
        Args:
            model_path: Path to pretrained model checkpoint
            config: Optional model configuration override
            device: Device to run inference on
            
        Returns:
            AVHubertInference instance with loaded model
        """
        model = AVHubertModel.from_pretrained(model_path, config)
        return cls(model=model, device=device)
    
    def preprocess_inputs(
        self,
        audio: Optional[Union[np.ndarray, torch.Tensor, str]] = None,
        video: Optional[Union[np.ndarray, str, List[np.ndarray]]] = None,
        audio_target_length: Optional[int] = None,
        video_target_length: Optional[int] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Preprocess audio and video inputs"""
        processed_audio, processed_video = self.preprocessor(
            audio=audio,
            video=video,
            audio_target_length=audio_target_length,
            video_target_length=video_target_length,
        )
        
        # Move to device and add batch dimension
        if processed_audio is not None:
            processed_audio = processed_audio.unsqueeze(0).to(self.device)
        
        if processed_video is not None:
            processed_video = processed_video.unsqueeze(0).to(self.device)
        
        return processed_audio, processed_video
    
    @torch.no_grad()
    def extract_features(
        self,
        audio: Optional[Union[np.ndarray, torch.Tensor, str]] = None,
        video: Optional[Union[np.ndarray, str, List[np.ndarray]]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features from audio and/or video
        
        Args:
            audio: Audio input (raw waveform, file path, or tensor)
            video: Video input (frames, file path, or tensor)
            **kwargs: Additional arguments for preprocessing
            
        Returns:
            Dictionary containing extracted features
        """
        # Preprocess inputs
        processed_audio, processed_video = self.preprocess_inputs(
            audio=audio, video=video, **kwargs
        )
        
        # Extract features without masking
        outputs = self.model(
            audio=processed_audio,
            video=processed_video,
            mask=False,  # No masking during inference
        )
        
        return {
            "last_hidden_state": outputs["last_hidden_state"],
            "projected_states": outputs["projected_states"],
        }
    
    @torch.no_grad()
    def predict(
        self,
        audio: Optional[Union[np.ndarray, torch.Tensor, str]] = None,
        video: Optional[Union[np.ndarray, str, List[np.ndarray]]] = None,
        return_features: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions with AVHuBERT
        
        Args:
            audio: Audio input
            video: Video input
            return_features: Whether to return intermediate features
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing predictions and optionally features
        """
        # Extract features
        outputs = self.extract_features(audio=audio, video=video, **kwargs)
        
        result = {
            "predictions": outputs["projected_states"],
        }
        
        if return_features:
            result["features"] = outputs["last_hidden_state"]
        
        return result
    
    @torch.no_grad()
    def lip_reading(
        self,
        video: Union[np.ndarray, str, List[np.ndarray]],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Perform lip reading (video-only speech recognition)
        
        Args:
            video: Video input containing lip movements
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing lip reading predictions
        """
        return self.predict(audio=None, video=video, **kwargs)
    
    @torch.no_grad()
    def audio_speech_recognition(
        self,
        audio: Union[np.ndarray, torch.Tensor, str],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Perform audio-only speech recognition
        
        Args:
            audio: Audio input
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing ASR predictions
        """
        return self.predict(audio=audio, video=None, **kwargs)
    
    @torch.no_grad()
    def audio_visual_speech_recognition(
        self,
        audio: Union[np.ndarray, torch.Tensor, str],
        video: Union[np.ndarray, str, List[np.ndarray]],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Perform audio-visual speech recognition
        
        Args:
            audio: Audio input
            video: Video input
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing AVSR predictions
        """
        return self.predict(audio=audio, video=video, **kwargs)
    
    def get_similarity(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute similarity between feature representations
        
        Args:
            features1: First set of features (B, T, D)
            features2: Second set of features (B, T, D)
            temperature: Temperature for similarity computation
            
        Returns:
            Similarity scores
        """
        # L2 normalize features
        features1 = F.normalize(features1, dim=-1)
        features2 = F.normalize(features2, dim=-1)
        
        # Compute cosine similarity
        similarity = torch.sum(features1 * features2, dim=-1) / temperature
        
        return similarity
    
    def set_model_mode(self, training: bool = False):
        """Set model training/evaluation mode"""
        if training:
            self.model.train()
        else:
            self.model.eval()


class LipReadingInference(AVHubertInference):
    """Specialized inference class for lip reading tasks"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Configure for video-only processing
        self.config.input_modality = "video"
    
    def __call__(
        self, 
        video: Union[np.ndarray, str, List[np.ndarray]],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Convenient call interface for lip reading"""
        return self.lip_reading(video, **kwargs)


class AudioSpeechRecognitionInference(AVHubertInference):
    """Specialized inference class for audio speech recognition tasks"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Configure for audio-only processing
        self.config.input_modality = "audio"
    
    def __call__(
        self, 
        audio: Union[np.ndarray, torch.Tensor, str],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Convenient call interface for ASR"""
        return self.audio_speech_recognition(audio, **kwargs)


class AudioVisualSpeechRecognitionInference(AVHubertInference):
    """Specialized inference class for audio-visual speech recognition tasks"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Configure for audio-visual processing
        self.config.input_modality = "audio_video"
    
    def __call__(
        self,
        audio: Union[np.ndarray, torch.Tensor, str],
        video: Union[np.ndarray, str, List[np.ndarray]],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Convenient call interface for AVSR"""
        return self.audio_visual_speech_recognition(audio, video, **kwargs)


# Factory functions for easy creation
def create_lip_reading_model(
    model_path: Optional[str] = None,
    config: Optional[AVHubertConfig] = None,
    device: str = "auto",
) -> LipReadingInference:
    """Create a lip reading inference model"""
    if model_path is not None:
        return LipReadingInference.from_pretrained(model_path, config, device)
    else:
        return LipReadingInference(config=config, device=device)


def create_asr_model(
    model_path: Optional[str] = None,
    config: Optional[AVHubertConfig] = None,
    device: str = "auto",
) -> AudioSpeechRecognitionInference:
    """Create an audio speech recognition inference model"""
    if model_path is not None:
        return AudioSpeechRecognitionInference.from_pretrained(model_path, config, device)
    else:
        return AudioSpeechRecognitionInference(config=config, device=device)


def create_avsr_model(
    model_path: Optional[str] = None,
    config: Optional[AVHubertConfig] = None,
    device: str = "auto",
) -> AudioVisualSpeechRecognitionInference:
    """Create an audio-visual speech recognition inference model"""
    if model_path is not None:
        return AudioVisualSpeechRecognitionInference.from_pretrained(model_path, config, device)
    else:
        return AudioVisualSpeechRecognitionInference(config=config, device=device)


def create_general_model(
    model_path: Optional[str] = None,
    config: Optional[AVHubertConfig] = None,
    device: str = "auto",
) -> AVHubertInference:
    """Create a general-purpose AVHuBERT inference model"""
    if model_path is not None:
        return AVHubertInference.from_pretrained(model_path, config, device)
    else:
        return AVHubertInference(config=config, device=device)