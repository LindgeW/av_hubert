"""
Processor for AVHuBERT model.
Combines feature extraction and model inference for a complete interface.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from .feature_extractor import AVHuBERTFeatureExtractor
from .config import AVHuBERTConfig
from .model import AVHuBERTModel


class AVHuBERTProcessor:
    """
    Processor for AVHuBERT model.
    
    This class combines feature extraction and model inference to provide
    a complete HuggingFace-style interface for AVHuBERT.
    """
    
    def __init__(
        self,
        feature_extractor: AVHuBERTFeatureExtractor,
        model: AVHuBERTModel,
        **kwargs
    ):
        """
        Initialize the processor.
        
        Args:
            feature_extractor: Feature extractor instance
            model: AVHuBERT model instance
        """
        self.feature_extractor = feature_extractor
        self.model = model

    def __call__(
        self,
        audio: Optional[Union[str, np.ndarray, List[np.ndarray]]] = None,
        video: Optional[Union[str, np.ndarray, List[np.ndarray]]] = None,
        audio_path: Optional[str] = None,
        video_path: Optional[str] = None,
        return_tensors: Optional[str] = "pt",
        padding: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Process inputs and run model inference.
        
        Args:
            audio: Audio input or path
            video: Video input or path
            audio_path: Path to audio file
            video_path: Path to video file
            return_tensors: Type of tensors to return
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            max_length: Maximum sequence length
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing model outputs
        """
        # Extract features
        features = self.feature_extractor(
            audio=audio,
            video=video,
            audio_path=audio_path,
            video_path=video_path,
            return_tensors=return_tensors,
            **kwargs
        )
        
        # Run model inference
        with torch.no_grad():
            outputs = self.model(
                audio=features["audio"],
                video=features["video"],
                mask=False,  # No masking for inference
                features_only=True,
            )
        
        # Combine features and outputs
        result = {**features, **outputs}
        return result

    def extract_features(
        self,
        audio: Optional[Union[str, np.ndarray, List[np.ndarray]]] = None,
        video: Optional[Union[str, np.ndarray, List[np.ndarray]]] = None,
        audio_path: Optional[str] = None,
        video_path: Optional[str] = None,
        mask: bool = False,
        output_layer: Optional[int] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features from audio and video inputs.
        
        Args:
            audio: Audio input or path
            video: Video input or path
            audio_path: Path to audio file
            video_path: Path to video file
            mask: Whether to apply masking
            output_layer: Specific layer to extract features from
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing extracted features
        """
        # Extract features
        features = self.feature_extractor(
            audio=audio,
            video=video,
            audio_path=audio_path,
            video_path=video_path,
            return_tensors="pt",
            **kwargs
        )
        
        # Extract model features
        with torch.no_grad():
            model_features = self.model.extract_features(
                audio=features["audio"],
                video=features["video"],
                mask=mask,
                output_layer=output_layer,
            )
        
        return {
            "features": model_features[0],
            "padding_mask": model_features[1],
        }

    def predict(
        self,
        audio: Optional[Union[str, np.ndarray, List[np.ndarray]]] = None,
        video: Optional[Union[str, np.ndarray, List[np.ndarray]]] = None,
        audio_path: Optional[str] = None,
        video_path: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run prediction on audio and video inputs.
        
        Args:
            audio: Audio input or path
            video: Video input or path
            audio_path: Path to audio file
            video_path: Path to video file
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing predictions
        """
        # Extract features
        features = self.extract_features(
            audio=audio,
            video=video,
            audio_path=audio_path,
            video_path=video_path,
            mask=False,
            **kwargs
        )
        
        return {
            "features": features["features"],
            "embeddings": features["features"].mean(dim=1),  # Pool over time
        }

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        config: Optional[AVHuBERTConfig] = None,
        **kwargs
    ):
        """
        Load processor from pretrained model.
        
        Args:
            model_name_or_path: Path to pretrained model or model identifier
            config: Model configuration
            **kwargs: Additional arguments
            
        Returns:
            Processor instance
        """
        # Load configuration
        if config is None:
            config = AVHuBERTConfig()
        
        # Create feature extractor
        feature_extractor = AVHuBERTFeatureExtractor.from_pretrained(
            model_name_or_path,
            **kwargs
        )
        
        # Create model
        model = AVHuBERTModel(config)
        
        # Load pretrained weights if available
        try:
            checkpoint = torch.load(f"{model_name_or_path}/pytorch_model.bin", map_location="cpu")
            model.load_state_dict(checkpoint, strict=False)
        except:
            print(f"Could not load pretrained weights from {model_name_or_path}")
        
        return cls(feature_extractor=feature_extractor, model=model)

    def save_pretrained(self, save_directory: str):
        """
        Save processor to directory.
        
        Args:
            save_directory: Directory to save the processor
        """
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model weights
        torch.save(self.model.state_dict(), f"{save_directory}/pytorch_model.bin")
        
        # Save configuration
        import json
        config_dict = {
            "audio_sample_rate": self.feature_extractor.audio_sample_rate,
            "video_size": self.feature_extractor.video_size,
            "video_mean": self.feature_extractor.video_mean,
            "video_std": self.feature_extractor.video_std,
            "do_normalize": self.feature_extractor.do_normalize,
            "do_center_crop": self.feature_extractor.do_center_crop,
            "do_random_crop": self.feature_extractor.do_random_crop,
            "do_horizontal_flip": self.feature_extractor.do_horizontal_flip,
            "flip_ratio": self.feature_extractor.flip_ratio,
        }
        
        with open(f"{save_directory}/config.json", "w") as f:
            json.dump(config_dict, f, indent=2)