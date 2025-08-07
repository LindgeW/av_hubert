"""
Data processor for AV-HuBERT
"""

import torch
import numpy as np
from typing import Union, List, Dict, Any, Optional
from pathlib import Path

from ..utils.audio_utils import AudioProcessor
from ..utils.video_utils import VideoTransforms


class AVHubertProcessor:
    """
    Processor for AV-HuBERT that handles both audio and video preprocessing.
    """
    
    def __init__(
        self,
        # Audio parameters
        audio_sr: int = 16000,
        n_mfcc: int = 13,
        stack_order: int = 4,
        normalize_audio: bool = True,
        target_db: float = -25.0,
        # Video parameters
        video_size: tuple = (96, 96),
        crop_size: tuple = (88, 88),
        video_mean: float = 0.5,
        video_std: float = 0.5,
        target_fps: int = 25,
        # General parameters
        return_tensors: str = "pt",
    ):
        self.audio_processor = AudioProcessor(
            sr=audio_sr,
            n_mfcc=n_mfcc,
            stack_order=stack_order,
            normalize=normalize_audio,
            target_db=target_db,
        )
        
        self.video_processor = VideoTransforms(
            target_size=video_size,
            crop_size=crop_size,
            mean=video_mean,
            std=video_std,
            target_fps=target_fps,
        )
        
        self.return_tensors = return_tensors
    
    def __call__(
        self,
        audio: Union[str, np.ndarray, torch.Tensor] = None,
        video: Union[str, np.ndarray, torch.Tensor] = None,
        sampling_rate: Optional[int] = None,
        return_tensors: Optional[str] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Process audio and/or video inputs.
        
        Args:
            audio: Audio input (file path, numpy array, or tensor)
            video: Video input (file path, numpy array, or tensor)
            sampling_rate: Audio sampling rate (if audio is raw waveform)
            return_tensors: Format to return tensors in
            
        Returns:
            Dictionary with processed audio and video features
        """
        return_tensors = return_tensors or self.return_tensors
        processed = {}
        
        # Process audio
        if audio is not None:
            if isinstance(audio, str):
                # Load from file
                audio_features = self.audio_processor(audio)
            elif isinstance(audio, (np.ndarray, torch.Tensor)):
                # Process raw audio
                if isinstance(audio, torch.Tensor):
                    audio = audio.numpy()
                
                # Extract MFCC features
                from ..utils.audio_utils import get_mfcc_features
                audio_features = get_mfcc_features(
                    audio,
                    sr=sampling_rate or self.audio_processor.sr,
                    n_mfcc=self.audio_processor.n_mfcc,
                    stack_order=self.audio_processor.stack_order,
                )
                audio_features = torch.from_numpy(audio_features).float()
            else:
                raise ValueError(f"Unsupported audio type: {type(audio)}")
            
            processed['audio_features'] = audio_features
        
        # Process video
        if video is not None:
            if isinstance(video, str):
                # Load from file
                from ..utils.video_utils import load_video
                video_frames = load_video(video)
                video_features = self.video_processor(video_frames)
            elif isinstance(video, (np.ndarray, torch.Tensor)):
                # Process raw video
                if isinstance(video, torch.Tensor):
                    video = video.numpy()
                video_features = self.video_processor(video)
            else:
                raise ValueError(f"Unsupported video type: {type(video)}")
            
            processed['video_features'] = video_features
        
        # Ensure tensors are in correct format
        if return_tensors == "pt":
            for key, value in processed.items():
                if not isinstance(value, torch.Tensor):
                    processed[key] = torch.tensor(value)
        
        return processed
    
    def batch_decode(self, sequences: torch.Tensor, **kwargs) -> List[str]:
        """
        Decode token sequences to text.
        
        Args:
            sequences: Token sequences to decode
            
        Returns:
            List of decoded strings
        """
        # This would implement actual decoding logic
        # For now, return placeholder
        return [f"decoded_sequence_{i}" for i in range(len(sequences))]
    
    def decode(self, token_ids: torch.Tensor, **kwargs) -> str:
        """
        Decode single token sequence to text.
        
        Args:
            token_ids: Token sequence to decode
            
        Returns:
            Decoded string
        """
        return self.batch_decode([token_ids], **kwargs)[0]