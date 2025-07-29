"""
Preprocessing utilities for AVHuBERT audio and video features.
Simplified version without fairseq dependencies.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, Tuple, Union, List
import logging

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logging.warning("librosa not available, audio preprocessing will be limited")

try:
    from python_speech_features import mfcc
    SPEECH_FEATURES_AVAILABLE = True
except ImportError:
    SPEECH_FEATURES_AVAILABLE = False
    logging.warning("python_speech_features not available, using alternative MFCC computation")

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Audio preprocessing for AVHuBERT"""
    
    def __init__(
        self, 
        sample_rate: int = 16000,
        n_mfcc: int = 80,
        normalize: bool = False,
        apply_cmvn: bool = True,
    ):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.normalize = normalize
        self.apply_cmvn = apply_cmvn
        
    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """Extract MFCC features from audio"""
        if SPEECH_FEATURES_AVAILABLE:
            # Use python_speech_features (preferred)
            features = mfcc(
                audio, 
                samplerate=self.sample_rate,
                numcep=self.n_mfcc,
                nfilt=self.n_mfcc,
                nfft=512,
                lowfreq=0,
                highfreq=None,
                preemph=0.97,
                ceplifter=22,
                appendEnergy=False
            )
        elif LIBROSA_AVAILABLE:
            # Use librosa as fallback
            features = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=512,
                hop_length=160,  # 10ms hop
                n_mels=self.n_mfcc
            ).T  # Transpose to (time, features)
        else:
            raise ImportError("Either python_speech_features or librosa is required for MFCC extraction")
        
        return features.astype(np.float32)
    
    def apply_cmvn_normalization(self, features: np.ndarray) -> np.ndarray:
        """Apply Cepstral Mean and Variance Normalization"""
        if not self.apply_cmvn:
            return features
            
        # Compute mean and std across time dimension
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True)
        
        # Avoid division by zero
        std = np.where(std == 0, 1.0, std)
        
        # Normalize
        features = (features - mean) / std
        
        return features
    
    def pad_or_trim(self, features: np.ndarray, target_length: Optional[int] = None) -> np.ndarray:
        """Pad or trim features to target length"""
        if target_length is None:
            return features
            
        current_length = features.shape[0]
        
        if current_length < target_length:
            # Pad with zeros
            pad_width = ((0, target_length - current_length), (0, 0))
            features = np.pad(features, pad_width, mode='constant', constant_values=0)
        elif current_length > target_length:
            # Trim from the end
            features = features[:target_length]
        
        return features
    
    def preprocess(
        self, 
        audio: Union[np.ndarray, torch.Tensor],
        target_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Complete audio preprocessing pipeline
        
        Args:
            audio: Raw audio waveform (1D array) or path to audio file
            target_length: Target sequence length for padding/trimming
            
        Returns:
            Processed MFCC features as torch.Tensor of shape (seq_len, n_mfcc)
        """
        # Convert to numpy if needed
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
        
        # Load audio file if path is provided
        if isinstance(audio, str):
            if not LIBROSA_AVAILABLE:
                raise ImportError("librosa is required to load audio files")
            audio, _ = librosa.load(audio, sr=self.sample_rate)
        
        # Normalize audio
        if self.normalize:
            audio = audio / (np.abs(audio).max() + 1e-8)
        
        # Extract MFCC features
        features = self.extract_mfcc(audio)
        
        # Apply CMVN
        features = self.apply_cmvn_normalization(features)
        
        # Pad or trim to target length
        features = self.pad_or_trim(features, target_length)
        
        # Convert to tensor
        return torch.from_numpy(features)


class VideoPreprocessor:
    """Video preprocessing for AVHuBERT"""
    
    def __init__(
        self,
        image_size: int = 88,
        image_mean: float = 0.421,
        image_std: float = 0.165,
        crop_size: Optional[int] = None,
        apply_augmentation: bool = False,
    ):
        self.image_size = image_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.crop_size = crop_size or image_size
        self.apply_augmentation = apply_augmentation
        
    def resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to target size"""
        return cv2.resize(frame, (self.image_size, self.image_size))
    
    def center_crop(self, frame: np.ndarray) -> np.ndarray:
        """Apply center crop to frame"""
        h, w = frame.shape[:2]
        
        # Calculate crop coordinates
        start_x = (w - self.crop_size) // 2
        start_y = (h - self.crop_size) // 2
        
        return frame[start_y:start_y + self.crop_size, start_x:start_x + self.crop_size]
    
    def normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Normalize frame pixel values"""
        # Convert to float and normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        
        # Apply mean and std normalization
        frame = (frame - self.image_mean) / self.image_std
        
        return frame
    
    def apply_random_augmentation(self, frame: np.ndarray) -> np.ndarray:
        """Apply random augmentations during training"""
        if not self.apply_augmentation:
            return frame
        
        # Random horizontal flip
        if np.random.random() > 0.5:
            frame = cv2.flip(frame, 1)
        
        # Random brightness adjustment
        if np.random.random() > 0.5:
            brightness_factor = np.random.uniform(0.8, 1.2)
            frame = np.clip(frame * brightness_factor, 0, 255)
        
        # Random crop (slight variations)
        if np.random.random() > 0.5:
            h, w = frame.shape[:2]
            max_offset = min(h, w) // 20  # 5% of image size
            offset_x = np.random.randint(-max_offset, max_offset + 1)
            offset_y = np.random.randint(-max_offset, max_offset + 1)
            
            # Apply offset crop
            start_x = max(0, min(w - self.crop_size, (w - self.crop_size) // 2 + offset_x))
            start_y = max(0, min(h - self.crop_size, (h - self.crop_size) // 2 + offset_y))
            
            frame = frame[start_y:start_y + self.crop_size, start_x:start_x + self.crop_size]
        
        return frame
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess a single video frame"""
        # Resize
        frame = self.resize_frame(frame)
        
        # Apply augmentation
        frame = self.apply_random_augmentation(frame)
        
        # Center crop
        if frame.shape[0] != self.crop_size or frame.shape[1] != self.crop_size:
            frame = self.center_crop(frame)
        
        # Normalize
        frame = self.normalize_frame(frame)
        
        return frame
    
    def preprocess(
        self, 
        video: Union[np.ndarray, str, List[np.ndarray]],
        target_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Complete video preprocessing pipeline
        
        Args:
            video: Video frames as (T, H, W, C) array, path to video file, or list of frames
            target_length: Target sequence length for padding/trimming
            
        Returns:
            Processed video as torch.Tensor of shape (T, C, H, W)
        """
        # Load video file if path is provided
        if isinstance(video, str):
            video = self.load_video_file(video)
        
        # Convert list to array
        if isinstance(video, list):
            video = np.stack(video, axis=0)
        
        # Ensure 4D array (T, H, W, C)
        if video.ndim == 3:
            video = video[None, ...]  # Add time dimension
        
        processed_frames = []
        
        for i in range(video.shape[0]):
            frame = video[i]
            
            # Convert grayscale to RGB if needed
            if frame.ndim == 2:
                frame = np.stack([frame] * 3, axis=-1)
            elif frame.shape[-1] == 1:
                frame = np.repeat(frame, 3, axis=-1)
            
            # Preprocess frame
            processed_frame = self.preprocess_frame(frame)
            processed_frames.append(processed_frame)
        
        # Stack frames
        video_tensor = np.stack(processed_frames, axis=0)
        
        # Pad or trim to target length
        if target_length is not None:
            video_tensor = self.pad_or_trim_video(video_tensor, target_length)
        
        # Convert to tensor and rearrange to (T, C, H, W)
        video_tensor = torch.from_numpy(video_tensor)
        if video_tensor.ndim == 4:  # (T, H, W, C)
            video_tensor = video_tensor.permute(0, 3, 1, 2)  # (T, C, H, W)
        
        return video_tensor
    
    def load_video_file(self, video_path: str) -> np.ndarray:
        """Load video file using OpenCV"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        if not frames:
            raise ValueError(f"No frames found in video: {video_path}")
        
        return np.stack(frames, axis=0)
    
    def pad_or_trim_video(self, video: np.ndarray, target_length: int) -> np.ndarray:
        """Pad or trim video to target length"""
        current_length = video.shape[0]
        
        if current_length < target_length:
            # Pad by repeating last frame
            last_frame = video[-1:] if current_length > 0 else np.zeros((1,) + video.shape[1:])
            padding = np.repeat(last_frame, target_length - current_length, axis=0)
            video = np.concatenate([video, padding], axis=0)
        elif current_length > target_length:
            # Trim from the end
            video = video[:target_length]
        
        return video


class AVHubertPreprocessor:
    """Combined audio-visual preprocessor for AVHuBERT"""
    
    def __init__(
        self,
        audio_config: Optional[dict] = None,
        video_config: Optional[dict] = None,
    ):
        # Initialize audio preprocessor
        audio_config = audio_config or {}
        self.audio_preprocessor = AudioPreprocessor(**audio_config)
        
        # Initialize video preprocessor  
        video_config = video_config or {}
        self.video_preprocessor = VideoPreprocessor(**video_config)
    
    def __call__(
        self,
        audio: Optional[Union[np.ndarray, torch.Tensor, str]] = None,
        video: Optional[Union[np.ndarray, str, List[np.ndarray]]] = None,
        audio_target_length: Optional[int] = None,
        video_target_length: Optional[int] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Preprocess audio and/or video for AVHuBERT
        
        Args:
            audio: Raw audio or path to audio file
            video: Video frames or path to video file
            audio_target_length: Target length for audio features
            video_target_length: Target length for video frames
            
        Returns:
            Tuple of (processed_audio, processed_video)
        """
        processed_audio = None
        processed_video = None
        
        if audio is not None:
            processed_audio = self.audio_preprocessor.preprocess(
                audio, target_length=audio_target_length
            )
        
        if video is not None:
            processed_video = self.video_preprocessor.preprocess(
                video, target_length=video_target_length
            )
        
        return processed_audio, processed_video
    
    def create_padding_mask(
        self, 
        audio: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Create padding mask for processed features"""
        # For now, return None (no padding mask)
        # In practice, you'd track the original lengths and create appropriate masks
        return None


# Factory functions
def create_audio_preprocessor(
    sample_rate: int = 16000,
    n_mfcc: int = 80,
    normalize: bool = False,
    apply_cmvn: bool = True,
) -> AudioPreprocessor:
    """Create audio preprocessor with specified parameters"""
    return AudioPreprocessor(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        normalize=normalize,
        apply_cmvn=apply_cmvn,
    )


def create_video_preprocessor(
    image_size: int = 88,
    image_mean: float = 0.421,
    image_std: float = 0.165,
    apply_augmentation: bool = False,
) -> VideoPreprocessor:
    """Create video preprocessor with specified parameters"""
    return VideoPreprocessor(
        image_size=image_size,
        image_mean=image_mean,
        image_std=image_std,
        apply_augmentation=apply_augmentation,
    )


def create_avhubert_preprocessor(
    audio_config: Optional[dict] = None,
    video_config: Optional[dict] = None,
) -> AVHubertPreprocessor:
    """Create combined AVHuBERT preprocessor"""
    return AVHubertPreprocessor(
        audio_config=audio_config,
        video_config=video_config,
    )