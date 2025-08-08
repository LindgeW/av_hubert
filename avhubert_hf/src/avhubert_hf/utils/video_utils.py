"""
Video utility functions for AV-HuBERT
"""

import cv2
import torch
import numpy as np
from typing import Union, Tuple


def load_video(path: str) -> np.ndarray:
    """
    Load video frames from file.
    
    Args:
        path: Path to video file
        
    Returns:
        numpy array of video frames with shape (num_frames, height, width)
    """
    for i in range(3):
        try:
            cap = cv2.VideoCapture(path)
            frames = []
            while True:
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frames.append(frame)
                else:
                    break
            cap.release()
            frames = np.stack(frames)
            return frames
        except Exception as e:
            print(f"Failed loading {path} ({i} / 3): {e}")
            if i == 2:
                raise ValueError(f"Unable to load {path}")


def resize_video(frames: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize video frames to target size.
    
    Args:
        frames: Video frames with shape (num_frames, height, width)
        target_size: Target (height, width)
        
    Returns:
        Resized frames
    """
    target_h, target_w = target_size
    resized_frames = []
    
    for frame in frames:
        resized_frame = cv2.resize(frame, (target_w, target_h))
        resized_frames.append(resized_frame)
    
    return np.stack(resized_frames)


def normalize_video(frames: np.ndarray, mean: float = 0.5, std: float = 0.5) -> np.ndarray:
    """
    Normalize video frames.
    
    Args:
        frames: Video frames
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        Normalized frames
    """
    frames = frames.astype(np.float32) / 255.0
    frames = (frames - mean) / std
    return frames


def center_crop_video(frames: np.ndarray, crop_size: Tuple[int, int]) -> np.ndarray:
    """
    Center crop video frames.
    
    Args:
        frames: Video frames with shape (num_frames, height, width)
        crop_size: Target crop (height, width)
        
    Returns:
        Center-cropped frames
    """
    t, h, w = frames.shape
    th, tw = crop_size
    delta_w = int(round((w - tw) / 2.))
    delta_h = int(round((h - th) / 2.))
    frames = frames[:, delta_h:delta_h+th, delta_w:delta_w+tw]
    return frames


def temporal_subsample(frames: np.ndarray, target_fps: int, original_fps: int = 25) -> np.ndarray:
    """
    Temporally subsample video frames.
    
    Args:
        frames: Video frames
        target_fps: Target frames per second
        original_fps: Original frames per second
        
    Returns:
        Subsampled frames
    """
    if target_fps >= original_fps:
        return frames
    
    step = original_fps // target_fps
    return frames[::step]


class VideoTransforms:
    """Video preprocessing transforms."""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (96, 96),
        crop_size: Tuple[int, int] = (88, 88),
        mean: float = 0.5,
        std: float = 0.5,
        target_fps: int = 25,
    ):
        self.target_size = target_size
        self.crop_size = crop_size
        self.mean = mean
        self.std = std
        self.target_fps = target_fps
    
    def __call__(self, frames: np.ndarray) -> torch.Tensor:
        """
        Apply preprocessing transforms to video frames.
        
        Args:
            frames: Raw video frames
            
        Returns:
            Preprocessed frames as torch tensor
        """
        # Resize
        frames = resize_video(frames, self.target_size)
        
        # Center crop
        frames = center_crop_video(frames, self.crop_size)
        
        # Normalize
        frames = normalize_video(frames, self.mean, self.std)
        
        # Convert to tensor and add channel dimension
        frames = torch.from_numpy(frames).unsqueeze(0)  # Add channel dim
        
        return frames