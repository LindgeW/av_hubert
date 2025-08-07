"""
Audio utility functions for AV-HuBERT
"""

import numpy as np
import torch
import librosa
from python_speech_features import mfcc
from typing import Union, Optional


def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    """
    Load audio from file.
    
    Args:
        path: Path to audio file
        sr: Target sample rate
        
    Returns:
        Audio waveform
    """
    audio, _ = librosa.load(path, sr=sr)
    return audio


def get_mfcc_features(
    audio: np.ndarray,
    sr: int = 16000,
    n_mfcc: int = 13,
    n_fft: int = 400,
    hop_length: int = 160,
    stack_order: int = 4,
) -> np.ndarray:
    """
    Extract MFCC features from audio.
    
    Args:
        audio: Audio waveform
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients
        n_fft: FFT window size
        hop_length: Hop length for STFT
        stack_order: Number of frames to stack
        
    Returns:
        MFCC features with shape (num_frames, n_mfcc * (1 + 2 * stack_order))
    """
    # Compute MFCC features
    mfcc_feats = mfcc(
        audio,
        samplerate=sr,
        winlen=n_fft / sr,
        winstep=hop_length / sr,
        numcep=n_mfcc,
        nfilt=26,
        nfft=n_fft,
        preemph=0.97,
        ceplifter=22,
        appendEnergy=False,
    )
    
    # Compute deltas and double-deltas
    delta1 = librosa.feature.delta(mfcc_feats.T, order=1).T
    delta2 = librosa.feature.delta(mfcc_feats.T, order=2).T
    
    # Stack features
    features = np.concatenate([mfcc_feats, delta1, delta2], axis=1)
    
    # Apply stacking if specified
    if stack_order > 1:
        stacked_features = []
        for i in range(0, len(features) - stack_order + 1, stack_order):
            stacked = features[i:i + stack_order].flatten()
            stacked_features.append(stacked)
        features = np.stack(stacked_features)
    
    return features


def normalize_audio(audio: np.ndarray, target_db: float = -25.0) -> np.ndarray:
    """
    Normalize audio to target dB level.
    
    Args:
        audio: Audio waveform
        target_db: Target dB level
        
    Returns:
        Normalized audio
    """
    # Calculate current RMS
    rms = np.sqrt(np.mean(audio ** 2))
    
    # Convert to dB
    current_db = 20 * np.log10(rms + 1e-8)
    
    # Calculate scaling factor
    scale = 10 ** ((target_db - current_db) / 20)
    
    return audio * scale


def add_noise(
    audio: np.ndarray,
    noise: Optional[np.ndarray] = None,
    snr_db: float = 20.0,
) -> np.ndarray:
    """
    Add noise to audio signal.
    
    Args:
        audio: Clean audio signal
        noise: Noise signal (if None, white noise is used)
        snr_db: Signal-to-noise ratio in dB
        
    Returns:
        Noisy audio
    """
    if noise is None:
        # Generate white noise
        noise = np.random.normal(0, 1, len(audio))
    else:
        # Repeat or truncate noise to match audio length
        if len(noise) < len(audio):
            reps = int(np.ceil(len(audio) / len(noise)))
            noise = np.tile(noise, reps)[:len(audio)]
        else:
            noise = noise[:len(audio)]
    
    # Calculate signal and noise power
    signal_power = np.mean(audio ** 2)
    noise_power = np.mean(noise ** 2)
    
    # Calculate noise scaling factor for desired SNR
    snr_linear = 10 ** (snr_db / 10)
    noise_scale = np.sqrt(signal_power / (snr_linear * noise_power))
    
    # Add scaled noise to signal
    return audio + noise_scale * noise


def apply_speed_perturbation(audio: np.ndarray, sr: int, speed_factor: float) -> np.ndarray:
    """
    Apply speed perturbation to audio.
    
    Args:
        audio: Audio waveform
        sr: Sample rate
        speed_factor: Speed factor (>1 = faster, <1 = slower)
        
    Returns:
        Speed-perturbed audio
    """
    return librosa.effects.time_stretch(audio, rate=speed_factor)


class AudioProcessor:
    """Audio preprocessing pipeline."""
    
    def __init__(
        self,
        sr: int = 16000,
        n_mfcc: int = 13,
        stack_order: int = 4,
        normalize: bool = True,
        target_db: float = -25.0,
    ):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.stack_order = stack_order
        self.normalize = normalize
        self.target_db = target_db
    
    def __call__(self, audio_path: str) -> torch.Tensor:
        """
        Process audio file to features.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Processed audio features as tensor
        """
        # Load audio
        audio = load_audio(audio_path, self.sr)
        
        # Normalize if requested
        if self.normalize:
            audio = normalize_audio(audio, self.target_db)
        
        # Extract MFCC features
        features = get_mfcc_features(
            audio,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            stack_order=self.stack_order,
        )
        
        return torch.from_numpy(features).float()