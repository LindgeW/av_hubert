"""
Dataset implementation for AV-HuBERT
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging

from ..utils.audio_utils import AudioProcessor
from ..utils.video_utils import VideoTransforms, load_video
from ..utils.mask_utils import compute_mask_indices

logger = logging.getLogger(__name__)


class AVHubertDataset(Dataset):
    """
    Dataset for AV-HuBERT that handles audio-visual data for pretraining and fine-tuning.
    
    This dataset can handle:
    1. Pretraining with masked language modeling
    2. Fine-tuning for downstream tasks (ASR, lip reading, etc.)
    3. Both audio-only, video-only, and audio-visual modalities
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        manifest_path: Optional[str] = None,
        split: str = "train",
        # Audio parameters
        audio_sr: int = 16000,
        n_mfcc: int = 13,
        stack_order: int = 4,
        normalize_audio: bool = True,
        # Video parameters
        video_size: tuple = (96, 96),
        crop_size: tuple = (88, 88),
        video_mean: float = 0.5,
        video_std: float = 0.5,
        target_fps: int = 25,
        # Masking parameters (for pretraining)
        mask_prob_audio: float = 0.8,
        mask_length_audio: int = 10,
        mask_prob_video: float = 0.8,
        mask_length_video: int = 10,
        # General parameters
        max_length: Optional[int] = None,
        modalities: List[str] = ["audio", "video"],
        task_type: str = "pretraining",  # "pretraining", "asr", "lip_reading"
        return_attention_mask: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_length = max_length
        self.modalities = modalities
        self.task_type = task_type
        self.return_attention_mask = return_attention_mask
        
        # Initialize processors
        self.audio_processor = AudioProcessor(
            sr=audio_sr,
            n_mfcc=n_mfcc,
            stack_order=stack_order,
            normalize=normalize_audio,
        )
        
        self.video_processor = VideoTransforms(
            target_size=video_size,
            crop_size=crop_size,
            mean=video_mean,
            std=video_std,
            target_fps=target_fps,
        )
        
        # Masking parameters
        self.mask_prob_audio = mask_prob_audio
        self.mask_length_audio = mask_length_audio
        self.mask_prob_video = mask_prob_video
        self.mask_length_video = mask_length_video
        
        # Load manifest
        self.manifest = self._load_manifest(manifest_path)
        
        logger.info(f"Loaded {len(self.manifest)} samples for {split} split")
    
    def _load_manifest(self, manifest_path: Optional[str]) -> pd.DataFrame:
        """Load dataset manifest file."""
        if manifest_path is None:
            # Default manifest path
            manifest_path = self.data_dir / f"{self.split}.tsv"
        
        if not Path(manifest_path).exists():
            # Create dummy manifest for demonstration
            logger.warning(f"Manifest {manifest_path} not found. Creating dummy data.")
            return self._create_dummy_manifest()
        
        # Load TSV manifest (AV-HuBERT format)
        manifest = pd.read_csv(manifest_path, sep='\t')
        
        # Ensure required columns exist
        required_cols = ['audio_path']
        if "video" in self.modalities:
            required_cols.append('video_path')
        
        for col in required_cols:
            if col not in manifest.columns:
                raise ValueError(f"Required column '{col}' not found in manifest")
        
        return manifest
    
    def _create_dummy_manifest(self) -> pd.DataFrame:
        """Create dummy manifest for testing purposes."""
        num_samples = 1000
        data = {
            'audio_path': [f"dummy_audio_{i}.wav" for i in range(num_samples)],
            'video_path': [f"dummy_video_{i}.mp4" for i in range(num_samples)],
            'duration': np.random.uniform(1.0, 10.0, num_samples),
            'text': [f"dummy text {i}" for i in range(num_samples)] if self.task_type != "pretraining" else [None] * num_samples,
        }
        return pd.DataFrame(data)
    
    def __len__(self) -> int:
        return len(self.manifest)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample from the dataset."""
        row = self.manifest.iloc[idx]
        sample = {}
        
        # Load and process audio
        if "audio" in self.modalities:
            audio_features = self._load_audio_features(row)
            sample['audio_features'] = audio_features
        
        # Load and process video
        if "video" in self.modalities:
            video_features = self._load_video_features(row)
            sample['video_features'] = video_features
        
        # Determine sequence length (use audio length as reference)
        if "audio" in self.modalities:
            seq_len = sample['audio_features'].shape[0]
        elif "video" in self.modalities:
            seq_len = sample['video_features'].shape[0]
        else:
            seq_len = 100  # Default
        
        # Apply length constraints
        if self.max_length and seq_len > self.max_length:
            seq_len = self.max_length
            if "audio" in self.modalities:
                sample['audio_features'] = sample['audio_features'][:seq_len]
            if "video" in self.modalities:
                sample['video_features'] = sample['video_features'][:seq_len]
        
        # Create attention mask
        if self.return_attention_mask:
            sample['attention_mask'] = torch.ones(seq_len, dtype=torch.bool)
        
        # Add task-specific labels/targets
        if self.task_type == "pretraining":
            # For pretraining, create masked targets
            sample.update(self._create_pretraining_targets(sample, seq_len))
        elif self.task_type in ["asr", "lip_reading"]:
            # For downstream tasks, add text labels
            if pd.notna(row.get('text')):
                sample['labels'] = self._encode_text(row['text'])
        
        return sample
    
    def _load_audio_features(self, row: pd.Series) -> torch.Tensor:
        """Load and process audio features."""
        audio_path = row['audio_path']
        
        # Check if it's a dummy path
        if audio_path.startswith('dummy_'):
            # Generate dummy MFCC features
            seq_len = int(np.random.uniform(50, 200))
            features = torch.randn(seq_len, 104)  # 13 MFCC * 8 (with deltas and stacking)
            return features
        
        # Load real audio
        full_path = self.data_dir / audio_path
        if not full_path.exists():
            logger.warning(f"Audio file {full_path} not found. Using dummy features.")
            seq_len = int(np.random.uniform(50, 200))
            return torch.randn(seq_len, 104)
        
        try:
            features = self.audio_processor(str(full_path))
            return features
        except Exception as e:
            logger.error(f"Error loading audio {full_path}: {e}")
            seq_len = int(np.random.uniform(50, 200))
            return torch.randn(seq_len, 104)
    
    def _load_video_features(self, row: pd.Series) -> torch.Tensor:
        """Load and process video features."""
        video_path = row['video_path']
        
        # Check if it's a dummy path
        if video_path.startswith('dummy_'):
            # Generate dummy video features
            seq_len = int(np.random.uniform(20, 100))
            features = torch.randn(seq_len, 512)  # ResNet features
            return features
        
        # Load real video
        full_path = self.data_dir / video_path
        if not full_path.exists():
            logger.warning(f"Video file {full_path} not found. Using dummy features.")
            seq_len = int(np.random.uniform(20, 100))
            return torch.randn(seq_len, 512)
        
        try:
            # Load video frames
            frames = load_video(str(full_path))
            
            # Apply video transforms
            processed_frames = self.video_processor(frames)
            
            # For this implementation, we'll return dummy ResNet features
            # In a real implementation, you'd pass through the ResNet encoder
            seq_len = processed_frames.shape[1]  # temporal dimension
            features = torch.randn(seq_len, 512)  # Simulated ResNet features
            
            return features
        except Exception as e:
            logger.error(f"Error loading video {full_path}: {e}")
            seq_len = int(np.random.uniform(20, 100))
            return torch.randn(seq_len, 512)
    
    def _create_pretraining_targets(self, sample: Dict[str, torch.Tensor], seq_len: int) -> Dict[str, torch.Tensor]:
        """Create masked targets for pretraining."""
        targets = {}
        
        # Create audio masks if audio is present
        if "audio" in self.modalities and 'audio_features' in sample:
            audio_mask = compute_mask_indices(
                shape=(1, seq_len),
                mask_prob=self.mask_prob_audio,
                mask_length=self.mask_length_audio,
            )[0]  # Remove batch dimension
            targets['audio_mask'] = torch.from_numpy(audio_mask)
            
            # Create dummy cluster targets (in real implementation, these would be k-means clusters)
            targets['audio_targets'] = torch.randint(0, 32, (seq_len,))
        
        # Create video masks if video is present
        if "video" in self.modalities and 'video_features' in sample:
            video_mask = compute_mask_indices(
                shape=(1, seq_len),
                mask_prob=self.mask_prob_video,
                mask_length=self.mask_length_video,
            )[0]  # Remove batch dimension
            targets['video_mask'] = torch.from_numpy(video_mask)
            
            # Create dummy cluster targets
            targets['video_targets'] = torch.randint(0, 32, (seq_len,))
        
        return targets
    
    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode text for downstream tasks."""
        # This is a simplified encoding - in practice you'd use a proper tokenizer
        # For now, just create dummy token IDs
        words = text.split()
        token_ids = torch.randint(0, 1000, (len(words),))
        return token_ids
    
    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate function for DataLoader."""
        # Find maximum sequence length in batch
        max_len = max(
            sample['audio_features'].shape[0] if 'audio_features' in sample else 0
            for sample in batch
        )
        
        if max_len == 0:
            max_len = max(
                sample['video_features'].shape[0] if 'video_features' in sample else 0
                for sample in batch
            )
        
        batch_size = len(batch)
        collated = {}
        
        # Collate audio features
        if 'audio_features' in batch[0]:
            audio_dim = batch[0]['audio_features'].shape[-1]
            audio_batch = torch.zeros(batch_size, max_len, audio_dim)
            for i, sample in enumerate(batch):
                seq_len = sample['audio_features'].shape[0]
                audio_batch[i, :seq_len] = sample['audio_features']
            collated['audio_features'] = audio_batch
        
        # Collate video features
        if 'video_features' in batch[0]:
            video_dim = batch[0]['video_features'].shape[-1]
            video_batch = torch.zeros(batch_size, max_len, video_dim)
            for i, sample in enumerate(batch):
                seq_len = sample['video_features'].shape[0]
                video_batch[i, :seq_len] = sample['video_features']
            collated['video_features'] = video_batch
        
        # Collate attention masks
        if 'attention_mask' in batch[0]:
            mask_batch = torch.zeros(batch_size, max_len, dtype=torch.bool)
            for i, sample in enumerate(batch):
                seq_len = sample['attention_mask'].shape[0]
                mask_batch[i, :seq_len] = sample['attention_mask']
            collated['attention_mask'] = mask_batch
        
        # Collate other tensors (masks, targets, labels)
        for key in ['audio_mask', 'video_mask', 'audio_targets', 'video_targets']:
            if key in batch[0]:
                tensor_batch = torch.zeros(batch_size, max_len, dtype=batch[0][key].dtype)
                for i, sample in enumerate(batch):
                    seq_len = sample[key].shape[0]
                    tensor_batch[i, :seq_len] = sample[key]
                collated[key] = tensor_batch
        
        # Handle variable-length labels
        if 'labels' in batch[0]:
            labels = [sample['labels'] for sample in batch]
            # For simplicity, just pad to max label length
            max_label_len = max(len(label) for label in labels)
            label_batch = torch.full((batch_size, max_label_len), -100, dtype=torch.long)
            for i, label in enumerate(labels):
                label_batch[i, :len(label)] = label
            collated['labels'] = label_batch
        
        return collated


class AVHubertPretrainingDataset(AVHubertDataset):
    """Specialized dataset for AV-HuBERT pretraining."""
    
    def __init__(self, *args, **kwargs):
        kwargs['task_type'] = 'pretraining'
        super().__init__(*args, **kwargs)


class AVHubertASRDataset(AVHubertDataset):
    """Specialized dataset for ASR fine-tuning."""
    
    def __init__(self, *args, **kwargs):
        kwargs['task_type'] = 'asr'
        kwargs['modalities'] = ['audio']
        super().__init__(*args, **kwargs)


class AVHubertLipReadingDataset(AVHubertDataset):
    """Specialized dataset for lip reading fine-tuning."""
    
    def __init__(self, *args, **kwargs):
        kwargs['task_type'] = 'lip_reading'
        kwargs['modalities'] = ['video']
        super().__init__(*args, **kwargs)