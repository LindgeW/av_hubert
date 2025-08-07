import logging
import os
import time
from typing import Any, List, Optional, Union, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from python_speech_features import logfbank
from scipy.io import wavfile
import cv2

from ..configs.data_config import DataConfig
from ..utils.dictionary import Dictionary

logger = logging.getLogger(__name__)


def load_audio_visual(
    manifest_path: str,
    max_keep: Optional[int],
    min_keep: Optional[int],
    frame_rate: float,
    label_paths: List[str],
    label_rates: List[float],
    tol: float = 0.1,
) -> Tuple[str, List[Tuple[str, str]], List[int], int, List[int]]:
    """Load audio-visual data from manifest file."""
    
    def is_audio_label_aligned(audio_dur: float, label_durs: List[float]) -> bool:
        return all([abs(audio_dur - label_dur) < tol for label_dur in label_durs])
    
    n_long, n_short, n_unaligned = 0, 0, 0
    names, inds, sizes = [], [], []
    dur_from_label_list = []
    is_seq_label = any([x == -1 for x in label_rates])
    
    for label_path, label_rate in zip(label_paths, label_rates):
        label_lengths = [len(line.rstrip().split()) / label_rate for line in open(label_path).readlines()]
        dur_from_label_list.append(label_lengths)
    dur_from_label_list = list(zip(*dur_from_label_list))
    
    with open(manifest_path) as f:
        root = f.readline().strip()
        for ind, line in enumerate(f):
            items = line.strip().split("\t")
            sz = int(items[-2])
            if min_keep is not None and sz < min_keep:
                n_short += 1
            elif max_keep is not None and sz > max_keep:
                n_long += 1
            elif (not is_seq_label) and (not is_audio_label_aligned(sz / frame_rate, dur_from_label_list[ind])):
                n_unaligned += 1
            else:
                video_path = items[1]
                audio_path = items[2]
                audio_id = items[0]
                names.append((video_path, audio_path + ':' + audio_id))
                inds.append(ind)
                sizes.append(sz)
    
    tot = ind + 1
    logger.info(
        f"max_keep={max_keep}, min_keep={min_keep}, "
        f"loaded {len(names)}, skipped {n_short} short and {n_long} long and {n_unaligned} unaligned, "
        f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
    )
    return root, names, inds, tot, sizes


def load_label(label_path: str, inds: List[int], tot: int) -> List[str]:
    """Load labels from file."""
    with open(label_path) as f:
        labels = [line.rstrip() for line in f]
        assert len(labels) == tot, f"number of labels does not match ({len(labels)} != {tot})"
        labels = [labels[i] for i in inds]
    return labels


class AVHubertDataset(Dataset):
    """Dataset for AV-HuBERT training and evaluation."""
    
    def __init__(
        self,
        config: DataConfig,
        split: str = "train",
        dictionaries: Optional[List[Dictionary]] = None,
    ):
        self.config = config
        self.split = split
        self.dictionaries = dictionaries or []
        
        # Load data
        manifest_path = os.path.join(config.data_path, f"{split}.tsv")
        label_paths = [os.path.join(config.label_dir or config.data_path, f"{split}.{label}") for label in config.labels]
        
        self.root, self.names, self.inds, self.tot, self.sizes = load_audio_visual(
            manifest_path=manifest_path,
            max_keep=config.max_sample_size,
            min_keep=config.min_sample_size,
            frame_rate=config.sample_rate,
            label_paths=label_paths,
            label_rates=[config.label_rate] * len(config.labels),
        )
        
        # Load labels
        self.labels = []
        for label_path in label_paths:
            self.labels.append(load_label(label_path, self.inds, self.tot))
        
        # Load tokenizer if needed
        self.tokenizer = None
        if config.is_s2s and config.tokenizer_bpe_model:
            import sentencepiece as spm
            self.tokenizer = spm.SentencePieceProcessor()
            self.tokenizer.load(config.tokenizer_bpe_model)
        
        # Noise settings
        self.noise_fn = None
        if config.noise_wav and config.noise_prob > 0:
            self.noise_fn = self._load_noise_data(config.noise_wav)
    
    def _load_noise_data(self, noise_wav_path: str) -> Dict[str, List[str]]:
        """Load noise data for augmentation."""
        noise_data = {}
        for split in ["valid", "test"]:
            noise_manifest = os.path.join(noise_wav_path, f"{split}.tsv")
            if os.path.exists(noise_manifest):
                with open(noise_manifest) as f:
                    noise_data[split] = [line.strip().split("\t")[0] for line in f]
        return noise_data
    
    def __len__(self) -> int:
        return len(self.names)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get a single sample."""
        video_path, audio_path = self.names[index]
        audio_path, audio_id = audio_path.split(":", 1)
        
        # Load audio
        audio = self._load_audio(os.path.join(self.root, audio_path))
        
        # Load video
        video = self._load_video(os.path.join(self.root, video_path))
        
        # Load labels
        labels = []
        for label_list in self.labels:
            labels.append(label_list[index])
        
        # Apply noise if configured
        if self.noise_fn and np.random.random() < self.config.noise_prob:
            audio = self._add_noise(audio)
        
        return {
            "id": audio_id,
            "audio": audio,
            "video": video,
            "labels": labels,
            "audio_path": audio_path,
            "video_path": video_path,
        }
    
    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio."""
        sample_rate, audio = wavfile.read(audio_path)
        
        # Resample if needed
        if sample_rate != self.config.sample_rate:
            # Simple resampling - in practice, you'd want to use librosa or similar
            ratio = self.config.sample_rate / sample_rate
            audio = audio[::int(1/ratio)]
        
        # Convert to float
        audio = audio.astype(np.float32) / 32768.0
        
        # Extract features
        if self.config.stack_order_audio > 1:
            audio = self._stack_audio_frames(audio)
        else:
            # Extract MFCC features
            audio = logfbank(audio, sample_rate=self.config.sample_rate, nfilt=80)
        
        return torch.from_numpy(audio).float()
    
    def _load_video(self, video_path: str) -> torch.Tensor:
        """Load and preprocess video."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale and resize
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (self.config.image_crop_size, self.config.image_crop_size))
            
            # Normalize
            frame = (frame.astype(np.float32) - self.config.image_mean * 255) / (self.config.image_std * 255)
            frames.append(frame)
        
        cap.release()
        
        if not frames:
            # Create dummy frame if video is empty
            frames = [np.zeros((self.config.image_crop_size, self.config.image_crop_size), dtype=np.float32)]
        
        return torch.from_numpy(np.stack(frames)).float()
    
    def _stack_audio_frames(self, audio: np.ndarray) -> np.ndarray:
        """Stack consecutive audio frames."""
        # This is a simplified implementation
        # In practice, you'd want to implement proper frame stacking
        return audio
    
    def _add_noise(self, audio: torch.Tensor) -> torch.Tensor:
        """Add noise to audio."""
        if not self.noise_fn:
            return audio
        
        # Select random noise file
        noise_split = np.random.choice(list(self.noise_fn.keys()))
        noise_file = np.random.choice(self.noise_fn[noise_split])
        
        # Load noise
        sample_rate, noise = wavfile.read(noise_file)
        noise = noise.astype(np.float32) / 32768.0
        
        # Resample noise if needed
        if sample_rate != self.config.sample_rate:
            ratio = self.config.sample_rate / sample_rate
            noise = noise[::int(1/ratio)]
        
        # Adjust noise length
        if len(noise) > len(audio):
            start = np.random.randint(0, len(noise) - len(audio))
            noise = noise[start:start + len(audio)]
        else:
            # Repeat noise if it's shorter
            noise = np.tile(noise, int(np.ceil(len(audio) / len(noise))))
            noise = noise[:len(audio)]
        
        # Mix audio and noise
        snr = float(self.config.noise_snr)
        noise_power = np.mean(noise ** 2)
        audio_power = np.mean(audio.numpy() ** 2)
        
        if noise_power > 0:
            noise_factor = np.sqrt(audio_power / (noise_power * (10 ** (snr / 10))))
            noise = noise * noise_factor
        
        mixed_audio = audio + torch.from_numpy(noise).float()
        return mixed_audio
    
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate function for batching."""
        # This is a simplified collate function
        # In practice, you'd want to implement proper padding and batching
        
        audio_list = [item["audio"] for item in batch]
        video_list = [item["video"] for item in batch]
        labels_list = [item["labels"] for item in batch]
        
        # Pad audio sequences
        max_audio_len = max(len(audio) for audio in audio_list)
        padded_audio = []
        for audio in audio_list:
            if len(audio) < max_audio_len:
                padding = torch.zeros(max_audio_len - len(audio), audio.size(-1))
                audio = torch.cat([audio, padding], dim=0)
            padded_audio.append(audio)
        
        # Pad video sequences
        max_video_len = max(len(video) for video in video_list)
        padded_video = []
        for video in video_list:
            if len(video) < max_video_len:
                padding = torch.zeros(max_video_len - len(video), *video.size()[1:])
                video = torch.cat([video, padding], dim=0)
            padded_video.append(video)
        
        return {
            "audio": torch.stack(padded_audio),
            "video": torch.stack(padded_video),
            "labels": labels_list,
            "audio_lengths": torch.tensor([len(audio) for audio in audio_list]),
            "video_lengths": torch.tensor([len(video) for video in video_list]),
        }