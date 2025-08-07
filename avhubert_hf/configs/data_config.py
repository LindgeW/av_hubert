from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    
    # Data paths
    data_path: str = field(metadata={"help": "path to data directory"})
    label_dir: Optional[str] = field(default=None, metadata={"help": "if set, looks for labels in this directory instead"})
    
    # Labels
    labels: List[str] = field(default_factory=lambda: ["ltr"], metadata={"help": "extension of the label files to load"})
    label_rate: int = field(default=-1, metadata={"help": "label frame rate. -1 for sequence label"})
    
    # Audio settings
    sample_rate: int = field(default=16000, metadata={"help": "target sample rate"})
    normalize: bool = field(default=False, metadata={"help": "if set, normalizes input to have 0 mean and unit variance"})
    enable_padding: bool = field(default=False, metadata={"help": "pad shorter samples instead of cropping"})
    max_sample_size: Optional[int] = field(default=None, metadata={"help": "max sample size to keep in training"})
    min_sample_size: Optional[int] = field(default=None, metadata={"help": "min sample size to keep in training"})
    max_trim_sample_size: Optional[int] = field(default=None, metadata={"help": "max sample size to trim to for batching"})
    
    # Processing
    single_target: bool = field(default=False, metadata={"help": "if set, AddTargetDatasets outputs same keys as AddTargetDataset"})
    random_crop: bool = field(default=True, metadata={"help": "always crop from the beginning if false"})
    pad_audio: bool = field(default=False, metadata={"help": "pad audio to the longest one in the batch if true"})
    stack_order_audio: int = field(default=1, metadata={"help": "concatenate n consecutive audio frames for one step"})
    skip_verify: bool = field(default=False, metadata={"help": "skip verifying label-audio alignment"})
    
    # Image settings
    image_aug: bool = field(default=False, metadata={"help": "image data augmentation"})
    image_crop_size: int = field(default=88, metadata={"help": "image ROI size"})
    image_mean: float = field(default=0.421, metadata={"help": "image mean"})
    image_std: float = field(default=0.165, metadata={"help": "image std"})
    
    # Modalities
    modalities: List[str] = field(default_factory=lambda: ["audio", "video"], metadata={"help": "modalities to load"})
    
    # Seq2Seq settings
    is_s2s: bool = field(default=False, metadata={"help": "seq2seq fine-tuning only"})
    tokenizer_bpe_name: Optional[str] = field(default=None, metadata={"help": "tokenizer model name"})
    tokenizer_bpe_model: Optional[str] = field(default=None, metadata={"help": "tokenizer model path"})
    
    # Noise settings
    noise_wav: Optional[str] = field(default=None, metadata={"help": "manifest of noise wav files"})
    noise_prob: float = field(default=0, metadata={"help": "noise probability"})
    noise_snr: str = field(default='0', metadata={"help": "noise SNR in audio"})
    noise_num: int = field(default=1, metadata={"help": "number of noise wav files to mix"})
    
    # Fine-tuning
    fine_tuning: bool = field(default=False, metadata={"help": "set to true if fine-tuning AV-Hubert"})
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_trim_sample_size is None:
            self.max_trim_sample_size = self.max_sample_size