"""
AV-HuBERT: Audio-Visual Hidden Unit BERT

A self-supervised representation learning framework for audio-visual speech.
This is a refactored version that removes fairseq dependencies and provides
a clean, Hugging Face-style interface.
"""

from .models import AVHubertModel, ResEncoder, TransformerEncoder, TransformerDecoder
from .configs import AVHubertConfig, DataConfig, TrainingConfig
from .data import AVHubertDataset, AVHubertCollator, create_dataloader
from .training import AVHubertTrainer, AVHubertCriterion
from .utils import Dictionary, compute_mask_indices, get_activation_fn, PositionalEncoding

__version__ = "1.0.0"

__all__ = [
    # Models
    "AVHubertModel",
    "ResEncoder", 
    "TransformerEncoder",
    "TransformerDecoder",
    
    # Configs
    "AVHubertConfig",
    "DataConfig",
    "TrainingConfig",
    
    # Data
    "AVHubertDataset",
    "AVHubertCollator",
    "create_dataloader",
    
    # Training
    "AVHubertTrainer",
    "AVHubertCriterion",
    
    # Utils
    "Dictionary",
    "compute_mask_indices",
    "get_activation_fn",
    "PositionalEncoding",
]