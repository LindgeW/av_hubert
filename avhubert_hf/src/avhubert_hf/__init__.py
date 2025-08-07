"""
AV-HuBERT: Audio-Visual Hidden Unit BERT
A HuggingFace-compatible implementation of AV-HuBERT for self-supervised audio-visual speech representation learning.
"""

__version__ = "0.1.0"

from .models import (
    AVHubertConfig,
    AVHubertModel,
    AVHubertForSequenceClassification,
    AVHubertForCTC,
)
from .data import AVHubertProcessor, AVHubertDataset
from .utils import compute_mask_indices

__all__ = [
    "AVHubertConfig",
    "AVHubertModel", 
    "AVHubertForSequenceClassification",
    "AVHubertForCTC",
    "AVHubertProcessor",
    "AVHubertDataset",
    "compute_mask_indices",
]