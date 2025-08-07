"""
AV-HuBERT: Audio-Visual Hidden Unit BERT

A self-supervised representation learning framework for audio-visual speech.
"""

from .configuration_avhubert import AVHubertConfig
from .modeling_avhubert import AVHubertModel, AVHubertForPreTraining, AVHubertForCTC
from .tokenization_avhubert import AVHubertTokenizer
from .feature_extraction_avhubert import AVHubertFeatureExtractor
from .processor_avhubert import AVHubertProcessor

__version__ = "0.1.0"

__all__ = [
    "AVHubertConfig",
    "AVHubertModel", 
    "AVHubertForPreTraining",
    "AVHubertForCTC",
    "AVHubertTokenizer",
    "AVHubertFeatureExtractor",
    "AVHubertProcessor",
]