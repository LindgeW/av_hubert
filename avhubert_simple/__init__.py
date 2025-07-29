"""
Simplified AVHuBERT implementation with HuggingFace-style interface.
Removes fairseq dependency and provides a cleaner, more user-friendly API.
"""

from .config import AVHuBERTConfig
from .model import AVHuBERTModel
from .feature_extractor import AVHuBERTFeatureExtractor
from .processor import AVHuBERTProcessor

__version__ = "1.0.0"
__all__ = ["AVHuBERTModel", "AVHuBERTConfig", "AVHuBERTFeatureExtractor", "AVHuBERTProcessor"]