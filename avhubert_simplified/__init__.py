"""
Simplified AVHuBERT implementation.
HuggingFace-style interface without fairseq dependencies.

This package provides a clean, easy-to-use implementation of AVHuBERT
for audio-visual speech recognition tasks including:
- Lip reading (video-only)
- Audio speech recognition (audio-only)
- Audio-visual speech recognition (both modalities)

Example usage:
    # For lip reading
    >>> from avhubert_simplified import create_lip_reading_model
    >>> model = create_lip_reading_model()
    >>> result = model(video_frames)
    
    # For general purpose
    >>> from avhubert_simplified import AVHubertInference, AVHubertConfig
    >>> config = AVHubertConfig()
    >>> model = AVHubertInference(config=config)
    >>> result = model.predict(audio=audio_data, video=video_data)
"""

__version__ = "1.0.0"
__author__ = "Simplified by Assistant"

# Core model components
from .config import AVHubertConfig, AVHubertPretrainConfig
from .model import (
    AVHubertModel,
    SubModel,
    create_avhubert_base,
    create_avhubert_large,
)

# Inference interfaces
from .inference import (
    AVHubertInference,
    LipReadingInference,
    AudioSpeechRecognitionInference,
    AudioVisualSpeechRecognitionInference,
    create_lip_reading_model,
    create_asr_model,
    create_avsr_model,
    create_general_model,
)

# Preprocessing utilities
from .preprocessors import (
    AudioPreprocessor,
    VideoPreprocessor,
    AVHubertPreprocessor,
    create_audio_preprocessor,
    create_video_preprocessor,
    create_avhubert_preprocessor,
)

# Core components (for advanced users)
from .resnet import ResEncoder, create_resnet_encoder
from .transformer import (
    TransformerEncoder,
    TransformerEncoderLayer,
    MultiheadAttention,
    LayerNorm,
)

# Utility functions
from .utils import (
    compute_mask_indices,
    apply_mask,
    sample_negatives,
    compute_cosine_similarity,
    compute_accuracy,
)

# Main exports for convenience
__all__ = [
    # Version info
    "__version__",
    "__author__",
    
    # Configuration
    "AVHubertConfig",
    "AVHubertPretrainConfig",
    
    # Main model
    "AVHubertModel",
    "create_avhubert_base",
    "create_avhubert_large",
    
    # Inference interfaces (most commonly used)
    "AVHubertInference",
    "LipReadingInference", 
    "AudioSpeechRecognitionInference",
    "AudioVisualSpeechRecognitionInference",
    
    # Factory functions (recommended for beginners)
    "create_lip_reading_model",
    "create_asr_model", 
    "create_avsr_model",
    "create_general_model",
    
    # Preprocessing
    "AudioPreprocessor",
    "VideoPreprocessor", 
    "AVHubertPreprocessor",
    "create_audio_preprocessor",
    "create_video_preprocessor",
    "create_avhubert_preprocessor",
    
    # Core components
    "ResEncoder",
    "TransformerEncoder",
    "LayerNorm",
    
    # Utilities
    "compute_mask_indices",
    "apply_mask",
    "compute_cosine_similarity",
]


def get_model_info():
    """Get information about available models and configurations"""
    return {
        "models": {
            "base": "Standard AVHuBERT model with 12 layers, 768 dimensions",
            "large": "Large AVHuBERT model with 24 layers, 1024 dimensions",
        },
        "tasks": {
            "lip_reading": "Video-only speech recognition from lip movements",
            "asr": "Audio-only speech recognition", 
            "avsr": "Audio-visual speech recognition using both modalities",
        },
        "modalities": {
            "audio": "MFCC features extracted from raw audio waveforms",
            "video": "ResNet features extracted from lip region frames",
        }
    }


def quick_start_guide():
    """Print a quick start guide for using the package"""
    print("""
    AVHuBERT Simplified - Quick Start Guide
    ======================================
    
    1. Lip Reading (Video Only):
    ----------------------------
    from avhubert_simplified import create_lip_reading_model
    
    model = create_lip_reading_model("path/to/pretrained/model.pt")
    result = model(video_frames)  # video_frames: (T, H, W, C) numpy array
    
    
    2. Audio Speech Recognition:
    ---------------------------
    from avhubert_simplified import create_asr_model
    
    model = create_asr_model("path/to/pretrained/model.pt") 
    result = model(audio_waveform)  # audio_waveform: 1D numpy array
    
    
    3. Audio-Visual Speech Recognition:
    ----------------------------------
    from avhubert_simplified import create_avsr_model
    
    model = create_avsr_model("path/to/pretrained/model.pt")
    result = model(audio_waveform, video_frames)
    
    
    4. General Purpose (All Tasks):
    ------------------------------
    from avhubert_simplified import AVHubertInference
    
    model = AVHubertInference.from_pretrained("path/to/pretrained/model.pt")
    
    # Extract features only
    features = model.extract_features(audio=audio, video=video)
    
    # Make predictions
    predictions = model.predict(audio=audio, video=video)
    
    # Task-specific methods
    lip_reading_result = model.lip_reading(video)
    asr_result = model.audio_speech_recognition(audio) 
    avsr_result = model.audio_visual_speech_recognition(audio, video)
    
    
    5. Custom Configuration:
    -----------------------
    from avhubert_simplified import AVHubertConfig, AVHubertInference
    
    config = AVHubertConfig(
        encoder_layers=24,
        encoder_embed_dim=1024,
        # ... other parameters
    )
    model = AVHubertInference(config=config)
    
    
    For more examples and detailed documentation, see the examples/ directory.
    """)


# Convenience function for common use cases
def load_model(model_path: str, task: str = "general", device: str = "auto"):
    """
    Load a pretrained model for a specific task
    
    Args:
        model_path: Path to pretrained model checkpoint
        task: Task type ("lip_reading", "asr", "avsr", "general")
        device: Device to run on ("auto", "cpu", "cuda")
        
    Returns:
        Loaded model ready for inference
    """
    if task == "lip_reading":
        return create_lip_reading_model(model_path=model_path, device=device)
    elif task == "asr":
        return create_asr_model(model_path=model_path, device=device)
    elif task == "avsr":
        return create_avsr_model(model_path=model_path, device=device)
    elif task == "general":
        return create_general_model(model_path=model_path, device=device)
    else:
        raise ValueError(f"Unknown task: {task}. Choose from: lip_reading, asr, avsr, general")


# Setup logging
import logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)