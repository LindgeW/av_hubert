# AVHuBERT Simplified Implementation

A simplified, user-friendly implementation of AVHuBERT (Audio-Visual HuBERT) that removes the dependency on fairseq and provides a HuggingFace-style interface.

## Overview

This implementation provides a clean, easy-to-use interface for AVHuBERT, a multimodal model that learns representations from both audio and visual inputs. The original implementation was heavily dependent on fairseq, making it difficult to use and understand. This simplified version:

- **Removes fairseq dependency**: All fairseq-specific code has been replaced with clean PyTorch implementations
- **Provides HuggingFace-style interface**: Easy-to-use classes similar to HuggingFace transformers
- **Maintains full functionality**: All core features of AVHuBERT are preserved
- **Improves readability**: Clean, well-documented code that's easy to understand and modify

## Features

- **Audio-Visual Fusion**: Combines audio and visual features using concatenation or addition
- **Masked Self-Supervised Learning**: Supports masked prediction for pretraining
- **Flexible Architecture**: Configurable transformer encoder with customizable layers
- **ResNet Visual Encoder**: Pre-trained ResNet for visual feature extraction
- **Convolutional Audio Encoder**: Multi-layer convolutional network for audio features
- **Easy-to-use Interface**: Simple API similar to HuggingFace transformers

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd avhubert_simple

# Install dependencies
pip install torch torchvision opencv-python numpy
```

## Quick Start

### Basic Usage

```python
import torch
from avhubert_simple import AVHuBERTModel, AVHuBERTConfig

# Create configuration
config = AVHuBERTConfig(
    encoder_layers=6,
    encoder_embed_dim=256,
    encoder_ffn_embed_dim=1024,
    encoder_attention_heads=8,
)

# Create model
model = AVHuBERTModel(config)

# Create dummy data
audio = torch.randn(2, 16000)  # [batch_size, audio_length]
video = torch.randn(2, 30, 1, 88, 88)  # [batch_size, frames, channels, height, width]

# Forward pass
with torch.no_grad():
    outputs = model(audio, video, mask=False, features_only=True)

print(f"Output features shape: {outputs['features'].shape}")
```

### Using the Processor

```python
from avhubert_simple import AVHuBERTProcessor, AVHuBERTFeatureExtractor, AVHuBERTConfig
import numpy as np

# Create processor
config = AVHuBERTConfig(encoder_layers=4, encoder_embed_dim=128)
feature_extractor = AVHuBERTFeatureExtractor()
model = AVHuBERTModel(config)
processor = AVHuBERTProcessor(feature_extractor, model)

# Create dummy data
audio = np.random.randn(16000).astype(np.float32)
video = np.random.randn(30, 88, 88).astype(np.float32)

# Run inference
with torch.no_grad():
    outputs = processor(audio=audio, video=video)

print(f"Features shape: {outputs['features'].shape}")
```

### Feature Extraction

```python
# Extract features from specific layers
features = processor.extract_features(
    audio=audio,
    video=video,
    mask=False,
    output_layer=6  # Extract from layer 6
)

print(f"Extracted features: {features['features'].shape}")
```

## Architecture

### Model Components

1. **Audio Feature Extractor**: Convolutional layers that process raw audio
2. **Visual Feature Extractor**: ResNet-18 that processes video frames
3. **Sub-Encoders**: Optional transformer layers for single modality processing
4. **Main Encoder**: Transformer encoder that fuses audio and visual features
5. **Output Projection**: Final linear layer for feature projection

### Modality Fusion

The model supports two fusion strategies:

- **Concatenation**: Concatenates audio and visual features along the feature dimension
- **Addition**: Adds audio and visual features (requires same dimensionality)

### Masking Strategy

For self-supervised learning, the model supports:

- **Input Masking**: Masks input features before encoding
- **Feature Masking**: Masks features after encoding
- **Channel Masking**: Masks specific feature channels

## Configuration

The `AVHuBERTConfig` class allows you to customize all aspects of the model:

```python
config = AVHuBERTConfig(
    # Model architecture
    encoder_layers=12,
    encoder_embed_dim=768,
    encoder_ffn_embed_dim=3072,
    encoder_attention_heads=12,
    activation_fn="gelu",
    
    # Dropouts
    dropout=0.1,
    attention_dropout=0.1,
    activation_dropout=0.0,
    
    # Masking
    mask_length_audio=10,
    mask_prob_audio=0.65,
    mask_length_image=10,
    mask_prob_image=0.65,
    
    # Modality fusion
    modality_fuse="concat",  # or "add"
    modality_dropout=0.0,
    
    # ResNet
    resnet_relu_type="prelu",
)
```

## API Reference

### AVHuBERTModel

Main model class for audio-visual representation learning.

**Methods:**
- `forward(audio, video, padding_mask=None, mask=True, features_only=False, output_layer=None)`: Forward pass
- `extract_features(audio, video, padding_mask=None, mask=False, output_layer=None)`: Extract features
- `apply_masking(features, padding_mask)`: Apply masking to features
- `apply_positional_embeddings(features)`: Apply positional embeddings

### AVHuBERTFeatureExtractor

Preprocesses audio and video inputs.

**Methods:**
- `__call__(audio, video, audio_path, video_path, return_tensors="pt")`: Process inputs
- `batch_decode(features)`: Decode features back to original format

### AVHuBERTProcessor

Combines feature extraction and model inference.

**Methods:**
- `__call__(audio, video, ...)`: End-to-end processing
- `extract_features(audio, video, mask=False, output_layer=None)`: Extract features
- `predict(audio, video, ...)`: Run predictions
- `from_pretrained(model_name_or_path, config=None)`: Load from pretrained model
- `save_pretrained(save_directory)`: Save model to directory

## Examples

See `example_usage.py` for comprehensive examples demonstrating:

- Basic model usage
- Feature extraction
- Processor usage
- Custom configurations
- Prediction examples

## Differences from Original

### Removed Dependencies
- **fairseq**: All fairseq-specific code replaced with clean PyTorch implementations
- **Complex configuration**: Simplified configuration system
- **Task-specific code**: Removed task-specific implementations (ASR, pretraining)

### Improvements
- **Cleaner API**: HuggingFace-style interface
- **Better documentation**: Comprehensive docstrings and examples
- **Modular design**: Separated concerns into different classes
- **Easier customization**: Simple configuration system

### Preserved Features
- **Core architecture**: All essential components maintained
- **Masking strategies**: Full masking functionality preserved
- **Modality fusion**: Both concatenation and addition supported
- **ResNet integration**: Visual feature extraction maintained

## Training

For training, you can use the model in a standard PyTorch training loop:

```python
import torch.nn as nn
import torch.optim as optim

# Create model and optimizer
model = AVHuBERTModel(config)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        audio, video = batch
        
        # Forward pass with masking
        outputs = model(audio, video, mask=True)
        
        # Compute loss (implement your loss function)
        loss = criterion(outputs['features'], targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:

- Bug fixes
- Performance improvements
- Additional features
- Documentation improvements

## License

This implementation is based on the original AVHuBERT paper and follows the same license terms.

## Citation

If you use this implementation, please cite the original AVHuBERT paper:

```bibtex
@inproceedings{shi2022avhubert,
  title={AV-HuBERT: Self-Supervised Learning of Audio-Visual Speech Representations},
  author={Shi, Bowen and Hsu, Wei-Ning and Lakhotia, Kushal and Yilmaz, Abdelrahman Mohamed and Garg, Anurag and Duan, Yuning and Schubert, Junkun and Wang, Chunyu and Gelly, Sylvain and Schalkwyk, Johan and others},
  booktitle={ICLR},
  year={2022}
}
```