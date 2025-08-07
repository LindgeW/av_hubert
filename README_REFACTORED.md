# AV-HuBERT: Audio-Visual Hidden Unit BERT (Refactored)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.20+-green.svg)](https://huggingface.co/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A refactored version of AV-HuBERT following Hugging Face patterns, making it easy to understand and use without the fairseq dependency.

## 🚀 Features

- **Hugging Face Style**: Follows standard Hugging Face patterns and conventions
- **No Fairseq Dependency**: Completely independent of fairseq
- **Easy to Use**: Simple, intuitive API
- **Modular Design**: Clean separation of components
- **Comprehensive Examples**: Ready-to-run examples for various use cases
- **Modern Dependencies**: Uses latest PyTorch and Transformers libraries

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Model Architecture](#model-architecture)
- [Usage Examples](#usage-examples)
- [Training](#training)
- [Inference](#inference)
- [API Reference](#api-reference)
- [Migration Guide](#migration-guide)
- [Contributing](#contributing)

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.9 or higher
- CUDA (optional, for GPU acceleration)

### Install the Package

```bash
# Clone the repository
git clone <repository-url>
cd av_hubert_refactored

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Verify Installation

```python
import torch
from avhubert_hf import AVHubertConfig, AVHubertModel

# Create a simple model
config = AVHubertConfig(encoder_layers=6, encoder_embed_dim=256)
model = AVHubertModel(config)
print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
```

## 🚀 Quick Start

### Basic Usage

```python
import torch
from avhubert_hf import AVHubertConfig, AVHubertModel

# Create configuration
config = AVHubertConfig(
    encoder_layers=12,
    encoder_embed_dim=768,
    encoder_attention_heads=12,
    final_dim=768,
)

# Create model
model = AVHubertModel(config)

# Create input
audio_input = torch.randn(2, 16000)  # 2 samples, 1 second each at 16kHz
video_input = torch.randn(2, 1, 25, 88, 88)  # 2 samples, 25 frames, 88x88 resolution

# Forward pass
with torch.no_grad():
    outputs = model(
        input_values=audio_input,
        video_values=video_input,
    )

print(f"Output shape: {outputs.last_hidden_state.shape}")
```

### Using the Processor

```python
import numpy as np
from avhubert_hf import AVHubertFeatureExtractor, AVHubertTokenizer, AVHubertProcessor

# Create processor components
feature_extractor = AVHubertFeatureExtractor(sampling_rate=16000)
tokenizer = AVHubertTokenizer("path/to/vocab.txt")
processor = AVHubertProcessor(feature_extractor, tokenizer)

# Process audio and text
audio = np.random.randn(16000)  # 1 second of audio
text = "hello world"

inputs = processor(
    audio,
    text=text,
    sampling_rate=16000,
    return_tensors="pt",
)

print(f"Input keys: {inputs.keys()}")
```

## 🏗️ Model Architecture

The refactored AV-HuBERT consists of several key components:

### 1. Feature Extraction
- **Audio**: Convolutional feature extractor for raw audio
- **Video**: ResNet-based encoder for video frames

### 2. Modality Fusion
- **Concatenation**: Combine audio and video features
- **Addition**: Add audio and video features
- **Configurable**: Choose fusion strategy via configuration

### 3. Transformer Encoder
- **Multi-head Attention**: Standard transformer attention
- **Positional Embeddings**: Convolutional positional embeddings
- **Layer Normalization**: Pre-norm or post-norm options

### 4. Pre-training Tasks
- **Masked Prediction**: Predict masked audio-visual features
- **Contrastive Learning**: Learn representations through contrastive loss

## 📚 Usage Examples

### Example 1: Basic Model

```python
from avhubert_hf import AVHubertConfig, AVHubertModel

# Create a small model for testing
config = AVHubertConfig(
    encoder_layers=6,
    encoder_embed_dim=256,
    encoder_attention_heads=8,
    final_dim=256,
)

model = AVHubertModel(config)

# Audio-only input
audio = torch.randn(1, 16000)
outputs = model(input_values=audio)
print(f"Audio output shape: {outputs.last_hidden_state.shape}")

# Video-only input
video = torch.randn(1, 1, 25, 88, 88)
outputs = model(video_values=video)
print(f"Video output shape: {outputs.last_hidden_state.shape}")

# Audio-video input
outputs = model(input_values=audio, video_values=video)
print(f"Audio-video output shape: {outputs.last_hidden_state.shape}")
```

### Example 2: Pre-training

```python
from avhubert_hf import AVHubertConfig, AVHubertForPreTraining

config = AVHubertConfig(
    encoder_layers=6,
    encoder_embed_dim=256,
    encoder_attention_heads=8,
    final_dim=256,
    mask_prob_audio=0.65,
    mask_length_audio=10,
)

model = AVHubertForPreTraining(config)

# Create inputs with masking
audio = torch.randn(2, 16000)
video = torch.randn(2, 1, 25, 88, 88)
mask_time_indices = torch.zeros(2, 50, dtype=torch.bool)  # After feature extraction
mask_time_indices[:, :10] = True  # Mask first 10 frames
labels = torch.randint(0, 100, (2, 50))

# Forward pass with loss computation
outputs = model(
    input_values=audio,
    video_values=video,
    mask_time_indices=mask_time_indices,
    labels=labels,
)

print(f"Loss: {outputs.loss.item():.4f}")
```

### Example 3: Feature Extraction

```python
from avhubert_hf import AVHubertFeatureExtractor
import numpy as np

feature_extractor = AVHubertFeatureExtractor(
    sampling_rate=16000,
    do_normalize=True,
)

# Extract features from audio
audio = np.random.randn(16000)
features = feature_extractor(
    audio,
    sampling_rate=16000,
    return_tensors="pt",
)

print(f"Features shape: {features['input_values'].shape}")
```

## 🎯 Training

### Training Configuration

```python
from avhubert_hf import AVHubertConfig, AVHubertForPreTraining
import torch.optim as optim

# Create configuration
config = AVHubertConfig(
    encoder_layers=12,
    encoder_embed_dim=768,
    encoder_attention_heads=12,
    encoder_ffn_embed_dim=3072,
    final_dim=768,
    modality_fuse="concat",
    mask_prob_audio=0.65,
    mask_length_audio=10,
    mask_prob_image=0.65,
    mask_length_image=10,
)

# Create model
model = AVHubertForPreTraining(config)

# Create optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
```

### Training Loop

```python
import torch
from torch.utils.data import DataLoader

# Training loop
model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        
        outputs = model(
            input_values=batch['audio'],
            video_values=batch['video'],
            mask_time_indices=batch['mask_time_indices'],
            labels=batch['labels'],
        )
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()
```

## 🔍 Inference

### Loading a Trained Model

```python
from avhubert_hf import AVHubertModel

# Load configuration and model
config = AVHubertConfig.from_pretrained("path/to/config")
model = AVHubertModel.from_pretrained("path/to/model")

# Set to evaluation mode
model.eval()

# Inference
with torch.no_grad():
    outputs = model(
        input_values=audio_input,
        video_values=video_input,
    )
```

### Feature Extraction

```python
# Extract features from a specific layer
with torch.no_grad():
    outputs = model(
        input_values=audio_input,
        video_values=video_input,
        output_hidden_states=True,
    )

# Get features from the last layer
features = outputs.last_hidden_state

# Get features from all layers
all_features = outputs.hidden_states
```

## 📖 API Reference

### Configuration

#### `AVHubertConfig`

Main configuration class for AV-HuBERT models.

**Key Parameters:**
- `encoder_layers`: Number of transformer encoder layers
- `encoder_embed_dim`: Embedding dimension
- `encoder_attention_heads`: Number of attention heads
- `encoder_ffn_embed_dim`: Feed-forward network dimension
- `final_dim`: Final output dimension
- `modality_fuse`: Fusion strategy ("concat" or "add")
- `mask_prob_audio`: Audio masking probability
- `mask_length_audio`: Audio mask length
- `mask_prob_image`: Video masking probability
- `mask_length_image`: Video mask length

### Models

#### `AVHubertModel`

Base model for feature extraction.

**Inputs:**
- `input_values`: Audio input tensor
- `video_values`: Video input tensor
- `attention_mask`: Attention mask
- `mask_time_indices`: Mask indices for pre-training

**Outputs:**
- `last_hidden_state`: Final hidden states
- `hidden_states`: All layer hidden states (optional)
- `attentions`: Attention weights (optional)

#### `AVHubertForPreTraining`

Model for pre-training with masked prediction.

**Additional Inputs:**
- `labels`: Target labels for masked prediction

**Additional Outputs:**
- `loss`: Training loss
- `logits`: Prediction logits

#### `AVHubertForCTC`

Model for CTC-based speech recognition.

### Processors

#### `AVHubertFeatureExtractor`

Extracts features from raw audio.

#### `AVHubertTokenizer`

Tokenizes text inputs.

#### `AVHubertProcessor`

Combines feature extractor and tokenizer.

## 🔄 Migration Guide

### From Original AV-HuBERT

The refactored version maintains compatibility while providing a cleaner API:

**Before (Original):**
```python
import fairseq
import hubert_pretraining, hubert

ckpt_path = "/path/to/checkpoint.pt"
models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
model = models[0]
```

**After (Refactored):**
```python
from avhubert_hf import AVHubertModel

model = AVHubertModel.from_pretrained("/path/to/model")
```

### Key Changes

1. **No Fairseq Dependency**: All fairseq components have been replaced
2. **Hugging Face Style**: Follows standard Hugging Face patterns
3. **Simplified API**: Cleaner, more intuitive interface
4. **Better Documentation**: Comprehensive docstrings and examples
5. **Modern Dependencies**: Uses latest PyTorch and Transformers

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd av_hubert_refactored

# Install in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linting
black avhubert_hf/
isort avhubert_hf/
flake8 avhubert_hf/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original AV-HuBERT authors for the research and implementation
- Hugging Face team for the excellent transformers library
- The open-source community for contributions and feedback

## 📞 Support

- **Issues**: Please use the [GitHub issue tracker](https://github.com/your-repo/issues)
- **Discussions**: Join our [GitHub Discussions](https://github.com/your-repo/discussions)
- **Documentation**: Check our [documentation](https://your-docs-url.com)

---

**Note**: This is a refactored version of the original AV-HuBERT implementation. For the original implementation, please refer to the [original repository](https://github.com/facebookresearch/av_hubert).