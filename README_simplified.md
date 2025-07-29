# AVHuBERT Simplified

A clean, HuggingFace-style implementation of AVHuBERT (Audio-Visual Hidden Unit BERT) without fairseq dependencies.

## 🎯 Overview

This simplified implementation provides an easy-to-use interface for AVHuBERT, supporting:

- **Lip Reading** (video-only speech recognition)
- **Audio Speech Recognition** (audio-only)
- **Audio-Visual Speech Recognition** (both modalities)

## ✨ Key Features

- 🚀 **HuggingFace-style API** - Simple and intuitive interface
- 🔥 **No fairseq dependency** - Standalone PyTorch implementation
- 📦 **Modular design** - Use only what you need
- 🎨 **Clean code** - Well-documented and readable
- 🔧 **Flexible configuration** - Easy to customize model architecture
- 📱 **Multiple inference modes** - Task-specific and general-purpose interfaces

## 📦 Installation

```bash
# Clone the repository
git clone <repo-url>
cd avhubert_simplified

# Install dependencies
pip install -r requirements_simplified.txt

# Optional: Install in development mode
pip install -e .
```

### Requirements

- Python 3.7+
- PyTorch 1.8+
- OpenCV (for video processing)
- librosa or python-speech-features (for audio processing)
- NumPy, SciPy

See `requirements_simplified.txt` for detailed dependencies.

## 🚀 Quick Start

### 1. Lip Reading

```python
from avhubert_simplified import create_lip_reading_model

# Load pretrained model
model = create_lip_reading_model("path/to/pretrained/model.pt")

# Process video (shape: T, H, W, C)
import numpy as np
video_frames = np.random.randint(0, 255, (50, 88, 88, 3), dtype=np.uint8)

# Perform lip reading
result = model(video_frames)
print(f"Lip reading features: {result['predictions'].shape}")
```

### 2. Audio Speech Recognition

```python
from avhubert_simplified import create_asr_model

# Load pretrained model
model = create_asr_model("path/to/pretrained/model.pt")

# Process audio waveform
audio_waveform = np.random.randn(16000 * 3).astype(np.float32)  # 3 seconds

# Perform ASR
result = model(audio_waveform)
print(f"ASR features: {result['predictions'].shape}")
```

### 3. Audio-Visual Speech Recognition

```python
from avhubert_simplified import create_avsr_model

# Load pretrained model
model = create_avsr_model("path/to/pretrained/model.pt")

# Process both modalities
result = model(audio_waveform, video_frames)
print(f"AVSR features: {result['predictions'].shape}")
```

### 4. General Purpose Interface

```python
from avhubert_simplified import AVHubertInference

# Load model
model = AVHubertInference.from_pretrained("path/to/pretrained/model.pt")

# Extract features
features = model.extract_features(audio=audio_waveform, video=video_frames)

# Make predictions
predictions = model.predict(audio=audio_waveform, video=video_frames)

# Task-specific methods
lip_result = model.lip_reading(video_frames)
asr_result = model.audio_speech_recognition(audio_waveform)
avsr_result = model.audio_visual_speech_recognition(audio_waveform, video_frames)
```

## 📖 Detailed Usage

### Configuration

```python
from avhubert_simplified import AVHubertConfig, AVHubertInference

# Custom configuration
config = AVHubertConfig(
    encoder_layers=24,           # Number of transformer layers
    encoder_embed_dim=1024,      # Hidden dimension
    encoder_attention_heads=16,  # Number of attention heads
    mask_prob_audio=0.8,         # Audio masking probability
    mask_prob_image=0.75,        # Video masking probability
    modality_fuse="concat",      # How to fuse modalities ("concat" or "add")
)

# Create model with custom config
model = AVHubertInference(config=config)
```

### Preprocessing

```python
from avhubert_simplified import create_avhubert_preprocessor

# Create preprocessor
preprocessor = create_avhubert_preprocessor(
    audio_config={
        "sample_rate": 16000,
        "n_mfcc": 80,
        "normalize": True,
        "apply_cmvn": True,
    },
    video_config={
        "image_size": 88,
        "image_mean": 0.421,
        "image_std": 0.165,
        "apply_augmentation": False,
    }
)

# Preprocess data
audio_features, video_features = preprocessor(
    audio="path/to/audio.wav",
    video="path/to/video.mp4"
)
```

### Loading Pretrained Models

The simplified implementation is designed to be compatible with original AVHuBERT checkpoints:

```python
# Load from fairseq checkpoint (automatic conversion)
model = AVHubertInference.from_pretrained("checkpoint.pt")

# Save in simplified format
model.save_pretrained("simplified_model.pt")

# Load simplified format
model = AVHubertInference.from_pretrained("simplified_model.pt")
```

### Model Variants

```python
from avhubert_simplified import create_avhubert_base, create_avhubert_large

# Base model (12 layers, 768 dims)
base_model = create_avhubert_base()

# Large model (24 layers, 1024 dims)
large_model = create_avhubert_large()
```

## 🏗️ Architecture

The simplified implementation maintains the core AVHuBERT architecture:

```
Input (Audio/Video)
        ↓
Feature Extractors
├── Audio: Conv1D layers → MFCC features
└── Video: ResNet-18 → Visual features
        ↓
Modality Fusion (concat/add)
        ↓
Transformer Encoder (12-24 layers)
        ↓
Output Projections
```

### Key Components

- **ResNet Visual Encoder**: Extracts features from lip region frames
- **Conv Audio Encoder**: Processes MFCC features from audio
- **Transformer Encoder**: Multi-layer attention mechanism
- **Modality Fusion**: Combines audio and visual features
- **Masking**: Self-supervised learning through masked prediction

## 📂 Project Structure

```
avhubert_simplified/
├── __init__.py              # Main package imports
├── config.py                # Configuration classes
├── model.py                 # Core AVHuBERT model
├── transformer.py           # Transformer encoder
├── resnet.py                # Visual encoder (ResNet)
├── utils.py                 # Utility functions
├── preprocessors.py         # Audio/video preprocessing
└── inference.py             # Inference interfaces

examples/
├── simple_usage.py          # Basic usage examples
├── advanced_usage.py        # Advanced features
└── model_conversion.py      # Convert fairseq models

requirements_simplified.txt  # Dependencies
README_simplified.md         # This file
```

## 🔧 Advanced Usage

### Custom Training Loop

```python
from avhubert_simplified import AVHubertModel, AVHubertConfig

config = AVHubertConfig()
model = AVHubertModel(config)

# Training mode
model.train()

# Forward pass with masking
outputs = model(
    audio=audio_batch,
    video=video_batch,
    mask=True,  # Enable masking for training
)

# Compute loss
loss = model.compute_mask_loss(
    outputs['projected_states'],
    targets,
    outputs['mask_indices']
)

# Backward pass
loss.backward()
```

### Feature Extraction for Downstream Tasks

```python
# Extract contextualized representations
model.eval()
with torch.no_grad():
    features = model.extract_features(audio=audio, video=video)
    
# Use features for classification, etc.
hidden_states = features['last_hidden_state']  # (B, T, D)
projected_states = features['projected_states']  # (B, T, final_dim)
```

### Similarity Computation

```python
# Compare two utterances
features1 = model.extract_features(audio=audio1, video=video1)
features2 = model.extract_features(audio=audio2, video=video2)

similarity = model.get_similarity(
    features1['projected_states'],
    features2['projected_states']
)
```

## 🆚 Comparison with Original

| Feature | Original AVHuBERT | Simplified AVHuBERT |
|---------|-------------------|---------------------|
| Dependencies | fairseq, complex setup | PyTorch only |
| Interface | fairseq CLI/API | HuggingFace-style |
| Code Complexity | High | Low |
| Customization | Difficult | Easy |
| Documentation | Limited | Comprehensive |
| Maintenance | Complex | Simple |

## 🤝 Migration from Original

### Loading Original Checkpoints

```python
# The simplified version can load original fairseq checkpoints
model = AVHubertInference.from_pretrained("original_checkpoint.pt")

# Convert and save in simplified format
model.save_pretrained("simplified_checkpoint.pt")
```

### API Mapping

```python
# Original fairseq style
import fairseq
models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
model = models[0]

# Simplified style
from avhubert_simplified import AVHubertInference
model = AVHubertInference.from_pretrained(ckpt_path)
```

## 📚 Examples

See the `examples/` directory for comprehensive usage examples:

- `simple_usage.py` - Basic inference examples
- `advanced_usage.py` - Custom training and fine-tuning
- `model_conversion.py` - Converting between formats

## 🐛 Troubleshooting

### Common Issues

1. **ImportError for audio libraries**
   ```bash
   pip install librosa python-speech-features
   ```

2. **CUDA out of memory**
   - Reduce batch size or sequence length
   - Use `device="cpu"` for testing

3. **Video processing issues**
   ```bash
   pip install opencv-python
   ```

### Getting Help

1. Check the examples in `examples/`
2. Run `python -c "from avhubert_simplified import quick_start_guide; quick_start_guide()"`
3. Review the configuration options in `config.py`

## 📄 License

This simplified implementation follows the same license as the original AVHuBERT project.

## 🙏 Acknowledgments

- Original AVHuBERT paper and implementation by Facebook Research
- HuggingFace for the inspiration on clean ML interfaces
- PyTorch team for the excellent framework

## 📈 Performance Notes

The simplified implementation maintains the same model architecture and should provide equivalent performance to the original, while being much easier to use and maintain.

## 🚧 Future Improvements

- [ ] Integration with HuggingFace Hub
- [ ] ONNX export support
- [ ] TorchScript compatibility
- [ ] Quantization support
- [ ] Streaming inference
- [ ] Web demo interface

---

**Happy coding with AVHuBERT Simplified! 🎉**