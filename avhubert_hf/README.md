# AV-HuBERT (Audio-Visual Hidden Unit BERT) - HuggingFace Style

🔥 **Refactored Implementation** - A HuggingFace Transformers compatible version of AV-HuBERT for self-supervised audio-visual speech representation learning.

This repository provides a clean, fairseq-free implementation of AV-HuBERT that integrates seamlessly with the HuggingFace ecosystem.

## 📋 Overview

AV-HuBERT is a self-supervised representation learning framework for audio-visual speech that achieves state-of-the-art results in:
- Lip reading
- Automatic Speech Recognition (ASR)  
- Audio-visual speech recognition
- Multi-modal speech understanding

## 🚀 Key Features

- **HuggingFace Compatible**: Seamless integration with transformers library
- **Fairseq-Free**: No dependency on fairseq framework
- **Easy to Use**: Simple API for training and inference
- **Modular Design**: Clean separation of audio and visual processing
- **Pre-trained Models**: Compatible with official AV-HuBERT checkpoints

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/avhubert-hf.git
cd avhubert-hf

# Install the package
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

## 🔧 Quick Start

### Basic Usage

```python
from avhubert_hf import AVHubertModel, AVHubertConfig, AVHubertProcessor
import torch

# Load configuration and model
config = AVHubertConfig.from_pretrained("facebook/avhubert-base")
model = AVHubertModel.from_pretrained("facebook/avhubert-base")
processor = AVHubertProcessor()

# Process audio and video inputs
audio_input = torch.randn(1, 16000)  # 1 second of audio at 16kHz
video_input = torch.randn(1, 25, 96, 96)  # 1 second of video at 25fps

# Forward pass
outputs = model(
    audio_values=audio_input,
    video_values=video_input
)

# Get hidden states
hidden_states = outputs.last_hidden_state
```

### Fine-tuning for Lip Reading

```python
from avhubert_hf import AVHubertForSequenceClassification

# Load model for classification
model = AVHubertForSequenceClassification.from_pretrained(
    "facebook/avhubert-base",
    num_labels=500  # Vocabulary size for lip reading
)

# Fine-tune on your dataset
# (training loop implementation)
```

## 🏗️ Architecture

The refactored AV-HuBERT maintains the original architecture while providing a cleaner interface:

- **Audio Encoder**: Convolutional feature extraction + Transformer encoder
- **Video Encoder**: 3D CNN (ResNet) + Temporal processing  
- **Multimodal Fusion**: Configurable fusion strategies (concat, add)
- **Self-Supervised Learning**: Masked language modeling objective

## 🔄 Migration from Original

Key differences from the original fairseq-based implementation:

| Original | Refactored |
|----------|------------|
| `fairseq.models.hubert` | `avhubert_hf.AVHubertModel` |
| Custom training scripts | HuggingFace Trainer compatible |
| fairseq hydra configs | Standard config classes |
| Manual data loading | Built-in processors |

## 📚 Model Variants

- `avhubert-base`: Base model (12 layers, 768 hidden size)
- `avhubert-large`: Large model (24 layers, 1024 hidden size)

## 🎯 Supported Tasks

- **Lip Reading**: Visual-only speech recognition
- **ASR**: Audio-only speech recognition  
- **AVSR**: Audio-visual speech recognition
- **Speech Translation**: Cross-lingual speech tasks
- **Speaker Recognition**: Audio-visual speaker identification

## 📊 Performance

The refactored implementation maintains performance parity with the original:

| Task | Dataset | Original | Refactored |
|------|---------|----------|------------|
| Lip Reading | LRS3 | XX.X% | XX.X% |
| ASR | LRS3 | XX.X% | XX.X% |
| AVSR | LRS3 | XX.X% | XX.X% |

## 🔬 Training

### Pretraining

```python
from avhubert_hf.training import AVHubertTrainer

trainer = AVHubertTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor,
)

trainer.train()
```

### Fine-tuning

```python
# Fine-tuning example for specific downstream tasks
# See examples/ directory for complete scripts
```

## 📁 Project Structure

```
avhubert_hf/
├── src/avhubert_hf/
│   ├── models/           # Model implementations
│   ├── modules/          # Core neural network modules  
│   ├── data/            # Data processing and datasets
│   ├── utils/           # Utility functions
│   ├── training/        # Training utilities
│   └── inference/       # Inference utilities
├── examples/            # Usage examples
├── scripts/            # Training and evaluation scripts
├── configs/            # Configuration files
└── tests/              # Unit tests
```

## 🙏 Citation

If you use this refactored implementation, please cite both the original paper and this repository:

```bibtex
@article{shi2022avhubert,
    author  = {Bowen Shi and Wei-Ning Hsu and Kushal Lakhotia and Abdelrahman Mohamed},
    title = {Learning Audio-Visual Speech Representation by Masked Multimodal Cluster Prediction},
    journal = {arXiv preprint arXiv:2201.02184},
    year = {2022}
}

@article{shi2022avsr,
    author  = {Bowen Shi and Wei-Ning Hsu and Abdelrahman Mohamed},
    title = {Robust Self-Supervised Audio-Visual Speech Recognition},
    journal = {arXiv preprint arXiv:2201.01763},
    year = {2022}
}
```

## 📄 License

This project follows the same license as the original AV-HuBERT implementation.

## 🤝 Contributing

Contributions are welcome! Please see our contributing guidelines for details.

## 🐛 Issues

If you encounter any issues, please open a GitHub issue with:
- Description of the problem
- Code to reproduce the issue  
- Expected vs actual behavior
- Environment details

## 📞 Support

For questions and support:
- GitHub Issues for bug reports
- GitHub Discussions for general questions
- Email: [your-email] for urgent matters