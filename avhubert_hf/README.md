# AV-HuBERT: Audio-Visual Hidden Unit BERT

[![PyPI version](https://badge.fury.io/py/avhubert.svg)](https://badge.fury.io/py/avhubert)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A self-supervised representation learning framework for audio-visual speech. This is a refactored version that removes fairseq dependencies and provides a clean, Hugging Face-style interface.

## 🚀 Features

- **No fairseq dependency**: Clean, standalone implementation
- **Hugging Face-style API**: Easy to use and understand
- **Modular design**: Separate components for models, data, and training
- **Comprehensive configuration**: Flexible configuration system
- **Multi-modal support**: Audio, video, and audio-visual fusion
- **Pre-training and fine-tuning**: Support for both training modes
- **Distributed training**: Multi-GPU training support

## 📖 Paper

This implementation is based on the following papers:

- [Learning Audio-Visual Speech Representation by Masked Multimodal Cluster Prediction](https://arxiv.org/abs/2201.02184)
- [Robust Self-Supervised Audio-Visual Speech Recognition](https://arxiv.org/abs/2201.01763)

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/your-username/avhubert-hf.git
cd avhubert-hf

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## 🚀 Quick Start

### Basic Usage

```python
import torch
from avhubert_hf import AVHubertModel, AVHubertConfig, DataConfig, TrainingConfig

# Create configuration
config = AVHubertConfig(
    encoder_layers=12,
    encoder_embed_dim=768,
    encoder_attention_heads=12,
)

# Create model
model = AVHubertModel(config)

# Create dummy input
batch_size, seq_len, channels = 2, 1000, 80
audio_input = torch.randn(batch_size, seq_len, channels)

# Forward pass
with torch.no_grad():
    output = model(audio_input, features_only=True)
    features = output["features"]
    print(f"Feature shape: {features.shape}")
```

### Training Example

```python
from avhubert_hf import (
    AVHubertModel, AVHubertConfig, DataConfig, TrainingConfig,
    AVHubertDataset, AVHubertTrainer
)

# Create configurations
model_config = AVHubertConfig(
    encoder_layers=12,
    encoder_embed_dim=768,
    encoder_attention_heads=12,
)

data_config = DataConfig(
    data_path="/path/to/your/data",
    label_dir="/path/to/your/labels",
    sample_rate=16000,
    label_rate=100,
)

training_config = TrainingConfig(
    batch_size=8,
    learning_rate=5e-4,
    max_epochs=100,
)

# Create datasets
train_dataset = AVHubertDataset(data_config, split="train")
val_dataset = AVHubertDataset(data_config, split="valid")

# Create model
model = AVHubertModel(model_config)

# Create trainer
trainer = AVHubertTrainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    config=training_config,
    device="cuda",
)

# Start training
trainer.train()
```

### Loading Pre-trained Models

```python
from avhubert_hf import AVHubertModel, AVHubertConfig

# Load configuration
config = AVHubertConfig.from_pretrained("facebook/avhubert-base")

# Load model
model = AVHubertModel.from_pretrained("facebook/avhubert-base")

# Use for inference
audio_input = torch.randn(1, 1000, 80)
with torch.no_grad():
    features = model(audio_input, features_only=True)["features"]
```

## 📁 Project Structure

```
avhubert_hf/
├── configs/           # Configuration classes
│   ├── model_config.py
│   ├── data_config.py
│   └── training_config.py
├── models/            # Model implementations
│   ├── avhubert.py
│   ├── resnet.py
│   └── transformer.py
├── data/              # Data loading and processing
│   ├── dataset.py
│   └── collator.py
├── training/          # Training utilities
│   ├── trainer.py
│   └── criterion.py
├── utils/             # Utility functions
│   ├── masking.py
│   ├── dictionary.py
│   ├── activations.py
│   └── positional_encoding.py
└── examples/          # Example scripts
    ├── pretrain.py
    ├── finetune.py
    └── inference.py
```

## ⚙️ Configuration

### Model Configuration

```python
from avhubert_hf import AVHubertConfig

config = AVHubertConfig(
    # Architecture
    encoder_layers=12,
    encoder_embed_dim=768,
    encoder_ffn_embed_dim=3072,
    encoder_attention_heads=12,
    
    # Dropout
    dropout=0.1,
    attention_dropout=0.1,
    
    # Masking
    mask_prob_audio=0.65,
    mask_length_audio=10,
    mask_prob_image=0.65,
    mask_length_image=10,
    
    # Modality fusion
    modality_fuse="concat",
    modality_dropout=0.0,
)
```

### Data Configuration

```python
from avhubert_hf import DataConfig

config = DataConfig(
    data_path="/path/to/data",
    label_dir="/path/to/labels",
    sample_rate=16000,
    label_rate=100,
    modalities=["audio", "video"],
    image_crop_size=88,
    normalize=True,
)
```

### Training Configuration

```python
from avhubert_hf import TrainingConfig

config = TrainingConfig(
    batch_size=8,
    learning_rate=5e-4,
    max_epochs=100,
    warmup_steps=4000,
    gradient_clip_val=1.0,
    save_dir="./checkpoints",
)
```

## 🔧 Advanced Usage

### Custom Dataset

```python
from avhubert_hf import AVHubertDataset, DataConfig

class CustomDataset(AVHubertDataset):
    def __init__(self, config: DataConfig, split: str = "train"):
        super().__init__(config, split)
    
    def _load_audio(self, audio_path: str) -> torch.Tensor:
        # Custom audio loading logic
        return super()._load_audio(audio_path)
    
    def _load_video(self, video_path: str) -> torch.Tensor:
        # Custom video loading logic
        return super()._load_video(video_path)
```

### Custom Training Loop

```python
from avhubert_hf import AVHubertTrainer

class CustomTrainer(AVHubertTrainer):
    def train_epoch(self):
        # Custom training logic
        return super().train_epoch()
    
    def validate(self):
        # Custom validation logic
        return super().validate()
```

### Distributed Training

```python
import torch.distributed as dist
from avhubert_hf import AVHubertTrainer

# Initialize distributed training
dist.init_process_group(backend="nccl")

# Create trainer with distributed=True
trainer = AVHubertTrainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    config=training_config,
    device="cuda",
    distributed=True,
)

# Start training
trainer.train()
```

## 📊 Performance

The refactored implementation maintains the same performance as the original fairseq-based version:

- **Pre-training**: Comparable convergence and final performance
- **Fine-tuning**: Same accuracy on downstream tasks
- **Inference**: Similar speed and memory usage

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original AV-HuBERT implementation by Facebook Research
- Hugging Face for the excellent library design patterns
- The open-source community for various utilities and tools

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{shi2022avhubert,
    author  = {Bowen Shi and Wei-Ning Hsu and Kushal Lakhotia and Abdelrahman Mohamed},
    title = {Learning Audio-Visual Speech Representation by Masked Multimodal Cluster Prediction},
    journal = {arXiv preprint arXiv:2201.02184}
    year = {2022}
}

@article{shi2022avsr,
    author  = {Bowen Shi and Wei-Ning Hsu and Abdelrahman Mohamed},
    title = {Robust Self-Supervised Audio-Visual Speech Recognition},
    journal = {arXiv preprint arXiv:2201.01763}
    year = {2022}
}
```

## 📞 Support

If you have any questions or issues, please:

1. Check the [documentation](docs/)
2. Search existing [issues](https://github.com/your-username/avhubert-hf/issues)
3. Create a new issue with a detailed description

## 🔄 Migration from Original AV-HuBERT

If you're migrating from the original fairseq-based AV-HuBERT:

1. **Installation**: No more fairseq dependency
2. **Configuration**: Use dataclass-based configs instead of fairseq configs
3. **Model loading**: Use `from_pretrained()` method
4. **Training**: Use the new `AVHubertTrainer` class
5. **Data loading**: Use the new `AVHubertDataset` class

See the [Migration Guide](docs/migration.md) for detailed instructions.