# Migration Guide: From Original AV-HuBERT to AV-HuBERT HF

This guide helps you migrate from the original fairseq-based AV-HuBERT implementation to the new Hugging Face-style implementation.

## 🎯 Key Changes

### 1. No More fairseq Dependency

**Before (Original):**
```bash
pip install fairseq
cd fairseq
pip install --editable ./
```

**After (New):**
```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Configuration System

**Before (Original):**
```python
# Used fairseq dataclasses and OmegaConf
from fairseq.dataclass import FairseqDataclass
from omegaconf import II

@dataclass
class AVHubertConfig(FairseqDataclass):
    label_rate: int = II("task.label_rate")
    # ... other fields
```

**After (New):**
```python
# Clean Python dataclasses
from dataclasses import dataclass, field

@dataclass
class AVHubertConfig:
    label_rate: int = field(default=-1, metadata={"help": "label frame rate"})
    # ... other fields
```

### 3. Model Loading

**Before (Original):**
```python
import fairseq
import hubert_pretraining, hubert

ckpt_path = "/path/to/checkpoint.pt"
models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
model = models[0]
```

**After (New):**
```python
from avhubert_hf import AVHubertModel, AVHubertConfig

# Load configuration
config = AVHubertConfig.from_pretrained("facebook/avhubert-base")

# Load model
model = AVHubertModel.from_pretrained("facebook/avhubert-base")
```

### 4. Training

**Before (Original):**
```bash
fairseq-hydra-train --config-dir /path/to/conf/ --config-name conf-name \
  task.data=/path/to/data task.label_dir=/path/to/label \
  model.label_rate=100 hydra.run.dir=/path/to/experiment/pretrain/ \
  common.user_dir=`pwd`
```

**After (New):**
```python
from avhubert_hf import AVHubertTrainer, TrainingConfig

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

### 5. Data Loading

**Before (Original):**
```python
# Used fairseq datasets and tasks
from fairseq.tasks import FairseqTask
from fairseq.data import FairseqDataset

class AVHubertDataset(FairseqDataset):
    # ... implementation
```

**After (New):**
```python
# Standard PyTorch Dataset
from torch.utils.data import Dataset

class AVHubertDataset(Dataset):
    # ... implementation
```

## 📋 Migration Checklist

### Step 1: Environment Setup

- [ ] Remove fairseq from requirements
- [ ] Install new dependencies: `pip install -r requirements.txt`
- [ ] Install the new package: `pip install -e .`

### Step 2: Configuration Migration

- [ ] Convert fairseq config files to Python dataclasses
- [ ] Update configuration loading code
- [ ] Test configuration creation

### Step 3: Model Migration

- [ ] Update model initialization code
- [ ] Replace fairseq model loading with new API
- [ ] Test model forward pass

### Step 4: Data Pipeline Migration

- [ ] Update dataset creation code
- [ ] Replace fairseq data loaders with PyTorch DataLoader
- [ ] Test data loading pipeline

### Step 5: Training Migration

- [ ] Replace fairseq training commands with Python code
- [ ] Update training loops
- [ ] Test training pipeline

### Step 6: Inference Migration

- [ ] Update inference code
- [ ] Replace fairseq decoding with new API
- [ ] Test inference pipeline

## 🔄 Code Migration Examples

### Configuration Migration

**Before:**
```python
# fairseq config
@dataclass
class AVHubertConfig(FairseqDataclass):
    label_rate: int = II("task.label_rate")
    encoder_layers: int = field(default=12)
    encoder_embed_dim: int = field(default=768)
```

**After:**
```python
# New config
@dataclass
class AVHubertConfig:
    label_rate: int = field(default=-1, metadata={"help": "label frame rate"})
    encoder_layers: int = field(default=12, metadata={"help": "num encoder layers"})
    encoder_embed_dim: int = field(default=768, metadata={"help": "encoder embedding dimension"})
```

### Model Usage Migration

**Before:**
```python
# Original usage
import fairseq
models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
model = models[0]

# Forward pass
output = model(source, target_list, padding_mask)
```

**After:**
```python
# New usage
from avhubert_hf import AVHubertModel, AVHubertConfig

# Load model
config = AVHubertConfig()
model = AVHubertModel(config)

# Forward pass
output = model(source, target_list=target_list, padding_mask=padding_mask)
```

### Training Migration

**Before:**
```python
# Original training (command line)
fairseq-hydra-train --config-dir ./conf/ --config-name pretrain.yaml \
  task.data=/path/to/data task.label_dir=/path/to/labels \
  model.label_rate=100 hydra.run.dir=/path/to/experiment/
```

**After:**
```python
# New training (Python)
from avhubert_hf import AVHubertTrainer, TrainingConfig

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

## 🧪 Testing Migration

### 1. Unit Tests

Create tests to verify each component works correctly:

```python
import pytest
from avhubert_hf import AVHubertModel, AVHubertConfig

def test_model_creation():
    config = AVHubertConfig(encoder_layers=6)
    model = AVHubertModel(config)
    assert model is not None

def test_forward_pass():
    config = AVHubertConfig(encoder_layers=6)
    model = AVHubertModel(config)
    
    # Test input
    batch_size, seq_len, features = 2, 1000, 80
    input_tensor = torch.randn(batch_size, seq_len, features)
    
    # Forward pass
    output = model(input_tensor, features_only=True)
    assert "features" in output
```

### 2. Integration Tests

Test the complete pipeline:

```python
def test_training_pipeline():
    # Create configurations
    model_config = AVHubertConfig(encoder_layers=6)
    data_config = DataConfig(data_path="./test_data")
    training_config = TrainingConfig(max_epochs=1)
    
    # Create components
    model = AVHubertModel(model_config)
    dataset = AVHubertDataset(data_config, split="train")
    trainer = AVHubertTrainer(model, dataset, config=training_config)
    
    # Test training
    trainer.train()
```

## 🚨 Common Issues and Solutions

### Issue 1: Import Errors

**Problem:** `ModuleNotFoundError: No module named 'fairseq'`

**Solution:** Remove all fairseq imports and replace with new imports:

```python
# Remove these
from fairseq import utils
from fairseq.models import BaseFairseqModel

# Use these instead
from avhubert_hf.utils import compute_mask_indices
from avhubert_hf.models import AVHubertModel
```

### Issue 2: Configuration Errors

**Problem:** Configuration fields not found

**Solution:** Update configuration field names to match new API:

```python
# Old
config.label_rate

# New
config.label_rate  # Same name, different structure
```

### Issue 3: Data Loading Errors

**Problem:** Dataset not compatible with new DataLoader

**Solution:** Ensure dataset returns the expected format:

```python
# Expected format
{
    "audio": torch.Tensor,
    "video": torch.Tensor,
    "labels": List[str],
    "audio_lengths": torch.Tensor,
    "video_lengths": torch.Tensor,
}
```

### Issue 4: Training Errors

**Problem:** Loss computation errors

**Solution:** Check that target format matches expected format:

```python
# Ensure targets are properly formatted
target_list = [torch.tensor(target) for target in targets]
```

## 📚 Additional Resources

- [Original AV-HuBERT Paper](https://arxiv.org/abs/2201.02184)
- [Original Implementation](https://github.com/facebookresearch/av_hubert)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PyTorch Documentation](https://pytorch.org/docs/)

## 🤝 Getting Help

If you encounter issues during migration:

1. Check the [documentation](docs/)
2. Search existing [issues](https://github.com/your-username/avhubert-hf/issues)
3. Create a new issue with:
   - Original code snippet
   - New code snippet
   - Error message
   - Expected behavior

## 🎉 Migration Complete!

Once you've completed the migration:

1. ✅ All fairseq dependencies removed
2. ✅ New configuration system working
3. ✅ Model loading and inference working
4. ✅ Training pipeline working
5. ✅ All tests passing

Congratulations! You now have a clean, maintainable AV-HuBERT implementation without fairseq dependencies.