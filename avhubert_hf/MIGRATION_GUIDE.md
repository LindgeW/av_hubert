# Migration Guide: From Fairseq to HuggingFace AV-HuBERT

This guide helps you migrate from the original fairseq-based AV-HuBERT implementation to our HuggingFace-compatible version.

## 🔄 Quick Migration

### Before (Fairseq)
```python
import fairseq
import hubert_pretraining, hubert

# Load model
ckpt_path = "/path/to/checkpoint.pt"
models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
model = models[0]

# Training with fairseq-hydra
fairseq-hydra-train --config-dir /path/to/conf/ --config-name conf-name \
  task.data=/path/to/data task.label_dir=/path/to/label \
  model.label_rate=100 hydra.run.dir=/path/to/experiment/pretrain/ \
  common.user_dir=`pwd`
```

### After (HuggingFace)
```python
from avhubert_hf import AVHubertModel, AVHubertConfig, AVHubertProcessor

# Load model
config = AVHubertConfig.from_pretrained("facebook/avhubert-base")
model = AVHubertModel.from_pretrained("facebook/avhubert-base")
processor = AVHubertProcessor()

# Training with HuggingFace Trainer
from transformers import Trainer, TrainingArguments

trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="./results"),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor,
)
trainer.train()
```

## 📊 Component Mapping

| Fairseq Component | HuggingFace Equivalent | Notes |
|-------------------|------------------------|-------|
| `fairseq.models.hubert.AVHubertModel` | `avhubert_hf.AVHubertModel` | Main model class |
| `fairseq.tasks.av_hubert_pretraining` | `avhubert_hf.AVHubertProcessor` | Data processing |
| `fairseq.criterions.av_hubert` | Built into model | Loss computation |
| `fairseq-hydra-train` | `transformers.Trainer` | Training loop |
| Hydra configs | `AVHubertConfig` | Configuration |

## 🔧 Configuration Changes

### Fairseq Config (YAML)
```yaml
model:
  _name: av_hubert
  label_rate: 100
  encoder_layers: 12
  encoder_embed_dim: 768
  final_dim: 256
  modality_fuse: concat
```

### HuggingFace Config (Python)
```python
config = AVHubertConfig(
    label_rate=100.0,
    num_hidden_layers=12,
    hidden_size=768,
    final_dim=256,
    modality_fuse="concat"
)
```

## 🏋️ Training Migration

### 1. Data Loading

**Before:**
```python
# Fairseq dataset
from fairseq.tasks.av_hubert_pretraining import AVHubertPretrainingTask
task = AVHubertPretrainingTask.setup_task(cfg.task)
```

**After:**
```python
# HuggingFace dataset
from avhubert_hf.data import AVHubertDataset
dataset = AVHubertDataset(data_dir="/path/to/data")
```

### 2. Model Initialization

**Before:**
```python
# Fairseq model
from fairseq.models import build_model
model = build_model(cfg.model, task)
```

**After:**
```python
# HuggingFace model
from avhubert_hf import AVHubertModel, AVHubertConfig
config = AVHubertConfig(**config_dict)
model = AVHubertModel(config)
```

### 3. Training Loop

**Before:**
```bash
# Command line training
fairseq-hydra-train --config-dir conf/ --config-name base_lrs3_iter1
```

**After:**
```python
# Python training
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

## 🎯 Inference Migration

### Before (Fairseq)
```python
# Fairseq inference
import fairseq
models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
model = models[0]

# Manual preprocessing required
# ... complex preprocessing code ...

# Forward pass
with torch.no_grad():
    output = model(source, target)
```

### After (HuggingFace)
```python
# HuggingFace inference
from avhubert_hf import AVHubertModel, AVHubertProcessor

model = AVHubertModel.from_pretrained("facebook/avhubert-base")
processor = AVHubertProcessor()

# Simple preprocessing
inputs = processor(
    audio="path/to/audio.wav",
    video="path/to/video.mp4",
    return_tensors="pt"
)

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)
    hidden_states = outputs.last_hidden_state
```

## 🔧 Key Differences

### 1. Dependencies
- **Removed**: fairseq, hydra-core (for training)
- **Added**: transformers, torch, numpy, opencv-python

### 2. Configuration
- **Before**: YAML-based Hydra configs
- **After**: Python-based config classes

### 3. Training
- **Before**: Command-line hydra training
- **After**: Python-based Trainer API

### 4. Model Loading
- **Before**: `fairseq.checkpoint_utils.load_model_ensemble_and_task`
- **After**: `AVHubertModel.from_pretrained()`

### 5. Data Processing
- **Before**: Manual preprocessing with fairseq utilities
- **After**: Unified `AVHubertProcessor`

## ⚡ Performance Considerations

1. **Memory Usage**: Similar to original implementation
2. **Speed**: Comparable inference speed, potentially faster training with Trainer optimizations
3. **Compatibility**: Full backward compatibility with original checkpoints (with conversion)

## 🐛 Common Migration Issues

### Issue 1: Import Errors
```python
# ❌ Old import
from fairseq.models.hubert import AVHubertModel

# ✅ New import  
from avhubert_hf import AVHubertModel
```

### Issue 2: Config Format
```python
# ❌ Old config loading
cfg = OmegaConf.load("config.yaml")

# ✅ New config loading
config = AVHubertConfig.from_pretrained("facebook/avhubert-base")
```

### Issue 3: Training Script
```bash
# ❌ Old training
fairseq-hydra-train --config-dir conf/ --config-name base

# ✅ New training
python train.py --model_name_or_path facebook/avhubert-base
```

## 📚 Additional Resources

- [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers)
- [Original AV-HuBERT Paper](https://arxiv.org/abs/2201.02184)
- [Example Scripts](./examples/)

## 🆘 Getting Help

If you encounter issues during migration:

1. Check the [examples](./examples/) directory
2. Open an issue on GitHub
3. Refer to the original fairseq documentation for model details