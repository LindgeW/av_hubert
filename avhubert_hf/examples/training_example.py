#!/usr/bin/env python3
"""
Training example for AV-HuBERT.

This script demonstrates how to:
1. Set up configurations for model, data, and training
2. Create datasets and data loaders
3. Initialize the model and trainer
4. Run training with validation
"""

import os
import torch
import logging
from pathlib import Path

from avhubert_hf import (
    AVHubertModel, AVHubertConfig, DataConfig, TrainingConfig,
    AVHubertDataset, AVHubertTrainer
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dummy_data(data_dir: str):
    """Create dummy data for demonstration purposes."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy manifest files
    for split in ["train", "valid"]:
        manifest_file = data_dir / f"{split}.tsv"
        label_file = data_dir / f"{split}.ltr"
        
        # Create dummy manifest
        with open(manifest_file, "w") as f:
            f.write("/dummy/path\n")  # Root path
            for i in range(10):  # 10 dummy samples
                f.write(f"sample_{i}\t/path/to/video_{i}.mp4\t/path/to/audio_{i}.wav\t16000\n")
        
        # Create dummy labels
        with open(label_file, "w") as f:
            for i in range(10):
                f.write("a b c d e\n")  # Dummy labels
    
    logger.info(f"Created dummy data in {data_dir}")


def main():
    """Main function demonstrating AV-HuBERT training."""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create dummy data for demonstration
    data_dir = "./dummy_data"
    create_dummy_data(data_dir)
    
    # Model configuration
    model_config = AVHubertConfig(
        encoder_layers=6,  # Smaller model for demo
        encoder_embed_dim=256,
        encoder_ffn_embed_dim=1024,
        encoder_attention_heads=8,
        dropout=0.1,
        attention_dropout=0.1,
        mask_prob_audio=0.65,
        mask_length_audio=10,
        final_dim=256,
    )
    
    # Data configuration
    data_config = DataConfig(
        data_path=data_dir,
        label_dir=data_dir,
        sample_rate=16000,
        label_rate=100,
        labels=["ltr"],
        max_sample_size=16000,  # 1 second at 16kHz
        min_sample_size=8000,   # 0.5 second at 16kHz
        modalities=["audio"],  # Only audio for demo
    )
    
    # Training configuration
    training_config = TrainingConfig(
        batch_size=2,
        learning_rate=1e-4,
        max_epochs=2,  # Small number for demo
        warmup_steps=100,
        gradient_clip_val=1.0,
        save_dir="./checkpoints",
        log_every_n_steps=1,
    )
    
    logger.info("Created configurations")
    
    try:
        # Create datasets
        train_dataset = AVHubertDataset(data_config, split="train")
        val_dataset = AVHubertDataset(data_config, split="valid")
        
        logger.info(f"Created datasets - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        
        # Create model
        model = AVHubertModel(model_config)
        
        logger.info("Created model")
        
        # Create trainer
        trainer = AVHubertTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=training_config,
            device=device,
        )
        
        logger.info("Created trainer")
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        logger.info("Training completed!")
        
        # Test evaluation
        logger.info("Running evaluation...")
        eval_results = trainer.evaluate(val_dataset)
        logger.info(f"Evaluation results: {eval_results}")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        logger.info("This is expected since we're using dummy data")
    
    # Clean up
    import shutil
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    if os.path.exists("./checkpoints"):
        shutil.rmtree("./checkpoints")
    
    logger.info("Training example completed!")


if __name__ == "__main__":
    main()