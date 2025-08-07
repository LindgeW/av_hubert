#!/usr/bin/env python3
"""
Basic usage example for AV-HuBERT.

This script demonstrates how to:
1. Create a model configuration
2. Initialize the AV-HuBERT model
3. Perform a forward pass
4. Extract features
"""

import torch
import logging

from avhubert_hf import AVHubertModel, AVHubertConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main function demonstrating basic AV-HuBERT usage."""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create model configuration
    config = AVHubertConfig(
        encoder_layers=6,  # Smaller model for demo
        encoder_embed_dim=256,
        encoder_ffn_embed_dim=1024,
        encoder_attention_heads=8,
        dropout=0.1,
        attention_dropout=0.1,
        mask_prob_audio=0.65,
        mask_length_audio=10,
    )
    
    logger.info("Created model configuration")
    
    # Create model
    model = AVHubertModel(config)
    model = model.to(device)
    model.eval()
    
    logger.info("Created and moved model to device")
    
    # Create dummy input (batch_size, sequence_length, features)
    batch_size, seq_len, features = 2, 1000, 80
    audio_input = torch.randn(batch_size, seq_len, features).to(device)
    
    logger.info(f"Created input tensor with shape: {audio_input.shape}")
    
    # Forward pass - feature extraction only
    with torch.no_grad():
        output = model(
            source=audio_input,
            features_only=True,
            modality="audio"
        )
        
        features = output["features"]
        logger.info(f"Extracted features with shape: {features.shape}")
        
        # Print feature statistics
        logger.info(f"Feature mean: {features.mean().item():.4f}")
        logger.info(f"Feature std: {features.std().item():.4f}")
        logger.info(f"Feature min: {features.min().item():.4f}")
        logger.info(f"Feature max: {features.max().item():.4f}")
    
    # Forward pass - with masking (for training)
    with torch.no_grad():
        output = model(
            source=audio_input,
            features_only=False,
            mask=True,
            modality="audio"
        )
        
        if "logits" in output:
            logits = output["logits"]
            logger.info(f"Logits shape: {logits.shape}")
    
    logger.info("Basic usage example completed successfully!")


if __name__ == "__main__":
    main()