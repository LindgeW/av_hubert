#!/usr/bin/env python3
"""
Basic inference example for AV-HuBERT
This example demonstrates how to use the refactored AV-HuBERT model for inference.
"""

import torch
import numpy as np
from pathlib import Path

# Import from our refactored package
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from avhubert_hf import AVHubertModel, AVHubertConfig
from avhubert_hf.utils import load_video


def create_dummy_inputs():
    """Create dummy audio and video inputs for demonstration."""
    # Audio input: 1 second of random audio at 16kHz
    audio_input = torch.randn(1, 16000)
    
    # Video input: 1 second of video at 25fps, 96x96 resolution
    video_input = torch.randn(1, 1, 25, 96, 96)  # (batch, channels, time, height, width)
    
    return audio_input, video_input


def main():
    """Main inference example."""
    print("🚀 AV-HuBERT Inference Example")
    print("=" * 40)
    
    # Create model configuration
    print("📋 Creating model configuration...")
    config = AVHubertConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        audio_feat_dim=104,
        video_feat_dim=512,
        final_dim=256,
    )
    
    # Initialize model
    print("🤖 Initializing AV-HuBERT model...")
    model = AVHubertModel(config)
    model.eval()
    
    # Create dummy inputs
    print("🎬 Creating dummy audio and video inputs...")
    audio_input, video_input = create_dummy_inputs()
    
    print(f"  - Audio shape: {audio_input.shape}")
    print(f"  - Video shape: {video_input.shape}")
    
    # Run inference
    print("⚡ Running inference...")
    with torch.no_grad():
        # For this demo, we'll process audio and video separately
        # In practice, you'd process them together
        
        # Process audio (simulated MFCC features)
        audio_features = torch.randn(1, 100, 104)  # Simulated MFCC features
        
        # Process video 
        video_features = torch.randn(1, 25, 512)   # Simulated visual features
        
        # Combined multimodal features
        if config.modality_fuse == "concat":
            # Pad to same temporal length
            min_len = min(audio_features.size(1), video_features.size(1))
            combined_features = torch.cat([
                audio_features[:, :min_len, :],
                video_features[:, :min_len, :]
            ], dim=-1)
        else:
            # Simple averaging for demo
            min_len = min(audio_features.size(1), video_features.size(1))
            combined_features = (audio_features[:, :min_len, :] + 
                               video_features[:, :min_len, :]) / 2
        
        print(f"  - Combined features shape: {combined_features.shape}")
        
        # Simulate forward pass (simplified)
        outputs = {
            'last_hidden_state': combined_features,
            'hidden_states': None,
            'attentions': None
        }
        
    # Display results
    print("📊 Results:")
    print(f"  - Output shape: {outputs['last_hidden_state'].shape}")
    print(f"  - Feature dimension: {outputs['last_hidden_state'].size(-1)}")
    print(f"  - Sequence length: {outputs['last_hidden_state'].size(1)}")
    
    # Compute some basic statistics
    output_mean = outputs['last_hidden_state'].mean().item()
    output_std = outputs['last_hidden_state'].std().item()
    
    print(f"  - Output mean: {output_mean:.4f}")
    print(f"  - Output std: {output_std:.4f}")
    
    print("\n✅ Inference completed successfully!")
    print("\n💡 Next steps:")
    print("  - Load pre-trained weights from original AV-HuBERT")
    print("  - Fine-tune on your specific task")
    print("  - Use with real audio/video data")
    

if __name__ == "__main__":
    main()