"""
Basic test for the refactored AV-HuBERT package.
"""

import torch
import numpy as np
import sys
import os

# Add the package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'avhubert_hf'))

from avhubert_hf import (
    AVHubertConfig,
    AVHubertModel,
    AVHubertForPreTraining,
    AVHubertFeatureExtractor,
)


def test_config():
    """Test configuration creation."""
    print("Testing configuration...")
    
    config = AVHubertConfig(
        encoder_layers=6,
        encoder_embed_dim=256,
        encoder_attention_heads=8,
        final_dim=256,
    )
    
    assert config.encoder_layers == 6
    assert config.encoder_embed_dim == 256
    assert config.encoder_attention_heads == 8
    assert config.final_dim == 256
    
    print("✓ Configuration test passed")


def test_basic_model():
    """Test basic model creation and forward pass."""
    print("Testing basic model...")
    
    config = AVHubertConfig(
        encoder_layers=6,
        encoder_embed_dim=256,
        encoder_attention_heads=8,
        final_dim=256,
    )
    
    model = AVHubertModel(config)
    
    # Test audio-only input
    audio_input = torch.randn(2, 16000)
    with torch.no_grad():
        outputs = model(input_values=audio_input)
    
    expected_shape = (2, 50, 256)  # batch_size, seq_len_after_feature_extraction, final_dim
    assert outputs.last_hidden_state.shape == expected_shape
    
    print("✓ Basic model test passed")


def test_audio_video_model():
    """Test model with both audio and video inputs."""
    print("Testing audio-video model...")
    
    config = AVHubertConfig(
        encoder_layers=6,
        encoder_embed_dim=256,
        encoder_attention_heads=8,
        final_dim=256,
        modality_fuse="concat",
    )
    
    model = AVHubertModel(config)
    
    # Test audio-video input
    audio_input = torch.randn(2, 16000)
    video_input = torch.randn(2, 1, 25, 88, 88)
    
    with torch.no_grad():
        outputs = model(
            input_values=audio_input,
            video_values=video_input,
        )
    
    expected_shape = (2, 50, 256)
    assert outputs.last_hidden_state.shape == expected_shape
    
    print("✓ Audio-video model test passed")


def test_pretraining_model():
    """Test pre-training model."""
    print("Testing pre-training model...")
    
    config = AVHubertConfig(
        encoder_layers=6,
        encoder_embed_dim=256,
        encoder_attention_heads=8,
        final_dim=256,
        mask_prob_audio=0.65,
        mask_length_audio=10,
    )
    
    model = AVHubertForPreTraining(config)
    
    # Create inputs
    audio_input = torch.randn(2, 16000)
    video_input = torch.randn(2, 1, 25, 88, 88)
    mask_time_indices = torch.zeros(2, 50, dtype=torch.bool)
    mask_time_indices[:, :10] = True
    labels = torch.randint(0, 100, (2, 50))
    
    # Forward pass
    outputs = model(
        input_values=audio_input,
        video_values=video_input,
        mask_time_indices=mask_time_indices,
        labels=labels,
    )
    
    assert outputs.loss is not None
    assert outputs.logits.shape == (2, 50, 256)
    
    print("✓ Pre-training model test passed")


def test_feature_extractor():
    """Test feature extractor."""
    print("Testing feature extractor...")
    
    feature_extractor = AVHubertFeatureExtractor(
        sampling_rate=16000,
        do_normalize=True,
    )
    
    # Create dummy audio
    audio = np.random.randn(16000).astype(np.float32)
    
    # Extract features
    features = feature_extractor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
    )
    
    assert "input_values" in features
    assert "attention_mask" in features
    
    print("✓ Feature extractor test passed")


def test_model_parameters():
    """Test that model parameters are properly initialized."""
    print("Testing model parameters...")
    
    config = AVHubertConfig(
        encoder_layers=6,
        encoder_embed_dim=256,
        encoder_attention_heads=8,
        final_dim=256,
    )
    
    model = AVHubertModel(config)
    
    # Check that parameters exist and are trainable
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    assert total_params > 0
    assert trainable_params > 0
    assert total_params == trainable_params  # All parameters should be trainable
    
    print(f"✓ Model has {total_params:,} parameters")
    print("✓ Model parameters test passed")


def main():
    """Run all tests."""
    print("Running AV-HuBERT refactored package tests...")
    print("=" * 50)
    
    try:
        test_config()
        test_basic_model()
        test_audio_video_model()
        test_pretraining_model()
        test_feature_extractor()
        test_model_parameters()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed successfully!")
        print("The refactored AV-HuBERT package is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)