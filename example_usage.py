#!/usr/bin/env python3
"""
Example usage of the simplified AVHuBERT implementation.

This script demonstrates how to use the HuggingFace-style AVHuBERT interface
for audio-visual representation learning.
"""

import torch
import numpy as np
from avhubert_simple import AVHuBERTModel, AVHuBERTConfig, AVHuBERTFeatureExtractor, AVHuBERTProcessor


def create_dummy_data(batch_size=2, audio_length=16000, video_frames=30, video_size=(88, 88)):
    """Create dummy audio and video data for testing."""
    # Create dummy audio data (16kHz, 1 second)
    audio = torch.randn(batch_size, audio_length)
    
    # Create dummy video data (30 frames, grayscale, 88x88)
    video = torch.randn(batch_size, video_frames, 1, video_size[0], video_size[1])
    
    return audio, video


def example_basic_usage():
    """Example of basic model usage."""
    print("=== Basic Model Usage ===")
    
    # Create configuration
    config = AVHuBERTConfig(
        encoder_layers=6,  # Smaller model for demo
        encoder_embed_dim=256,
        encoder_ffn_embed_dim=1024,
        encoder_attention_heads=8,
    )
    
    # Create model
    model = AVHuBERTModel(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create dummy data
    audio, video = create_dummy_data()
    print(f"Audio shape: {audio.shape}")
    print(f"Video shape: {video.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(audio, video, mask=False, features_only=True)
    
    print(f"Output features shape: {outputs['features'].shape}")
    print()


def example_feature_extraction():
    """Example of feature extraction."""
    print("=== Feature Extraction ===")
    
    # Create feature extractor
    feature_extractor = AVHuBERTFeatureExtractor(
        video_size=(88, 88),
        video_mean=0.421,
        video_std=0.1652,
        do_normalize=True,
        do_center_crop=True,
    )
    
    # Create dummy data
    audio = np.random.randn(16000).astype(np.float32)
    video = np.random.randn(30, 88, 88).astype(np.float32)  # [T, H, W]
    
    # Extract features
    features = feature_extractor(
        audio=audio,
        video=video,
        return_tensors="pt"
    )
    
    print(f"Processed audio shape: {features['audio'].shape}")
    print(f"Processed video shape: {features['video'].shape}")
    print()


def example_processor_usage():
    """Example of using the processor for end-to-end inference."""
    print("=== Processor Usage ===")
    
    # Create configuration
    config = AVHuBERTConfig(
        encoder_layers=4,  # Small model for demo
        encoder_embed_dim=128,
        encoder_ffn_embed_dim=512,
        encoder_attention_heads=4,
    )
    
    # Create processor
    feature_extractor = AVHuBERTFeatureExtractor()
    model = AVHuBERTModel(config)
    processor = AVHuBERTProcessor(feature_extractor, model)
    
    # Create dummy data
    audio = np.random.randn(16000).astype(np.float32)
    video = np.random.randn(30, 88, 88).astype(np.float32)
    
    # Run inference
    with torch.no_grad():
        outputs = processor(
            audio=audio,
            video=video,
            return_tensors="pt"
        )
    
    print(f"Model outputs: {list(outputs.keys())}")
    print(f"Features shape: {outputs['features'].shape}")
    print()


def example_feature_extraction_with_processor():
    """Example of feature extraction using the processor."""
    print("=== Feature Extraction with Processor ===")
    
    # Create processor
    config = AVHuBERTConfig(
        encoder_layers=4,
        encoder_embed_dim=128,
        encoder_ffn_embed_dim=512,
        encoder_attention_heads=4,
    )
    
    feature_extractor = AVHuBERTFeatureExtractor()
    model = AVHuBERTModel(config)
    processor = AVHuBERTProcessor(feature_extractor, model)
    
    # Create dummy data
    audio = np.random.randn(16000).astype(np.float32)
    video = np.random.randn(30, 88, 88).astype(np.float32)
    
    # Extract features
    with torch.no_grad():
        features = processor.extract_features(
            audio=audio,
            video=video,
            mask=False,
        )
    
    print(f"Extracted features shape: {features['features'].shape}")
    print(f"Padding mask shape: {features['padding_mask'].shape}")
    print()


def example_prediction():
    """Example of running predictions."""
    print("=== Prediction Example ===")
    
    # Create processor
    config = AVHuBERTConfig(
        encoder_layers=4,
        encoder_embed_dim=128,
        encoder_ffn_embed_dim=512,
        encoder_attention_heads=4,
    )
    
    feature_extractor = AVHuBERTFeatureExtractor()
    model = AVHuBERTModel(config)
    processor = AVHuBERTProcessor(feature_extractor, model)
    
    # Create dummy data
    audio = np.random.randn(16000).astype(np.float32)
    video = np.random.randn(30, 88, 88).astype(np.float32)
    
    # Run prediction
    with torch.no_grad():
        predictions = processor.predict(
            audio=audio,
            video=video,
        )
    
    print(f"Prediction keys: {list(predictions.keys())}")
    print(f"Features shape: {predictions['features'].shape}")
    print(f"Embeddings shape: {predictions['embeddings'].shape}")
    print()


def example_custom_configuration():
    """Example of custom model configuration."""
    print("=== Custom Configuration ===")
    
    # Create custom configuration
    config = AVHuBERTConfig(
        # Model architecture
        encoder_layers=12,
        encoder_embed_dim=768,
        encoder_ffn_embed_dim=3072,
        encoder_attention_heads=12,
        activation_fn="gelu",
        
        # Dropouts
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.0,
        
        # Masking
        mask_length_audio=10,
        mask_prob_audio=0.65,
        mask_length_image=10,
        mask_prob_image=0.65,
        
        # Modality fusion
        modality_fuse="concat",  # or "add"
        modality_dropout=0.0,
        
        # ResNet
        resnet_relu_type="prelu",
    )
    
    # Create model with custom config
    model = AVHuBERTModel(config)
    print(f"Custom model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    audio, video = create_dummy_data()
    with torch.no_grad():
        outputs = model(audio, video, mask=True, features_only=True)
    
    print(f"Output features shape: {outputs['features'].shape}")
    print()


if __name__ == "__main__":
    print("AVHuBERT Simplified Implementation Examples")
    print("=" * 50)
    
    try:
        example_basic_usage()
        example_feature_extraction()
        example_processor_usage()
        example_feature_extraction_with_processor()
        example_prediction()
        example_custom_configuration()
        
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()