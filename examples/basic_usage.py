"""
Basic usage example for the refactored AV-HuBERT package.
"""

import torch
import numpy as np
from avhubert_hf import (
    AVHubertConfig,
    AVHubertModel,
    AVHubertForPreTraining,
    AVHubertForCTC,
    AVHubertFeatureExtractor,
    AVHubertTokenizer,
    AVHubertProcessor,
)


def example_basic_model():
    """Example of creating and using the basic AV-HuBERT model."""
    print("=== Basic AV-HuBERT Model Example ===")
    
    # Create configuration
    config = AVHubertConfig(
        encoder_layers=6,  # Smaller model for demo
        encoder_embed_dim=256,
        encoder_attention_heads=8,
        final_dim=256,
    )
    
    # Create model
    model = AVHubertModel(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create dummy input
    batch_size, seq_len = 2, 16000  # 1 second of audio at 16kHz
    audio_input = torch.randn(batch_size, seq_len)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_values=audio_input)
    
    print(f"Input shape: {audio_input.shape}")
    print(f"Output shape: {outputs.last_hidden_state.shape}")
    print()


def example_pretraining_model():
    """Example of using the pre-training model."""
    print("=== Pre-training Model Example ===")
    
    # Create configuration
    config = AVHubertConfig(
        encoder_layers=6,
        encoder_embed_dim=256,
        encoder_attention_heads=8,
        final_dim=256,
    )
    
    # Create pre-training model
    model = AVHubertForPreTraining(config)
    
    # Create dummy input
    batch_size, seq_len = 2, 16000
    audio_input = torch.randn(batch_size, seq_len)
    
    # Create mask indices for pre-training
    mask_time_indices = torch.zeros(batch_size, seq_len // 320, dtype=torch.bool)  # After feature extraction
    mask_time_indices[:, :10] = True  # Mask first 10 frames
    
    # Create dummy labels
    labels = torch.randint(0, 100, (batch_size, seq_len // 320))
    
    # Forward pass
    outputs = model(
        input_values=audio_input,
        mask_time_indices=mask_time_indices,
        labels=labels,
    )
    
    print(f"Loss: {outputs.loss.item():.4f}")
    print(f"Logits shape: {outputs.logits.shape}")
    print()


def example_audio_video_model():
    """Example of using the model with both audio and video inputs."""
    print("=== Audio-Video Model Example ===")
    
    # Create configuration
    config = AVHubertConfig(
        encoder_layers=6,
        encoder_embed_dim=256,
        encoder_attention_heads=8,
        final_dim=256,
        modality_fuse="concat",  # Concatenate audio and video features
    )
    
    # Create model
    model = AVHubertModel(config)
    
    # Create dummy inputs
    batch_size, seq_len = 2, 16000
    audio_input = torch.randn(batch_size, seq_len)
    
    # Video input: (batch_size, channels, time, height, width)
    video_input = torch.randn(batch_size, 1, 25, 88, 88)  # 25 frames, 88x88 resolution
    
    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_values=audio_input,
            video_values=video_input,
        )
    
    print(f"Audio input shape: {audio_input.shape}")
    print(f"Video input shape: {video_input.shape}")
    print(f"Output shape: {outputs.last_hidden_state.shape}")
    print()


def example_feature_extractor():
    """Example of using the feature extractor."""
    print("=== Feature Extractor Example ===")
    
    # Create feature extractor
    feature_extractor = AVHubertFeatureExtractor(
        sampling_rate=16000,
        do_normalize=True,
    )
    
    # Create dummy audio
    audio = np.random.randn(16000).astype(np.float32)  # 1 second of audio
    
    # Extract features
    features = feature_extractor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
    )
    
    print(f"Audio shape: {audio.shape}")
    print(f"Features keys: {features.keys()}")
    print(f"Input values shape: {features['input_values'].shape}")
    print()


def example_processor():
    """Example of using the processor."""
    print("=== Processor Example ===")
    
    # Create a simple vocabulary for demo
    vocab = {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3, "hello": 4, "world": 5}
    
    # Save vocabulary to file
    with open("temp_vocab.txt", "w") as f:
        for token in vocab.keys():
            f.write(f"{token}\n")
    
    # Create tokenizer
    tokenizer = AVHubertTokenizer("temp_vocab.txt")
    
    # Create feature extractor
    feature_extractor = AVHubertFeatureExtractor()
    
    # Create processor
    processor = AVHubertProcessor(feature_extractor, tokenizer)
    
    # Process audio and text
    audio = np.random.randn(16000).astype(np.float32)
    text = "hello world"
    
    inputs = processor(
        audio,
        text=text,
        sampling_rate=16000,
        return_tensors="pt",
    )
    
    print(f"Input keys: {inputs.keys()}")
    print(f"Input values shape: {inputs['input_values'].shape}")
    print(f"Labels: {inputs['labels']}")
    
    # Clean up
    import os
    os.remove("temp_vocab.txt")
    print()


def main():
    """Run all examples."""
    print("AV-HuBERT Refactored Package Examples")
    print("=" * 50)
    
    try:
        example_basic_model()
        example_pretraining_model()
        example_audio_video_model()
        example_feature_extractor()
        example_processor()
        
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()