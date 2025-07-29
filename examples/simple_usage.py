"""
Example usage of the simplified AVHuBERT implementation.

This script demonstrates how to use the HuggingFace-style interface
for various audio-visual speech recognition tasks.
"""

import torch
import numpy as np
from avhubert_simplified import (
    create_lip_reading_model,
    create_asr_model,
    create_avsr_model,
    AVHubertInference,
    AVHubertConfig,
    quick_start_guide,
    get_model_info,
)


def demo_lip_reading():
    """Demonstrate lip reading (video-only speech recognition)"""
    print("\n" + "="*50)
    print("LIP READING DEMO")
    print("="*50)
    
    # Create a lip reading model
    model = create_lip_reading_model()
    
    # Create dummy video data (in real use, load from file or camera)
    # Video shape: (time_steps, height, width, channels)
    dummy_video = np.random.randint(0, 255, (50, 88, 88, 3), dtype=np.uint8)
    
    print(f"Input video shape: {dummy_video.shape}")
    print("Processing video for lip reading...")
    
    # Perform lip reading
    result = model(dummy_video)
    
    print(f"Output shape: {result['predictions'].shape}")
    print(f"Feature dimension: {result['predictions'].shape[-1]}")
    print("Lip reading completed successfully!")


def demo_audio_speech_recognition():
    """Demonstrate audio-only speech recognition"""
    print("\n" + "="*50)
    print("AUDIO SPEECH RECOGNITION DEMO")
    print("="*50)
    
    # Create an ASR model
    model = create_asr_model()
    
    # Create dummy audio data (in real use, load from file)
    # Audio: 1D waveform at 16kHz sample rate
    sample_rate = 16000
    duration = 3  # 3 seconds
    dummy_audio = np.random.randn(sample_rate * duration).astype(np.float32)
    
    print(f"Input audio shape: {dummy_audio.shape}")
    print(f"Sample rate: {sample_rate} Hz, Duration: {duration}s")
    print("Processing audio for speech recognition...")
    
    # Perform ASR
    result = model(dummy_audio)
    
    print(f"Output shape: {result['predictions'].shape}")
    print(f"Feature dimension: {result['predictions'].shape[-1]}")
    print("Audio speech recognition completed successfully!")


def demo_audio_visual_speech_recognition():
    """Demonstrate audio-visual speech recognition"""
    print("\n" + "="*50)
    print("AUDIO-VISUAL SPEECH RECOGNITION DEMO")
    print("="*50)
    
    # Create an AVSR model
    model = create_avsr_model()
    
    # Create dummy audio and video data
    sample_rate = 16000
    duration = 3
    dummy_audio = np.random.randn(sample_rate * duration).astype(np.float32)
    dummy_video = np.random.randint(0, 255, (50, 88, 88, 3), dtype=np.uint8)
    
    print(f"Input audio shape: {dummy_audio.shape}")
    print(f"Input video shape: {dummy_video.shape}")
    print("Processing audio and video for AVSR...")
    
    # Perform AVSR
    result = model(dummy_audio, dummy_video)
    
    print(f"Output shape: {result['predictions'].shape}")
    print(f"Feature dimension: {result['predictions'].shape[-1]}")
    print("Audio-visual speech recognition completed successfully!")


def demo_general_inference():
    """Demonstrate general-purpose inference interface"""
    print("\n" + "="*50) 
    print("GENERAL INFERENCE DEMO")
    print("="*50)
    
    # Create a general-purpose model with custom config
    config = AVHubertConfig(
        encoder_layers=6,  # Smaller model for demo
        encoder_embed_dim=512,
        encoder_attention_heads=8,
    )
    model = AVHubertInference(config=config)
    
    # Create dummy data
    dummy_audio = np.random.randn(16000 * 2).astype(np.float32)  # 2 seconds
    dummy_video = np.random.randint(0, 255, (25, 88, 88, 3), dtype=np.uint8)  # 1 second at 25fps
    
    print(f"Model config: {config.encoder_layers} layers, {config.encoder_embed_dim} dims")
    print(f"Audio shape: {dummy_audio.shape}")
    print(f"Video shape: {dummy_video.shape}")
    
    # Extract features only
    print("\n1. Extracting features...")
    features = model.extract_features(audio=dummy_audio, video=dummy_video)
    print(f"Hidden states shape: {features['last_hidden_state'].shape}")
    print(f"Projected states shape: {features['projected_states'].shape}")
    
    # Make predictions with features returned
    print("\n2. Making predictions...")
    predictions = model.predict(audio=dummy_audio, video=dummy_video, return_features=True)
    print(f"Predictions shape: {predictions['predictions'].shape}")
    print(f"Features shape: {predictions['features'].shape}")
    
    # Task-specific inference
    print("\n3. Task-specific inference...")
    lip_result = model.lip_reading(dummy_video)
    print(f"Lip reading result shape: {lip_result['predictions'].shape}")
    
    asr_result = model.audio_speech_recognition(dummy_audio)
    print(f"ASR result shape: {asr_result['predictions'].shape}")
    
    avsr_result = model.audio_visual_speech_recognition(dummy_audio, dummy_video)
    print(f"AVSR result shape: {avsr_result['predictions'].shape}")


def demo_feature_similarity():
    """Demonstrate feature similarity computation"""
    print("\n" + "="*50)
    print("FEATURE SIMILARITY DEMO") 
    print("="*50)
    
    model = AVHubertInference()
    
    # Create two different audio samples
    audio1 = np.random.randn(16000).astype(np.float32)
    audio2 = np.random.randn(16000).astype(np.float32)
    
    # Extract features
    features1 = model.extract_features(audio=audio1)['projected_states']
    features2 = model.extract_features(audio=audio2)['projected_states']
    
    # Compute similarity
    similarity = model.get_similarity(features1, features2)
    
    print(f"Features 1 shape: {features1.shape}")
    print(f"Features 2 shape: {features2.shape}")
    print(f"Similarity shape: {similarity.shape}")
    print(f"Average similarity: {similarity.mean().item():.4f}")


def demo_preprocessing():
    """Demonstrate preprocessing utilities"""
    print("\n" + "="*50)
    print("PREPROCESSING DEMO")
    print("="*50)
    
    from avhubert_simplified import (
        create_audio_preprocessor,
        create_video_preprocessor,
        create_avhubert_preprocessor,
    )
    
    # Create preprocessors
    audio_preprocessor = create_audio_preprocessor(
        sample_rate=16000,
        n_mfcc=80,
        normalize=True,
        apply_cmvn=True,
    )
    
    video_preprocessor = create_video_preprocessor(
        image_size=88,
        apply_augmentation=False,
    )
    
    combined_preprocessor = create_avhubert_preprocessor()
    
    # Dummy data
    raw_audio = np.random.randn(32000).astype(np.float32)  # 2 seconds at 16kHz
    raw_video = np.random.randint(0, 255, (50, 112, 112, 3), dtype=np.uint8)
    
    print(f"Raw audio shape: {raw_audio.shape}")
    print(f"Raw video shape: {raw_video.shape}")
    
    # Process audio
    processed_audio = audio_preprocessor.preprocess(raw_audio)
    print(f"Processed audio shape: {processed_audio.shape}")
    
    # Process video
    processed_video = video_preprocessor.preprocess(raw_video)
    print(f"Processed video shape: {processed_video.shape}")
    
    # Combined preprocessing
    audio_feat, video_feat = combined_preprocessor(audio=raw_audio, video=raw_video)
    print(f"Combined audio features shape: {audio_feat.shape}")
    print(f"Combined video features shape: {video_feat.shape}")


def demo_model_configuration():
    """Demonstrate different model configurations"""
    print("\n" + "="*50)
    print("MODEL CONFIGURATION DEMO")
    print("="*50)
    
    # Show model info
    info = get_model_info()
    print("Available models:")
    for name, desc in info["models"].items():
        print(f"  {name}: {desc}")
    
    print("\nAvailable tasks:")
    for name, desc in info["tasks"].items():
        print(f"  {name}: {desc}")
    
    # Create different sized models
    configs = {
        "tiny": AVHubertConfig(
            encoder_layers=4,
            encoder_embed_dim=256,
            encoder_attention_heads=4,
        ),
        "small": AVHubertConfig(
            encoder_layers=8,
            encoder_embed_dim=512,
            encoder_attention_heads=8,
        ),
        "base": AVHubertConfig(),  # Default config
    }
    
    print("\nModel configurations:")
    for name, config in configs.items():
        model = AVHubertInference(config=config)
        total_params = sum(p.numel() for p in model.model.parameters())
        print(f"  {name}: {config.encoder_layers} layers, {config.encoder_embed_dim} dims, "
              f"{total_params:,} parameters")


def main():
    """Run all demos"""
    print("AVHuBERT Simplified - Example Usage")
    print("=====================================")
    
    # Print quick start guide
    quick_start_guide()
    
    # Run demos
    try:
        demo_lip_reading()
        demo_audio_speech_recognition()
        demo_audio_visual_speech_recognition()
        demo_general_inference()
        demo_feature_similarity()
        demo_preprocessing()
        demo_model_configuration()
        
        print("\n" + "="*50)
        print("ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*50)
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install -r requirements_simplified.txt")


if __name__ == "__main__":
    main()