#!/usr/bin/env python3
"""
Simple test script for the simplified AVHuBERT implementation.
"""

import sys
import os

# Add the avhubert_simple directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'avhubert_simple'))

def test_imports():
    """Test that all modules can be imported successfully."""
    print("Testing imports...")
    
    try:
        from avhubert_simple import AVHuBERTConfig, AVHuBERTModel, AVHuBERTFeatureExtractor, AVHuBERTProcessor
        print("✓ All imports successful!")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_config():
    """Test configuration creation."""
    print("\nTesting configuration...")
    
    try:
        from avhubert_simple import AVHuBERTConfig
        
        config = AVHuBERTConfig(
            encoder_layers=4,
            encoder_embed_dim=128,
            encoder_ffn_embed_dim=512,
            encoder_attention_heads=4,
        )
        print("✓ Configuration created successfully!")
        return True
    except Exception as e:
        print(f"✗ Configuration creation failed: {e}")
        return False

def test_model():
    """Test model creation and forward pass."""
    print("\nTesting model...")
    
    try:
        import torch
        from avhubert_simple import AVHuBERTConfig, AVHuBERTModel
        
        # Create small model for testing
        config = AVHuBERTConfig(
            encoder_layers=2,
            encoder_embed_dim=64,
            encoder_ffn_embed_dim=256,
            encoder_attention_heads=4,
        )
        
        model = AVHuBERTModel(config)
        print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Create dummy data
        audio = torch.randn(1, 16000)  # 1 second of audio at 16kHz
        video = torch.randn(1, 30, 1, 88, 88)  # 30 frames, grayscale, 88x88
        
        # Forward pass
        with torch.no_grad():
            outputs = model(audio, video, mask=False, features_only=True)
        
        print(f"✓ Forward pass successful! Output shape: {outputs['features'].shape}")
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_extractor():
    """Test feature extractor."""
    print("\nTesting feature extractor...")
    
    try:
        import numpy as np
        from avhubert_simple import AVHuBERTFeatureExtractor
        
        feature_extractor = AVHuBERTFeatureExtractor(
            video_size=(88, 88),
            do_normalize=True,
        )
        
        # Create dummy data
        audio = np.random.randn(16000).astype(np.float32)
        video = np.random.randn(30, 88, 88).astype(np.float32)
        
        # Process data
        features = feature_extractor(
            audio=audio,
            video=video,
            return_tensors="pt"
        )
        
        print(f"✓ Feature extraction successful!")
        print(f"  Audio shape: {features['audio'].shape}")
        print(f"  Video shape: {features['video'].shape}")
        return True
        
    except Exception as e:
        print(f"✗ Feature extractor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_processor():
    """Test processor."""
    print("\nTesting processor...")
    
    try:
        import numpy as np
        from avhubert_simple import AVHuBERTProcessor, AVHuBERTFeatureExtractor, AVHuBERTConfig, AVHuBERTModel
        
        # Create processor
        config = AVHuBERTConfig(
            encoder_layers=2,
            encoder_embed_dim=64,
            encoder_ffn_embed_dim=256,
            encoder_attention_heads=4,
        )
        
        feature_extractor = AVHuBERTFeatureExtractor()
        model = AVHuBERTModel(config)
        processor = AVHuBERTProcessor(feature_extractor, model)
        
        # Create dummy data
        audio = np.random.randn(16000).astype(np.float32)
        video = np.random.randn(30, 88, 88).astype(np.float32)
        
        import torch
        # Run inference
        with torch.no_grad():
            outputs = processor(audio=audio, video=video)
        
        print(f"✓ Processor test successful!")
        print(f"  Output keys: {list(outputs.keys())}")
        return True
        
    except Exception as e:
        print(f"✗ Processor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("AVHuBERT Simplified Implementation - Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config,
        test_model,
        test_feature_extractor,
        test_processor,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The implementation is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main())