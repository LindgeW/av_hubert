#!/usr/bin/env python3
"""
Test script to verify all imports work correctly in the refactored AV-HuBERT package
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test all major imports"""
    
    print("🧪 Testing AV-HuBERT Imports")
    print("=" * 40)
    
    try:
        # Test main package import
        print("📦 Testing main package...")
        import avhubert_hf
        print("✅ avhubert_hf imported successfully")
        
        # Test configuration
        print("📋 Testing configuration...")
        from avhubert_hf.models import AVHubertConfig
        config = AVHubertConfig()
        print(f"✅ AVHubertConfig created: {config.model_type}")
        
        # Test modules
        print("🔧 Testing modules...")
        from avhubert_hf.modules import (
            LayerNorm, GradMultiply, ConvFeatureExtractionModel, 
            ResEncoder, MultiheadAttention, TransformerEncoder
        )
        print("✅ All modules imported successfully")
        
        # Test data components
        print("📊 Testing data components...")
        from avhubert_hf.data import AVHubertProcessor, AVHubertDataset
        processor = AVHubertProcessor()
        print("✅ Data components imported successfully")
        
        # Test utilities
        print("🛠️ Testing utilities...")
        from avhubert_hf.utils import compute_mask_indices
        print("✅ Utilities imported successfully")
        
        # Test specific module functionality
        print("🔍 Testing module instantiation...")
        
        # Test attention
        attention = MultiheadAttention(embed_dim=768, num_heads=12)
        print(f"✅ MultiheadAttention created: {attention.num_heads} heads")
        
        # Test transformer encoder
        encoder = TransformerEncoder(embedding_dim=768, num_layers=12)
        print(f"✅ TransformerEncoder created: {encoder.num_layers} layers")
        
        # Test ResNet encoder
        resnet = ResEncoder()
        print(f"✅ ResEncoder created: {resnet.backend_out} output dim")
        
        # Test layer norm
        layer_norm = LayerNorm(768)
        print("✅ LayerNorm created")
        
        print("\n🎉 All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality"""
    
    print("\n🧪 Testing Basic Functionality")
    print("=" * 40)
    
    try:
        import torch
        from avhubert_hf.utils import compute_mask_indices
        
        # Test mask computation
        print("🎭 Testing mask computation...")
        shape = (2, 100)  # batch_size=2, seq_len=100
        mask = compute_mask_indices(
            shape=shape,
            mask_prob=0.8,
            mask_length=10
        )
        print(f"✅ Mask computed: shape={mask.shape}, ratio={mask.mean():.3f}")
        
        # Test dataset creation
        print("📊 Testing dataset...")
        from avhubert_hf.data import AVHubertDataset
        dataset = AVHubertDataset(data_dir="dummy_data")
        print(f"✅ Dataset created: {len(dataset)} samples")
        
        # Test getting a sample
        print("🎵 Testing sample retrieval...")
        sample = dataset[0]
        print(f"✅ Sample retrieved: keys={list(sample.keys())}")
        
        print("\n🎉 All functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 AV-HuBERT Import Test Suite")
    print("=" * 50)
    
    # Run import tests
    import_success = test_imports()
    
    # Run functionality tests  
    func_success = test_basic_functionality()
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 20)
    print(f"Imports: {'✅ PASS' if import_success else '❌ FAIL'}")
    print(f"Functionality: {'✅ PASS' if func_success else '❌ FAIL'}")
    
    if import_success and func_success:
        print("\n🎉 All tests passed! The refactored package is working correctly.")
        exit_code = 0
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        exit_code = 1
    
    print(f"\n📊 Final Status: {'SUCCESS' if exit_code == 0 else 'FAILED'}")
    exit(exit_code)