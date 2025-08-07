"""
Simple test to verify the refactored AV-HuBERT package structure.
"""

import os
import sys
import importlib.util

def test_file_structure():
    """Test that all required files exist."""
    print("Testing file structure...")
    
    required_files = [
        "avhubert_hf/__init__.py",
        "avhubert_hf/configuration_avhubert.py",
        "avhubert_hf/modeling_avhubert.py",
        "avhubert_hf/utils.py",
        "avhubert_hf/resnet.py",
        "avhubert_hf/transformer.py",
        "avhubert_hf/tokenization_avhubert.py",
        "avhubert_hf/feature_extraction_avhubert.py",
        "avhubert_hf/processor_avhubert.py",
        "setup.py",
        "requirements.txt",
        "README_REFACTORED.md",
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path} exists")
        else:
            print(f"✗ {file_path} missing")
            return False
    
    return True


def test_import_structure():
    """Test that the package can be imported (without external dependencies)."""
    print("\nTesting import structure...")
    
    # Add the package to the path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'avhubert_hf'))
    
    try:
        # Test importing the main package
        import avhubert_hf
        print("✓ avhubert_hf package imported successfully")
        
        # Test importing individual modules
        modules_to_test = [
            "configuration_avhubert",
            "utils",
            "resnet",
            "transformer",
        ]
        
        for module_name in modules_to_test:
            try:
                module = importlib.import_module(module_name)
                print(f"✓ {module_name} module imported successfully")
            except ImportError as e:
                print(f"✗ Failed to import {module_name}: {e}")
                return False
        
        return True
        
    except ImportError as e:
        print(f"✗ Failed to import avhubert_hf: {e}")
        return False


def test_class_definitions():
    """Test that key classes are defined."""
    print("\nTesting class definitions...")
    
    try:
        # Test configuration class
        from configuration_avhubert import AVHubertConfig
        print("✓ AVHubertConfig class found")
        
        # Test utility functions
        from utils import compute_mask_indices, get_activation_fn
        print("✓ Utility functions found")
        
        # Test ResNet classes
        from resnet import ResEncoder, BasicBlock, ResNet
        print("✓ ResNet classes found")
        
        # Test transformer classes
        from transformer import TransformerEncoder, TransformerDecoder, MultiheadAttention
        print("✓ Transformer classes found")
        
        return True
        
    except ImportError as e:
        print(f"✗ Failed to import classes: {e}")
        return False


def test_configuration():
    """Test configuration creation."""
    print("\nTesting configuration...")
    
    try:
        from configuration_avhubert import AVHubertConfig
        
        # Create a simple configuration
        config = AVHubertConfig(
            encoder_layers=6,
            encoder_embed_dim=256,
            encoder_attention_heads=8,
            final_dim=256,
        )
        
        # Test that configuration attributes are set correctly
        assert config.encoder_layers == 6
        assert config.encoder_embed_dim == 256
        assert config.encoder_attention_heads == 8
        assert config.final_dim == 256
        
        print("✓ Configuration creation and attribute access works")
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_utility_functions():
    """Test utility functions."""
    print("\nTesting utility functions...")
    
    try:
        from utils import get_activation_fn
        
        # Test activation function retrieval
        relu_fn = get_activation_fn("relu")
        gelu_fn = get_activation_fn("gelu")
        
        print("✓ Activation function retrieval works")
        return True
        
    except Exception as e:
        print(f"✗ Utility functions test failed: {e}")
        return False


def main():
    """Run all structure tests."""
    print("Testing AV-HuBERT Refactored Package Structure")
    print("=" * 50)
    
    tests = [
        test_file_structure,
        test_import_structure,
        test_class_definitions,
        test_configuration,
        test_utility_functions,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All structure tests passed!")
        print("The refactored AV-HuBERT package structure is correct.")
        return True
    else:
        print("❌ Some tests failed.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)