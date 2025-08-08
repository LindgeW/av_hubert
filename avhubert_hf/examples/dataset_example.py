#!/usr/bin/env python3
"""
Dataset usage example for AV-HuBERT
This example demonstrates how to use the AVHubertDataset for different tasks.
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys

# Import from our refactored package
sys.path.append(str(Path(__file__).parent.parent / "src"))

from avhubert_hf.data import AVHubertDataset, AVHubertPretrainingDataset, AVHubertASRDataset


def demo_pretraining_dataset():
    """Demo pretraining dataset usage."""
    print("🚀 Pretraining Dataset Demo")
    print("=" * 40)
    
    # Create pretraining dataset (will use dummy data since no real data is available)
    dataset = AVHubertPretrainingDataset(
        data_dir="./dummy_data",  # This will trigger dummy data creation
        split="train",
        modalities=["audio", "video"],
        max_length=100,
        mask_prob_audio=0.8,
        mask_length_audio=10,
        mask_prob_video=0.8,
        mask_length_video=10,
    )
    
    print(f"📊 Dataset size: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    print(f"📋 Sample keys: {list(sample.keys())}")
    
    if 'audio_features' in sample:
        print(f"🎵 Audio features shape: {sample['audio_features'].shape}")
    if 'video_features' in sample:
        print(f"🎬 Video features shape: {sample['video_features'].shape}")
    if 'attention_mask' in sample:
        print(f"👀 Attention mask shape: {sample['attention_mask'].shape}")
    if 'audio_mask' in sample:
        print(f"🎭 Audio mask shape: {sample['audio_mask'].shape}")
        print(f"🎭 Audio mask ratio: {sample['audio_mask'].float().mean():.3f}")
    if 'video_mask' in sample:
        print(f"🎭 Video mask shape: {sample['video_mask'].shape}")
        print(f"🎭 Video mask ratio: {sample['video_mask'].float().mean():.3f}")
    
    # Create DataLoader
    print("\n📦 Creating DataLoader...")
    dataloader = DataLoader(
        dataset, 
        batch_size=4, 
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=0
    )
    
    # Get a batch
    batch = next(iter(dataloader))
    print(f"📦 Batch keys: {list(batch.keys())}")
    
    if 'audio_features' in batch:
        print(f"🎵 Batch audio features shape: {batch['audio_features'].shape}")
    if 'video_features' in batch:
        print(f"🎬 Batch video features shape: {batch['video_features'].shape}")
    
    print("✅ Pretraining dataset demo completed!\n")


def demo_asr_dataset():
    """Demo ASR dataset usage."""
    print("🎤 ASR Dataset Demo")
    print("=" * 40)
    
    # Create ASR dataset (audio-only)
    dataset = AVHubertASRDataset(
        data_dir="./dummy_data",
        split="train",
        max_length=150,
    )
    
    print(f"📊 Dataset size: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    print(f"📋 Sample keys: {list(sample.keys())}")
    
    if 'audio_features' in sample:
        print(f"🎵 Audio features shape: {sample['audio_features'].shape}")
    if 'labels' in sample:
        print(f"📝 Labels shape: {sample['labels'].shape}")
        print(f"📝 Labels: {sample['labels']}")
    
    print("✅ ASR dataset demo completed!\n")


def demo_multimodal_dataset():
    """Demo custom multimodal dataset configuration."""
    print("🎭 Custom Multimodal Dataset Demo")
    print("=" * 40)
    
    # Create custom dataset with specific parameters
    dataset = AVHubertDataset(
        data_dir="./dummy_data",
        split="train",
        modalities=["audio", "video"],
        task_type="pretraining",
        max_length=80,
        # Audio parameters
        audio_sr=16000,
        n_mfcc=13,
        stack_order=4,
        # Video parameters
        video_size=(96, 96),
        crop_size=(88, 88),
        target_fps=25,
        # Masking parameters
        mask_prob_audio=0.75,
        mask_length_audio=8,
        mask_prob_video=0.75,
        mask_length_video=8,
    )
    
    print(f"📊 Dataset size: {len(dataset)}")
    print(f"🎯 Modalities: {dataset.modalities}")
    print(f"🎯 Task type: {dataset.task_type}")
    print(f"📏 Max length: {dataset.max_length}")
    
    # Test with DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=2, 
        shuffle=False,
        collate_fn=dataset.collate_fn
    )
    
    # Process a few batches
    print("\n📦 Processing batches...")
    for i, batch in enumerate(dataloader):
        if i >= 3:  # Just process first 3 batches
            break
        
        print(f"  Batch {i+1}:")
        if 'audio_features' in batch:
            print(f"    🎵 Audio: {batch['audio_features'].shape}")
        if 'video_features' in batch:
            print(f"    🎬 Video: {batch['video_features'].shape}")
        if 'attention_mask' in batch:
            print(f"    👀 Attention: {batch['attention_mask'].shape}")
    
    print("✅ Custom multimodal dataset demo completed!\n")


def demo_dataset_variations():
    """Demo different dataset configurations."""
    print("🔄 Dataset Variations Demo")
    print("=" * 40)
    
    configurations = [
        {"modalities": ["audio"], "name": "Audio-only"},
        {"modalities": ["video"], "name": "Video-only"},  
        {"modalities": ["audio", "video"], "name": "Audio-Visual"},
    ]
    
    for config in configurations:
        print(f"\n📋 {config['name']} Dataset:")
        
        dataset = AVHubertDataset(
            data_dir="./dummy_data",
            split="train",
            modalities=config["modalities"],
            max_length=50,
        )
        
        sample = dataset[0]
        print(f"  📊 Sample keys: {list(sample.keys())}")
        
        for key, tensor in sample.items():
            if isinstance(tensor, torch.Tensor):
                print(f"  📐 {key}: {tensor.shape}")
    
    print("\n✅ Dataset variations demo completed!")


def main():
    """Main demo function."""
    print("🎬 AV-HuBERT Dataset Examples")
    print("=" * 50)
    
    try:
        # Run different demos
        demo_pretraining_dataset()
        demo_asr_dataset()
        demo_multimodal_dataset()
        demo_dataset_variations()
        
        print("\n🎉 All dataset demos completed successfully!")
        print("\n💡 Next steps:")
        print("  - Replace dummy data with real audio/video files")
        print("  - Create proper TSV manifest files")
        print("  - Implement real feature extraction pipelines")
        print("  - Add proper tokenization for text labels")
        
    except Exception as e:
        print(f"❌ Error running demos: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()