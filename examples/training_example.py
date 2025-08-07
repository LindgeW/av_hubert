"""
Training example for the refactored AV-HuBERT package.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

from avhubert_hf import (
    AVHubertConfig,
    AVHubertForPreTraining,
    AVHubertFeatureExtractor,
)


class DummyAudioVideoDataset(Dataset):
    """Dummy dataset for demonstration purposes."""
    
    def __init__(self, num_samples=100, audio_length=16000, video_frames=25):
        self.num_samples = num_samples
        self.audio_length = audio_length
        self.video_frames = video_frames
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate dummy audio and video data
        audio = torch.randn(self.audio_length)
        video = torch.randn(1, self.video_frames, 88, 88)  # 1 channel, 25 frames, 88x88
        
        # Generate dummy labels (cluster assignments)
        feature_length = self.audio_length // 320  # After feature extraction
        labels = torch.randint(0, 100, (feature_length,))
        
        # Generate mask indices
        mask_time_indices = torch.zeros(feature_length, dtype=torch.bool)
        mask_time_indices[:10] = True  # Mask first 10 frames
        
        return {
            'audio': audio,
            'video': video,
            'labels': labels,
            'mask_time_indices': mask_time_indices,
        }


def collate_fn(batch):
    """Collate function for the dataloader."""
    audio = torch.stack([item['audio'] for item in batch])
    video = torch.stack([item['video'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    mask_time_indices = torch.stack([item['mask_time_indices'] for item in batch])
    
    return {
        'audio': audio,
        'video': video,
        'labels': labels,
        'mask_time_indices': mask_time_indices,
    }


def train_model():
    """Train the AV-HuBERT model."""
    print("=== AV-HuBERT Training Example ===")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create configuration
    config = AVHubertConfig(
        encoder_layers=6,  # Smaller model for demo
        encoder_embed_dim=256,
        encoder_attention_heads=8,
        encoder_ffn_embed_dim=1024,
        final_dim=256,
        modality_fuse="concat",
        mask_prob_audio=0.65,
        mask_length_audio=10,
        mask_prob_image=0.65,
        mask_length_image=10,
    )
    
    # Create model
    model = AVHubertForPreTraining(config)
    model.to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create dataset and dataloader
    dataset = DummyAudioVideoDataset(num_samples=50)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    
    # Create optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    num_epochs = 3
    model.train()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Training epoch {epoch + 1}")
        
        for batch in progress_bar:
            # Move data to device
            audio = batch['audio'].to(device)
            video = batch['video'].to(device)
            labels = batch['labels'].to(device)
            mask_time_indices = batch['mask_time_indices'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
            outputs = model(
                input_values=audio,
                video_values=video,
                mask_time_indices=mask_time_indices,
                labels=labels,
            )
            
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / num_batches:.4f}'
            })
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")
    
    print("\nTraining completed!")
    
    # Save model
    torch.save(model.state_dict(), "avhubert_trained.pth")
    print("Model saved to avhubert_trained.pth")


def inference_example():
    """Example of using the trained model for inference."""
    print("\n=== Inference Example ===")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create configuration (same as training)
    config = AVHubertConfig(
        encoder_layers=6,
        encoder_embed_dim=256,
        encoder_attention_heads=8,
        encoder_ffn_embed_dim=1024,
        final_dim=256,
        modality_fuse="concat",
    )
    
    # Create model
    model = AVHubertForPreTraining(config)
    
    # Load trained weights
    try:
        model.load_state_dict(torch.load("avhubert_trained.pth", map_location=device))
        print("Loaded trained model weights")
    except FileNotFoundError:
        print("No trained weights found, using random initialization")
    
    model.to(device)
    model.eval()
    
    # Create dummy input
    audio = torch.randn(1, 16000).to(device)  # Single sample
    video = torch.randn(1, 1, 25, 88, 88).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(
            input_values=audio,
            video_values=video,
        )
    
    print(f"Input audio shape: {audio.shape}")
    print(f"Input video shape: {video.shape}")
    print(f"Output shape: {outputs.logits.shape}")
    print(f"Output logits mean: {outputs.logits.mean().item():.4f}")


def main():
    """Run training and inference examples."""
    print("AV-HuBERT Training and Inference Examples")
    print("=" * 60)
    
    try:
        train_model()
        inference_example()
        
        print("\nAll examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()