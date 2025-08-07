#!/usr/bin/env python3
"""
Training example for AV-HuBERT
This example demonstrates how to set up training for the refactored AV-HuBERT model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import numpy as np

# Import from our refactored package
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from avhubert_hf import AVHubertModel, AVHubertConfig


class DummyAVDataset(Dataset):
    """Dummy dataset for demonstration purposes."""
    
    def __init__(self, num_samples=1000, seq_len=100):
        self.num_samples = num_samples
        self.seq_len = seq_len
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate dummy audio features (MFCC-like)
        audio_features = torch.randn(self.seq_len, 104)
        
        # Generate dummy video features  
        video_features = torch.randn(self.seq_len, 512)
        
        # Generate dummy labels (for masked language modeling)
        labels = torch.randint(0, 32, (self.seq_len,))
        
        # Create attention mask
        attention_mask = torch.ones(self.seq_len)
        
        return {
            'audio_features': audio_features,
            'video_features': video_features,
            'labels': labels,
            'attention_mask': attention_mask
        }


class AVHubertTrainer:
    """Simple trainer for AV-HuBERT."""
    
    def __init__(self, model, train_dataloader, optimizer, device='cpu'):
        self.model = model
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.device = device
        self.model.to(device)
        
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in self.train_dataloader:
            # Move batch to device
            audio_features = batch['audio_features'].to(self.device)
            video_features = batch['video_features'].to(self.device) 
            labels = batch['labels'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass (simplified for demo)
            # In practice, you'd implement the full forward pass with masking
            batch_size, seq_len = audio_features.shape[:2]
            
            # Combine features (simplified)
            combined_features = torch.cat([audio_features, video_features], dim=-1)
            
            # Simulate model forward pass
            # In real implementation, this would be: outputs = self.model(combined_features)
            hidden_states = combined_features  # Placeholder
            
            # Compute loss (dummy cross-entropy for demo)
            loss_fn = nn.CrossEntropyLoss()
            
            # Project to vocab size for loss computation
            vocab_proj = nn.Linear(hidden_states.size(-1), 32).to(self.device)
            logits = vocab_proj(hidden_states)
            
            loss = loss_fn(logits.view(-1, 32), labels.view(-1))
            
            # Backward pass
            loss.backward()
            
            # Update weights
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if num_batches % 10 == 0:
                print(f"  Batch {num_batches}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        return avg_loss


def main():
    """Main training example."""
    print("🚀 AV-HuBERT Training Example")
    print("=" * 40)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
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
        modality_fuse="concat",  # Concatenate audio and video features
    )
    
    # Initialize model
    print("🤖 Initializing AV-HuBERT model...")
    model = AVHubertModel(config)
    
    # Create dataset and dataloader
    print("📚 Creating training dataset...")
    train_dataset = DummyAVDataset(num_samples=1000, seq_len=100)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=4, 
        shuffle=True,
        num_workers=0  # Set to 0 for demo
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Create trainer
    trainer = AVHubertTrainer(model, train_dataloader, optimizer, device)
    
    # Training loop
    num_epochs = 2
    print(f"🏋️ Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        print(f"\n📈 Epoch {epoch + 1}/{num_epochs}")
        avg_loss = trainer.train_epoch()
        print(f"  Average loss: {avg_loss:.4f}")
    
    print("\n✅ Training completed!")
    
    # Save model (optional)
    print("💾 Saving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'avhubert_checkpoint.pt')
    
    print("\n💡 Next steps:")
    print("  - Implement real data loading pipeline")
    print("  - Add proper masking strategy")
    print("  - Use learning rate scheduling")
    print("  - Add validation loop")
    print("  - Implement checkpointing")
    

if __name__ == "__main__":
    main()