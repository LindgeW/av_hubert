import logging
import os
import time
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ..models.avhubert import AVHubertModel
from ..data.dataset import AVHubertDataset
from ..data.collator import create_dataloader
from ..training.criterion import AVHubertCriterion
from ..configs.model_config import AVHubertConfig
from ..configs.data_config import DataConfig
from ..configs.training_config import TrainingConfig

logger = logging.getLogger(__name__)


class AVHubertTrainer:
    """Trainer for AV-HuBERT model."""
    
    def __init__(
        self,
        model: AVHubertModel,
        train_dataset: AVHubertDataset,
        val_dataset: Optional[AVHubertDataset] = None,
        config: TrainingConfig = None,
        device: str = "cuda",
        distributed: bool = False,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config or TrainingConfig()
        self.device = device
        self.distributed = distributed
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Setup distributed training
        if distributed:
            self.model = DDP(self.model, find_unused_parameters=self.config.find_unused_parameters)
        
        # Create criterion
        self.criterion = AVHubertCriterion(model.config).to(device)
        
        # Create optimizers and schedulers
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Create data loaders
        self.train_loader = create_dataloader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )
        
        if val_dataset:
            self.val_loader = create_dataloader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
            )
        else:
            self.val_loader = None
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_val_loss = float('inf')
        
        # Create save directory
        os.makedirs(self.config.save_dir, exist_ok=True)
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        if self.config.optimizer.lower() == "adam":
            return AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        if self.config.scheduler.lower() == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.max_epochs,
                eta_min=0,
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            audio = batch["audio"].to(self.device)
            video = batch["video"].to(self.device)
            labels = batch["labels"]
            audio_lengths = batch["audio_lengths"].to(self.device)
            video_lengths = batch["video_lengths"].to(self.device)
            
            # Create padding mask
            padding_mask = self._create_padding_mask(audio_lengths, audio.size(1))
            
            # Forward pass
            self.optimizer.zero_grad()
            
            model_output = self.model(
                source=audio,
                target_list=labels,
                padding_mask=padding_mask,
                mask=True,
                features_only=False,
            )
            
            # Compute loss
            loss_output = self.criterion(model_output, labels, padding_mask)
            loss = loss_output["loss"]
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_val)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_accuracy += loss_output["accuracy"].item()
            num_batches += 1
            self.current_step += 1
            
            # Log progress
            if batch_idx % self.config.log_every_n_steps == 0:
                logger.info(
                    f"Epoch {self.current_epoch}, Step {self.current_step}, "
                    f"Loss: {loss.item():.4f}, Accuracy: {loss_output['accuracy'].item():.4f}"
                )
        
        return {
            "train_loss": total_loss / num_batches,
            "train_accuracy": total_accuracy / num_batches,
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                audio = batch["audio"].to(self.device)
                video = batch["video"].to(self.device)
                labels = batch["labels"]
                audio_lengths = batch["audio_lengths"].to(self.device)
                video_lengths = batch["video_lengths"].to(self.device)
                
                # Create padding mask
                padding_mask = self._create_padding_mask(audio_lengths, audio.size(1))
                
                # Forward pass
                model_output = self.model(
                    source=audio,
                    target_list=labels,
                    padding_mask=padding_mask,
                    mask=False,
                    features_only=False,
                )
                
                # Compute loss
                loss_output = self.criterion(model_output, labels, padding_mask)
                loss = loss_output["loss"]
                
                # Update metrics
                total_loss += loss.item()
                total_accuracy += loss_output["accuracy"].item()
                num_batches += 1
        
        return {
            "val_loss": total_loss / num_batches,
            "val_accuracy": total_accuracy / num_batches,
        }
    
    def _create_padding_mask(self, lengths: torch.Tensor, max_length: int) -> torch.Tensor:
        """Create padding mask from sequence lengths."""
        batch_size = lengths.size(0)
        mask = torch.arange(max_length, device=lengths.device).expand(batch_size, max_length) >= lengths.unsqueeze(1)
        return mask
    
    def save_checkpoint(self, filename: str, metrics: Dict[str, float] = None):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "current_epoch": self.current_epoch,
            "current_step": self.current_step,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        
        if metrics:
            checkpoint["metrics"] = metrics
        
        torch.save(checkpoint, os.path.join(self.config.save_dir, filename))
        logger.info(f"Checkpoint saved: {filename}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["current_epoch"]
        self.current_step = checkpoint["current_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        for epoch in range(self.current_epoch, self.config.max_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            
            # Log metrics
            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Train Accuracy: {train_metrics['train_accuracy']:.4f}"
            )
            
            if val_metrics:
                logger.info(
                    f"Epoch {epoch}: "
                    f"Val Loss: {val_metrics['val_loss']:.4f}, "
                    f"Val Accuracy: {val_metrics['val_accuracy']:.4f}"
                )
                
                # Save best model
                if val_metrics["val_loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["val_loss"]
                    self.save_checkpoint("best_model.pt", {**train_metrics, **val_metrics})
            
            # Save checkpoint periodically
            if (epoch + 1) % self.config.save_steps == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt", {**train_metrics, **val_metrics})
        
        logger.info("Training completed!")
    
    def evaluate(self, test_dataset: AVHubertDataset) -> Dict[str, float]:
        """Evaluate the model on test dataset."""
        test_loader = create_dataloader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )
        
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in test_loader:
                # Move batch to device
                audio = batch["audio"].to(self.device)
                video = batch["video"].to(self.device)
                labels = batch["labels"]
                audio_lengths = batch["audio_lengths"].to(self.device)
                video_lengths = batch["video_lengths"].to(self.device)
                
                # Create padding mask
                padding_mask = self._create_padding_mask(audio_lengths, audio.size(1))
                
                # Forward pass
                model_output = self.model(
                    source=audio,
                    target_list=labels,
                    padding_mask=padding_mask,
                    mask=False,
                    features_only=False,
                )
                
                # Compute loss
                loss_output = self.criterion(model_output, labels, padding_mask)
                loss = loss_output["loss"]
                
                # Update metrics
                total_loss += loss.item()
                total_accuracy += loss_output["accuracy"].item()
                num_batches += 1
        
        return {
            "test_loss": total_loss / num_batches,
            "test_accuracy": total_accuracy / num_batches,
        }