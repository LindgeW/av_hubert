from typing import Dict, List, Any
import torch
from torch.utils.data import DataLoader

from .dataset import AVHubertDataset


class AVHubertCollator:
    """Collator for AV-HuBERT dataset batching."""
    
    def __init__(self, dataset: AVHubertDataset):
        self.dataset = dataset
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate a batch of samples."""
        return self.dataset.collate_fn(batch)


def create_dataloader(
    dataset: AVHubertDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    collate_fn: Any = None,
) -> DataLoader:
    """Create a DataLoader for the AV-HuBERT dataset."""
    if collate_fn is None:
        collate_fn = AVHubertCollator(dataset)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=shuffle,
    )