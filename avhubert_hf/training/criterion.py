import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from ..configs.model_config import AVHubertConfig


class AVHubertCriterion(nn.Module):
    """Criterion for AV-HuBERT training."""
    
    def __init__(self, config: AVHubertConfig):
        super().__init__()
        self.config = config
        self.logit_temp = config.logit_temp
        self.skip_masked = config.skip_masked
        self.skip_nomask = config.skip_nomask
    
    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        target_list: List[torch.Tensor],
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute loss for AV-HuBERT training.
        
        Args:
            model_output: Output from the model
            target_list: List of target tensors
            padding_mask: Boolean mask indicating padded positions
        
        Returns:
            Dictionary containing loss and other metrics
        """
        logits = model_output["logits"]
        target = model_output["target"]
        features = model_output["features"]
        
        # Apply temperature scaling
        logits = logits / self.logit_temp
        
        # Compute cross-entropy loss
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), reduction="none")
            
            # Apply padding mask if provided
            if padding_mask is not None:
                loss = loss * (1 - padding_mask.view(-1).float())
            
            # Skip masked or unmasked frames if configured
            if self.skip_masked:
                # This would require mask indices from the model
                pass
            if self.skip_nomask:
                # This would require mask indices from the model
                pass
            
            loss = loss.sum() / (loss != 0).sum().clamp(min=1)
        else:
            loss = torch.tensor(0.0, device=logits.device)
        
        # Compute accuracy
        if target is not None:
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                correct = (pred == target).float()
                if padding_mask is not None:
                    correct = correct * (1 - padding_mask.float())
                accuracy = correct.sum() / (correct != 0).sum().clamp(min=1)
        else:
            accuracy = torch.tensor(0.0, device=logits.device)
        
        return {
            "loss": loss,
            "accuracy": accuracy,
            "logits": logits,
            "target": target,
        }