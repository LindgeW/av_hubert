"""
Checkpoint utilities for loading and converting AVHuBERT models.
Provides compatibility between fairseq and simplified implementations.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
import logging
import re
from pathlib import Path

from .model import AVHubertModel
from .config import AVHubertConfig

logger = logging.getLogger(__name__)


class CheckpointConverter:
    """Convert between fairseq and simplified checkpoint formats"""
    
    def __init__(self):
        # Mapping from fairseq keys to simplified keys
        self.key_mapping = {
            # Model root
            "encoder.": "",
            
            # Feature extractors
            "feature_extractor_audio.": "feature_extractor_audio.",
            "feature_extractor_video.": "feature_extractor_video.", 
            
            # Encoder layers
            "layers.": "encoder.layers.",
            "self_attn.": "self_attn.",
            "fc1.": "fc1.",
            "fc2.": "fc2.",
            "self_attn_layer_norm.": "self_attn_layer_norm.",
            "final_layer_norm.": "final_layer_norm.",
            
            # Projection layers
            "final_proj.": "final_proj.",
            "post_extract_proj.": "post_extract_proj.",
            
            # Masking
            "mask_emb": "mask_emb",
            
            # Layer norm
            "layer_norm.": "layer_norm.",
            
            # ResNet components
            "resnet.": "resnet.backbone.",
        }
        
        # Keys to skip during conversion
        self.skip_keys = {
            "label_embs_concat",  # Task-specific, will be recreated
            "num_updates",        # Training state
            "_float_tensor",      # Internal fairseq tensor
        }
    
    def convert_fairseq_to_simplified(
        self, 
        fairseq_state_dict: Dict[str, torch.Tensor],
        config: Optional[AVHubertConfig] = None
    ) -> Tuple[Dict[str, torch.Tensor], AVHubertConfig]:
        """
        Convert fairseq state dict to simplified format
        
        Args:
            fairseq_state_dict: State dict from fairseq checkpoint
            config: Optional config override
            
        Returns:
            Tuple of (simplified_state_dict, config)
        """
        logger.info("Converting fairseq checkpoint to simplified format...")
        
        # Extract model state dict if nested
        if "model" in fairseq_state_dict:
            model_state = fairseq_state_dict["model"]
        else:
            model_state = fairseq_state_dict
        
        # Extract configuration if available
        if config is None:
            config = self._extract_config_from_checkpoint(fairseq_state_dict)
        
        # Convert state dict keys
        simplified_state = {}
        
        for key, value in model_state.items():
            # Skip unwanted keys
            if any(skip_key in key for skip_key in self.skip_keys):
                logger.debug(f"Skipping key: {key}")
                continue
            
            # Convert key name
            new_key = self._convert_key(key)
            
            # Handle special cases
            new_key, value = self._handle_special_conversions(new_key, value, config)
            
            if new_key is not None:
                simplified_state[new_key] = value
                logger.debug(f"Converted: {key} -> {new_key}")
        
        logger.info(f"Converted {len(simplified_state)} parameters")
        return simplified_state, config
    
    def _convert_key(self, fairseq_key: str) -> str:
        """Convert a fairseq key to simplified format"""
        new_key = fairseq_key
        
        # Apply key mappings
        for fairseq_pattern, simplified_pattern in self.key_mapping.items():
            if fairseq_pattern in new_key:
                new_key = new_key.replace(fairseq_pattern, simplified_pattern)
        
        # Handle multihead attention projection weights
        if "in_proj_weight" in new_key or "in_proj_bias" in new_key:
            # These are already in the correct format
            pass
        elif "q_proj" in new_key or "k_proj" in new_key or "v_proj" in new_key:
            # Convert separate q/k/v projections to combined in_proj
            # This would need special handling in _handle_special_conversions
            pass
        
        return new_key
    
    def _handle_special_conversions(
        self, 
        key: str, 
        value: torch.Tensor, 
        config: AVHubertConfig
    ) -> Tuple[Optional[str], torch.Tensor]:
        """Handle special cases in parameter conversion"""
        
        # Skip empty keys
        if not key:
            return None, value
        
        # Handle attention projections
        if "q_proj" in key or "k_proj" in key or "v_proj" in key:
            # These need to be combined into in_proj_weight/bias
            # For simplicity, we'll skip individual projections and assume
            # the fairseq checkpoint already has in_proj_weight/bias
            logger.warning(f"Skipping separate attention projection: {key}")
            return None, value
        
        # Handle ResNet weights
        if "resnet.backbone." in key:
            # Adjust ResNet layer names if needed
            if "module." in key:
                key = key.replace("module.", "")
        
        # Handle conv feature extraction layers
        if "feature_extractor" in key and "conv_layers" in key:
            # Ensure proper mapping of conv layer indices
            pass
        
        return key, value
    
    def _extract_config_from_checkpoint(
        self, 
        checkpoint: Dict[str, Any]
    ) -> AVHubertConfig:
        """Extract configuration from fairseq checkpoint"""
        
        # Try to get config from checkpoint
        if "cfg" in checkpoint:
            fairseq_cfg = checkpoint["cfg"]
            config = self._convert_fairseq_config(fairseq_cfg)
        elif "args" in checkpoint:
            fairseq_args = checkpoint["args"]
            config = self._convert_fairseq_args(fairseq_args)
        else:
            logger.warning("No config found in checkpoint, using default")
            config = AVHubertConfig()
        
        return config
    
    def _convert_fairseq_config(self, fairseq_cfg) -> AVHubertConfig:
        """Convert fairseq config to simplified config"""
        
        # Extract model config if nested
        if hasattr(fairseq_cfg, "model"):
            model_cfg = fairseq_cfg.model
        else:
            model_cfg = fairseq_cfg
        
        # Map fairseq config to simplified config
        config_kwargs = {}
        
        # Model architecture
        if hasattr(model_cfg, "encoder_layers"):
            config_kwargs["encoder_layers"] = model_cfg.encoder_layers
        if hasattr(model_cfg, "encoder_embed_dim"):
            config_kwargs["encoder_embed_dim"] = model_cfg.encoder_embed_dim
        if hasattr(model_cfg, "encoder_ffn_embed_dim"):
            config_kwargs["encoder_ffn_embed_dim"] = model_cfg.encoder_ffn_embed_dim
        if hasattr(model_cfg, "encoder_attention_heads"):
            config_kwargs["encoder_attention_heads"] = model_cfg.encoder_attention_heads
        
        # Dropouts
        if hasattr(model_cfg, "dropout"):
            config_kwargs["dropout"] = model_cfg.dropout
        if hasattr(model_cfg, "attention_dropout"):
            config_kwargs["attention_dropout"] = model_cfg.attention_dropout
        if hasattr(model_cfg, "activation_dropout"):
            config_kwargs["activation_dropout"] = model_cfg.activation_dropout
        
        # Masking
        if hasattr(model_cfg, "mask_prob"):
            config_kwargs["mask_prob_audio"] = model_cfg.mask_prob
            config_kwargs["mask_prob_image"] = model_cfg.mask_prob
        if hasattr(model_cfg, "mask_length"):
            config_kwargs["mask_length_audio"] = model_cfg.mask_length
            config_kwargs["mask_length_image"] = model_cfg.mask_length
        
        # Other parameters
        if hasattr(model_cfg, "final_dim"):
            config_kwargs["final_dim"] = model_cfg.final_dim
        if hasattr(model_cfg, "activation_fn"):
            config_kwargs["activation_fn"] = model_cfg.activation_fn
        
        return AVHubertConfig(**config_kwargs)
    
    def _convert_fairseq_args(self, fairseq_args) -> AVHubertConfig:
        """Convert fairseq args (older format) to simplified config"""
        
        config_kwargs = {}
        
        # Map common argument names
        arg_mapping = {
            "encoder_layers": "encoder_layers",
            "encoder_embed_dim": "encoder_embed_dim", 
            "encoder_ffn_embed_dim": "encoder_ffn_embed_dim",
            "encoder_attention_heads": "encoder_attention_heads",
            "dropout": "dropout",
            "attention_dropout": "attention_dropout",
            "activation_dropout": "activation_dropout",
            "mask_prob": "mask_prob_audio",
            "mask_length": "mask_length_audio",
        }
        
        for fairseq_name, simplified_name in arg_mapping.items():
            if hasattr(fairseq_args, fairseq_name):
                config_kwargs[simplified_name] = getattr(fairseq_args, fairseq_name)
        
        return AVHubertConfig(**config_kwargs)


def load_fairseq_checkpoint(
    checkpoint_path: str,
    config: Optional[AVHubertConfig] = None,
    device: str = "cpu",
) -> AVHubertModel:
    """
    Load a fairseq checkpoint and convert to simplified model
    
    Args:
        checkpoint_path: Path to fairseq checkpoint
        config: Optional config override
        device: Device to load model on
        
    Returns:
        AVHubertModel instance with loaded weights
    """
    logger.info(f"Loading fairseq checkpoint from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Convert to simplified format
    converter = CheckpointConverter()
    simplified_state, extracted_config = converter.convert_fairseq_to_simplified(
        checkpoint, config
    )
    
    # Use provided config or extracted config
    final_config = config if config is not None else extracted_config
    
    # Create model
    model = AVHubertModel(final_config)
    
    # Load state dict with flexibility for missing/unexpected keys
    missing_keys, unexpected_keys = model.load_state_dict(simplified_state, strict=False)
    
    if missing_keys:
        logger.warning(f"Missing keys in checkpoint: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")
    
    logger.info("Successfully loaded fairseq checkpoint")
    return model


def save_simplified_checkpoint(
    model: AVHubertModel,
    save_path: str,
    config: Optional[AVHubertConfig] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Save model in simplified checkpoint format
    
    Args:
        model: AVHubertModel to save
        save_path: Path to save checkpoint
        config: Optional config to save (defaults to model.config)
        metadata: Optional metadata to include
    """
    logger.info(f"Saving simplified checkpoint to {save_path}")
    
    checkpoint = {
        "model": model.state_dict(),
        "config": config if config is not None else model.config,
    }
    
    if metadata is not None:
        checkpoint["metadata"] = metadata
    
    # Ensure directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(checkpoint, save_path)
    logger.info("Checkpoint saved successfully")


def load_simplified_checkpoint(
    checkpoint_path: str,
    device: str = "cpu",
) -> Tuple[AVHubertModel, AVHubertConfig]:
    """
    Load a simplified checkpoint
    
    Args:
        checkpoint_path: Path to simplified checkpoint
        device: Device to load model on
        
    Returns:
        Tuple of (model, config)
    """
    logger.info(f"Loading simplified checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config
    if "config" in checkpoint:
        config = checkpoint["config"]
    else:
        logger.warning("No config found in checkpoint, using default")
        config = AVHubertConfig()
    
    # Create and load model
    model = AVHubertModel(config)
    
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    
    logger.info("Successfully loaded simplified checkpoint")
    return model, config


def convert_checkpoint(
    input_path: str,
    output_path: str,
    config: Optional[AVHubertConfig] = None,
    format_type: str = "auto",
):
    """
    Convert between checkpoint formats
    
    Args:
        input_path: Input checkpoint path
        output_path: Output checkpoint path  
        config: Optional config override
        format_type: "fairseq_to_simplified", "simplified_to_fairseq", or "auto"
    """
    
    if format_type == "auto":
        # Detect format based on file content
        checkpoint = torch.load(input_path, map_location="cpu")
        if "cfg" in checkpoint or "args" in checkpoint:
            format_type = "fairseq_to_simplified"
        else:
            format_type = "simplified_to_fairseq"
    
    if format_type == "fairseq_to_simplified":
        logger.info("Converting fairseq checkpoint to simplified format")
        model = load_fairseq_checkpoint(input_path, config)
        save_simplified_checkpoint(model, output_path, config)
        
    elif format_type == "simplified_to_fairseq":
        logger.info("Converting simplified checkpoint to fairseq format")
        # This would require implementing the reverse conversion
        # For now, we'll raise an error
        raise NotImplementedError("Simplified to fairseq conversion not yet implemented")
    
    else:
        raise ValueError(f"Unknown format type: {format_type}")
    
    logger.info(f"Conversion completed: {input_path} -> {output_path}")


# Convenience functions
def is_fairseq_checkpoint(checkpoint_path: str) -> bool:
    """Check if a checkpoint is in fairseq format"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        return "cfg" in checkpoint or "args" in checkpoint
    except Exception:
        return False


def get_checkpoint_info(checkpoint_path: str) -> Dict[str, Any]:
    """Get information about a checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    info = {
        "format": "fairseq" if is_fairseq_checkpoint(checkpoint_path) else "simplified",
        "keys": list(checkpoint.keys()),
    }
    
    # Extract model info
    if "model" in checkpoint:
        model_state = checkpoint["model"]
        info["num_parameters"] = sum(p.numel() for p in model_state.values())
        info["parameter_keys"] = list(model_state.keys())[:10]  # First 10 keys
    
    # Extract config info
    if "config" in checkpoint:
        config = checkpoint["config"]
        if hasattr(config, "__dict__"):
            info["config"] = vars(config)
        else:
            info["config"] = str(config)
    elif "cfg" in checkpoint:
        info["config"] = str(checkpoint["cfg"])
    
    return info