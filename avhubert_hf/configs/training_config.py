from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration for training settings."""
    
    # Training settings
    batch_size: int = field(default=8, metadata={"help": "batch size for training"})
    learning_rate: float = field(default=5e-4, metadata={"help": "learning rate"})
    weight_decay: float = field(default=0.01, metadata={"help": "weight decay"})
    max_epochs: int = field(default=100, metadata={"help": "maximum number of epochs"})
    warmup_steps: int = field(default=4000, metadata={"help": "number of warmup steps"})
    max_steps: Optional[int] = field(default=None, metadata={"help": "maximum number of training steps"})
    
    # Optimization
    optimizer: str = field(default="adam", metadata={"help": "optimizer type"})
    scheduler: str = field(default="cosine", metadata={"help": "learning rate scheduler"})
    gradient_clip_val: float = field(default=1.0, metadata={"help": "gradient clipping value"})
    accumulate_grad_batches: int = field(default=1, metadata={"help": "number of batches to accumulate gradients"})
    
    # Checkpointing
    save_dir: str = field(default="./checkpoints", metadata={"help": "directory to save checkpoints"})
    save_steps: int = field(default=1000, metadata={"help": "save checkpoint every N steps"})
    save_top_k: int = field(default=3, metadata={"help": "save top k checkpoints"})
    monitor: str = field(default="val_loss", metadata={"help": "metric to monitor for saving best model"})
    mode: str = field(default="min", metadata={"help": "minimize or maximize the monitored metric"})
    
    # Validation
    val_check_interval: float = field(default=1.0, metadata={"help": "validation check interval"})
    num_workers: int = field(default=4, metadata={"help": "number of data loader workers"})
    
    # Logging
    log_every_n_steps: int = field(default=100, metadata={"help": "log every N steps"})
    tensorboard: bool = field(default=True, metadata={"help": "use tensorboard logging"})
    
    # Mixed precision
    precision: str = field(default="32", metadata={"help": "training precision (16, 32, bf16)"})
    
    # Distributed training
    distributed_backend: str = field(default="nccl", metadata={"help": "distributed backend"})
    find_unused_parameters: bool = field(default=False, metadata={"help": "find unused parameters in distributed training"})