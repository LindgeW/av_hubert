import torch
import torch.nn.functional as F
from typing import Callable


def get_activation_fn(activation: str) -> Callable:
    """Returns the activation function corresponding to `activation`"""
    
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "swish":
        return F.silu
    elif activation == "silu":
        return F.silu
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    else:
        raise RuntimeError(f"activation {activation} not supported")


def gelu_accurate(x):
    """Accurate GELU implementation."""
    if not hasattr(gelu_accurate, "_a"):
        gelu_accurate._a = math.sqrt(2 / math.pi)
        gelu_accurate._b = 0.044715
        gelu_accurate._c = 1.70165
        gelu_accurate._d = -0.288675
    return (
        x * 0.5 * (1.0 + torch.tanh(gelu_accurate._a * (x + gelu_accurate._b * x * x * x)))
    )


def gelu(x, approximate=False):
    """GELU activation function."""
    if approximate:
        return x * 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    else:
        return gelu_accurate(x)


# Import math for gelu functions
import math