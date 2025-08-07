import sys
import types
from enum import Enum
from dataclasses import dataclass
import torch
import torch.nn as nn

"""Minimal internal stub for the parts of *fairseq* that AVHuBERT relies on.
This file is imported *before* any other AVHuBERT modules so that it can
populate ``sys.modules`` with fake *fairseq* sub-modules, thereby removing the
runtime dependency on the external *fairseq* package while keeping the public
API that the original code expects.

NOTE:  This is **not** a full re-implementation of fairseq.  It only provides
the tiny subset of classes / functions that AVHuBERT touches during inference
and is **not** suitable for training.  Add new symbols here on demand whenever
an ``ImportError`` surfaces.
"""

# -----------------------------------------------------------------------------
# Build the top-level *fairseq* module and register it so that subsequent
# ``import fairseq`` statements resolve to this in-memory stub.
# -----------------------------------------------------------------------------
_fairseq_root = types.ModuleType("fairseq")
sys.modules["fairseq"] = _fairseq_root

# -----------------------------------------------------------------------------
# fairseq.utils
# -----------------------------------------------------------------------------
_utils = types.ModuleType("fairseq.utils")

def get_available_activation_fns():
    """Return the activation functions supported by the stub.
    This is only the subset used by AVHuBERT.  Extend as needed."""
    return [
        "relu",
        "gelu",
        "gelu_fast",
        "gelu_accurate",
        "tanh",
        "sigmoid",
    ]

_utils.get_available_activation_fns = get_available_activation_fns
sys.modules["fairseq.utils"] = _utils
_fairseq_root.utils = _utils

# -----------------------------------------------------------------------------
# fairseq.dataclass (+ helpers)
# -----------------------------------------------------------------------------
_dataclass_mod = types.ModuleType("fairseq.dataclass")

class FairseqDataclass:  # pragma: no cover – behaviourally inert stub
    """Marker base-class used by the original code for type annotation only."""
    pass

class ChoiceEnum(str, Enum):  # type: ignore[misc]
    """String Enum helper used by the old fairseq config dataclasses."""

    def __new__(cls, value):  # noqa: D401
        obj = str.__new__(cls, value)
        obj._value_ = value  # type: ignore[attr-defined]
        return obj

_dataclass_mod.FairseqDataclass = FairseqDataclass
_dataclass_mod.ChoiceEnum = ChoiceEnum
sys.modules["fairseq.dataclass"] = _dataclass_mod
_fairseq_root.dataclass = _dataclass_mod

# Minimal *fairseq.dataclass.utils*
_dataclass_utils = types.ModuleType("fairseq.dataclass.utils")
_dataclass_utils.convert_namespace_to_omegaconf = lambda *args, **kwargs: None  # noqa: E731
sys.modules["fairseq.dataclass.utils"] = _dataclass_utils

# Minimal *fairseq.dataclass.configs*
sys.modules["fairseq.dataclass.configs"] = types.ModuleType("fairseq.dataclass.configs")

# -----------------------------------------------------------------------------
# fairseq.modules – provide the couple of modules used by AVHuBERT
# -----------------------------------------------------------------------------
_modules_mod = types.ModuleType("fairseq.modules")

class GradMultiply(torch.autograd.Function):
    """Implements ``GradMultiply.apply(x, scale)`` that scales the gradient."""

    @staticmethod
    def forward(ctx, x, scale):  # type: ignore[override]
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore[override]
        return grad_output * ctx.scale, None

_modules_mod.GradMultiply = GradMultiply
_modules_mod.LayerNorm = nn.LayerNorm
sys.modules["fairseq.modules"] = _modules_mod
_fairseq_root.modules = _modules_mod

# -----------------------------------------------------------------------------
# fairseq.data stubs – only what is imported by AVHuBERT
# -----------------------------------------------------------------------------
_data_mod = types.ModuleType("fairseq.data")
sys.modules["fairseq.data"] = _data_mod
_fairseq_root.data = _data_mod

# sub-module: fairseq.data.data_utils
from avhubert.utils import compute_mask_indices as _compute_mask_indices  # noqa: E402
_data_utils_mod = types.ModuleType("fairseq.data.data_utils")
_data_utils_mod.compute_mask_indices = _compute_mask_indices
sys.modules["fairseq.data.data_utils"] = _data_utils_mod

# sub-module: fairseq.data.dictionary
_dictionary_mod = types.ModuleType("fairseq.data.dictionary")

class Dictionary:  # pragma: no cover – placeholder with minimal API
    def __len__(self):
        return 0

    @classmethod
    def load(cls, *args, **kwargs):  # noqa: D401
        return cls()

_dictionary_mod.Dictionary = Dictionary
sys.modules["fairseq.data.dictionary"] = _dictionary_mod

# -----------------------------------------------------------------------------
# fairseq.models – basic stubs so that type-checking passes
# -----------------------------------------------------------------------------
_models_mod = types.ModuleType("fairseq.models")

class BaseFairseqModel(nn.Module):
    """Thin shim around ``torch.nn.Module`` used for inheritance only."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    # The original BaseFairseqModel offers a few helpers – omit for stub.


def register_model(_name: str, dataclass=None):  # noqa: D401
    """Decorator used in the legacy codebase.  No-op for the stub."""

    def _inner(cls):  # type: ignore[NestedBlock]
        return cls

    return _inner

_models_mod.BaseFairseqModel = BaseFairseqModel
_models_mod.register_model = register_model
sys.modules["fairseq.models"] = _models_mod
_fairseq_root.models = _models_mod

# Provide minimal wav2vec2 building blocks that AVHuBERT expects
_wav2vec_root = types.ModuleType("fairseq.models.wav2vec")
_wav2vec2_mod = types.ModuleType("fairseq.models.wav2vec.wav2vec2")

class ConvFeatureExtractionModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):  # noqa: D401
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):  # noqa: D401
        return x


_wav2vec2_mod.ConvFeatureExtractionModel = ConvFeatureExtractionModel
_wav2vec2_mod.TransformerEncoder = TransformerEncoder
sys.modules["fairseq.models.wav2vec"] = _wav2vec_root
sys.modules["fairseq.models.wav2vec.wav2vec2"] = _wav2vec2_mod

# Glue into *fairseq.models* hierarchy
_models_mod.wav2vec = _wav2vec_root
_wav2vec_root.wav2vec2 = _wav2vec2_mod

# -----------------------------------------------------------------------------
# Additional high-level stubs that the code *imports* but does not need at
# inference time.  They are left mostly empty to avoid pulling extra deps.
# -----------------------------------------------------------------------------
for _name in [
    "fairseq.tasks",
    "fairseq.metrics",
    "fairseq.logging",
    "fairseq.logging.meters",
    "fairseq.logging.progress_bar",
    "fairseq.sequence_scorer",
    "fairseq.ngram_repeat_block",
    "fairseq.checkpoint_utils",
    "fairseq.options",
    "fairseq.distributed_utils",
]:
    sys.modules[_name] = types.ModuleType(_name)

# ----------------------------------------------------------------------------
# Clean-up namespace
# ----------------------------------------------------------------------------
del sys, types, Enum, dataclass, torch, nn  # type: ignore[var-annotated]