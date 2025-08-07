import torch
import torch.nn as nn
from typing import Optional, Tuple

from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_avhubert import AVHubertConfig

try:
    # Import the legacy implementation *after* the fairseq stub has populated
    # `sys.modules` – this is taken care of by `avhubert.__init__`.
    from avhubert.hubert import AVHubertModel as _LegacyAVHubertModel
except ImportError as exc:  # pragma: no cover – development aid
    raise RuntimeError(
        "AVHuBERT legacy model could not be imported. Make sure that "
        "`avhubert.fairseq_stub` is executed *before* this file."
    ) from exc

logger = logging.get_logger(__name__)

__all__ = ["AVHubertModel"]


class AVHubertModel(PreTrainedModel):
    """Hugging Face wrapper for the original *AVHuBERT* implementation.

    The goal is a *drop-in* experience for users familiar with the 🤗 *Transformers*
    eco-system.  Under the hood we still delegate all heavy-lifting to the legacy
    PyTorch model so that no accuracy is lost during the transition.
    """

    config_class = AVHubertConfig
    _keys_to_ignore_on_load_missing = [r"position_ids"]  # HF housekeeping attr

    def __init__(self, config: AVHubertConfig):  # noqa: D401
        super().__init__(config)
        # ------------------------------------------------------------------
        # Build the *legacy* model.  AVHuBERT expects three arguments:
        #    1. cfg         (AVHubertConfig dataclass)
        #    2. task_cfg    (AVHubertPretrainingConfig)
        #    3. dictionaries(list[Dictionary])
        #
        # We only need the forward pass for inference ⇒ provide *mock* objects.
        # ------------------------------------------------------------------
        self.core = _LegacyAVHubertModel(
            cfg=_FakeFairseqCfg(config),
            task_cfg=_FakeTaskCfg(),
            dictionaries=[],
        )

    # ---------------------------------------------------------------------
    # Public API – mirrors HF naming conventions as closely as possible.
    # ---------------------------------------------------------------------
    def forward(
        self,
        input_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        """Forward pass.

        Parameters
        ----------
        input_values: torch.Tensor
            Input tensor *either* with shape ``[B, T]`` (audio) *or*
            ``[B, C, T, H, W]`` (video) depending on ``config.input_modality``.
        padding_mask: torch.Tensor, optional
            Boolean mask where *True* indicates **padded** positions.
        mask: bool, default *False*
            Whether to apply AVHuBERT's internal random masking strategy.
        output_hidden_states: bool, default *False*
            If *True*, also return hidden-states from each transformer layer –
            not yet implemented.
        """
        legacy_out = self.core(
            source=input_values,
            padding_mask=padding_mask,
            mask=mask,
            features_only=not output_hidden_states,
        )

        if output_hidden_states:
            raise NotImplementedError(
                "output_hidden_states=True is not yet supported in the "
                "wrapper. Contributions welcome!"
            )

        return legacy_out["x"], None  # logits / features

    # ------------------------------------------------------------------
    # Convenience – expose *extract_features* from the legacy implementation.
    # ------------------------------------------------------------------
    def extract_features(
        self,
        input_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
        **kwargs,
    ):
        return self.core.extract_features(
            source=input_values, padding_mask=padding_mask, mask=mask
        )


# -------------------------------------------------------------------------
# Helper mocks – keep them out of the public API. They just satisfy the
# signature of the original AVHuBERT constructor.
# -------------------------------------------------------------------------


class _FakeFairseqCfg:
    """Simple shim to present a *dict-like* interface on top of HF config."""

    def __init__(self, hf_config: AVHubertConfig):
        for key, value in hf_config.to_dict().items():
            setattr(self, key, value)


class _FakeTaskCfg:
    """Placeholder that fulfils the attribute contract of *task_cfg*."""

    label_rate: int = 50  # dummy value – not used in inference
    input_modality: str = "audio"