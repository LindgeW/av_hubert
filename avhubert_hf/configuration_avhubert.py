from transformers.configuration_utils import PretrainedConfig

__all__ = ["AVHubertConfig"]


class AVHubertConfig(PretrainedConfig):
    """Minimal `PretrainedConfig` for AVHuBERT.

    This configuration intentionally keeps the surface identical to the
    original `AVHubertConfig` dataclass where possible, but it does **not** aim
    for feature-parity.  Only the arguments that are strictly required for
    inference survive here.  Add new attributes as needed.
    """

    model_type = "avhubert"

    def __init__(
        self,
        input_modality: str = "audio",
        encoder_embed_dim: int = 768,
        encoder_layers: int = 12,
        encoder_attention_heads: int = 12,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        masking_type: str = "input",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.input_modality = input_modality
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.masking_type = masking_type