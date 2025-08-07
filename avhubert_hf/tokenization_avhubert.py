"""
Tokenization classes for AV-HuBERT.
"""

import os
from typing import List, Optional, Union, Tuple

from transformers import PreTrainedTokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)


class AVHubertTokenizer(PreTrainedTokenizer):
    """
    Constructs an AV-HuBERT tokenizer.
    
    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods.
    Users should refer to this superclass for more information regarding those methods.
    
    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        word_delimiter_token (`str`, *optional*, defaults to `"|"`):
            The token used for word boundaries.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        **kwargs
            Additional keyword arguments passed along to [`PreTrainedTokenizer`]
    """
    
    vocab_files_names = {"vocab_file": "vocab.json"}
    pretrained_vocab_files_map = {
        "vocab_file": {
            "facebook/avhubert-base": "https://huggingface.co/facebook/avhubert-base/resolve/main/vocab.json",
        }
    }
    max_model_input_sizes = {
        "facebook/avhubert-base": 1024,
    }
    model_input_names = ["input_ids", "attention_mask"]
    
    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|",
        do_lower_case=True,
        **kwargs,
    ):
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            word_delimiter_token=word_delimiter_token,
            do_lower_case=do_lower_case,
            **kwargs,
        )
        
        self.do_lower_case = do_lower_case
        self.word_delimiter_token = word_delimiter_token
        
        # Load vocabulary
        if not os.path.isfile(vocab_file):
            raise ValueError(f"Can't find a vocabulary file at path '{vocab_file}'")
        
        self.vocab = self.load_vocab(vocab_file)
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
    
    def load_vocab(self, vocab_file):
        """Load vocabulary from file."""
        vocab = {}
        with open(vocab_file, "r", encoding="utf-8") as f:
            for line in f:
                token = line.strip()
                if token:
                    vocab[token] = len(vocab)
        return vocab
    
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
    
    def get_vocab(self):
        return dict(self.vocab)
    
    def _tokenize(self, text):
        """Tokenize a text."""
        if self.do_lower_case:
            text = text.lower()
        
        # Split on word delimiter
        tokens = text.split(self.word_delimiter_token)
        
        # Tokenize each word
        tokenized_tokens = []
        for token in tokens:
            if token in self.vocab:
                tokenized_tokens.append(token)
            else:
                # Handle unknown tokens
                tokenized_tokens.append(self.unk_token)
        
        return tokenized_tokens
    
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))
    
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)
    
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        return self.word_delimiter_token.join(tokens)
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """Save the vocabulary to a file."""
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + self.vocab_files_names["vocab_file"]
            )
        else:
            vocab_file = save_directory
        
        with open(vocab_file, "w", encoding="utf-8") as f:
            for token, token_id in sorted(self.vocab.items(), key=lambda x: x[1]):
                f.write(f"{token}\n")
        
        return (vocab_file,)
    
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An AV-HuBERT sequence has the following format:
        - single sequence: `<s> X </s>`
        - pair of sequences: `<s> A </s> B: <cls> </s>`
        
        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        
        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep
    
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.
        
        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.
        
        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )
        
        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]
    
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. An AV-HuBERT
        sequence pair mask has the following format:
        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```
        
        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).
        
        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        
        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]