"""
Masking utilities adapted from fairseq
"""

import torch
import numpy as np
from typing import Optional


def compute_mask_indices(
    shape: torch.Size,
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[torch.Tensor] = None,
    min_masks: int = 0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape. Used to implement SpecAugment mask
    and other masking strategies.
    
    Args:
        shape: Shape for which to compute masks
        mask_prob: Probability for each token to be masked
        mask_length: Size of each mask
        attention_mask: Attention mask to avoid masking padded areas
        min_masks: Minimum number of masks
        
    Returns:
        An array of shape [batch_size, seq_len] with 1 indicating masked positions
    """
    
    batch_size, sequence_length = shape
    
    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )

    # epsilon is used for probabilistic rounding
    epsilon = np.random.rand(1).item()

    def compute_num_masked_spans(input_length):
        """Given input length, compute how many spans should be masked"""
        num_masked_spans = int(mask_prob * input_length / mask_length + epsilon)
        num_masked_spans = max(num_masked_spans, min_masks)

        # make sure num masked spans <= sequence_length
        if num_masked_spans * mask_length > sequence_length:
            num_masked_spans = sequence_length // mask_length

        # make sure num_masked spans is also <= input_length - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_spans:
            num_masked_spans = max(input_length - (mask_length - 1), 0)

        return num_masked_spans

    # compute number of masked spans in batch
    input_lengths = (
        attention_mask.sum(-1).detach().tolist()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
    )

    # SpecAugment mask to fill
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)
    spec_aug_mask_idxs = []

    max_num_masked_spans = compute_num_masked_spans(sequence_length)

    if max_num_masked_spans == 0:
        return spec_aug_mask

    for input_length in input_lengths:
        # compute num of masked spans for this input
        num_masked_spans = compute_num_masked_spans(input_length)

        # get random indices to mask
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_spans, replace=False
        )

        # pick first sampled index that will serve as a dummy index to pad vector
        # to ensure same dimension for all batches due to probabilistic rounding
        # Picking first sample just pads those vectors twice.
        if len(spec_aug_mask_idx) == 0:
            # this case can only happen if `input_length` is strictly smaller then
            # `sequence_length` in which case the last token has to be a padding
            # token which we can use as a dummy mask id
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_spans - num_masked_spans, dtype=np.int32) * dummy_mask_idx]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # expand masked indices to masked spans
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_spans, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_spans * mask_length)

    # add offset to the starting indexes so that indexes now create a span
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_spans, mask_length)).reshape(
        batch_size, max_num_masked_spans * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # ensure that we cannot have indices larger than sequence_length
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

    # scatter indices to mask
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    return spec_aug_mask