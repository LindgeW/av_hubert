import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def compute_mask_indices(
    shape: Tuple[int, int],
    padding_mask: Optional[torch.Tensor],
    mask_prob: float,
    mask_length: int,
    mask_type: str = "static",
    mask_other: float = 0.0,
    min_masks: int = 0,
    no_overlap: bool = False,
    min_space: int = 0,
    require_same_masks: bool = True,
    mask_dropout: float = 0.0,
) -> torch.Tensor:
    """Computes random mask spans for a given shape.
    
    This is the main function to create masks for masked training.
    
    Args:
        shape: the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        padding_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_length: size of the mask
        mask_type: how to compute mask lengths
            static = fixed size
            uniform = sample from uniform distribution [mask_other, mask_length*2]
            normal = sample from normal distribution with mean mask_length and stdev mask_other. mask is min 1 element
            poisson = sample from possion distribution with lambda = mask length
        min_masks: minimum number of masked spans
        no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
        min_space: only used if no_overlap is True, this is how many elements to keep unmasked between spans
        require_same_masks: if true, will randomly drop out masks until same number of masks remain for each sample
        mask_dropout: randomly dropout this percentage of masks in each example
    """
    bsz, all_sz = shape
    mask = torch.full((bsz, all_sz), False, dtype=torch.bool, device=padding_mask.device if padding_mask is not None else None)

    all_num_mask = int(
        # add a random number for probabilistic rounding
        mask_prob * all_sz / float(mask_length)
        + torch.rand(1).item()
    )

    all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    for i in range(bsz):
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            assert sz >= 0, sz
        else:
            sz = all_sz

        if sz == 0:
            continue

        if mask_type == "static":
            lengths = torch.full(size=(sz,), fill_value=mask_length, dtype=torch.long)
        elif mask_type == "uniform":
            lengths = torch.randint(mask_other, mask_length * 2 + 1, size=(sz,))
        elif mask_type == "normal":
            lengths = torch.normal(mask_length, mask_other, size=(sz,))
            lengths = torch.clamp(lengths, min=1)
        elif mask_type == "poisson":
            lengths = torch.poisson(mask_length, size=(sz,))
            lengths = torch.clamp(lengths, min=1)
        else:
            raise Exception(f"unknown mask selection: {mask_type}")

        if no_overlap:
            mask_idc = _compute_mask_indices_no_overlap(
                sz, lengths, min_space, require_same_masks
            )
        else:
            mask_idc = _compute_mask_indices_with_overlap(
                sz, lengths, require_same_masks
            )

        if len(mask_idc) == 0:
            continue

        mask_idc = torch.from_numpy(mask_idc).to(dtype=torch.long)
        mask_idcs.append(mask_idc)

    if len(mask_idcs) == 0:
        return mask

    # randomly drop out masks
    if mask_dropout > 0:
        mask_idcs = [
            mask_idc[torch.rand(len(mask_idc)) > mask_dropout]
            for mask_idc in mask_idcs
        ]

    target_len = min([len(mask_idc) for mask_idc in mask_idcs])

    if target_len == 0:
        return mask

    if require_same_masks:
        mask_idcs = [mask_idc[:target_len] for mask_idc in mask_idcs]

    for i, mask_idc in enumerate(mask_idcs):
        mask[i, mask_idc] = True

    return mask


def _compute_mask_indices_with_overlap(
    sz: int, lengths: torch.Tensor, require_same_masks: bool
) -> torch.Tensor:
    """Computes random mask spans for a given 1d tensor with overlap.
    
    Args:
        sz: size of the tensor
        lengths: lengths of each mask
        require_same_masks: if true, will randomly drop out masks until same number of masks remain for each sample
    """
    mask_idcs = []

    for i in range(len(lengths)):
        length = lengths[i]

        # Try to find a valid span
        for _ in range(100):
            start = torch.randint(0, sz - length + 1, (1,)).item()
            end = start + length
            mask_idc = torch.arange(start, end, dtype=torch.long)
            
            # Check if this span overlaps with existing ones
            overlap = False
            for existing_mask in mask_idcs:
                if torch.any(torch.isin(mask_idc, existing_mask)):
                    overlap = True
                    break
            
            if not overlap:
                mask_idcs.append(mask_idc)
                break

    if len(mask_idcs) == 0:
        return torch.tensor([], dtype=torch.long)

    return torch.cat(mask_idcs)


def _compute_mask_indices_no_overlap(
    sz: int, lengths: torch.Tensor, min_space: int, require_same_masks: bool
) -> torch.Tensor:
    """Computes random mask spans for a given 1d tensor without overlap.
    
    Args:
        sz: size of the tensor
        lengths: lengths of each mask
        min_space: minimum space between masks
        require_same_masks: if true, will randomly drop out masks until same number of masks remain for each sample
    """
    mask_idcs = []
    start = 0

    for length in lengths:
        if start + length > sz:
            break
        
        mask_idc = torch.arange(start, start + length, dtype=torch.long)
        mask_idcs.append(mask_idc)
        start = start + length + min_space

    if len(mask_idcs) == 0:
        return torch.tensor([], dtype=torch.long)

    return torch.cat(mask_idcs)