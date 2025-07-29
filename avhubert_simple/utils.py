"""
Utility functions for AVHuBERT.
Simplified version without fairseq dependencies.
"""

import cv2
import torch
import random
import numpy as np
from typing import Dict, List, Optional, Tuple


def load_video(path: str) -> np.ndarray:
    """Load video and convert to grayscale frames."""
    for i in range(3):
        try:
            cap = cv2.VideoCapture(path)
            frames = []
            while True:
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frames.append(frame)
                else:
                    break
            cap.release()
            frames = np.stack(frames)
            return frames
        except Exception as e:
            print(f"Failed loading {path} ({i+1}/3): {e}")
            if i == 2:
                raise ValueError(f"Unable to load {path}")
    return None


class Compose:
    """Compose several transforms together."""
    
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Normalize:
    """Normalize frames with mean and standard deviation."""
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, frames):
        frames = (frames - self.mean) / self.std
        return frames

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CenterCrop:
    """Crop frames at the center."""
    
    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = int(round((w - tw)) / 2.)
        delta_h = int(round((h - th)) / 2.)
        frames = frames[:, delta_h:delta_h+th, delta_w:delta_w+tw]
        return frames


class RandomCrop:
    """Randomly crop frames."""
    
    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = random.randint(0, w - tw)
        delta_h = random.randint(0, h - th)
        frames = frames[:, delta_h:delta_h+th, delta_w:delta_w+tw]
        return frames


class HorizontalFlip:
    """Randomly flip frames horizontally."""
    
    def __init__(self, flip_ratio=0.5):
        self.flip_ratio = flip_ratio

    def __call__(self, frames):
        if random.random() < self.flip_ratio:
            frames = np.flip(frames, axis=2)  # flip width dimension
        return frames


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
) -> np.ndarray:
    """
    Computes random mask spans for a given shape.
    
    Args:
        shape: The shape for which to compute masks.
        padding_mask: The padding mask of the same shape as shape.
        mask_prob: Probability for each token to be chosen as start of the span to be masked.
        mask_length: Size of the span to be masked.
        mask_type: How to compute mask lengths.
        mask_other: Secondary mask argument (used for more complex distributions).
        min_masks: Minimum number of masked spans.
        no_overlap: If false, will switch to an alternative recursive algorithm that prevents spans from overlapping.
        min_space: Minimum space between spans (if no_overlap is enabled).
    
    Returns:
        A boolean tensor with the same shape as shape.
    """
    
    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)
    
    all_num_mask = int(
        mask_prob * all_sz / float(mask_length)
        + np.random.rand()
    )
    all_num_mask = max(min_masks, all_num_mask)
    
    mask_idcs = []
    for i in range(bsz):
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            num_mask = int(
                mask_prob * sz / float(mask_length)
                + np.random.rand()
            )
            num_mask = max(min_masks, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask
        
        lengths = np.full(num_mask, mask_length)
        
        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)
        
        if no_overlap:
            mask_idc = []
            
            def arrange(s, e, length, keep_length):
                span_start = np.random.randint(s, e - length)
                mask_idc.extend(span_start + i for i in range(length))
                
                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - keep_length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts
            
            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = np.array([e - s for s, e in parts], dtype=np.int64)
                lens = lens[lens >= length]
                if len(lens) == 0:
                    break
                lens_idc = np.random.randint(0, len(lens))
                lens_idc = lens_idc[lens >= length][lens_idc]
                parts = arrange(parts[lens_idc][0][0], parts[lens_idc][0][1], length, min_length)
                if len(parts) == 0:
                    break
        else:
            min_len = min(lengths)
            if sz - min_len <= num_mask:
                min_len = sz - num_mask - 1
            
            mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)
            
            lengths = np.array(lengths)
            lengths = np.minimum(lengths, sz - mask_idc)
            
            for i, (idc, length) in enumerate(zip(mask_idc, lengths)):
                mask_idc.extend(idc + i for i in range(length))
        
        mask_idcs.append(np.array(mask_idc))
    
    min_len = min(len(m) for m in mask_idcs)
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        mask[i, mask_idc] = True
    
    return mask


def find_runs(x):
    """Find runs of consecutive items in an array."""
    n = x.shape[0]
    y = x[1:] != x[:-1]
    i = np.append(np.where(y), n - 1)
    z = np.diff(np.append(-1, i))
    p = np.cumsum(np.append(0, z))[:-1]
    return p, i, z


def get_buckets(sizes, num_buckets):
    """Get bucket assignments for given sizes."""
    buckets = {}
    for i, sz in enumerate(sizes):
        buckets[sz] = buckets.get(sz, []) + [i]
    return buckets


def get_bucket_info(bucketed_sizes):
    """Get information about buckets."""
    buckets = get_buckets(bucketed_sizes, len(bucketed_sizes))
    bucketed_ids = [buckets[sz] for sz in bucketed_sizes]
    bucket_sizes = [len(bucket) for bucket in bucketed_ids]
    return bucketed_ids, bucket_sizes


def get_bucketed_sizes(orig_sizes, num_buckets, max_len=1024):
    """Get bucketed sizes for given original sizes."""
    def get_bucket(sz):
        bucket = np.log(sz) / np.log(max_len) * num_buckets
        return min(int(bucket), num_buckets - 1)
    
    return [get_bucket(sz) for sz in orig_sizes]