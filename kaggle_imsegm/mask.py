from typing import Dict

import numpy as np
import pandas as pd


def rle_decode(mask_rle: str, img: np.ndarray = None, img_shape: tuple = None, label: int = 1) -> np.ndarray:
    """Create a single label mask for Run-length encoding.

    >>> mask = rle_decode("3 2 11 5 23 3 35 1", img_shape=(8, 10))
    >>> mask = rle_decode("55 3 66 2 77 1", img=mask, label=2)
    >>> mask = rle_decode("26 3 36 2", img=mask, label=3)
    >>> mask
    array([[0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 3, 3, 3, 0],
           [0, 0, 0, 0, 0, 1, 3, 3, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 2, 2, 2, 0, 0],
           [0, 0, 0, 0, 0, 0, 2, 2, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 2, 0, 0]], dtype=uint16)
    >>> from pprint import pprint
    >>> pprint(rle_encode(mask))
    {1: '3 2 11 5 23 3 35 1', 2: '55 3 66 2 77 1', 3: '26 3 36 2'}
    """
    seq = mask_rle.split()
    starts = np.array(list(map(int, seq[0::2])))
    lengths = np.array(list(map(int, seq[1::2])))
    assert len(starts) == len(lengths)
    ends = starts + lengths

    if img is None:
        img = np.zeros((np.product(img_shape),), dtype=np.uint16)
    else:
        img_shape = img.shape
        img = img.flatten()
    for begin, end in zip(starts, ends):
        img[begin:end] = label
    return img.reshape(img_shape)


def rle_encode(mask: np.ndarray, bg: int = 0) -> Dict[int, str]:
    """Encode mask to Run-length encoding.

    >>> from pprint import pprint
    >>> mask = np.array([[0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    ...                  [0, 0, 0, 1, 1, 1, 2, 2, 2, 0],
    ...                  [0, 0, 0, 0, 0, 1, 3, 3, 0, 0],])
    >>> pprint(rle_encode(mask))
    {1: '1 5 13 3 25 1', 2: '16 3', 3: '26 2'}
    """
    vec = mask.flatten()
    rle = {}
    running_lb = None
    running_idx = None
    running_count = 0
    # iterate complete vector
    for i, v in enumerate(vec):
        if v == bg and running_lb in (None, bg):
            continue
        if running_lb == v:
            running_count += 1
            continue
        if running_lb not in (None, bg):
            if running_lb not in rle:
                rle[running_lb] = []
            rle[running_lb] += [running_idx, running_count]
        running_lb = v
        running_idx = i
        running_count = 1
    # post processing
    rle = {lb: " ".join(map(str, idx_counts)) for lb, idx_counts in rle.items()}
    return rle


def create_cells_instances_mask(df_image: pd.DataFrame) -> np.ndarray:
    """Aggregate multiple encoding to single multi-label mask."""
    assert len(df_image["id"].unique()) == 1
    sizes = list({(row["height"], row["width"]) for _, row in df_image.iterrows()})
    assert len(sizes) == 1
    mask = np.zeros(sizes[0], dtype=np.uint16)
    df_image.reset_index(inplace=True)
    for idx, row in df_image.iterrows():
        mask = rle_decode(row["annotation"], img=mask, label=idx + 1)
    return mask
