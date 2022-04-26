from typing import Dict

import numpy as np


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


def rle_encode(mask: np.ndarray, label_bg: int = 0) -> Dict[int, str]:
    """Encode mask to Run-length encoding.

    Inspiration took from: https://gist.github.com/nvictus/66627b580c13068589957d6ab0919e66

    >>> from pprint import pprint
    >>> mask = np.array([[0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    ...                  [0, 0, 0, 1, 1, 1, 2, 2, 2, 0],
    ...                  [0, 0, 0, 0, 0, 1, 3, 3, 0, 0],])
    >>> pprint(rle_encode(mask))
    {1: '1 5 13 3 25 1', 2: '16 3', 3: '26 2'}
    """
    vec = mask.flatten()
    nb = len(vec)
    where = np.flatnonzero
    starts = np.r_[0, where(~np.isclose(vec[1:], vec[:-1], equal_nan=True)) + 1]
    lengths = np.diff(np.r_[starts, nb])
    values = vec[starts]
    assert len(starts) == len(lengths) == len(values)
    rle = {}
    for start, length, val in zip(starts, lengths, values):
        if val == label_bg:
            continue
        rle[val] = rle.get(val, []) + [str(start), length]
    # post-processing
    rle = {lb: " ".join(map(str, id_lens)) for lb, id_lens in rle.items()}
    return rle
