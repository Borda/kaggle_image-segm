import numpy as np
import pandas as pd


def rle_decode(mask_rle: str, img: np.ndarray = None, img_shape: tuple = None, label: int = 1) -> np.ndarray:
    """create a single label mask for encoding.

    >>> mask = rle_decode("3 2 11 5 23 3 35 1", img_shape=(8, 10))
    >>> mask = rle_decode("55 3 66 2", img=mask, label=2)
    >>> mask
    array([[0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 2, 2, 2, 0, 0],
           [0, 0, 0, 0, 0, 0, 2, 2, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint16)
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


def create_mask(df_image: pd.DataFrame) -> np.ndarray:
    """Aggregate multiple encoding to single multi-label mask."""
    assert len(df_image["id"].unique()) == 1
    sizes = list({(row["height"], row["width"]) for _, row in df_image.iterrows()})
    assert len(sizes) == 1
    mask = np.zeros(sizes[0], dtype=np.uint16)
    df_image.reset_index(inplace=True)
    for idx, row in df_image.iterrows():
        mask = rle_decode(row["annotation"], img=mask, label=idx + 1)
    return mask
