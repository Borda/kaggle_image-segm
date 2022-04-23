import glob
import os
from typing import Tuple

import numpy as np
from pandas import DataFrame
from PIL import Image

from kaggle_imsegm.mask import rle_decode


def load_image_volume(img_dir: str, quant: float = 0.01) -> np.ndarray:
    """
    Args:
        img_dir: path to folder with images, where each image is volume slice
        quant: remove some intensity extreme
    """
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    imgs = [np.array(Image.open(p)).tolist() for p in imgs]
    # print([np.max(im) for im in imgs])
    vol = np.array(imgs)
    if quant:
        q_low, q_high = np.percentile(vol, [quant * 100, (1 - quant) * 100])
        vol = np.clip(vol, q_low, q_high)
    v_min, v_max = np.min(vol), np.max(vol)
    vol = (vol - v_min) / (v_max - v_min)
    vol = (vol * 255).astype(np.uint8)
    return vol


def create_organs_segm(df_vol: DataFrame, vol_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Args:
        df_vol: dataframe with filtered resorts just for one volume and segmentation ans RLE
        vol_shape: shape of the resulting volume
    """
    assert all(c in df_vol.columns for c in ["Slice", "class", "segmentation"])
    df_vol = df_vol.replace(np.nan, "")
    segm = np.zeros(vol_shape, dtype=np.uint8)
    lbs = sorted(df_vol["class"].unique())
    for idx_, dfg in df_vol.groupby("Slice"):
        idx = int(idx_) - 1
        mask = segm[idx, :, :]
        for _, (lb, rle) in dfg[["class", "segmentation"]].iterrows():
            lb = lbs.index(lb)
            if not rle:
                continue
            mask = rle_decode(rle, img=mask, label=lb)
        segm[idx, :, :] = mask
        # plt.figure(); plt.imshow(mask)
    return segm
