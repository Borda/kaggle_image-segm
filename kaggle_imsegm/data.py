import glob
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch.nn.functional as F
from pandas import DataFrame
from PIL import Image
from torch import Tensor

from kaggle_imsegm.mask import rle_decode


def load_volume_from_images(img_dir: str, quant: float = 0.01) -> np.ndarray:
    """Load X-ray volume constructed from images/scans in vertical direction.

    Args:
        img_dir: path to folder with images, where each image is volume slice
        quant: remove some intensity extreme
    """
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    imgs = [np.array(Image.open(p)).tolist() for p in img_paths]
    # print([np.max(im) for im in imgs])
    vol = np.array(imgs)
    if quant:
        q_low, q_high = np.percentile(vol, [quant * 100, (1 - quant) * 100])
        vol = np.clip(vol, q_low, q_high)
    v_min, v_max = np.min(vol), np.max(vol)
    vol = (vol - v_min) / (v_max - v_min)
    vol = (vol * 255).astype(np.uint8)
    return vol


def create_tract_segm(
    df_vol: DataFrame, vol_shape: Tuple[int, int, int], labels: Optional[Sequence[str]] = None
) -> np.ndarray:
    """Create 3D segmentation if tracts.

    Args:
        df_vol: dataframe with filtered resorts just for one volume and segmentation ans RLE
        vol_shape: shape of the resulting volume
    """
    assert all(c in df_vol.columns for c in ["Slice", "class", "segmentation"])
    df_vol = df_vol.replace(np.nan, "")
    segm = np.zeros(vol_shape, dtype=np.uint8)
    if not labels:
        labels = sorted(df_vol["class"].unique())
    else:
        df_labels = tuple(df_vol["class"].unique())
        assert all(lb in labels for lb in df_labels), df_labels
    for idx_, dfg in df_vol.groupby("Slice"):
        idx = int(idx_) - 1
        mask = segm[idx, :, :]
        for _, (lb, rle) in dfg[["class", "segmentation"]].iterrows():
            lb = labels.index(lb) + 1
            if not rle or not isinstance(rle, str):
                continue
            mask = rle_decode(rle, img=mask, label=lb)
        segm[idx, :, :] = mask
        # plt.figure(); plt.imshow(mask)
    return segm


def extract_tract_details(id_: str, dataset_dir: str, folder: str = "train") -> Dict[str, Any]:
    """Enrich dataframe by information from image name."""
    id_fields = id_.split("_")
    case = id_fields[0].replace("case", "")
    day = id_fields[1].replace("day", "")
    slice_id = id_fields[3]
    # ../input/uw-madison-gi-tract-image-segmentation/train/case101/case101_day20/scans/slice_0001_266_266_1.50_1.50.png
    img_dir = os.path.join(dataset_dir, folder, f"case{case}", f"case{case}_day{day}", "scans")
    imgs = glob.glob(os.path.join(img_dir, f"slice_{slice_id}_*.png"))
    assert len(imgs) == 1
    img_path = imgs[0].replace(dataset_dir + "/", "")
    img = os.path.basename(img_path)
    # slice_0001_266_266_1.50_1.50.png
    im_fields = img.split("_")
    return {
        "Case": int(case),
        "Day": int(day),
        "Slice": slice_id,
        "image": img,
        "image_path": img_path,
        "height": int(im_fields[3]),
        "width": int(im_fields[2]),
    }


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


def interpolate_volume(volume: Tensor, vol_size: Optional[Tuple[int, int, int]], mode: str = "nearest") -> Tensor:
    """Interpolate volume in last (Z) dimension.

    >>> import torch
    >>> vol = torch.rand(64, 64, 12)
    >>> vol2 = interpolate_volume(vol, vol_size=(64, 64, 24), mode="trilinear")
    >>> vol2.shape
    torch.Size([64, 64, 24])
    """
    vol_shape = tuple(volume.shape)
    # assert vol_shape[0] == vol_shape[1], f"mixed shape: {vol_shape}"
    if vol_shape == vol_size:
        return volume
    return F.interpolate(volume.unsqueeze(0).unsqueeze(0), size=vol_size, mode=mode, align_corners=False)[0, 0]


def preprocess_tract_scan(
    df_, labels: List[str], dir_data: str, dir_imgs: str, dir_segm: Optional[str] = None, sfolder: str = "train"
) -> List[str]:
    case, day, image_path = df_.iloc[0][["Case", "Day", "image_path"]]
    img_dirs = os.path.dirname(image_path).split(os.path.sep)

    img_folder = os.path.join(dir_data, f"case{case}", f"case{case}_day{day}", "scans")
    vol = load_volume_from_images(img_dir=img_folder)
    if dir_segm:
        seg = create_tract_segm(df_vol=df_, vol_shape=vol.shape, labels=labels)

    imgs = []
    for _, row in df_.drop_duplicates("image_path").iterrows():
        idx, image, img_path = row[["Slice", "image", "image_path"]]
        idx = int(idx) - 1
        img_name_local = f"{img_dirs[-2]}_{image}"
        img_path_local = os.path.join(dir_imgs, sfolder, img_name_local)
        Image.fromarray(vol[idx, :, :]).convert("RGB").save(img_path_local)
        if dir_segm:
            segm_path = os.path.join(dir_segm, sfolder, img_name_local)
            Image.fromarray(seg[idx, :, :]).save(segm_path)
        imgs.append(img_path_local)
    return imgs
