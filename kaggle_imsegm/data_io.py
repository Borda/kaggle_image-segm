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


def load_volume_from_images(img_dir: str, quantile: float = 0.01, norm: bool = True) -> np.ndarray:
    """Load X-ray volume constructed from images/scans in vertical direction.

    Args:
        img_dir: path to folder with images, where each image is volume slice
        quantile: remove some intensity extreme
        norm: normalize image in range 0 - 255
    """
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    imgs = [np.array(Image.open(p)).tolist() for p in img_paths]
    # print([np.max(im) for im in imgs])
    vol = np.array(imgs)
    if quantile:
        q_low, q_high = np.percentile(vol, [quantile * 100, (1 - quantile) * 100])
        vol = np.clip(vol, q_low, q_high)
    if norm:
        v_min, v_max = np.min(vol), np.max(vol)
        vol = (vol - v_min) / float(v_max - v_min)
        vol = (vol * 255).astype(np.uint8)
    return vol


def create_tract_segmentation(
    df_vol: DataFrame,
    vol_shape: Tuple[int, int, int],
    mode: str = "multiclass",
    labels: Optional[Sequence[str]] = None,
    label_dtype=np.uint8,
) -> np.ndarray:
    """Create 3D segmentation if tracts.

    Args:
        df_vol: dataframe with filtered resorts just for one volume and segmentation ans RLE
        mode: select type of annotation
        vol_shape: shape of the resulting volume
        labels: list of selected classes
        label_dtype: data type
    """
    assert all(c in df_vol.columns for c in ["Slice", "class", "segmentation"])
    df_vol = df_vol.replace(np.nan, "")
    if not labels:
        labels = sorted(df_vol["class"].unique())
    else:
        df_labels = tuple(df_vol["class"].unique())
        assert all(lb in labels for lb in df_labels), df_labels
    if mode == "multilabel":
        vol_shape = (len(labels),) + vol_shape
    segm = np.zeros(vol_shape, dtype=label_dtype)
    for idx_, dfg in df_vol.groupby("Slice"):
        idx = int(idx_) - 1
        for _, (lb, rle) in dfg[["class", "segmentation"]].iterrows():
            lb = labels.index(lb)
            if not rle or not isinstance(rle, str):
                continue
            if mode == "multilabel":
                segm[lb, idx, :, :] = rle_decode(rle, img=segm[lb, idx, :, :], label=1)
            else:
                segm[idx, :, :] = rle_decode(rle, img=segm[idx, :, :], label=lb + 1)
        # plt.figure(); plt.imshow(mask)
    return segm


def extract_tract_details(id_: str, dataset_dir: str, folder: str = "train") -> Dict[str, Any]:
    """Enrich dataframe by information from image name.

    Args:
        id_: ID from the provided table
        dataset_dir: path to the dataset folder
        folder: sub-folder as train/test
    """
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

    Args:
        volume: input volume of any shape
        vol_size: the output volume shape
        mode: interpolation mode (reccomended "nearest" for segmentation and "trilinear" for images)

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
    df_scan,
    labels: List[str],
    dir_data: str,
    dir_imgs: str,
    dir_segm: Optional[str] = None,
    sfolder: str = "train",
    quantile: float = 0.03,
) -> List[str]:
    """Prepare set of images and segmentation with respect to whole scan volume.

    Args:
        df_scan: filtered DataFrame with scans
        labels: list of all possible labels
        dir_data: input dataset folder
        dir_imgs: output folder for images
        dir_segm: output folder for segmentations
        sfolder: sub-folder for separation between train/val/test
        quantile: for filtering noise/outliers in intensities
    """
    case, day, image_path = df_scan.iloc[0][["Case", "Day", "image_path"]]
    img_dirs = os.path.dirname(image_path).split(os.path.sep)

    img_folder = os.path.join(dir_data, f"case{case}", f"case{case}_day{day}", "scans")
    vol = load_volume_from_images(img_dir=img_folder, quantile=quantile)
    if dir_segm:
        seg = create_tract_segmentation(df_vol=df_scan, vol_shape=vol.shape, labels=labels)

    imgs = []
    for _, row in df_scan.drop_duplicates("image_path").iterrows():
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
