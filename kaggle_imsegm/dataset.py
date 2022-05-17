import os.path
from typing import Callable, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from kaggle_imsegm.mask import rle_decode


class TractDataset2D(Dataset):
    """Basic 2D dataset."""

    def __init__(
        self,
        df_data: pd.DataFrame,
        path_imgs: str,
        transform: Callable = None,
        img_quantile: float = 0.01,
        img_norm: bool = True,
        mode: str = "multi-label",
    ):
        self.labels = sorted(list(df_data["class"].unique()))
        self._df_data = self._convert_table(df_data)
        self.path_imgs = path_imgs
        self.quantile = img_quantile
        self.norm = img_norm
        self.mode = mode
        self.transform = transform

    @staticmethod
    def _convert_table(df):
        """Convert table to row per images and column per class."""
        rows = []
        for id_, dfg in tqdm(df.groupby("id")):
            row = dict(dfg.iloc[0])
            del row["class"]
            del row["segmentation"]
            for _, (cls, segm) in dfg[["class", "segmentation"]].iterrows():
                row[cls] = segm
            rows.append(row)
        return pd.DataFrame(rows)

    def _load_image(self, img_path: str) -> np.ndarray:
        img = np.array(Image.open(img_path))
        if self.quantile:
            q_low, q_high = np.percentile(img, [self.quantile * 100, (1 - self.quantile) * 100])
            img = np.clip(img, q_low, q_high)
        if self.norm:
            v_min, v_max = np.min(img), np.max(img)
            img = (img - v_min) / float(v_max - v_min)
            img = (img * 255).astype(np.uint8)
        return img

    def _load_annot(self, row: pd.Series, img_size: Tuple[int, int]) -> np.ndarray:
        seg = np.zeros((len(self.labels), *img_size)) if self.mode == "multi-label" else np.zeros(img_size)
        for i, lb in enumerate(self.labels):
            rle = row[lb]
            if isinstance(rle, str):
                if self.mode == "multi-label":
                    seg[i, ...] = rle_decode(rle, img=seg[i, ...])
                else:
                    seg = rle_decode(rle, img=seg, label=i + 1)
        return seg

    def __getitem__(self, idx: int):
        item = self._df_data.iloc[idx]
        img_path = os.path.join(self.path_imgs, item["image_path"])
        img = self._load_image(img_path)
        seg = self._load_annot(item, img.shape)
        item = {"image": img, "mask": seg}
        if self.transform:
            item = self.transform(item)
        return item

    def __len__(self) -> int:
        return len(self._df_data)
