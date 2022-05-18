import os.path
from typing import Any, Callable, Dict, Sequence, Tuple, Type, Union

import albumentations as alb

import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from kaggle_imsegm.augment import FlashAlbumentationsAdapter
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
        labels: Sequence[str] = None,
        mode: str = "multilabel",
    ):
        self.labels = labels if labels else sorted(list(df_data["class"].unique()))
        self._df_data = self._convert_table(df_data)
        self.path_imgs = path_imgs
        self.quantile = img_quantile
        self.norm = img_norm
        self.mode = mode
        self.transform = transform if transform else FlashAlbumentationsAdapter([])

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
        seg = np.zeros((len(self.labels), *img_size)) if self.mode == "multilabel" else np.zeros(img_size)
        for i, lb in enumerate(self.labels):
            rle = row[lb]
            if isinstance(rle, str):
                if self.mode == "multilabel":
                    seg[i, ...] = rle_decode(rle, img=seg[i, ...])
                else:
                    seg = rle_decode(rle, img=seg, label=i + 1)
        return seg

    def __getitem__(self, idx: int):
        item = self._df_data.iloc[idx]
        img_path = os.path.join(self.path_imgs, item["image_path"])
        img = self._load_image(img_path)
        seg = self._load_annot(item, img.shape)
        item = {
            "input": torch.from_numpy(np.repeat(img[..., np.newaxis], 3, axis=2)),
            "target": torch.from_numpy(np.rollaxis(seg, 0, 3)),
        }
        if self.transform:
            item = self.transform(item)
        return item

    def __len__(self) -> int:
        return len(self._df_data)


COLOR_MEAN: float = 0.349977
COLOR_STD: float = 0.215829
DEFAULT_TRANSFORM = FlashAlbumentationsAdapter(
    [alb.Resize(224, 224), alb.Normalize(mean=COLOR_MEAN, std=COLOR_STD, max_pixel_value=255)]
)


class TractData(LightningDataModule):
    _df_train: pd.DataFrame
    _df_val: pd.DataFrame
    _dataset_cls: Union[Type[TractDataset2D]]
    dataset_train: Dataset
    dataset_val: Dataset

    def __init__(
        self,
        df_data: pd.DataFrame,
        dataset_dir: str,
        val_split: float = 0.1,
        train_transform: Callable = None,
        input_transform: Callable = DEFAULT_TRANSFORM,
        dataset_cls: Union[Type[TractDataset2D]] = TractDataset2D,
        dataset_kwargs: Dict[str, Any] = None,
        dataloader_kwargs: Dict[str, Any] = None,
    ):
        super().__init__()
        self._df_data = df_data
        self.dataset_dir = dataset_dir
        self.val_split = val_split
        self.train_transform = train_transform if train_transform else input_transform
        self.input_transform = input_transform
        self._dataset_cls = dataset_cls
        self._dataset_kwargs = dataset_kwargs if dataset_kwargs else {}
        self._dataloader_kwargs = dataloader_kwargs if dataloader_kwargs else {}

    # def prepare_data(self):
    #     pass

    def setup(self, stage=None) -> None:
        self._df_data["Case_Day"] = [f"case{r['Case']}_day{r['Day']}" for _, r in self._df_data.iterrows()]
        case_days = list(self._df_data["Case_Day"].unique())
        np.random.shuffle(case_days)
        val_offset = int(self.val_split * len(case_days))
        val_ = case_days[-val_offset:]
        labels = list(self._df_data["class"].unique())

        self._df_train = self._df_data[~self._df_data["Case_Day"].isin(val_)]
        self.dataset_train = self._dataset_cls(
            self._df_train, self.dataset_dir, transform=self.train_transform, labels=labels, **self._dataset_kwargs
        )
        self._df_val = self._df_data[self._df_data["Case_Day"].isin(val_)]
        self.dataset_val = self._dataset_cls(
            self._df_val, self.dataset_dir, transform=self.input_transform, labels=labels, **self._dataset_kwargs
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset_train, shuffle=True, **self._dataloader_kwargs)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset_val, shuffle=False, **self._dataloader_kwargs)

    def test_dataloader(self):
        # todo
        return None
