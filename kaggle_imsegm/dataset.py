import os
from typing import Any, Callable, Dict, Sequence, Tuple, Type, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from kaggle_imsegm.mask import rle_decode
from kaggle_imsegm.transform import DEFAULT_TRANSFORM, FlashAlbumentationsAdapter


class TractDataset2D(Dataset):
    """Basic 2D dataset."""

    labels: Sequence[str]

    def __init__(
        self,
        df_data: pd.DataFrame,
        path_imgs: str,
        transform: Callable = None,
        img_quantile: float = 0.01,
        img_norm: bool = True,
        labels: Sequence[str] = None,
        mode: str = "multilabel",
        label_dtype=np.uint8,
    ):
        assert "image_path" in df_data.columns
        self.with_annot = all(c in df_data.columns for c in ["class", "segmentation"])
        if self.with_annot:
            self.labels = labels or sorted(list(df_data["class"].unique()))
        self._df_data = self._convert_table(df_data) if self.with_annot else df_data
        self.path_imgs = path_imgs
        self.quantile = img_quantile
        self.norm = img_norm
        self.mode = mode
        self.transform = transform or FlashAlbumentationsAdapter([])
        self._label_dtype = label_dtype

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
        seg_size = (len(self.labels), *img_size) if self.mode == "multilabel" else img_size
        seg = np.zeros(seg_size, dtype=self._label_dtype)
        for i, lb in enumerate(self.labels):
            rle = row[lb]
            if isinstance(rle, str):
                if self.mode == "multilabel":
                    seg[i, ...] = rle_decode(rle, img=seg[i, ...])
                else:
                    seg = rle_decode(rle, img=seg, label=i + 1)
        return seg

    def __getitem__(self, idx: int):
        row = self._df_data.iloc[idx]
        img_path = os.path.join(self.path_imgs, row["image_path"])
        img = self._load_image(img_path)
        h, w = img.shape
        item = {
            "input": torch.from_numpy(np.repeat(img[..., np.newaxis], 3, axis=2)),
            "metadata": dict(size=(h, w), height=h, width=w),
        }
        if self.with_annot:
            seg = self._load_annot(row, img.shape)
            item["target"] = torch.from_numpy(np.rollaxis(seg, 0, 3))
        if self.transform:
            item = self.transform(item)
        if self.with_annot:
            item["target"] = item["target"].permute(2, 0, 1)
        return item

    def __len__(self) -> int:
        return len(self._df_data)


class TractData(LightningDataModule):
    _df_train: pd.DataFrame
    _df_predict: pd.DataFrame
    _dataset_cls: Union[Type[TractDataset2D]]
    dataset_train: Dataset = None
    dataset_val: Dataset = None
    dataset_pred: Dataset = None
    labels: Sequence[str] = None
    _setup_completed: bool = False

    def __init__(
        self,
        df_train: pd.DataFrame,
        dataset_dir: str,
        df_predict: pd.DataFrame = None,
        val_split: float = 0.1,
        train_transform: Callable = None,
        input_transform: Callable = DEFAULT_TRANSFORM,
        dataset_cls: Union[Type[TractDataset2D]] = TractDataset2D,
        dataset_kwargs: Dict[str, Any] = None,
        dataloader_kwargs: Dict[str, Any] = None,
    ):
        super().__init__()
        self._df_train = df_train
        self._df_predict = df_predict
        self.dataset_dir = dataset_dir
        self.val_split = val_split
        self.train_transform = train_transform or input_transform
        self.input_transform = input_transform
        self._dataset_cls = dataset_cls
        self._dataset_kwargs = dataset_kwargs or {}
        self._dataloader_kwargs = dataloader_kwargs or {}

    # def prepare_data(self):
    #     pass

    def setup(self, stage=None) -> None:
        if self._setup_completed:
            return
        self._df_train["Case_Day"] = [f"case{r['Case']}_day{r['Day']}" for _, r in self._df_train.iterrows()]
        case_days = list(self._df_train["Case_Day"].unique())
        np.random.shuffle(case_days)
        val_offset = int(self.val_split * len(case_days))
        val_ = case_days[-val_offset:]
        self.labels = list(self._df_train["class"].unique())

        df_train = self._df_train[~self._df_train["Case_Day"].isin(val_)]
        self.dataset_train = self._dataset_cls(
            df_train, self.dataset_dir, transform=self.train_transform, labels=self.labels, **self._dataset_kwargs
        )
        df_val = self._df_train[self._df_train["Case_Day"].isin(val_)]
        self.dataset_val = self._dataset_cls(
            df_val, self.dataset_dir, transform=self.input_transform, labels=self.labels, **self._dataset_kwargs
        )
        if self._df_predict is not None:
            self.dataset_pred = self._dataset_cls(
                self._df_predict, self.dataset_dir, transform=self.input_transform, **self._dataset_kwargs
            )
        self._setup_completed = True

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset_train, shuffle=True, **self._dataloader_kwargs)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset_val, shuffle=False, **self._dataloader_kwargs)

    def predict_dataloader(self) -> DataLoader:
        if self.dataset_pred:
            return DataLoader(self.dataset_pred, shuffle=False, **self._dataloader_kwargs)
