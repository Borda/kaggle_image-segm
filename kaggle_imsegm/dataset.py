import os
from typing import Any, Callable, Dict, List, Sequence, Tuple, Type, Union

import albumentations as alb
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from kaggle_imsegm.data_io import create_tract_segmentation, load_volume_from_images
from kaggle_imsegm.mask import rle_decode
from kaggle_imsegm.transform import COLOR_MEAN, COLOR_STD, FlashAlbumentationsAdapter

DEFAULT_TRANSFORM_2D = FlashAlbumentationsAdapter([
    alb.Resize(224, 224),
    alb.Normalize(mean=COLOR_MEAN, std=COLOR_STD, max_pixel_value=255),
])


class TractDataset(Dataset):
    """Basic dataset."""

    _df_data: Union[pd.DataFrame, Sequence[pd.DataFrame]]
    labels: Sequence[str]
    transform: Callable = None

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
        self.img_folder = path_imgs
        self.quantile = img_quantile
        self.norm = img_norm
        self.mode = mode
        if transform:
            self.transform = transform
        self._label_dtype = label_dtype

    def _load_image(self, idx: int) -> np.ndarray:
        raise NotImplementedError()

    def _load_annot(self, idx: int, img_shape: tuple) -> np.ndarray:
        raise NotImplementedError()

    def _metadata(self, img: np.ndarray) -> Dict[str, Any]:
        raise NotImplementedError()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img = self._load_image(idx)
        item = {
            "input": torch.from_numpy(img),
            "metadata": self._metadata(img),
        }
        if self.with_annot:
            item["target"] = torch.from_numpy(self._load_annot(idx, img.shape))
        if self.transform:
            item = self.transform(item)
        return item

    def __len__(self) -> int:
        return len(self._df_data)


class TractDataset2D(TractDataset):
    """2D dataset."""

    transform: Callable = FlashAlbumentationsAdapter([])

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
        super().__init__(df_data, path_imgs, transform, img_quantile, img_norm, labels, mode, label_dtype)
        self._df_data = self._convert_table(df_data) if self.with_annot else df_data

    @staticmethod
    def _convert_table(df: pd.DataFrame) -> pd.DataFrame:
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

    def _load_image(self, idx: int, as_rgb: bool = True) -> np.ndarray:
        img_path = os.path.join(self.img_folder, self._df_data.iloc[idx]["image_path"])
        img = np.array(Image.open(img_path))
        if self.quantile:
            q_low, q_high = np.percentile(img, [self.quantile * 100, (1 - self.quantile) * 100])
            img = np.clip(img, q_low, q_high)
        if self.norm:
            v_min, v_max = np.min(img), np.max(img)
            img = (img - v_min) / float(v_max - v_min)
            img = (img * 255).astype(np.uint8)
        if as_rgb:
            img = np.repeat(img[..., np.newaxis], 3, axis=2)
        return img

    def _load_annot(self, idx: int, img_shape: Tuple[int, int]) -> np.ndarray:
        row = self._df_data.iloc[idx]
        img_size = img_shape[:2]  # in case you pass RGB image
        seg_size = (*img_size, len(self.labels)) if self.mode == "multilabel" else img_size
        seg = np.zeros(seg_size, dtype=self._label_dtype)
        for i, lb in enumerate(self.labels):
            rle = row[lb]
            if isinstance(rle, str):
                if self.mode == "multilabel":
                    seg[..., i] = rle_decode(rle, img=seg[..., i])
                else:
                    seg = rle_decode(rle, img=seg, label=i + 1)
        return seg

    def _metadata(self, img: np.ndarray) -> Dict[str, Any]:
        h, w = img.shape[:2]
        return dict(size=(h, w), height=h, width=w)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = super().__getitem__(idx)
        if "target" in item and item["target"].ndim == 3:
            item["target"] = item["target"].permute(2, 0, 1)
        return item


class TractDataset3D(TractDataset):
    """3D dataset."""

    # transform: Callable = FlashAlbumentationsAdapter([])

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
        super().__init__(df_data, path_imgs, transform, img_quantile, img_norm, labels, mode, label_dtype)
        self._df_data = self._convert_table(df_data)

    @staticmethod
    def _convert_table(df: pd.DataFrame) -> List[pd.DataFrame]:
        """Convert table to row per images and column per class."""
        df["Case_Day"] = [p.split(os.path.sep)[-3] for p in df["image_path"]]
        dfs = [dfg for _, dfg in df.groupby("Case_Day")]
        return dfs

    def _load_image(self, idx: int) -> np.ndarray:
        img_path = os.path.join(self.img_folder, self._df_data[idx].iloc[0]["image_path"])
        return load_volume_from_images(os.path.dirname(img_path), quantile=self.quantile, norm=self.norm)

    def _load_annot(self, idx: int, img_shape: Tuple[int, int, int]) -> np.ndarray:
        return create_tract_segmentation(
            self._df_data[idx], vol_shape=img_shape, mode=self.mode, labels=self.labels, label_dtype=self._label_dtype
        )

    def _metadata(self, img: np.ndarray) -> Dict[str, Any]:
        return dict(shape=img.shape)


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
        input_transform: Callable = None,
        dataset_cls: Union[Type[TractDataset2D]] = TractDataset2D,
        dataset_kwargs: Dict[str, Any] = None,
        dataloader_kwargs: Dict[str, Any] = None,
    ):
        super().__init__()
        self._df_train = df_train
        self._df_predict = df_predict
        self.dataset_dir = dataset_dir
        self.val_split = val_split
        self._dataset_cls = dataset_cls
        self.input_transform = input_transform or self._default_transform()
        self.train_transform = train_transform or self.input_transform
        self._dataset_kwargs = dataset_kwargs or {}
        self._dataloader_kwargs = dataloader_kwargs or {}

    # def prepare_data(self):
    #     pass
    def _default_transform(self) -> Callable:
        if self._dataset_cls is TractDataset2D:
            return DEFAULT_TRANSFORM_2D
        if self._dataset_cls is TractDataset3D:
            # todo
            return lambda x: x

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
