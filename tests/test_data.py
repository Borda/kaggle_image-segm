import glob
import os

import numpy as np
import pandas as pd
import pytest
import torch

from kaggle_imsegm.data_io import create_tract_segmentation, extract_tract_details, load_volume_from_images
from kaggle_imsegm.dataset import TractData, TractDataset, TractDataset2D
from kaggle_imsegm.visual import show_tract_datamodule_samples_2d
from tests import _ROOT_DATA


def test_load_volume(data_dir: str = _ROOT_DATA):
    img_dir = os.path.join(data_dir, "train", "case102", "case102_day0", "scans")
    vol = load_volume_from_images(img_dir)
    assert vol.shape == (8, 310, 360)

    tab = pd.read_csv(os.path.join(data_dir, "train.csv"))
    tab[["Case", "Day", "Slice", "image", "image_path", "height", "width"]] = tab["id"].apply(
        lambda x: pd.Series(extract_tract_details(x, data_dir))
    )
    tab_cd = tab[tab["id"].str.startswith("case102_day0")]
    # HACK: reindex to align the small sample dataset
    tab_cd["Slice"] = tab_cd["Slice"].apply(lambda s: int((int(s) - 55) / 10 + 1))
    seg = create_tract_segmentation(tab_cd, vol_shape=vol.shape)
    assert seg.shape == (8, 310, 360)


@pytest.mark.parametrize("cls", [TractDataset2D])
def test_dataset(cls: TractDataset, data_dir: str = _ROOT_DATA):
    tab = pd.read_csv(os.path.join(data_dir, "train.csv"))
    tab[["Case", "Day", "Slice", "image", "image_path", "height", "width"]] = tab["id"].apply(
        lambda x: pd.Series(extract_tract_details(x, data_dir))
    )
    ds = cls(df_data=tab, path_imgs=data_dir)
    assert len(ds) == 18
    spl = ds[5]
    assert spl["input"].dtype == torch.uint8
    assert spl["input"].shape == torch.Size([3, 310, 360])
    assert spl["target"].dtype == torch.uint8
    assert spl["target"].shape == torch.Size([3, 310, 360])


def test_dataset_predict_2d(data_dir: str = _ROOT_DATA):
    ls_imgs = glob.glob(os.path.join(data_dir, "**", "*.png"), recursive=True)
    ls_imgs = [p.replace(data_dir + os.path.sep, "") for p in sorted(ls_imgs)]
    tab = pd.DataFrame({"image_path": ls_imgs})
    ds = TractDataset2D(df_data=tab, path_imgs=data_dir)
    assert len(ds) == 18
    spl = ds[5]
    assert spl["input"].dtype == torch.uint8
    assert spl["input"].shape == torch.Size([3, 310, 360])
    assert "target" not in spl


def test_datamodule(data_dir: str = _ROOT_DATA):
    np.random.seed(42)
    tab_train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    tab_train[["Case", "Day", "Slice", "image", "image_path", "height", "width"]] = tab_train["id"].apply(
        lambda x: pd.Series(extract_tract_details(x, data_dir))
    )

    ls_imgs = glob.glob(os.path.join(data_dir, "**", "*.png"), recursive=True)
    ls_imgs = [p.replace(data_dir + os.path.sep, "") for p in sorted(ls_imgs)]
    tab_pred = pd.DataFrame({"image_path": ls_imgs})

    dm = TractData(
        df_train=tab_train,
        df_predict=tab_pred,
        dataset_dir=data_dir,
        val_split=0.25,
        dataset_kwargs=dict(label_dtype=np.float32),
        dataloader_kwargs=dict(batch_size=3, num_workers=2),
    )
    dm.setup()
    assert len(dm.labels) == 3
    assert len(dm.train_dataloader()) == 5
    assert len(dm.val_dataloader()) == 2
    assert len(dm.predict_dataloader()) == 6
    assert list(dm.train_dataloader())
    assert list(dm.val_dataloader())
    assert list(dm.predict_dataloader())

    show_tract_datamodule_samples_2d(dm.predict_dataloader())
