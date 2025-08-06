import glob
import os

import numpy as np
import pandas as pd
import pytest
import torch

from kaggle_imsegm.data_io import create_tract_segmentation, extract_tract_details, load_volume_from_images
from kaggle_imsegm.dataset import TractData, TractDataset2D, TractDataset3D
from kaggle_imsegm.visual import show_tract_datamodule_samples_2d


def _reindex_slices(tab: pd.DataFrame) -> pd.DataFrame:
    """Replace slice index in table with sequence starting from 1."""
    tab["Case_Day"] = [f"case{r['Case']}_day{r['Day']}" for _, r in tab.iterrows()]
    dfg = []
    for cd, df in tab.groupby("Case_Day"):
        sl = sorted(df["Slice"].unique())
        ids = [i + 1 for i in range(len(sl))]
        df["Slice"] = df["Slice"].map(dict(zip(sl, ids)))
        dfg.append(df)
    return pd.concat(dfg)


def test_load_volume(data_dir):
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
    assert seg.dtype == np.uint8
    seg = create_tract_segmentation(tab_cd, vol_shape=vol.shape, mode="multilabel", label_dtype=bool)
    assert seg.shape == (3, 8, 310, 360)
    assert seg.dtype == bool


@pytest.mark.parametrize("annot_mode", ["multilabel", "multiclass"])
def test_dataset_2d(annot_mode: str, data_dir):
    tab = pd.read_csv(os.path.join(data_dir, "train.csv"))
    tab[["Case", "Day", "Slice", "image", "image_path", "height", "width"]] = tab["id"].apply(
        lambda x: pd.Series(extract_tract_details(x, data_dir))
    )
    ds = TractDataset2D(df_data=tab, path_imgs=data_dir, mode=annot_mode)
    assert len(ds) == 18
    spl = ds[5]
    assert spl["input"].dtype == torch.uint8
    assert spl["input"].shape == torch.Size([3, 310, 360])
    assert spl["target"].dtype == torch.uint8
    if annot_mode == "multilabel":
        assert spl["target"].shape == torch.Size([3, 310, 360])
    else:
        assert spl["target"].shape == torch.Size([310, 360])


def test_dataset_2d_predict(data_dir):
    ls_imgs = glob.glob(os.path.join(data_dir, "**", "*.png"), recursive=True)
    ls_imgs = [p.replace(data_dir + os.path.sep, "") for p in sorted(ls_imgs)]
    tab = pd.DataFrame({"image_path": ls_imgs})
    ds = TractDataset2D(df_data=tab, path_imgs=data_dir)
    assert len(ds) == 18
    spl = ds[3]
    assert spl["input"].dtype == torch.uint8
    assert spl["input"].shape == torch.Size([3, 310, 360])
    assert "target" not in spl


@pytest.mark.parametrize("annot_mode", ["multilabel", "multiclass"])
def test_dataset_3d(annot_mode: str, data_dir):
    tab = pd.read_csv(os.path.join(data_dir, "train.csv"))
    tab[["Case", "Day", "Slice", "image", "image_path", "height", "width"]] = tab["id"].apply(
        lambda x: pd.Series(extract_tract_details(x, data_dir))
    )
    # reset index for slices
    tab = _reindex_slices(tab)
    ds = TractDataset3D(df_data=tab, path_imgs=data_dir, mode=annot_mode)

    assert len(ds) == 4
    spl = ds[3]
    assert spl["input"].dtype == torch.uint8
    assert spl["input"].shape == torch.Size([4, 310, 360])
    assert spl["target"].dtype == torch.uint8
    if annot_mode == "multilabel":
        assert spl["target"].shape == torch.Size([3, 4, 310, 360])
    else:
        assert spl["target"].shape == torch.Size([4, 310, 360])


def test_dataset_3d_predict(data_dir):
    ls_imgs = glob.glob(os.path.join(data_dir, "**", "*.png"), recursive=True)
    ls_imgs = [p.replace(data_dir + os.path.sep, "") for p in sorted(ls_imgs)]
    tab = pd.DataFrame({"image_path": ls_imgs})
    ds = TractDataset3D(df_data=tab, path_imgs=data_dir)

    assert len(ds) == 4
    spl = ds[3]
    assert spl["input"].dtype == torch.uint8
    assert spl["input"].shape == torch.Size([4, 310, 360])
    assert "target" not in spl


@pytest.mark.parametrize("dataset_cls", [TractDataset2D])  # todo: TractDataset3D
def test_datamodule(dataset_cls, data_dir):
    tab_train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    tab_train[["Case", "Day", "Slice", "image", "image_path", "height", "width"]] = tab_train["id"].apply(
        lambda x: pd.Series(extract_tract_details(x, data_dir))
    )
    tab_train = _reindex_slices(tab_train)

    ls_imgs = glob.glob(os.path.join(data_dir, "**", "*.png"), recursive=True)
    ls_imgs = [p.replace(data_dir + os.path.sep, "") for p in sorted(ls_imgs)]
    tab_pred = pd.DataFrame({"image_path": ls_imgs})

    np.random.seed(42)
    dm = TractData(
        df_train=tab_train,
        df_predict=tab_pred,
        dataset_dir=data_dir,
        val_split=0.25,
        dataset_kwargs=dict(label_dtype=np.float32),
        dataloader_kwargs=dict(batch_size=3, num_workers=2),
        dataset_cls=dataset_cls,
    )
    dm.setup()
    assert len(dm.labels) == 3
    assert list(dm.train_dataloader())
    assert list(dm.val_dataloader())
    assert list(dm.predict_dataloader())
    if dataset_cls is TractDataset2D:
        assert len(dm.train_dataloader()) == 5
        assert len(dm.val_dataloader()) == 1
        assert len(dm.predict_dataloader()) == 6
    # elif dataset_cls is TractDataset3D:
    #     assert len(dm.train_dataloader()) == 1
    #     assert len(dm.val_dataloader()) == 1
    #     assert len(dm.predict_dataloader()) == 2
    else:
        raise TypeError()

    show_tract_datamodule_samples_2d(dm.predict_dataloader())
