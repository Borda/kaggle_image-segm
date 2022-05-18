import os

import pandas as pd
import pytest
from torch.utils.data import Dataset

from kaggle_imsegm.data_io import create_tract_segmentation, extract_tract_details, load_volume_from_images
from kaggle_imsegm.dataset import TractDataset2D
from tests import _ROOT_DATA


def test_load_volume(data_dir: str = _ROOT_DATA):
    img_dir = os.path.join(data_dir, "train", "case102", "case102_day0", "scans")
    vol = load_volume_from_images(img_dir)
    assert vol.shape == (8, 310, 360)

    tab = pd.read_csv(os.path.join(_ROOT_DATA, "train.csv"))
    tab[["Case", "Day", "Slice", "image", "image_path", "height", "width"]] = tab["id"].apply(
        lambda x: pd.Series(extract_tract_details(x, data_dir))
    )
    tab_cd = tab[tab["id"].str.startswith("case102_day0")]
    # HACK: reindex to align the small sample dataset
    tab_cd["Slice"] = tab_cd["Slice"].apply(lambda s: int((int(s) - 55) / 10 + 1))
    seg = create_tract_segmentation(tab_cd, vol_shape=vol.shape)
    assert seg.shape == (8, 310, 360)


@pytest.mark.parametrize("cls", [TractDataset2D])
def test_dataset(cls: Dataset, data_dir: str = _ROOT_DATA):
    tab = pd.read_csv(os.path.join(_ROOT_DATA, "train.csv"))
    tab[["Case", "Day", "Slice", "image", "image_path", "height", "width"]] = tab["id"].apply(
        lambda x: pd.Series(extract_tract_details(x, data_dir))
    )
    ds = cls(df_data=tab, path_imgs=data_dir)
    assert len(ds) == 18
