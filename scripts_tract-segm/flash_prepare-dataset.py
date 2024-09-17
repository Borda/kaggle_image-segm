import glob
import os
from typing import List

import fire as fire
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from kaggle_imsegm.data_io import extract_tract_details, preprocess_tract_scan


def _chose_sfolder(df_: pd.DataFrame, val_cases_days: List[str]) -> str:
    case, day = df_.iloc[0][["Case", "Day"]]
    case_day = f"case{case}_day{day}"
    return "val" if case_day in val_cases_days else "train"


def _color_means(img_path, thr: float = 0.05):
    img = plt.imread(img_path)
    if np.max(img) > 1.5:
        img = img / 255.0
    clr_mean = np.mean(img[img >= thr])
    clr_std = np.std(img)
    return clr_mean, clr_std


def main(
    dataset_folder: str = "/home/jirka/Datasets/uw-madison-gi-tract-image-segmentation",
    dataset_flash: str = "/home/jirka/Datasets/tract-image-segmentation-flash",
    val_split: float = 0.1,
    n_jobs: int = 6,
) -> None:
    assert 0.0 <= val_split <= 1.0
    assert n_jobs >= 1
    df_train = pd.read_csv(os.path.join(dataset_folder, "train.csv"))
    print(df_train.head())

    dir_flash_image = os.path.join(dataset_flash, "images")
    dir_flash_segm = os.path.join(dataset_flash, "segms")

    # pprint(extract_tract_details(df_train["id"].iloc[0], dataset_folder))
    df_train[["Case", "Day", "Slice", "image", "image_path", "height", "width"]] = df_train["id"].apply(
        lambda x: pd.Series(extract_tract_details(x, dataset_folder))
    )
    df_train["Case_Day"] = [f"case{r['Case']}_day{r['Day']}" for _, r in df_train.iterrows()]
    print(df_train.head())
    labels = sorted(df_train["class"].unique())
    print(labels)

    cases_days = list(df_train["Case_Day"].unique())
    nb_val = int(val_split * len(cases_days))
    val_cases_days = cases_days[-nb_val:]
    print(f"all case-day: {len(cases_days)}")
    print(f"val case-day: {len(val_cases_days)}")

    for rdir in (dir_flash_image, dir_flash_segm):
        for sdir in ("train", "val"):
            os.makedirs(os.path.join(rdir, sdir), exist_ok=True)

    df_train["Case_Day"] = [f"case{r['Case']}_day{r['Day']}" for _, r in df_train.iterrows()]
    _ = Parallel(n_jobs=n_jobs)(
        delayed(preprocess_tract_scan)(
            dfg,
            sfolder=_chose_sfolder(dfg, val_cases_days),
            dir_data=os.path.join(dataset_folder, "train"),
            dir_imgs=dir_flash_image,
            dir_segm=dir_flash_segm,
            labels=labels,
        )
        for _, dfg in tqdm(df_train.groupby("Case_Day"))
    )

    images = glob.glob(os.path.join(dataset_flash, "images", "**", "*.png"), recursive=True)
    # images += glob.glob(os.path.join(PATH_DATASET, "test_images", "*", "*.jpg"))
    clr_mean_std = Parallel(n_jobs=os.cpu_count())(delayed(_color_means)(fn) for fn in tqdm(images))

    img_color_mean = pd.DataFrame([c[0] for c in clr_mean_std]).describe()
    print(img_color_mean.T)
    img_color_std = pd.DataFrame([c[1] for c in clr_mean_std]).describe()
    print(img_color_std.T)

    img_color_mean = list(img_color_mean.T["mean"])
    img_color_std = list(img_color_std.T["mean"])
    print(f"MEAN: {img_color_mean} \n STD: {img_color_std}")


if __name__ == "__main__":
    fire.Fire(main)
