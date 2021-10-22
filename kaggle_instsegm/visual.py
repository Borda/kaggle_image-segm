import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from kaggle_instsegm.mask import create_mask


def show_image_annot(img_name: str, df_train: pd.DataFrame, img_folder: str):
    print(img_name)
    df_img = df_train[df_train["id"] == img_name]
    path_img = os.path.join(img_folder, f"{img_name}.png")
    img = plt.imread(path_img)
    mask = create_mask(df_img)
    fig, axarr = plt.subplots(ncols=3, figsize=(14, 6))
    axarr[0].imshow(img)
    axarr[1].imshow(img)
    axarr[1].contour(mask, levels=np.unique(mask).tolist(), cmap="inferno", linewidths=0.5)
    axarr[2].imshow(mask, cmap="inferno", interpolation="antialiased")
    return fig
