import glob
import os
from itertools import chain

import flash
import numpy as np
import pandas as pd
import pytest
import torch
from flash.image import SemanticSegmentation

from kaggle_imsegm.data_io import extract_tract_details
from kaggle_imsegm.dataset import TractData
from kaggle_imsegm.model import MixedLoss
from kaggle_imsegm.transform import SemanticSegmentationOutputTransform
from tests import _ROOT_DATA


@pytest.mark.parametrize("losses", ["dice", ("bce", "tversky")])
def test_losses(losses):
    ml = MixedLoss(losses)
    y_pred = torch.rand((5, 3, 64, 64))
    y_true = torch.rand((5, 3, 64, 64))
    ml(y_pred, y_true)


def test_model_train_predict(data_dir: str = _ROOT_DATA):
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
        dataloader_kwargs=dict(batch_size=3, num_workers=2),
    )

    model = SemanticSegmentation(
        backbone="efficientnet-b0",
        head="unetplusplus",
        pretrained=True,
        learning_rate=2e-3,
        loss_fn=MixedLoss("bce"),
        num_classes=3,
        multi_label=True,
        output_transform=SemanticSegmentationOutputTransform(),
    )

    trainer = flash.Trainer(
        max_epochs=2,
        gpus=torch.cuda.device_count(),
        precision=16 if torch.cuda.is_available() else 32,
        accumulate_grad_batches=2,
    )

    # Train the model
    trainer.finetune(model, datamodule=dm, strategy="no_freeze")

    preds = trainer.predict(model, datamodule=dm)  # , output="probabilities"
    preds = list(chain(*preds))
    assert len(preds) == 18
