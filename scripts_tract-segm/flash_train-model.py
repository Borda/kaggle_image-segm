import os

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import albumentations as alb
import fire

import flash
from flash.core.data.io.input_transform import InputTransform
from flash.image import SemanticSegmentation, SemanticSegmentationData
from flash.image.segmentation.input_transform import prepare_target, remove_extra_dimensions

from kaggle_imsegm.augment import FlashAlbumentationsAdapter
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


@dataclass
class SemanticSegmentationInputTransform(InputTransform):

    image_size: Tuple[int, int] = (256, 256)

    def train_per_sample_transform(self) -> Callable:
        return FlashAlbumentationsAdapter(
            [
                alb.Resize(*self.image_size),
                alb.VerticalFlip(p=0.5),
                alb.HorizontalFlip(p=0.5),
                alb.RandomRotate90(p=0.5),
                alb.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.05, rotate_limit=15, p=0.5),
                alb.GaussNoise(var_limit=(0.00, 0.03), mean=0, per_channel=False, p=1.0),
                # alb.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
                # alb.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
                # alb.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            ]
        )

    def per_sample_transform(self) -> Callable:
        return FlashAlbumentationsAdapter([alb.Resize(*self.image_size)])

    def predict_input_per_sample_transform(self) -> Callable:
        return FlashAlbumentationsAdapter([alb.Resize(*self.image_size)])

    def target_per_batch_transform(self) -> Callable:
        return prepare_target

    def predict_per_batch_transform(self) -> Callable:
        return remove_extra_dimensions

    def serve_per_batch_transform(self) -> Callable:
        return remove_extra_dimensions


def main(
    dataset_flash: str = "/home/jirka/Datasets/tract-image-segmentation-flash",
    checkpoints_dir: str = "/home/jirka/Workspace/checkpoints_tract-image-segm-flash",
    batch_size: int = 72,
    num_workers: int = 24,
    model_backbone: str = "resnext50_32x4d",
    model_head: str = "deeplabv3",
    model_pretrained: bool = False,
    optimizer: str = "AdamW",
    learning_rate: float = 7e-3,
    max_epochs: int = 50,
    gpus: int = 1,
    accumulate_grad_batches: int = 1,
    early_stopping: Optional[float] = None,
) -> None:
    dir_flash_image = os.path.join(dataset_flash, "images")
    assert os.path.isdir(dir_flash_image)
    dir_flash_segm = os.path.join(dataset_flash, "segms")
    assert os.path.isdir(dir_flash_segm)
    os.makedirs(checkpoints_dir, exist_ok=True)

    datamodule = SemanticSegmentationData.from_folders(
        train_folder=os.path.join(dir_flash_image, "train"),
        train_target_folder=os.path.join(dir_flash_segm, "train"),
        val_folder=os.path.join(dir_flash_image, "val"),
        val_target_folder=os.path.join(dir_flash_segm, "val"),
        # val_split=0.1,
        train_transform=SemanticSegmentationInputTransform,
        val_transform=SemanticSegmentationInputTransform,
        predict_transform=SemanticSegmentationInputTransform,
        transform_kwargs=dict(image_size=(256, 256)),
        num_classes=4,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = SemanticSegmentation(
        backbone=model_backbone,
        head=model_head,
        pretrained=model_pretrained,
        optimizer=optimizer,
        learning_rate=learning_rate,
        num_classes=datamodule.num_classes,
    )

    logger = WandbLogger(project="Flash_tract-image-segmentation")
    log_id = str(logger.experiment.id)
    monitor = "val_cross_entropy"
    cbs = [ModelCheckpoint(dirpath=checkpoints_dir, filename=f"{log_id}", monitor=monitor, mode="max", verbose=True)]
    if early_stopping is not None:
        cbs.append(EarlyStopping(monitor=monitor, min_delta=early_stopping, mode="max", verbose=True))
    trainer = flash.Trainer(
        callbacks=cbs,
        max_epochs=max_epochs,
        # precision="bf16",
        gpus=gpus,
        accelerator="ddp" if gpus > 1 else None,
        benchmark=True,
        logger=logger,
        accumulate_grad_batches=accumulate_grad_batches,
        # limit_train_batches=0.25,
        # limit_val_batches=0.25,
    )

    # Train the model
    trainer.finetune(model, datamodule=datamodule, strategy="no_freeze")

    # Save the model!
    checkpoint_name = f"tract-segm-{log_id}_{model_head}-{model_backbone}.pt"
    trainer.save_checkpoint(os.path.join(checkpoints_dir, checkpoint_name))


if __name__ == "__main__":
    fire.Fire(main)
