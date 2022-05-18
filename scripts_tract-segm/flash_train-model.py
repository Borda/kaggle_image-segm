import os

from typing import Optional

import fire

import flash

from flash.image import SemanticSegmentation, SemanticSegmentationData
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import WandbLogger

from kaggle_imsegm.augment import TractFlashSegmentationTransform
from kaggle_imsegm.models import create_loss


def main(
    dataset_flash: str = "/home/jirka/Datasets/tract-image-segmentation-flash",
    checkpoints_dir: str = "/home/jirka/Workspace/checkpoints_tract-image-segm-flash",
    batch_size: int = 24,
    num_workers: int = 12,
    model_backbone: str = "se_resnext50_32x4d",
    model_head: str = "deeplabv3plus",
    model_pretrained: bool = False,
    image_size: int = 224,
    loss: Optional[str] = None,
    optimizer: str = "AdamW",
    lr_scheduler: Optional[str] = None,
    learning_rate: float = 5e-3,
    max_epochs: int = 20,
    gpus: int = 1,
    accumulate_grad_batches: int = 1,
    early_stopping: Optional[float] = None,
    swa: Optional[float] = None,
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
        train_transform=TractFlashSegmentationTransform,
        val_transform=TractFlashSegmentationTransform,
        predict_transform=TractFlashSegmentationTransform,
        transform_kwargs=dict(image_size=(image_size, image_size)),
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
        lr_scheduler=lr_scheduler,
        loss_fn=create_loss().get(loss),
        num_classes=datamodule.num_classes,
    )

    logger = WandbLogger(project="Flash_tract-image-segmentation")
    log_id = str(logger.experiment.id)
    monitor = "val_jaccardindex"
    cbs = [ModelCheckpoint(dirpath=checkpoints_dir, filename=f"{log_id}", monitor=monitor, mode="max", verbose=True)]
    if early_stopping is not None:
        cbs.append(EarlyStopping(monitor=monitor, min_delta=early_stopping, mode="max", verbose=True))
    if isinstance(swa, float):
        cbs.append(StochasticWeightAveraging(swa_epoch_start=swa))

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
