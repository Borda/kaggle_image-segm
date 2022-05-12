import segmentation_models_pytorch as smp

LOSS_FNS = {
    "dice": smp.losses.DiceLoss(mode="multiclass"),
    "focal": smp.losses.FocalLoss(mode="multiclass"),
    "jaccard": smp.losses.JaccardLoss(mode="multiclass"),
    "lovasz": smp.losses.LovaszLoss(mode="multiclass"),
    "tversky": smp.losses.TverskyLoss(mode="multiclass"),
}
