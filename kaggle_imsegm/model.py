import segmentation_models_pytorch as smp


def create_loss(mode: str = "multiclass", smooth: float = 0.01):
    return {
        "bce": smp.losses.SoftBCEWithLogitsLoss(smooth_factor=smooth),
        "dice": smp.losses.DiceLoss(mode=mode, smooth=smooth),
        "focal": smp.losses.FocalLoss(mode=mode),
        "jaccard": smp.losses.JaccardLoss(mode=mode, smooth=smooth),
        "lovasz": smp.losses.LovaszLoss(mode=mode),
        "tversky": smp.losses.TverskyLoss(mode=mode),
    }
