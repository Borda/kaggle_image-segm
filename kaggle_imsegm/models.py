import segmentation_models_pytorch as smp


def create_loss(mode: str = "multiclass"):
    return {
        "dice": smp.losses.DiceLoss(mode=mode),
        "focal": smp.losses.FocalLoss(mode=mode),
        "jaccard": smp.losses.JaccardLoss(mode=mode),
        "lovasz": smp.losses.LovaszLoss(mode=mode),
        "tversky": smp.losses.TverskyLoss(mode=mode),
    }
