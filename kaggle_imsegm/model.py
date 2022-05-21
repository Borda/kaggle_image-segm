from typing import Sequence

import numpy as np
import segmentation_models_pytorch as smp
from torch import Tensor


class MixedLoss:
    def __init__(
        self, name: Sequence[str], mode: str = "multilabel", smooth: float = 0.01, ratio: Sequence[float] = None
    ) -> None:
        losses = {
            "bce": smp.losses.SoftBCEWithLogitsLoss(smooth_factor=smooth),
            "dice": smp.losses.DiceLoss(mode=mode, smooth=smooth),
            "focal": smp.losses.FocalLoss(mode=mode),
            "jaccard": smp.losses.JaccardLoss(mode=mode, smooth=smooth),
            "lovasz": smp.losses.LovaszLoss(mode=mode),
            "tversky": smp.losses.TverskyLoss(mode=mode),
        }
        self.names = [name] if isinstance(name, str) else name
        assert all(n in losses for n in self.names)
        self.losses = [losses[n] for n in self.names]
        if not ratio:
            ratio = [1.0 / len(self.names)] * len(self.names)
        self.ratio = np.array(ratio, dtype=float) / sum(ratio)

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return sum(r * loss(y_pred, y_true) for r, loss in zip(self.ratio, self.losses))
