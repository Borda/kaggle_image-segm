from typing import Any

import numpy as np
from albumentations import Compose
from albumentations.pytorch import ToTensorV2
from torch import nn, Tensor


class FlashAlbumentationsAdapter(nn.Module):
    """This is temporary class for Flash new release."""

    # mapping from albumentations to Flash
    TRANSFORM_INPUT_MAPPING = {"image": "input", "mask": "target"}

    def __init__(self, transform, mapping: dict = None, image_key: str = "input"):
        super().__init__()
        if not isinstance(transform, (list, tuple)):
            transform = [transform]
        self.transform = Compose(list(transform) + [ToTensorV2()])
        self._img_key = image_key
        if not mapping:
            mapping = self.TRANSFORM_INPUT_MAPPING
        self._mapping_rev = mapping
        self._mapping = {v: k for k, v in mapping.items()}

    @staticmethod
    def _image_transform(x: Tensor) -> np.ndarray:
        if x.ndim == 3 and x.shape[0] < 4:
            return x.permute(1, 2, 0).numpy()
        return x.numpy()

    def forward(self, x: Any) -> Any:
        if isinstance(x, dict):
            x_ = {self._mapping.get(k, k): x[k].numpy() for k in self._mapping if k in x and k != self._img_key}
            if self._img_key in self._mapping and self._img_key in x:
                x_.update({self._mapping[self._img_key]: self._image_transform(x[self._img_key])})
        else:
            x_ = {"image": self._image_transform(x)}
        x_ = self.transform(**x_)
        if isinstance(x, dict):
            x.update({self._mapping_rev.get(k, k): x_[k] for k in self._mapping_rev if k in x_})
        else:
            x = x_["image"]
        return x
