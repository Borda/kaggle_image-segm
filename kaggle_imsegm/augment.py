from dataclasses import dataclass
from typing import Any, Callable, Tuple, Union

import albumentations as alb

import numpy as np
from albumentations import Compose
from albumentations.pytorch import ToTensorV2

from flash.core.data.io.input_transform import InputTransform
from flash.image.segmentation.input_transform import prepare_target, remove_extra_dimensions
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

    @staticmethod
    def _to_numpy(t: Union[Tensor, np.ndarray]) -> np.ndarray:
        return t.numpy() if isinstance(t, Tensor) else t

    def forward(self, x: Any) -> Any:
        if isinstance(x, dict):
            x_ = {self._mapping.get(k, k): self._to_numpy(x[k]) for k in self._mapping if k in x and k != self._img_key}
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


@dataclass
class TractFlashSegmentationTransform(InputTransform):
    # https://albumentations.ai/docs/examples/pytorch_semantic_segmentation

    image_size: Tuple[int, int] = (224, 224)
    color_mean: float = 0.349977
    color_std: float = 0.215829

    def train_per_sample_transform(self) -> Callable:
        return FlashAlbumentationsAdapter(
            [
                alb.Resize(*self.image_size),
                alb.VerticalFlip(p=0.5),
                alb.HorizontalFlip(p=0.5),
                alb.RandomRotate90(p=0.5),
                alb.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.03, rotate_limit=5, p=1.0),
                alb.GaussNoise(var_limit=(0.001, 0.005), mean=0, per_channel=False, p=1.0),
                alb.OneOf(
                    [
                        alb.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                        alb.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
                    ],
                    p=0.25,
                ),
                alb.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
                alb.Normalize(mean=[self.color_mean] * 3, std=[self.color_std] * 3, max_pixel_value=1.0),
            ]
        )

    def per_sample_transform(self) -> Callable:
        return FlashAlbumentationsAdapter(
            [
                alb.Resize(*self.image_size),
                alb.Normalize(mean=[self.color_mean] * 3, std=[self.color_std] * 3, max_pixel_value=1.0),
            ]
        )

    def predict_input_per_sample_transform(self) -> Callable:
        return FlashAlbumentationsAdapter(
            [
                alb.Resize(*self.image_size),
                alb.Normalize(mean=[self.color_mean] * 3, std=[self.color_std] * 3, max_pixel_value=1.0),
            ]
        )

    def target_per_batch_transform(self) -> Callable:
        return prepare_target

    def predict_per_batch_transform(self) -> Callable:
        return remove_extra_dimensions

    def serve_per_batch_transform(self) -> Callable:
        return remove_extra_dimensions
