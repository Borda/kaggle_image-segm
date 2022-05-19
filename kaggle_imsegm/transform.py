from dataclasses import dataclass
from typing import Any, Callable, List, Tuple, Union

import albumentations as alb

import numpy as np
from albumentations import Compose
from albumentations.pytorch import ToTensorV2

from flash.core.data.io.input_transform import InputTransform
from flash.core.data.io.output_transform import OutputTransform
from flash.image.segmentation.input_transform import prepare_target, remove_extra_dimensions
from torch import nn, Tensor

COLOR_MEAN: float = 0.349977
COLOR_STD: float = 0.215829


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


DEFAULT_TRANSFORM = FlashAlbumentationsAdapter(
    [alb.Resize(224, 224), alb.Normalize(mean=COLOR_MEAN, std=COLOR_STD, max_pixel_value=255)]
)


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


def default_uncollate(batch: Any) -> List[Any]:
    """This function is used to uncollate a batch into samples. The following conditions are used.

    >>> import torch
    >>> from pprint import pprint
    >>> batch = {"input": torch.zeros([5, 3, 224, 224]), "target": torch.zeros([5, 3, 224, 224]),
    ...          "metadata": {
    ...             'size': [torch.tensor([266, 266, 266, 266, 266]), torch.tensor([266, 266, 266, 266, 266])],
    ...             'height': torch.tensor([266, 266, 266, 266, 266]),
    ...             'width': torch.tensor([266, 266, 266, 266, 266])
    ... }}
    >>> bbatch = default_uncollate(batch)
    >>> len(bbatch)
    5
    >>> print(bbatch[0].keys())
    dict_keys(['input', 'target', 'metadata'])
    >>> print(bbatch[0]["input"].size(), bbatch[0]["target"].size())
    torch.Size([3, 224, 224]) torch.Size([3, 224, 224])
    >>> pprint(bbatch[0]["metadata"])
    {'height': tensor(266),
     'size': (tensor(266), tensor(266)),
     'width': tensor(266)}
    """
    if isinstance(batch, dict):
        elements = [default_uncollate(element) for element in batch.values()]
        return [dict(zip(batch.keys(), element)) for element in zip(*elements)]
    if isinstance(batch, (list, tuple)):
        return list(zip(*batch))
    return list(batch)


class SemanticSegmentationOutputTransform(OutputTransform):
    def per_sample_transform(self, sample: Any) -> Any:
        resize = alb.Resize(*[s.item() for s in sample["metadata"]["size"]])
        sample["input"] = sample["input"].numpy()
        if sample["input"].ndim == 3:
            sample["input"] = np.rollaxis(sample["input"], 0, 3)
        sample["input"] = resize(image=sample["input"])["image"]
        sample["preds"] = [resize(image=pred)["image"] for pred in sample["preds"].numpy()]
        return super().per_sample_transform(sample)

    @staticmethod
    def uncollate(batch: Any) -> Any:
        return default_uncollate(batch)
