import pytest
import torch

from kaggle_imsegm.model import MixedLoss


@pytest.mark.parametrize("losses", ["dice", ("bce", "tversky")])
def test_losses(losses):
    ml = MixedLoss(losses)
    y_pred = torch.rand((5, 3, 64, 64))
    y_true = torch.rand((5, 3, 64, 64))
    ml(y_pred, y_true)
