"""Linear classifier: the simplest model, used to smoke-test the pipeline."""

from torch import nn

from rmlab.registry import MODELS


@MODELS.register("linear")
def linear(in_features: int, num_classes: int) -> nn.Module:
    """One affine layer, logits = W x + b (Understanding Deep Learning, ch. 2 and 5.5)."""
    return nn.Linear(in_features, num_classes)
