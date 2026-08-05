import gc

import pytest

from cvmodellearning.models.classification_model_utils import _DEFAULT_WEIGHTS, make_model
from cvmodellearning.models.registry import (
    CLASSIFIER_HEAD_PATHS,
    ClassificationModelId,
    model_ids,
)
from cvmodellearning.preprocessing.transformations import CLASSIFICATION_TRANSFORM_PROFILES
from cvmodellearning.training.training_utils import (
    classification_parameter_groups,
    classifier_head,
    set_backbone_trainable,
)


def test_classification_registry_keys_remain_aligned():
    registered = set(model_ids("classification"))

    assert registered == {model.value for model in ClassificationModelId}
    assert registered == set(_DEFAULT_WEIGHTS)
    assert registered == set(CLASSIFICATION_TRANSFORM_PROFILES)
    assert registered == set(CLASSIFIER_HEAD_PATHS)
    assert "vgg16" not in registered


@pytest.mark.parametrize("model_name", model_ids("classification"))
def test_every_registered_classifier_factory_exposes_its_declared_head(model_name):
    model, _ = make_model(model_name, "none", num_classes=3)

    assert classifier_head(model, model_name).out_features == 3

    del model
    gc.collect()


def test_vit_supports_head_only_and_staged_unfreezing_with_head_lr():
    model, _ = make_model("vit_b_16", "none", num_classes=3)
    set_backbone_trainable(model, "vit_b_16", False)

    head = classifier_head(model, "vit_b_16")
    assert all(parameter.requires_grad for parameter in head.parameters())
    assert not model.conv_proj.weight.requires_grad

    groups = classification_parameter_groups(
        model,
        "vit_b_16",
        {"learning_rate": 1e-4, "head_learning_rate_multiplier": 10.0},
    )
    assert [group["lr"] for group in groups] == [1e-4, 1e-3]

    set_backbone_trainable(model, "vit_b_16", True)
    assert all(parameter.requires_grad for parameter in model.parameters())
