# src/models/factory.py

import torchvision.models as tv_models

from src.models.modelTemplate import build_model


MODEL_REGISTRY = {

    "ResNet18":
        tv_models.resnet18,

    "ResNet50":
        tv_models.resnet50,

    "DenseNet121":
        tv_models.densenet121,

    "EfficientNetB0":
        tv_models.efficientnet_b0,
}


def create_model(config):

    model_config = config["model"]

    model_name = model_config[
        "name"
    ]

    pretrained = model_config.get(
        "pretrained",
        True
    )

    if model_name not in MODEL_REGISTRY:

        raise ValueError(
            f"Unsupported model: "
            f"{model_name}"
        )

    model_fn = MODEL_REGISTRY[
        model_name
    ]

    # --------------------------------------------------------
    # Create base model
    # --------------------------------------------------------

    if pretrained:

        model = model_fn(
            weights="DEFAULT"
        )

    else:

        model = model_fn(
            weights=None
        )

    # --------------------------------------------------------
    # Apply common template
    # --------------------------------------------------------

    model = build_model(
        model,
        config
    )

    return model
