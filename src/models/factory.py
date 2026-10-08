# src/models/factory.py

import torchvision.models as tv_models

from src.models.modelTemplate import build_model


def create_model(config):

    model_config = config["model"]

    model_name = model_config["name"]

    pretrained = model_config.get(
        "pretrained",
        True
    )

    # --------------------------------------------------
    # Chuyển tên YAML -> tên function torchvision
    # --------------------------------------------------

    torchvision_name = (
        model_name[0].lower()
        + model_name[1:]
    )

    # Ví dụ:
    #
    # ResNet18
    # -> resNet18   ❌
    #
    # DenseNet121
    # -> denseNet121 ❌
    #
    # nên cần mapping convention hoặc alias.
