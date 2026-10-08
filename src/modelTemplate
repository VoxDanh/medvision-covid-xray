# src/models/modelTemplate.py

import torch.nn as nn


def convert_first_conv_to_grayscale(
    old_conv
):
    """
    Convert pretrained RGB Conv2d
    from 3 input channels -> 1 input channel.
    """

    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=old_conv.bias is not None,
        padding_mode=old_conv.padding_mode
    )

    # RGB -> grayscale
    new_conv.weight.data = (
        old_conv.weight.data.mean(
            dim=1,
            keepdim=True
        )
    )

    if old_conv.bias is not None:
        new_conv.bias.data = (
            old_conv.bias.data
        )

    return new_conv


def replace_classifier(
    model,
    classifier_path,
    num_classes
):
    """
    Generic classifier replacement.

    classifier_path:
        Example:
        "fc"
        "classifier"
    """

    parent = model

    parts = classifier_path.split(".")

    for part in parts[:-1]:
        parent = getattr(
            parent,
            part
        )

    last_part = parts[-1]

    old_classifier = getattr(
        parent,
        last_part
    )

    if not isinstance(
        old_classifier,
        nn.Linear
    ):
        raise TypeError(
            f"{classifier_path} is not nn.Linear"
        )

    new_classifier = nn.Linear(
        old_classifier.in_features,
        num_classes
    )

    setattr(
        parent,
        last_part,
        new_classifier
    )

    return model


def build_model(
    model,
    config
):
    """
    Common model post-processing.
    """

    model_config = config["model"]

    num_classes = model_config[
        "num_classes"
    ]

    input_channels = model_config[
        "input_channels"
    ]

    # --------------------------------------------------------
    # First layer
    # --------------------------------------------------------

    first_layer_path = model_config.get(
        "first_layer"
    )

    if (
        input_channels == 1
        and first_layer_path
    ):

        parent = model

        parts = first_layer_path.split(".")

        for part in parts[:-1]:
            parent = getattr(
                parent,
                part
            )

        last_part = parts[-1]

        old_conv = getattr(
            parent,
            last_part
        )

        new_conv = (
            convert_first_conv_to_grayscale(
                old_conv
            )
        )

        setattr(
            parent,
            last_part,
            new_conv
        )

    # --------------------------------------------------------
    # Classifier
    # --------------------------------------------------------

    classifier_path = model_config.get(
        "classifier"
    )

    if classifier_path:

        model = replace_classifier(
            model,
            classifier_path,
            num_classes
        )

    return model
