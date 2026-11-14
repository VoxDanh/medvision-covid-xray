import torchvision.models as tv_models

from src.models.modelTemplate import build_model


def create_model(config):
    """
    Create a model from YAML configuration.

    Responsibilities:
    1. Read model configuration.
    2. Resolve the torchvision constructor.
    3. Instantiate the model.
    4. Delegate model customization to modelTemplate.
    5. Apply layer-wise freezing.
    6. Return the final model.
    """

    model_config = config["model"]

    model_name = model_config["name"]
    pretrained = model_config.get("pretrained", True)

    # --------------------------------------------------
    # Resolve model constructor dynamically
    # --------------------------------------------------

    torchvision_name = model_name.lower()

    if not hasattr(tv_models, torchvision_name):
        raise ValueError(
            f"Model '{model_name}' không tồn tại trong "
            f"torchvision.models với tên '{torchvision_name}'."
        )

    model_constructor = getattr(
        tv_models,
        torchvision_name
    )

    # --------------------------------------------------
    # Instantiate model
    # --------------------------------------------------

    weights = "DEFAULT" if pretrained else None

    model = model_constructor(
        weights=weights
    )

    # --------------------------------------------------
    # Apply YAML-driven customization
    # --------------------------------------------------

    model = build_model(
        model,
        config
    )

    if model is None:
        raise RuntimeError(
            f"Không thể tạo model '{model_name}': "
            "build_model() trả về None."
        )

    # --------------------------------------------------
    # Apply layer-wise freezing
    # --------------------------------------------------

    freezing_config = model_config.get("freezing", {})

    if freezing_config.get("enabled", False):

        freeze_layers = freezing_config.get(
            "freeze_layers", []
        )

        if not freeze_layers:
            raise ValueError(
                "Freezing is enabled but freeze_layers is empty."
            )

        module_map = dict(model.named_modules())

        for layer_name in freeze_layers:

            if layer_name not in module_map:
                raise ValueError(
                    f"Layer '{layer_name}' không tồn tại trong "
                    f"model '{model_name}'."
                )

            module = module_map[layer_name]

            for parameter in module.parameters():
                parameter.requires_grad = False

    return model
