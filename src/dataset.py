# ============================================================
# src/dataset.py
# DATASET + DATALOADERS
# - COVID vs Normal classification
# - Stratified train / validation / test split
# - Grayscale images
# - Scenario-based augmentation
# - Class weights for CrossEntropyLoss
# - Cache images in system RAM
# ============================================================

from pathlib import Path

import numpy as np
import torch

from PIL import Image
from tqdm.auto import tqdm

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


# ============================================================
# DATASET
# ============================================================

class LungROIDataset(Dataset):

    SUPPORTED_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
    }

    def __init__(
        self,
        root,
        classes,
        image_size=224,
        resize=False,
        cache_in_memory=True,
    ):
        self.root = Path(root)
        self.classes = list(classes)

        self.class_to_idx = {
            class_name: index
            for index, class_name in enumerate(self.classes)
        }

        if not self.root.is_dir():
            raise FileNotFoundError(
                f"Dataset directory does not exist: {self.root}"
            )

        self.samples = []

        # ----------------------------------------------------
        # Discover image files
        # ----------------------------------------------------

        for class_name in self.classes:

            class_dir = self.root / class_name

            if not class_dir.is_dir():
                raise FileNotFoundError(
                    f"Class directory does not exist: {class_dir}"
                )

            image_paths = sorted(
                path
                for path in class_dir.rglob("*")
                if path.is_file()
                and path.suffix.lower() in self.SUPPORTED_EXTENSIONS
            )

            if not image_paths:
                raise ValueError(
                    f"No images found in class directory: {class_dir}"
                )

            for image_path in image_paths:
                self.samples.append(
                    (
                        image_path,
                        self.class_to_idx[class_name],
                    )
                )

        if not self.samples:
            raise ValueError(
                f"No images found in dataset directory: {self.root}"
            )

        self.image_size = image_size
        self.resize = resize
        self.cache_in_memory = cache_in_memory

        # Cache contains grayscale uint8 NumPy arrays.
        self.cached_images = None

        # ----------------------------------------------------
        # Dataset information
        # ----------------------------------------------------

        print(f"Dataset root: {self.root}")
        print(f"Classes: {self.class_to_idx}")

        for class_name, class_idx in self.class_to_idx.items():

            count = sum(
                label == class_idx
                for _, label in self.samples
            )

            print(f"  {class_name}: {count:,} images")

        # ----------------------------------------------------
        # Load all images into system RAM
        # ----------------------------------------------------

        if self.cache_in_memory:

            print(
                f"\nLoading {len(self.samples):,} images into RAM..."
            )

            self.cached_images = []

            try:
                for image_path, _ in tqdm(
                    self.samples,
                    desc="Caching images",
                    unit="image",
                ):

                    with Image.open(image_path) as image:

                        image = image.convert("L")

                        if self.resize:
                            image = image.resize(
                                (
                                    self.image_size,
                                    self.image_size,
                                ),
                                Image.Resampling.BILINEAR,
                            )

                        image_array = np.array(
                            image,
                            dtype=np.uint8,
                            copy=True,
                        )

                    self.cached_images.append(image_array)

            except Exception as exc:

                self.cached_images = None

                raise RuntimeError(
                    f"Failed to cache image: {image_path}"
                ) from exc

            cached_bytes = sum(
                image.nbytes
                for image in self.cached_images
            )

            print(
                f"Successfully cached {len(self.cached_images):,} images."
            )

            print(
                f"Image pixel memory: "
                f"{cached_bytes / (1024 ** 3):.3f} GiB"
            )

    def __len__(self):
        return len(self.samples)

    def load_image(self, image_path):
        """
        Load an image directly from storage.
        Used when cache_in_memory=False.
        """

        with Image.open(image_path) as image:

            image = image.convert("L")

            if self.resize:
                image = image.resize(
                    (
                        self.image_size,
                        self.image_size,
                    ),
                    Image.Resampling.BILINEAR,
                )

            return image.copy()

    def get_pil_image(self, index):
        """
        Retrieve an independent PIL image.

        If caching is enabled, read from RAM.
        Otherwise, read from storage.
        """

        if self.cached_images is not None:

            return Image.fromarray(
                self.cached_images[index],
                mode="L",
            )

        image_path, _ = self.samples[index]

        return self.load_image(image_path)

    def __getitem__(self, index):

        _, label = self.samples[index]

        image = self.get_pil_image(index)

        # Convert PIL image to a grayscale tensor in [0, 1].
        image = transforms.functional.to_tensor(image)

        return image, label


# ============================================================
# NORMALIZATION STATISTICS
# ============================================================

def calculate_mean_std(dataset, indices):
    """
    Calculate grayscale mean and standard deviation using
    only training samples.
    """

    pixel_sum = 0.0
    pixel_squared_sum = 0.0
    pixel_count = 0

    for index in tqdm(
        indices,
        desc="Calculating train mean/std",
        unit="image",
    ):

        image, _ = dataset[index]

        pixel_sum += image.sum().item()
        pixel_squared_sum += image.square().sum().item()
        pixel_count += image.numel()

    if pixel_count == 0:
        raise ValueError(
            "Cannot calculate normalization statistics "
            "from an empty training dataset."
        )

    mean = pixel_sum / pixel_count

    variance = (
        pixel_squared_sum / pixel_count
        - mean ** 2
    )

    std = max(variance, 0.0) ** 0.5

    # Avoid division by zero during normalization.
    std = max(std, 1e-6)

    return float(mean), float(std)


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(labels, num_classes):
    """
    Inverse-frequency weighting.

    weight[c] = total_samples / (num_classes * class_count[c])
    """

    counts = np.bincount(
        np.asarray(labels, dtype=np.int64),
        minlength=num_classes,
    )

    if np.any(counts == 0):
        raise ValueError(
            "At least one class has no training samples: "
            f"{counts.tolist()}"
        )

    total = counts.sum()

    weights = total / (num_classes * counts)

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


# ============================================================
# DATALOADERS
# ============================================================

def create_dataloaders(base_config, scenario_config):

    # --------------------------------------------------------
    # Read configuration
    # --------------------------------------------------------

    data_config = base_config["data"]
    training_config = base_config["training"]

    dataset_root = data_config["root"]
    classes = data_config["classes"]

    num_classes = int(
        data_config["num_classes"]
    )

    image_size = int(
        data_config.get("image_size", 224)
    )

    batch_size = int(
        training_config["batch_size"]
    )

    num_workers = int(
        training_config.get("num_workers", 2)
    )

    split_config = data_config["split"]

    train_ratio = float(split_config["train"])
    val_ratio = float(split_config["val"])
    test_ratio = float(split_config["test"])

    if not np.isclose(
        train_ratio + val_ratio + test_ratio,
        1.0,
    ):
        raise ValueError(
            "Train, validation and test ratios must sum to 1.0."
        )

    if len(classes) != num_classes:
        raise ValueError(
            f"Configured classes ({len(classes)}) do not match "
            f"num_classes ({num_classes})."
        )

    # --------------------------------------------------------
    # Read scenario configuration
    # --------------------------------------------------------

    input_config = scenario_config.get(
        "input",
        {},
    )

    preprocessing_config = scenario_config.get(
        "preprocessing",
        {},
    )

    augmentation_config = scenario_config.get(
        "augmentation",
        {},
    )

    resize = bool(
        preprocessing_config.get("resize", False)
    )

    if input_config.get("grayscale", True):

        input_channels = 1

    else:

        input_channels = int(
            input_config.get("channels", 3)
        )

    if input_channels != 1:
        raise ValueError(
            "This dataset loader currently expects grayscale "
            "images with one input channel."
        )

    # --------------------------------------------------------
    # Load dataset and cache images
    # --------------------------------------------------------

    dataset = LungROIDataset(
        root=dataset_root,
        classes=classes,
        image_size=image_size,
        resize=resize,
        cache_in_memory=True,
    )

    labels = [
        label
        for _, label in dataset.samples
    ]

    indices = np.arange(len(dataset))

    if len(indices) < num_classes * 3:
        raise ValueError(
            "Dataset is too small to create stratified "
            "train/validation/test splits."
        )

    # --------------------------------------------------------
    # Stratified train / validation / test split
    # --------------------------------------------------------

    stratify_labels = (
        labels
        if data_config.get("stratify", True)
        else None
    )

    train_indices, remaining_indices = train_test_split(
        indices,
        train_size=train_ratio,
        random_state=int(base_config["project"]["seed"]),
        stratify=stratify_labels,
    )

    remaining_labels = [
        labels[index]
        for index in remaining_indices
    ]

    relative_test_ratio = test_ratio / (
        val_ratio + test_ratio
    )

    stratify_remaining = (
        remaining_labels
        if data_config.get("stratify", True)
        else None
    )

    val_indices, test_indices = train_test_split(
        remaining_indices,
        test_size=relative_test_ratio,
        random_state=int(base_config["project"]["seed"]),
        stratify=stratify_remaining,
    )

    train_indices = list(train_indices)
    val_indices = list(val_indices)
    test_indices = list(test_indices)

    print("\nDataset split:")
    print(f"  Train: {len(train_indices):,}")
    print(f"  Val:   {len(val_indices):,}")
    print(f"  Test:  {len(test_indices):,}")

    # --------------------------------------------------------
    # Calculate normalization using training samples only
    # --------------------------------------------------------

    normalize_config = preprocessing_config.get(
        "normalize",
        {},
    )

    normalize_enabled = bool(
        normalize_config.get("enabled", False)
    )

    mean = normalize_config.get("mean")
    std = normalize_config.get("std")

    if normalize_enabled:

        calculate_from_train = bool(
            normalize_config.get(
                "calculate_from_train",
                False,
            )
        )

        if (
            mean is None
            or std is None
        ) and calculate_from_train:

            mean, std = calculate_mean_std(
                dataset,
                train_indices,
            )

        elif mean is None or std is None:

            raise ValueError(
                "Normalization mean/std are missing. "
                "Set calculate_from_train=true or provide "
                "explicit mean and std values."
            )

        mean = float(mean)
        std = float(std)

        if std <= 0:
            raise ValueError(
                "Normalization std must be greater than zero."
            )

        print(
            f"Normalization: mean={mean:.6f}, std={std:.6f}"
        )

    # --------------------------------------------------------
    # Build transforms
    # --------------------------------------------------------

    train_transform_list = []
    eval_transform_list = []

    train_transform_list.append(
        transforms.ToTensor()
    )

    eval_transform_list.append(
        transforms.ToTensor()
    )

    if normalize_enabled:

        normalize_transform = transforms.Normalize(
            mean=[mean],
            std=[std],
        )

        train_transform_list.append(
            normalize_transform
        )

        eval_transform_list.append(
            normalize_transform
        )

    # --------------------------------------------------------
    # Training augmentation only
    # --------------------------------------------------------

    if augmentation_config.get("enabled", False):

        flip_config = augmentation_config.get(
            "horizontal_flip",
            {},
        )

        if flip_config.get("enabled", False):

            train_transform_list.append(
                transforms.RandomHorizontalFlip(
                    p=float(
                        flip_config.get("probability", 0.5)
                    )
                )
            )

        rotation_config = augmentation_config.get(
            "rotation",
            {},
        )

        if rotation_config.get("enabled", False):

            train_transform_list.append(
                transforms.RandomRotation(
                    degrees=float(
                        rotation_config.get("degrees", 10)
                    )
                )
            )

        affine_config = augmentation_config.get(
            "affine",
            {},
        )

        if affine_config.get("enabled", False):

            translate = float(
                affine_config.get("translate", 0.05)
            )

            affine_scale_config = affine_config.get(
                "scale",
                {},
            )

            scale_min = float(
                affine_scale_config.get("min", 0.95)
            )

            scale_max = float(
                affine_scale_config.get("max", 1.05)
            )

            train_transform_list.append(
                transforms.RandomAffine(
                    degrees=0,
                    translate=(translate, translate),
                    scale=(scale_min, scale_max),
                )
            )

    train_transform = transforms.Compose(
        train_transform_list
    )

    eval_transform = transforms.Compose(
        eval_transform_list
    )

    # --------------------------------------------------------
    # Dataset views
    # --------------------------------------------------------

    class TransformedSubset(Dataset):

        def __init__(
            self,
            parent_dataset,
            subset_indices,
            transform,
        ):
            self.parent_dataset = parent_dataset
            self.indices = list(subset_indices)
            self.transform = transform

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, index):

            parent_index = self.indices[index]

            _, label = self.parent_dataset.samples[
                parent_index
            ]

            # Read from RAM cache instead of Google Drive.
            image = self.parent_dataset.get_pil_image(
                parent_index
            )

            if self.transform is not None:
                image = self.transform(image)

            return image, label

    train_dataset = TransformedSubset(
        dataset,
        train_indices,
        train_transform,
    )

    val_dataset = TransformedSubset(
        dataset,
        val_indices,
        eval_transform,
    )

    test_dataset = TransformedSubset(
        dataset,
        test_indices,
        eval_transform,
    )

    # --------------------------------------------------------
    # Class weights: training split only
    # --------------------------------------------------------

    loss_config = base_config.get(
        "loss",
        {},
    )

    class_weights_config = loss_config.get(
        "class_weights",
        {},
    )

    class_weights_enabled = bool(
        class_weights_config.get("enabled", False)
    )

    if class_weights_enabled:

        strategy = class_weights_config.get(
            "strategy",
            "inverse_frequency",
        )

        if strategy != "inverse_frequency":
            raise ValueError(
                f"Unsupported class-weight strategy: {strategy}"
            )

        train_labels = [
            labels[index]
            for index in train_indices
        ]

        class_weights = calculate_class_weights(
            train_labels,
            num_classes,
        )

    else:

        class_weights = None

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print("\nDataLoaders created successfully.")
    print(f"  Batch size: {batch_size}")
    print(f"  Workers: {num_workers}")
    print(f"  RAM cache: {dataset.cached_images is not None}")
    print(f"  Train batches: {len(train_loader):,}")
    print(f"  Val batches: {len(val_loader):,}")
    print(f"  Test batches: {len(test_loader):,}")

    return (
        train_loader,
        val_loader,
        test_loader,
        class_weights,
    )
