"""Dataset identity shared by loaders, heads and cross-dataset evaluation."""
from collections.abc import Mapping
from math import isfinite
from numbers import Integral, Real

DATASET_IDS = {"bvcc": 1, "singmos": 2, "tmhint": 3, "brspeech": 4}


def block_expansion_enabled(specification):
    if specification is None:
        return False
    if not isinstance(specification, Mapping):
        raise ValueError("dataset_block_expansion must be a mapping")
    return bool(specification.get("enabled", False))


def validate_block_specification(specification):
    if specification.get("num_blocks", 4) != 4:
        raise ValueError("Dataset block expansion requires exactly four blocks")
    mapping = dict(specification.get("dataset_to_id", DATASET_IDS))
    if mapping != DATASET_IDS:
        raise ValueError(f"Dataset mapping must be {DATASET_IDS}, got {mapping}")
    block_expansion_multiplier(specification)
    block_width = specification.get("block_width")
    if block_width is not None and (
        isinstance(block_width, bool) or not isinstance(block_width, Integral) or block_width < 1
    ):
        raise ValueError("dataset_block_expansion.block_width must be a positive integer")
    block_expansion_padding_value(specification)
    return mapping


def block_expansion_multiplier(specification):
    """Return the configured output-width multiplier (legacy default: 4x)."""
    if not isinstance(specification, Mapping):
        raise ValueError("dataset_block_expansion must be a mapping")
    value = specification.get("output_multiplier", 4.0)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("dataset_block_expansion.output_multiplier must be a finite number")
    value = float(value)
    if not isfinite(value) or value < 1.0:
        raise ValueError("dataset_block_expansion.output_multiplier must be finite and >= 1")
    return value


def block_expansion_padding_value(specification):
    value = specification.get("padding_value", 0.0)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("dataset_block_expansion.padding_value must be a finite number")
    value = float(value)
    if not isfinite(value):
        raise ValueError("dataset_block_expansion.padding_value must be finite")
    return value


def validate_dataset_id(dataset_id):
    if isinstance(dataset_id, bool) or not isinstance(dataset_id, Integral) or not 1 <= dataset_id <= 4:
        raise ValueError(f"dataset_id must be an integer from 1 to 4, got {dataset_id!r}")
    return int(dataset_id)


def dataset_id_for_split(config, split):
    specification = config.model.get("dataset_block_expansion", None)
    if not block_expansion_enabled(specification):
        return None
    mapping = validate_block_specification(specification)
    name = split.get("dataset_name", None)
    # Dispatchers already running before deployment attach the dataset as a
    # canonical metadata tag. This keeps their future fine-tunings routable.
    # Explicit split identity takes precedence, including mixed split configs.
    if name is None:
        names = {tag for tag in config.get("tags", []) if isinstance(tag, str) and tag in mapping}
        if len(names) == 1:
            name = names.pop()
    if name not in mapping:
        raise ValueError(f"Dataset block expansion requires dataset_name in {tuple(mapping)} or one unambiguous dataset metadata tag, got {name!r}")
    return mapping[name]
