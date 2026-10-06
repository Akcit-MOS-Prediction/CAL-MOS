def resolve_layer_index(num_layers: int, specific_layer_idx: int = -1, last_layer: bool = False) -> int:
    """Resolve a hidden-state index, honoring last_layer when requested."""
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    if last_layer:
        return num_layers - 1
    index = specific_layer_idx
    if index < 0:
        index += num_layers
    if index < 0 or index >= num_layers:
        raise IndexError(
            f"specific_layer_idx={specific_layer_idx} is invalid for "
            f"num_layers={num_layers}"
        )
    return index
