from models.base_models import (
    BaseModel,
    ReLuKANBaseModel,
    CalMOSDynamicModel,
    CalMOSDynamicKANModel,
    CalMOSDynamicMelSpecModel,
    CalMOSDynamicMelSpecKANModel,
    CalMOSAllLayersEmbeddingModel,
    CalMOSOneLayerEmbeddingModel,
)

from typing import Union

ModelLike = Union[BaseModel, ReLuKANBaseModel]


def create_model(
    model_type: str,
    **kwargs
) -> ModelLike:
    """
    Factory function to create either a SERDynamicModel or a SEREmbeddingModel.
    model_type: "dynamic" or "embedding"
    kwargs: parameters to be passed to the model constructors
    """
    if model_type.lower() == "dynamic":
        return CalMOSDynamicModel(**kwargs)
    if model_type.lower() == "dynamic_kan":
        return CalMOSDynamicKANModel(**kwargs)
    if model_type.lower() == "dynamic_melspec":
        return CalMOSDynamicMelSpecModel(**kwargs)
    if model_type.lower() == "dynamic_kan_melspec":
        return CalMOSDynamicMelSpecKANModel(**kwargs)
    elif model_type.lower() == "all_layers_embedding":
        return CalMOSAllLayersEmbeddingModel(**kwargs)
    elif model_type.lower() == "one_layer_embedding":
        return CalMOSOneLayerEmbeddingModel(**kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be 'dynamic' or 'all_layers_embedding' or 'one_layer_embedding'.")