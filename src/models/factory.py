from models.base_models import (
    BaseModel,
    CalMOSDynamicModel,
    CalMOSAllLayersEmbeddingModel,
    CalMOSMelSpecModel,
    CalMOSOneLayerEmbeddingModel,
)


def create_model(
    model_type: str,
    **kwargs
) -> BaseModel:
    """
    Factory function to create either a SERDynamicModel or a SEREmbeddingModel.
    model_type: "dynamic" or "embedding"
    kwargs: parameters to be passed to the model constructors
    """
    kwargs.pop("model_name", None)
    if model_type.lower() == "dynamic":
        return CalMOSDynamicModel(**kwargs)
    elif model_type == "melspec":
        return CalMOSMelSpecModel(**kwargs) 
    elif model_type.lower() == "all_layers_embedding":
        return CalMOSAllLayersEmbeddingModel(**kwargs)
    elif model_type.lower() == "one_layer_embedding":
        return CalMOSOneLayerEmbeddingModel(**kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be 'dynamic' or 'all_layers_embedding' or 'one_layer_embedding'.")