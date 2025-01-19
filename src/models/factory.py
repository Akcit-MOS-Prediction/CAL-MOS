from models.base_models import (
    BaseModel,
    CalMOSDynamicModel,
    CalMOSEmbeddingModel,
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
    if model_type.lower() == "dynamic":
        return CalMOSDynamicModel(**kwargs)
    elif model_type.lower() == "embedding":
        return CalMOSEmbeddingModel(**kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be 'dynamic' or 'embedding'.")