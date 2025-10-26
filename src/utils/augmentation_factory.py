import numpy as np 
#métodos aqui 

def ClipAugmentation(waveform: np.ndarray, threshold: float) -> np.ndarray:
    return np.clip(waveform, -threshold, threshold)

# Continua aqui 

def criar_factory_augmentation(augmentation_type: str):

    if augmentation_type == "clip":
        return ClipAugmentation()
    elif augmentation_type == "passa_alta":
        return PassaAltaAugmentation()
    elif augmentation_type == "passa_banda":
        return PassaBandaAugmentation()
    elif augmentation_type == "passa_baixar":
        return PassaBaixarAugmentation()
    else:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}")