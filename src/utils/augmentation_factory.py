import numpy as np 
from scipy.signal import butter, lfilter    


#TODO more methods latter 

def ClipAugmentation(waveform: np.ndarray, threshold: float) -> np.ndarray:
    return np.clip(waveform, -threshold, threshold)

def PassaAltaAugmentation(waveform: np.ndarray, cutoff_freq: float, sample_rate: int) -> np.ndarray:
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(1, normal_cutoff, btype='high', analog=False)
    return lfilter(b, a, waveform)

def PassaBaixaAugmentation(waveform: np.ndarray, cutoff_freq: float, sample_rate: int) -> np.ndarray:
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(1, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, waveform)

def PassaBandaAugmentation(waveform: np.ndarray, lowcut: float, highcut: float, sample_rate: int) -> np.ndarray:
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(1, [low, high], btype='band')
    return lfilter(b, a, waveform)


def criar_factory_augmentation(augmentation_type: str , audio , **args):

    if augmentation_type == "clip":
        return ClipAugmentation(audio , **args)
    elif augmentation_type == "passa_alta":
        return PassaAltaAugmentation(audio , **args)
    elif augmentation_type == "passa_banda":
        return PassaBandaAugmentation(audio , **args)
    elif augmentation_type == "passa_baixa":
        return PassaBaixaAugmentation(audio , **args)
    else:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}")