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

def passa_nada(waveform: np.ndarray):
    return waveform

def apply_50pct(waveform: np.ndarray , augmentation_type , **args): 
    slice_audio = waveform.shape[-1]
    slice_point = slice_audio // 2
    audio_50pct = waveform[..., :slice_point]
    other_half = waveform[..., slice_point:]
    
    audio_augmented = criar_factory_augmentation(augmentation_type , audio_50pct, **args)
    return np.concatenate((audio_augmented , other_half), axis=-1)

def apply_randn_prct(waveform: np.ndarray , augmentation_type: str , **args):
    rand_slice_factor = np.random.uniform(0.1, 0.9)
    audio_size = waveform.shape[-1]
    slice_point = int(audio_size * rand_slice_factor)
    
    audio_rand_slice = waveform[..., :slice_point]
    audio_other_half = waveform[..., slice_point:]
    
    audio_augmented_slice = criar_factory_augmentation(augmentation_type, audio_rand_slice, **args)
    
    return np.concatenate((audio_augmented_slice,audio_other_half), axis=-1)

def criar_factory_augmentation(augmentation_type: str , audio , **args):

    if augmentation_type == "clip":
        return ClipAugmentation(audio , **args)
    elif augmentation_type == "passa_alta":
        return PassaAltaAugmentation(audio , **args)
    elif augmentation_type == "passa_banda":
        return PassaBandaAugmentation(audio , **args)
    elif augmentation_type == "passa_baixa":
        return PassaBaixaAugmentation(audio , **args)
    elif augmentation_type == "passa_nada":
        return passa_nada(audio)
    else:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}")