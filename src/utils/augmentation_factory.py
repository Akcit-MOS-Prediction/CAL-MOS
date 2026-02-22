import numpy as np 
from audiomentations import Compose, AddGaussianSNR, GainTransition, Limiter, HighPassFilter, LowPassFilter, TimeStretch, Shift, Trim, TimeMask, PitchShift, TanhDistortion

## ======= 1. Acoustic Variations Augmentations ======= ##

### ----- Noise Additions ----- ###
def GaussianSNRAugmentation(waveform: np.ndarray, audio_sr: int, min_snr_db: float = 5.0, max_snr_db: float = 40.0, p: float = 0.0) -> np.ndarray:
    augmenter = AddGaussianSNR(
        min_snr_db=min_snr_db,
        max_snr_db=max_snr_db,
        p=p
    )
    return augmenter(waveform, sample_rate=audio_sr)

### ----- Volume and Gain Adjustments ----- ### 
def GainTransitionAugmentation(waveform: np.ndarray, audio_sr: int, min_gain_db: float = -24.0, max_gain_db: float = 6.0, min_duration: float = 0.2, max_duration: float = 6.0, p: float = 0.0) -> np.ndarray:
    augmenter = GainTransition(
        min_gain_db=min_gain_db,
        max_gain_db=max_gain_db,
        min_duration=min_duration,
        max_duration=max_duration,
        p=p
    )
    return augmenter(waveform, sample_rate=audio_sr)

def LimiterAugmentation(waveform: np.ndarray, audio_sr: int, min_threshold_db: float = -24.0, max_threshold_db: float = -2.0, threshold_mode: str = "relative_to_signal_peak", p: float = 0.0) -> np.ndarray:
    augmenter = Limiter(
        min_threshold_db=min_threshold_db,
        max_threshold_db=max_threshold_db,
        threshold_mode=threshold_mode,
        p=p
    )
    return augmenter(waveform, sample_rate=audio_sr)

### ----- Filtering ----- ###
def HighPassFilterAugmentation(waveform: np.ndarray, audio_sr: int, min_cutoff_freq: float = 20.0, max_cutoff_freq: float = 2400.0, zero_phase: bool = False, p: float = 0.0) -> np.ndarray:
    augmenter = HighPassFilter(
        min_cutoff_freq=min_cutoff_freq,
        max_cutoff_freq=max_cutoff_freq,
        zero_phase=zero_phase,
        p=p
    )
    return augmenter(waveform, sample_rate=audio_sr)

def LowPassFilterAugmentation(waveform: np.ndarray, audio_sr: int, min_cutoff_freq: float = 150.0, max_cutoff_freq: float = 7500.0, zero_phase: bool = False, p: float = 0.0) -> np.ndarray:
    augmenter = LowPassFilter(
        min_cutoff_freq=min_cutoff_freq,
        max_cutoff_freq=max_cutoff_freq,
        zero_phase=zero_phase,
        p=p
    )
    return augmenter(waveform, sample_rate=audio_sr)

## ======= 2. Temporal Augmentations ======= ##

### ----- Time Manipulations ----- ###
def TimeStretchAugmentation(waveform: np.ndarray, audio_sr: int, min_rate: float = 0.8, max_rate: float = 1.25, leave_length_unchanged: bool = True, p: float = 0.0) -> np.ndarray:
    augmenter = TimeStretch(
        min_rate=min_rate,
        max_rate=max_rate,
        leave_length_unchanged=leave_length_unchanged,
        p=p
    )
    return augmenter(waveform, sample_rate=audio_sr)

def ShiftAugmentation(waveform: np.ndarray, audio_sr: int, min_shift: float = -0.5, max_shift: float = 0.5, shift_unit: str = "fraction", rollover: bool = False, p: float = 0.0) -> np.ndarray:
    augmenter = Shift(
        min_shift=min_shift,
        max_shift=max_shift,
        shift_unit=shift_unit,
        rollover=rollover,
        p=p
    )
    return augmenter(waveform, sample_rate=audio_sr)

def TrimAugmentation(waveform: np.ndarray, audio_sr: int, top_db: float = 30.0, p: float = 0.0) -> np.ndarray:
    augmenter = Trim(
        top_db=top_db,
        p=p
    )
    return augmenter(waveform, sample_rate=audio_sr)

### ----- Masking ----- ###
def TimeMaskAugmentation(waveform: np.ndarray, audio_sr: int, min_band_part: float = 0.01, max_band_part: float = 0.2, mask_location: str = "random", p: float = 0.0) -> np.ndarray:
    augmenter = TimeMask(
        min_band_part=min_band_part,
        max_band_part=max_band_part,
        mask_location=mask_location,
        p=p
    )
    return augmenter(waveform, sample_rate=audio_sr)

## ======= 3. Spectral Distortion Augmentations ======= ##

def PitchShiftAugmentation(waveform: np.ndarray, audio_sr: int, min_semitones: float = -4.0, max_semitones: float = 4.0, p: float = 0.0) -> np.ndarray:
    augmenter = PitchShift(
        min_semitones=min_semitones,
        max_semitones=max_semitones,
        p=p
    )
    return augmenter(waveform, sample_rate=audio_sr)

def TanhDistortionAugmentation(waveform: np.ndarray, audio_sr: int, min_distortion: float = 0.01, max_distortion: float = 0.7, p: float = 0.0) -> np.ndarray:
    augmenter = TanhDistortion(
        min_distortion=min_distortion,
        max_distortion=max_distortion,
        p=p
    )
    return augmenter(waveform, sample_rate=audio_sr)


def Compose1(waveform: np.ndarray, audio_sr: int, p: float = 0.0) -> np.ndarray:
    augment = Compose([
        Limiter(min_threshold_db=-24.0, max_threshold_db=-2.0, threshold_mode="relative_to_signal_peak", p=p),
        Trim(top_db=30.0, p=p)    
    ])

    return augment(waveform, sample_rate=audio_sr)

def Compose2(waveform: np.ndarray, audio_sr: int, p: float = 0.0) -> np.ndarray:
    augment = Compose([
        Limiter(min_threshold_db=-24.0, max_threshold_db=-2.0, threshold_mode="relative_to_signal_peak", p=p),
        GainTransition(min_gain_db=-24.0, max_gain_db=6.0, min_duration=0.2, max_duration=6.0, p=p)
    ])

    return augment(waveform, sample_rate=audio_sr)


## ======= Augmentation Factory ======= ##
def apply_augmentation(augmentation_type: str, audio: np.ndarray, audio_sr: int, prob: float):
    if augmentation_type == "gaussian_snr": 
        return GaussianSNRAugmentation(audio, audio_sr, p=prob)
    elif augmentation_type == "gain_transition": 
        return GainTransitionAugmentation(audio, audio_sr, p=prob)
    elif augmentation_type == "limiter": 
        return LimiterAugmentation(audio, audio_sr, p=prob)
    elif augmentation_type == "high_pass_filter": 
        return HighPassFilterAugmentation(audio, audio_sr, p=prob)
    elif augmentation_type == "low_pass_filter":
        return LowPassFilterAugmentation(audio, audio_sr, p=prob)
    elif augmentation_type == "time_stretch": 
        return TimeStretchAugmentation(audio, audio_sr, p=prob)
    elif augmentation_type == "shift": 
        return ShiftAugmentation(audio, audio_sr, p=prob)
    elif augmentation_type == "trim": 
        return TrimAugmentation(audio, audio_sr, p=prob)
    elif augmentation_type == "time_mask": 
        return TimeMaskAugmentation(audio, audio_sr, p=prob)
    elif augmentation_type == "pitch_shift": 
        return PitchShiftAugmentation(audio, audio_sr, p=prob)
    elif augmentation_type == "tanh_distortion": 
        return TanhDistortionAugmentation(audio, audio_sr, p=prob)
    elif augmentation_type == "compose1":
        return Compose1(audio, audio_sr, p=prob)
    elif augmentation_type == "compose2":
        return Compose2(audio, audio_sr, p=prob)
    else:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}")

