import numpy as np 
from audiomentations import AddBackgroundNoise, AddGaussianSNR, GainTransition, Limiter, HighPassFilter, LowPassFilter, TimeStretch, Shift, Trim, TimeMask, PitchShift, TanhDistortion

## ======= 1. Acoustic Variations Augmentations ======= ##

### ----- Noise Additions ----- ###
def BackgroundNoiseAugmentation(waveform: np.ndarray, audio_sr: int, sounds_path: str, min_snr_db: float = 3.0, max_snr_db: float = 30.0, noise_rms: str = "relative") -> np.ndarray:
    augmenter = AddBackgroundNoise(
        sounds_path=sounds_path,
        min_snr_db=min_snr_db,
        max_snr_db=max_snr_db,
        noise_rms=noise_rms,
        p=1.0
    )
    return augmenter(waveform, sample_rate=audio_sr)

def GaussianSNRAugmentation(waveform: np.ndarray, audio_sr: int, min_snr_db: float = 5.0, max_snr_db: float = 40.0) -> np.ndarray:
    augmenter = AddGaussianSNR(
        min_snr_db=min_snr_db,
        max_snr_db=max_snr_db,
        p=1.0
    )
    return augmenter(waveform, sample_rate=audio_sr)

### ----- Volume and Gain Adjustments ----- ### 
def GainTransitionAugmentation(waveform: np.ndarray, audio_sr: int, min_gain_db: float = -24.0, max_gain_db: float = 6.0, min_duration: float = 0.2, max_duration: float = 6.0) -> np.ndarray:
    augmenter = GainTransition(
        min_gain_db=min_gain_db,
        max_gain_db=max_gain_db,
        min_duration=min_duration,
        max_duration=max_duration,
        p=1.0
    )
    return augmenter(waveform, sample_rate=audio_sr)

def LimiterAugmentation(waveform: np.ndarray, audio_sr: int, min_threshold_db: float = -24.0, max_threshold_db: float = -2.0, threshold_mode: str = "relative_to_signal_peak") -> np.ndarray:
    augmenter = Limiter(
        min_threshold_db=min_threshold_db,
        max_threshold_db=max_threshold_db,
        threshold_mode=threshold_mode,
        p=1.0
    )
    return augmenter(waveform, sample_rate=audio_sr)

### ----- Filtering ----- ###
def HighPassFilterAugmentation(waveform: np.ndarray, audio_sr: int, min_cutoff_freq: float = 20.0, max_cutoff_freq: float = 2400.0, zero_phase: bool = False) -> np.ndarray:
    augmenter = HighPassFilter(
        min_cutoff_freq=min_cutoff_freq,
        max_cutoff_freq=max_cutoff_freq,
        zero_phase=zero_phase,
        p=1.0
    )
    return augmenter(waveform, sample_rate=audio_sr)

def LowPassFilterAugmentation(waveform: np.ndarray, audio_sr: int, min_cutoff_freq: float = 150.0, max_cutoff_freq: float = 7500.0, zero_phase: bool = False) -> np.ndarray:
    augmenter = LowPassFilter(
        min_cutoff_freq=min_cutoff_freq,
        max_cutoff_freq=max_cutoff_freq,
        zero_phase=zero_phase,
        p=1.0
    )
    return augmenter(waveform, sample_rate=audio_sr)

## ======= 2. Temporal Augmentations ======= ##

### ----- Time Manipulations ----- ###
def TimeStretchAugmentation(waveform: np.ndarray, audio_sr: int, min_rate: float = 0.8, max_rate: float = 1.25, leave_length_unchanged: bool = True) -> np.ndarray:
    augmenter = TimeStretch(
        min_rate=min_rate,
        max_rate=max_rate,
        leave_length_unchanged=leave_length_unchanged,
        p=1.0
    )
    return augmenter(waveform, sample_rate=audio_sr)

def ShiftAugmentation(waveform: np.ndarray, audio_sr: int, min_shift: float = -0.5, max_shift: float = 0.5, shift_unit: str = "fraction", rollover: bool = False) -> np.ndarray:
    augmenter = Shift(
        min_shift=min_shift,
        max_shift=max_shift,
        shift_unit=shift_unit,
        rollover=rollover,
        p=1.0
    )
    return augmenter(waveform, sample_rate=audio_sr)

def TrimAugmentation(waveform: np.ndarray, audio_sr: int, top_db: float = 30.0) -> np.ndarray:
    augmenter = Trim(
        top_db=top_db,
        p=1.0
    )
    return augmenter(waveform, sample_rate=audio_sr)

### ----- Masking ----- ###
def TimeMaskAugmentation(waveform: np.ndarray, audio_sr: int, min_band_part: float = 0.01, max_band_part: float = 0.2, mask_location: str = "random") -> np.ndarray:
    augmenter = TimeMask(
        min_band_part=min_band_part,
        max_band_part=max_band_part,
        mask_location=mask_location,
        p=1.0
    )
    return augmenter(waveform, sample_rate=audio_sr)

## ======= 3. Spectral Distortion Augmentations ======= ##

def PitchShiftAugmentation(waveform: np.ndarray, audio_sr: int, min_semitones: float = -4.0, max_semitones: float = 4.0) -> np.ndarray:
    augmenter = PitchShift(
        min_semitones=min_semitones,
        max_semitones=max_semitones,
        p=1.0
    )
    return augmenter(waveform, sample_rate=audio_sr)

def TanhDistortionAugmentation(waveform: np.ndarray, audio_sr: int, min_distortion: float = 0.01, max_distortion: float = 0.7) -> np.ndarray:
    augmenter = TanhDistortion(
        min_distortion=min_distortion,
        max_distortion=max_distortion,
        p=1.0
    )
    return augmenter(waveform, sample_rate=audio_sr)

## ======= Augmentation Factory ======= ##
def apply_augmentation(augmentation_type: str, audio: np.ndarray, audio_sr: int, **args):

    if augmentation_type == "background_noise": 
        return BackgroundNoiseAugmentation(audio, audio_sr, **args)
    elif augmentation_type == "gain_transition": 
        return GainTransitionAugmentation(audio, audio_sr, **args)
    elif augmentation_type == "limiter": 
        return LimiterAugmentation(audio, audio_sr, **args)
    elif augmentation_type == "high_pass_filter": 
        return HighPassFilterAugmentation(audio, audio_sr, **args)
    elif augmentation_type == "low_pass_filter":
        return LowPassFilterAugmentation(audio, audio_sr, **args)
    elif augmentation_type == "time_stretch": 
        return TimeStretchAugmentation(audio, audio_sr, **args)
    elif augmentation_type == "shift": 
        return ShiftAugmentation(audio, audio_sr, **args)
    elif augmentation_type == "trim": 
        return TrimAugmentation(audio, audio_sr, **args)
    elif augmentation_type == "time_mask": 
        return TimeMaskAugmentation(audio, audio_sr, **args)
    elif augmentation_type == "pitch_shift": 
        return PitchShiftAugmentation(audio, audio_sr, **args)
    elif augmentation_type == "tanh_distortion": 
        return TanhDistortionAugmentation(audio, audio_sr, **args)
    else:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}")
