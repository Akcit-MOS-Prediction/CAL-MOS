import numpy as np 
from scipy.signal import butter, lfilter    
import argparse
import scipy

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

def save_audio(waveform: np.ndarray, sample_rate: int, output_prefix: str, augmentation_type: str, params: dict):
    
    audio_augmented = criar_factory_augmentation(augmentation_type, waveform, sample_rate, params)
    
    output_path = f"{output_prefix}_{augmentation_type}.wav"
    original_path = f"{output_prefix}_original.wav"

    print(f"\n\nSalvando audio augmentado em: {output_path}")

    def normalize_and_convert_to_int16(arr: np.ndarray) -> np.ndarray:
        arr_f32 = arr.astype(np.float32)
        arr_max = np.abs(arr_f32).max()
        if arr_max > 0:
            arr_f32 = arr_f32 / arr_max
        return (arr_f32 * 32767).astype(np.int16)

    scipy.io.wavfile.write(output_path, sample_rate, normalize_and_convert_to_int16(audio_augmented))
    
    print(f"Salvando audio original em: {original_path}")
    scipy.io.wavfile.write(original_path, sample_rate, normalize_and_convert_to_int16(waveform))

def criar_factory_augmentation(augmentation_type: str, audio: np.ndarray, sample_rate: int, params: dict):
    if augmentation_type == "clip":
        return ClipAugmentation(audio, threshold=params['threshold'])
    elif augmentation_type == "passa_alta":
        return PassaAltaAugmentation(audio, cutoff_freq=params['cutoff_freq'], sample_rate=sample_rate)
    elif augmentation_type == "passa_banda":
        return PassaBandaAugmentation(audio, lowcut=params['lowcut'], highcut=params['highcut'], sample_rate=sample_rate)
    elif augmentation_type == "passa_baixa":
        return PassaBaixaAugmentation(audio, cutoff_freq=params['cutoff_freq'], sample_rate=sample_rate)
    else:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}")
    
def main():
    parser = argparse.ArgumentParser(description="Aplica augmentações em áudio e salva os resultados.")
    
    parser.add_argument("--input_file", type=str, required=True, help="Caminho para o arquivo WAV de entrada.")
    parser.add_argument("--output_prefix", type=str, required=True, help="Prefixo para os arquivos WAV de saída.")
    parser.add_argument("--augmentation_name", type=str, required=True, choices=["clip", "passa_alta", "passa_baixa", "passa_banda"])
    
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--cutoff_freq", type=float, default=300.0)
    parser.add_argument("--lowcut", type=float, default=300.0)
    parser.add_argument("--highcut", type=float, default=3000.0)
    
    args = parser.parse_args()
    
    sample_rate, waveform = scipy.io.wavfile.read(args.input_file)
    
    if waveform.dtype in (np.int16, np.int32):
        waveform = waveform.astype(np.float32) / np.iinfo(waveform.dtype).max
    
    params = {
        "threshold": args.threshold,
        "cutoff_freq": args.cutoff_freq,
        "lowcut": args.lowcut,
        "highcut": args.highcut
    }
    
    save_audio(waveform, sample_rate, args.output_prefix, args.augmentation_name, params)

if __name__ == "__main__":
    main()
    
'''
    LJ003-0304_078.wav
LJ045-0176_068.wav
booksent_2013_0064_005.wav
conv_2007_0014_006.wav
general_0007_071.wav
'''