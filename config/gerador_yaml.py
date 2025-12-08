from omegaconf import OmegaConf
import os 


yaml_padrao = OmegaConf.load('default_augmentation.yaml')

output_folder = "generated_configs"
os.makedirs(output_folder, exist_ok=True)

base_yaml_path = 'default_augmentation.yaml'
yaml_padrao = OmegaConf.load(base_yaml_path)

LIST_OF_AUG = [
    'background_noise', 'gaussian_snr', 'limiter', 'gain_transition',
    'high_pass_filter', 'low_pass_filter', 'time_stretch', 'shift',
    'trim', 'time_mask', 'pitch_shift', 'tanh_distortion'
]


LIST_OF_PCT = [0.0, 0.5, 1.0]

BACKBONE = [
    'facebook/mms-300m', 
    'facebook/wav2vec2-xls-r-300m', 
    'openai/whisper-large'
]

AUG_PARAMS = {
    'background_noise': {
        'sounds_path': "/caminho dos ruidos",
        'min_snr_db': 3.0,
        'max_snr_db': 30.0,
        'noise_rms': "relative"
    },
    'gaussian_snr': {
        'min_snr_db': 5.0,
        'max_snr_db': 40.0
    },
    'gain_transition': {
        'min_gain_db': -24.0,
        'max_gain_db': 6.0,
        'min_duration': 0.2,
        'max_duration': 6.0
    },
    'limiter': {
        'min_threshold_db': -24.0,
        'max_threshold_db': -2.0,
        'threshold_mode': "relative_to_signal_peak"
    },
    'high_pass_filter': {
        'min_cutoff_freq': 20.0,
        'max_cutoff_freq': 2400.0,
        'zero_phase': False
    },
    'low_pass_filter': {
        'min_cutoff_freq': 150.0,
        'max_cutoff_freq': 7500.0,
        'zero_phase': False
    },
    'time_stretch': {
        'min_rate': 0.8,
        'max_rate': 1.25,
        'leave_length_unchanged': True
    },
    'shift': {
        'min_shift': -0.5,
        'max_shift': 0.5,
        'shift_unit': "fraction",
        'rollover': False
    },
    'trim': {
        'top_db': 30.0
    },
    'time_mask': {
        'min_band_part': 0.01,
        'max_band_part': 0.2,
        'mask_location': "random"
    },
    'pitch_shift': {
        'min_semitones': -4.0,
        'max_semitones': 4.0
    },
    'tanh_distortion': {
        'min_distortion': 0.01,
        'max_distortion': 0.7
    }
}

print(f"Iniciando geração de arquivos em: {output_folder}/")

for model in BACKBONE:
    # Ajuste da mlp
    current_mlp_dim = 1024
    if "whisper-large" in model:
        current_mlp_dim = 1280 # Whisper
    elif "xls-r-300m" in model or "mms-300m" in model:
        current_mlp_dim = 1024 # Wav2Vec2 base/large e MMS
    
    model_file_name = model.replace("/", "_")

    for aug_name in LIST_OF_AUG:
        for pct in LIST_OF_PCT:
            
            cfg = yaml_padrao.copy()
            
            cfg.model.model_name = model
            cfg.model.mlp_input_dim = current_mlp_dim
            
            cfg.model.mlp_hidden_dim = current_mlp_dim
            cfg.model.transformer_hidden_size = current_mlp_dim

            cfg.augmentation.name = aug_name
            cfg.augmentation.probability = float(pct)
            
            # CAL-MOS-${model.model_name}-(epochs-${trainer.max_epochs})-(bs-${train.batch_size})-(LR-${optimizer.params.learning_rate})
            cfg.title = 'CAL-MOS-${model.model_name}-(augmentation-${augmentation.name})-(pct-${augmentation.probability})'
            
            if aug_name in AUG_PARAMS:
                cfg.augmentation.params = AUG_PARAMS[aug_name]
            else:
                print(f"AVISO: Parâmetros para {aug_name} não encontrados. Usando vazio.")
                cfg.augmentation.params = {}

            filename = f"{aug_name}_p{pct}_{model_file_name}.yaml"
            filepath = os.path.join(output_folder, filename)
            

            with open(filepath, 'w') as f:
                OmegaConf.save(cfg, f)
            
            print(f"Gerado: {filename}")

print("Concluído! Todos os arquivos .yaml foram gerados.")