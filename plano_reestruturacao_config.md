# Plano de Reestruturação de Configurações (CAL-MOS)

Este documento descreve o plano para dividir os arquivos de configuração monolíticos em partes modulares, separando as definições de dataset, modelos e garantindo total rastreabilidade dos experimentos.

## 1. Nova Estrutura de Diretórios

Proposta de organização para a pasta `config/`:

```text
config/
├── datasets/          # Apenas definições de caminhos e parâmetros de áudio e dados
│   ├── brspeech_v2.yaml
│   └── bvcc.yaml
├── models/            # Arquitetura, otimizador, scheduler e hiperparâmetros
│   ├── wav2vec2_base.yaml
│   └── wavlm_large.yaml
└── default.yaml       # Configurações globais, WandB, batch_size e templates de títulos
```

## 2. Divisão do Conteúdo

### Arquivo A: `config/datasets/brspeech_v2.yaml`
Foco: Onde estão os dados e como processá-los.
```yaml
datasets:
    train:
        - name: Train
          metadata_path: "/home/pedro-lustosa/mos-finetune-ssl/BRSPEECH_MOS_DATASET_v2/train.csv"
          base_dir: "/home/pedro-lustosa/mos-finetune-ssl/BRSPEECH_MOS_DATASET_v2"
          filename_column: "filepath"
          target_column: "mos"
    val:
        - name: Validation
          metadata_path: "/home/pedro-lustosa/mos-finetune-ssl/BRSPEECH_MOS_DATASET_v2/val.csv"
          base_dir: "/home/pedro-lustosa/mos-finetune-ssl/BRSPEECH_MOS_DATASET_v2"
          filename_column: "filepath"
          target_column: "mos"
    test:
        - name: Test
          metadata_path: "/home/pedro-lustosa/mos-finetune-ssl/data/sets/test_mos_list.txt"
          base_dir: "/home/pedro-lustosa/mos-finetune-ssl/data/wav"
          filename_column: "filepath"
          target_column: "mos"

data:
    use_seqaug: false
    num_classes: 1
    target_sr: 16000
    mixup_alpha: 0.0
    use_rand_truncation: false
    min_duration: 2.0
    min_white_noise_amp: 0.01
    max_white_noise_amp: 0.1
    insert_white_noise: false
```

### Arquivo B: `config/models/wav2vec2_base.yaml`
Foco: Arquitetura do modelo, otimizadores e treinamento.
```yaml
model:
    model_type: "dynamic"
    model_name: "facebook/wav2vec2-base"
    mlp_input_dim: 1536
    mlp_hidden_dim: ${model.mlp_input_dim}
    mlp_num_layers: 1
    mlp_output_size: ${data.num_classes}
    mlp_dropout: 0.1
    mlp_activation_func: "relu"
    layer_weight_strategy: "per_layer"
    num_feature_layers: 13
    specific_layer_idx: -1
    pooling_strategy: "attpool"
    freeze_backbone: true
    transformer_hidden_size: ${model.mlp_input_dim}
    transformer_nhead: 4
    transformer_dim_feedforward: 2048
    transformer_activation: "gelu"
    transformer_dropout: 0.1
    transformer_layer_norm_eps: 1e-5
    transformer_bias: true
    transformer_num_hidden_layers: 2

optimizer:
    name: "adamw"
    params:
        min_learning_rate: 1e-5
        learning_rate: 5e-5
        eps: 1e-8
        weight_decay: 1e-6
        betas: [0.9, 0.98]

scheduler:
    name: "CosineWarmupLR"
    params:
        warmup_lr: 500

trainer:
    accelerator: "gpu"
    max_epochs: 20
    num_sanity_val_steps: 2
    overfit_batches: 0.0
    log_every_n_steps: 10
    gradient_clip_val: 10.0
    gradient_clip_algorithm: "norm"
    val_check_interval: 1.0
    accumulate_grad_batches: 2
```

### Arquivo C: `config/default.yaml`
Foco: Configurações base e de infraestrutura.
```yaml
wandb_entity: proga150-ufrn
title: CAL-MOS-${model.model_name}-(epochs-${trainer.max_epochs})-(bs-${train.batch_size})-(LR-${optimizer.params.learning_rate})

train:
    batch_size: 32
    shuffle: True
    num_workers: 12

model_checkpoint:
    mode: "max"
    save_last: true
    save_weights_only: true
    monitor: "val/spearman"
    dirpath: "../checkpoints/mos-prediction"
    filename: "{epoch:02d}-{step:02d}-{val/spearman:.4f}"

tags:
    - ${train.batch_size}-BS
    - LR-${optimizer.params.learning_rate}
    - WD-${optimizer.params.weight_decay}
    - ${model.model_name}-HS
    - ${model.mlp_input_dim}-AH
    - ${model.mlp_hidden_dim}-HL
    - ${model.mlp_num_layers}-IS
```

## 3. Rastreabilidade de Experimentos (Checkpoints e Configs)

Para garantir que sempre saibamos exatamente quais hiperparâmetros e dados geraram um determinado modelo (checkpoint/cpk), o sistema deverá **salvar automaticamente uma cópia do arquivo de configuração final mesclado (resolvido) dentro da mesma pasta onde os checkpoints serão salvos**. 

Isso significa que na pasta `../checkpoints/mos-prediction/...`, além do arquivo `.ckpt`, você sempre encontrará um arquivo `.yaml` contendo a exata fusão de `default` + `dataset` + `modelo` usada na rodada.

## 4. Alteração no Código-Fonte (`src/main.py`)

Para que a aplicação suporte múltiplos arquivos de configuração mesclados e implemente a rastreabilidade:

```python
import os
import argparse
from omegaconf import OmegaConf

# Atualização do argparse para aceitar múltiplas flags -c
parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--config_path",
    required=True,
    action="append", # Permite múltiplos argumentos -c
    type=str,
    help="YAML file(s) with configurations"
)

# ... outros argumentos ...
args = parser.parse_args()

# 1. Carregamento e merge
configs = [OmegaConf.load(path) for path in args.config_path]
config = OmegaConf.merge(*configs)

# 2. Resolução de variáveis interpoladas (opcional, mas recomendado para o dump)
OmegaConf.resolve(config)

# 3. Rastreabilidade: Salvar o config resolvido na pasta de checkpoints
# Obter o caminho real onde o run será salvo (pode incluir timestamp ou nome do wandb para evitar sobrescrita)
checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir else config.model_checkpoint.dirpath
os.makedirs(checkpoint_dir, exist_ok=True)

# Salvar o .yaml consolidado
config_save_path = os.path.join(checkpoint_dir, "resolved_config.yaml")
OmegaConf.save(config=config, f=config_save_path)
print(f"Configuração consolidada salva em: {config_save_path}")

# ... resto do código (Trainer, etc) ...
```

**Exemplo de Execução (Scripts Bash):**
```bash
python src/main.py -c config/default.yaml -c config/datasets/brspeech_v2.yaml -c config/models/wav2vec2_base.yaml
```

## 5. Vantagens

1.  **Reutilização / Mix & Match**: É possível combinar qualquer modelo com qualquer dataset sem duplicar os metadados dos CSVs e configurações gerais.
2.  **Manutenibilidade**: Se o caminho do dataset mudar, apenas o arquivo em `config/datasets/` será modificado.
3.  **Rastreabilidade Completa**: Os checkpoints agora nunca ficarão "órfãos". Ao analisar os resultados no futuro, o arquivo `.yaml` consolidado e guardado junto com os pesos do modelo dirá exatamente como ele foi treinado.
4.  **Clean Architecture**: Ajuda a isolar a infraestrutura de dados da arquitetura da rede neural.
