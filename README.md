# CAL-MOS Bridging Layers with Adapters for Robust MOS Prediction Across Speech Foundation Models

[![ISCA Archive](https://img.shields.io/badge/ISCA%20Archive-Interspeech%202026-1a73e8.svg)](https://www.isca-archive.org/interspeech_2026/ferreira26_interspeech.html)
[![arXiv](https://img.shields.io/badge/arXiv-2609.14956-b31b1b.svg)](https://arxiv.org/abs/2609.14956)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Collection-ffd21e.svg)](https://huggingface.co/collections/alefiury/cal-mos)


Official repository for the paper: **CAL-MOS Bridging Layers with Adapters for Robust MOS Prediction Across Speech Foundation Models**, accepted at Interspeech 2026.

## Overview

Speech Foundation Models (SFMs) expose hidden representations at every encoder depth, and it is not obvious which of those depths carry the information a Mean Opinion Score predictor actually needs. This repository contains the benchmark and the models used to study that question.

We train non intrusive MOS predictors on top of ten SFMs across four MOS datasets under a single protocol, comparing five ways of using the encoder:

| Code | Strategy | What it does |
|------|----------|--------------|
| LL | Last Layer | Frozen backbone, only the final hidden state feeds the head |
| BL | Best Layer | Frozen backbone, one intermediate layer picked from the layer wise sweep |
| WS | Weighted Sum | Frozen backbone, all layers combined with learnable softmax normalized scalars |
| FT | Full Fine Tuning | Backbone and head trained jointly |
| A+M | Adapters + Mean | Frozen backbone, one adapter per layer, then concatenation and masked mean pooling |

## Architecture

<img src="resources/CAL-MOS.png" alt="CAL-MOS architecture" width="640">

Audio goes through a frozen SFM and all hidden states are collected. Each layer gets its own adapter. The adapted sequences are concatenated along time, reduced by masked mean pooling into one utterance vector, and passed to an MLP head. Predictions are clipped to the [1, 5] MOS range and trained against reference scores with MSE.

## Results

Averages over BRSpeech, BVCC, SingMOS, and TMHINT-QI from the 100 epoch comparison. Lower MSE is better, higher SRCC is better.

| Backbone | Strategy | Utt. MSE | Utt. SRCC | Sys. MSE | Sys. SRCC |
|----------|----------|----------|-----------|----------|-----------|
| WavLM Large | LL | 0.412 | 0.727 | 0.067 | 0.908 |
| WavLM Large | BL | 0.410 | 0.731 | 0.070 | 0.909 |
| WavLM Large | WS | 0.498 | 0.688 | 0.077 | 0.905 |
| WavLM Large | FT | 0.417 | 0.737 | 0.074 | 0.916 |
| WavLM Large | **A+M** | **0.392** | **0.747** | **0.065** | **0.917** |
| MMS 300M | LL | 0.444 | 0.684 | 0.085 | 0.842 |
| MMS 300M | BL | 0.483 | 0.734 | 0.148 | 0.911 |
| MMS 300M | WS | **0.431** | 0.716 | **0.072** | 0.865 |
| MMS 300M | FT | 0.439 | 0.719 | 0.074 | 0.897 |
| MMS 300M | A+M | 0.486 | **0.737** | 0.142 | **0.903** |
| XLS-R 300M | LL | 0.469 | 0.671 | 0.104 | 0.844 |
| XLS-R 300M | BL | 0.417 | 0.737 | 0.058 | 0.918 |
| XLS-R 300M | WS | 0.417 | 0.729 | 0.070 | 0.907 |
| XLS-R 300M | FT | **0.407** | 0.733 | **0.053** | **0.923** |
| XLS-R 300M | A+M | 0.408 | **0.745** | 0.078 | 0.913 |
| Wav2BERT 2.0 | LL | 0.466 | 0.660 | 0.116 | 0.817 |
| Wav2BERT 2.0 | BL | 0.395 | 0.735 | 0.061 | 0.920 |
| Wav2BERT 2.0 | WS | 0.396 | 0.739 | 0.059 | 0.919 |
| Wav2BERT 2.0 | FT | 0.458 | 0.716 | 0.095 | 0.906 |
| Wav2BERT 2.0 | **A+M** | **0.388** | **0.749** | **0.052** | **0.932** |
| HuBERT Large | LL | 0.422 | 0.715 | 0.066 | 0.895 |
| HuBERT Large | BL | 0.413 | 0.723 | 0.072 | 0.891 |
| HuBERT Large | WS | 0.485 | 0.689 | 0.065 | **0.925** |
| HuBERT Large | FT | 0.426 | 0.731 | 0.070 | 0.904 |
| HuBERT Large | **A+M** | **0.414** | **0.738** | **0.062** | 0.913 |

These five backbones were selected out of the ten screened at 20 epochs, based on average SRCC. Per dataset numbers and the full ten backbone screening table are in the paper.

## Repository layout

```
config/
  data/          dataset paths for raw audio training
  data_embs/     dataset paths for pre extracted embedding training
  model/         one directory per training regime, one YAML per backbone
  trainer/       batch size, epochs, optimizer, scheduler, logging
src/
  main.py                      train a run and evaluate it on the test split
  eval/inference.py            standalone inference from a checkpoint
  eval/inference_func.py       utterance and system level metric computation
  run_inferences.py            re run inference over a directory of finished experiments
  get_metrics.py               collect metrics from finished experiments
  rank_backbones_bootstrap.py  paired bootstrap ranking with Holm Bonferroni correction
  models/                      backbones, adapters, pooling, heads, PEFT
  utils/                       datasets, collators, schedulers, metrics
  recipes/                     shell scripts for batches of runs and layer sweeps
extract_embs/                  offline embedding extraction scripts
resources/                     architecture figure
```

Model configs are grouped by regime:

| Directory | Regime |
|-----------|--------|
| `config/model/frozen-last-layer/` | LL |
| `config/model/frozen-specific-layer/` | BL and the layer wise sweep |
| `config/model/weighted-sum/` | WS |
| `config/model/fine-tuning/` | FT |
| `config/model/adapters-mean-pooling/` | A+M |
| `config/model/weighted-sum-asp/` | WS with attentive statistics pooling |
| `config/model/adapters-att-pooling/` | Adapters with attentive statistics pooling |
| `config/model/fine-tuning-reinit/` | FT with the top k encoder blocks re initialized |
| `config/model/weighted_sum-acoustic_concat-pretrained/` | Adapters plus a pretrained CED mel spectrogram branch |
| `config/model/multiple_layer_embedding*/` | Training from pre extracted embeddings |


## Installation

With pip:

```bash
pip install -r requirements.txt
```

With conda, which pins the full environment used for the paper:

```bash
conda env create -f environment.yml
conda activate calmos
```

A CUDA image is also available:

```bash
docker build -t calmos .
```

Training logs go to Weights and Biases. Set `logger.wandb.entity` and `logger.wandb.project` in the trainer config and run `wandb login` before the first run.

## Data preparation

Each split is a CSV file. The dataset config points at it and names the relevant columns.

| Column | Required | Purpose |
|--------|----------|---------|
| filename column | yes | Path to the audio file, absolute or relative to `base_dir` |
| target column | yes | Reference MOS value |
| system id column | test only | Groups utterances into systems for system level metrics, defaults to `system_id` |
| sr column | no | Sampling rate class, used only when `use_sr_embeddings` is on |

Point the four dataset configs at your copies of BVCC, BRSpeech-MOS, SingMOS, and TMHINT-QI. The paths shipped in `config/data/` are from the original cluster and need to be replaced. A single dataset file looks like this:

```yaml
datasets:
    name: "BVCC"
    train:
        - name: Train
          metadata_path: "/path/to/bvcc_train.csv"
          base_dir: ""
          filename_column: "wav_path"
          target_column: "avg_score"
    val:
        - name: Validation
          metadata_path: "/path/to/bvcc_dev.csv"
          base_dir: ""
          filename_column: "wav_path"
          target_column: "avg_score"
    test:
        - name: Test
          metadata_path: "/path/to/bvcc_test.csv"
          base_dir: ""
          filename_column: "wav_path"
          target_column: "avg_score"
          sys_id_column: "system_id"
```

Audio is resampled to 16 kHz by the dataset class, so the source sampling rate does not need to be uniform.

## Training

A run is assembled from three configs: one dataset, one model, one trainer.

Adapters and mean pooling on Wav2BERT 2.0 and BVCC, which is the best configuration in the paper:

```bash
cd src
python main.py \
    -cd=../config/data/bvcc.yaml \
    -cm=../config/model/adapters-mean-pooling/wav2bert2.yaml \
    -ct=../config/trainer/default-16bs-weighted_sum.yaml \
    -g=0
```

Swap the model config to change regime. Last layer probing on the same data:

```bash
python main.py \
    -cd=../config/data/bvcc.yaml \
    -cm=../config/model/frozen-last-layer/wav2bert2.yaml \
    -ct=../config/trainer/default-16bs-weighted_sum.yaml \
    -g=0
```

Full fine tuning, which uses its own trainer config because the memory profile is different:

```bash
python main.py \
    -cd=../config/data/bvcc.yaml \
    -cm=../config/model/fine-tuning/wav2bert2.yaml \
    -ct=../config/trainer/default-16bs-ft.yaml \
    -g=0
```

A single YAML holding the dataset, model, and trainer keys together also works, through `-c`:

```bash
python main.py -c=/path/to/merged_config.yaml -g=0
```

`main.py` trains, picks the checkpoint with the best validation SRCC, runs inference on the test split, and writes `results.csv`, `results.txt`, `predictions_utt.csv`, and `predictions_sys.csv` into the run directory under `logger.runs_dir`.

### Trainer configs

All of them use AdamW with betas 0.9 and 0.98, eps 1e-8, weight decay 1e-6, gradient clipping at norm 10, and a cosine schedule with 500 warmup steps moving between 1e-5 and 5e-5. Checkpoint selection and early stopping both monitor validation SRCC. They differ in how they reach an effective batch of 64 and where they write output:

| Config | Batch | Accumulation | Epochs |
|--------|-------|--------------|--------|
| `default.yaml` | 32 | 2 | 20 |
| `default-2bs.yaml` | 2 | 32 | 20 |
| `default-1bs16bf.yaml` | 1 | 64 | 20 |
| `default-16bs-weighted_sum.yaml` | 16 | 4 | 100 |
| `default-16bs-specific_layer.yaml` | 16 | 4 | 100 |
| `default-16bs-ft.yaml` | 16 | 4 | 100 |
| `default-1bs-ft.yaml` | 1 | 64 | 100 |

The 20 epoch configs reproduce the screening stage and the 100 epoch configs reproduce the main comparison.

### Layer wise sweep

The Best Layer results come from training one probe per encoder depth. `config/model/frozen-specific-layer/` leaves `specific_layer_idx` empty so it can be set from the command line with `-l`, and `default-16bs-specific_layer.yaml` puts the layer index into the run title so the runs stay distinguishable:

```bash
for layer in $(seq 0 24)
do
    python main.py \
        -cd=../config/data/bvcc.yaml \
        -cm=../config/model/frozen-specific-layer/wavlm-large.yaml \
        -ct=../config/trainer/default-16bs-specific_layer.yaml \
        -l=$layer \
        -g=0
done
```

Use 24 for the 25 layer backbones, 32 for Whisper Large, and 48 for the 1B models. The shell scripts in `src/recipes/` follow the same pattern and are kept as templates, though their paths point at an older config layout.

## Evaluation

`main.py` already evaluates at the end of training. To score an existing checkpoint on its own:

```bash
cd src
python eval/inference.py \
    -c=/path/to/merged_config.yaml \
    -ckpt=/path/to/checkpoint/dir \
    -g=0
```

This entry point takes one merged config rather than the three part split, so pass the same YAML the run was trained with.

Metrics are reported at two levels. Utterance level compares per file predictions against per file MOS. System level first averages predictions and references inside each system, using the system id column, and then compares those averages. Both report MSE, LCC, SRCC, and KTAU.

## Pre extracted embeddings

Training directly on cached hidden states avoids repeated forward passes through a frozen backbone. Extract first:

```bash
cd extract_embs
python extract_automodel_embeddings.py \
    -b=/path/to/audio/base/dir \
    -o=/path/to/output/dir \
    -m="microsoft/wavlm-large" \
    -c=/path/to/metadata.csv \
    -col=wav_path
```

Add `-l` to keep a single layer instead of all of them, or use `extract_automodel_embeddings_save_layers_individually.py` to write one file per layer, which keeps the sweep cheap. Then train with a config from `config/data_embs/` and a model config from `config/model/multiple_layer_embedding/` or `config/model/multiple_layer_embedding_weighted_sum/`. See `extract_embs/README.md` for the older per family scripts.

## Backbones

Layer counts include the convolutional feature output, so a 24 block encoder is listed as 25 layers. The hidden size is what `mlp_input_dim` and `adapter_input_dim` must be set to.

Used in the paper:

| Backbone | Hugging Face id | Layers | Hidden size |
|----------|-----------------|--------|-------------|
| wav2vec 2.0 Large | `facebook/wav2vec2-large` | 25 | 1024 |
| XLS-R 300M | `facebook/wav2vec2-xls-r-300m` | 25 | 1024 |
| XLS-R 1B | `facebook/wav2vec2-xls-r-1b` | 49 | 1280 |
| MMS 300M | `facebook/mms-300m` | 25 | 1024 |
| MMS 1B | `facebook/mms-1b` | 49 | 1280 |
| W2V-BERT 2.0 | `facebook/w2v-bert-2.0` | 25 | 1024 |
| WavLM Large | `microsoft/wavlm-large` | 25 | 1024 |
| HuBERT Large | `facebook/hubert-large-ll60k` | 25 | 1024 |
| data2vec Large | `facebook/data2vec-audio-large` | 25 | 1024 |
| Whisper Large | `openai/whisper-large` | 33 | 1280 |

Other backbones the code handles, for adding new configs:

| Backbone | Hugging Face id | Layers | Hidden size |
|----------|-----------------|--------|-------------|
| wav2vec 2.0 Base 960h | `facebook/wav2vec2-base-960h` | 13 | 768 |
| wav2vec 2.0 Large XLSR-53 | `facebook/wav2vec2-large-xlsr-53` | 25 | 1024 |
| XLS-R 2B | `facebook/wav2vec2-xls-r-2b` | 49 | 1280 |
| HuBERT Large ASR | `facebook/hubert-large-ls960-ft` | 25 | 1024 |
| HuBERT XLarge | `facebook/hubert-xlarge-ls960-ft` | 49 | 1280 |
| WavLM Base Plus | `microsoft/wavlm-base-plus` | 13 | 768 |
| Whisper Tiny | `openai/whisper-tiny` | 5 | 384 |
| Whisper Base | `openai/whisper-base` | 7 | 512 |
| Whisper Small | `openai/whisper-small` | 13 | 768 |
| Whisper Medium | `openai/whisper-medium` | 25 | 1024 |
| Whisper Large v3 | `openai/whisper-large-v3` | 33 | 1280 |
| Whisper Large v2 | `openai/whisper-large-v2` | 33 | 1280 |

To add a backbone, copy a model YAML from the regime you want, set `model_name`, and match `num_feature_layers`, `mlp_input_dim`, and `adapter_input_dim` to the table above.

## Citation

```bibtex
@inproceedings{ferreira26_interspeech,
  title     = {{CAL-MOS: Bridging Layers with Adapters for Robust MOS Prediction Across Speech Foundation Models}},
  author    = {Alef Iury Ferreira and Pedro Botelho and Fernanda Silva and Daniel Casanova and Rafael Faustino and Frederico Oliveira and Arlindo Galvão Filho and Anderson da Silva Soares},
  year      = {2026},
  booktitle = {{Interspeech 2026}},
  pages     = {174--179},
  doi       = {10.21437/Interspeech.2026-2960},
  issn      = {2958-1796},
}
```

## License

MIT. See [LICENSE](LICENSE).
