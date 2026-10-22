# AGENTS.md

> Operacional durável deste repo. Status e progresso moram no plan ativo (`plans/PLAN_*.md`), não aqui.

## Build & Run
- Ambiente: `conda activate calmos` (ver `environment.yml`). Rodar tudo a partir de `src/` ou com `PYTHONPATH=src`.
- Protocolo sequencial (cabeça apenas, backbone congelado): `python src/train_sequence.py --config <model.yaml> --output <dir> --gpu 0` (`--dry-run` valida configs e caminhos sem carregar backbone).
- GPU: se o torch falhar com `CUBLAS_STATUS_NOT_INITIALIZED`, rode com `LD_LIBRARY_PATH=` vazio. Um `LD_LIBRARY_PATH` apontando para `/usr/local/cuda/lib64` faz o cuBLASLt do sistema substituir o do torch.

## Validation / Tests
- Cabeça KAN (sem dependências externas): `PYTHONPATH=src python -m unittest test_kan_input_squash -v`
- `PYTHONPATH=src python -m unittest test_dataset_blocks test_train_sequence`
- Quirks conhecidos:
  - `test_dataset_blocks.py` importa `gerenciador` e `grid`, scripts pessoais fora do repo (`.gitignore`). Sem eles o arquivo não importa; com stubs vazios, 3 testes erram por falta de atributos.
  - `src/train_sequence.py` usa `spearmanr(...).statistic`, que exige scipy ≥ 1.10. Com scipy 1.9.3 (env `calmos`), 3 testes de `test_train_sequence.py` erram.

## PR & Branch
- Nunca commitar na `main`. Trabalho da KAN parte de `feat/4times`.

## Codebase patterns
- Cabeças: `MLPBase` (`src/models/base_models.py`) e `ReLUKAN` (`src/models/torch_relu_kan.py`).
- Expansão por dataset: `DatasetBlockExpansion` (`src/models/dataset_blocks.py`); ids em `src/utils/dataset_identity.py`.
