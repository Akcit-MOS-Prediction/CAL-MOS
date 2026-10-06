# Embeddings de validação: última camada, backbone congelado

O script `extract_val_embeddings.py` descobre todos os backbones distintos em
`config/models/*.yaml`. Variantes de cabeça KAN/MLP do mesmo backbone são
deduplicadas. Atualmente há 12 backbones; a seleção abaixo corresponde aos oito
da lista solicitada. Somente os CSVs de `datasets.val` são lidos.

## Executar no ambiente CAL-MOS

Dentro do container/ambiente com PyTorch, Transformers, NumPy, PyYAML,
torchaudio e tqdm, execute a partir da raiz do projeto:

```bash
python extract_embs/extract_val_embeddings.py --gpu 0
```

Para extrair somente os oito modelos da lista:

```bash
python extract_embs/extract_val_embeddings.py --gpu 0 --models mHuBERT-147 mms_1b mms_300m w2vbert_2.0 wav2vec2_1b wav2vec2_300m wavlm_large whisper_large_v3
```

O padrão usa todos os modelos. Os quatro adicionais atuais são `hubert_large`,
`wav2vec2_base`, `wav2vec2_large` e `whisper_large`.

Para conferir o plano sem carregar backbones:

```bash
python extract_embs/extract_val_embeddings.py --dry-run
```

No servidor, os workers existentes têm os datasets montados em `/workspace/datasets`.
Um comando usando o container existente, após conferir que a GPU está livre:

```bash
docker exec -w /home/user_danielcasanova/mos/CAL-MOS calmos-grid-f4612771fa5a-gpu0 /opt/conda/envs/calmos/bin/python extract_embs/extract_val_embeddings.py --gpu 0
```

Isso executa a extração em primeiro plano. O script não modifica a fila de treinos.
Para iniciar todos os backbones em segundo plano na GPU 0:

```bash
docker exec -d -e HF_HUB_OFFLINE=0 -w /home/user_danielcasanova/mos/CAL-MOS calmos-grid-f4612771fa5a-gpu0 /opt/conda/envs/calmos/bin/python extract_embs/run_val_embeddings.py
```

O launcher usa uma trava para evitar duplicação por esse comando e grava
`extraction.log` e `runner.json` no diretório padrão de resultados. O JSON registra
início, término e código de saída. Uma falha de modelo não interrompe os demais.
O cache padrão é consultado primeiro. Modelos ausentes são baixados em
`CAL-MOS/.cache/embeddings_models`, pois o cache original do container é somente
leitura. Isso não altera o cache compartilhado nem o ambiente dos outros workers.

Modelos ausentes do cache são baixados pelo Transformers; use `--local-files-only`
para impedir downloads. Use `--output DIRETORIO` para escolher o destino.

## O que é extraído

1. Backbone pré-treinado em `eval()`, sem gradientes e com todos os parâmetros congelados.
2. Último `hidden_state` do encoder (Whisper usa somente seu encoder).
3. Média temporal para produzir um vetor por áudio, antes da cabeça MLP/KAN.
4. LayerNorm adicional sobre o vetor, com `eps` do YAML, nas versões afim e não afim.

As LayerNorms internas dos backbones permanecem como vieram no pré-treino.
A LayerNorm adicional corresponde à posição `input_layer_norm` da cabeça CAL-MOS,
após pooling. Não são embeddings de uma cabeça treinada.

O áudio vira mono e é reamostrado para a taxa do extrator (16 kHz nesses modelos).
Áudios longos são processados inteiros em segmentos de até 30 s. A média dos segmentos
é ponderada pela quantidade de frames. Isso limita memória, mas reinicia o contexto
do encoder em cada segmento; não equivale a uma única passagem de áudio longo.
Para Whisper, o pooling exclui frames do padding até 30 s. Um segmento final com
menos de 640 amostras é completado com zeros para viabilizar as convoluções.

## LayerNorm afim

Sem treino nem parâmetros carregados, `gamma=1` e `beta=0`; por isso os arquivos
`afim.npy` e `nao_afim.npy` são numericamente iguais. A versão não afim ainda aplica
normalização; não significa ausência de LayerNorm.

Para usar parâmetros afins aprendidos, passe `--affine-params affine_params.json`.
O JSON mapeia o nome do modelo ou ID Hugging Face para um checkpoint e seu prefixo:

```json
{
  "wavlm_large": {
    "checkpoint": "/caminho/checkpoint.ckpt",
    "prefix": "model.mlp.input_layer_norm"
  }
}
```

O prefixo deve corresponder às chaves `.weight` e `.bias` reais do checkpoint;
se omitido, o script procura um único `input_layer_norm.weight`. Também é possível
fornecer arrays `weight` e `bias` de tamanho igual à dimensão do embedding.
Modelos não especificados usam identidade. Escolha um checkpoint consistente com
o backbone e com a origem dos parâmetros que deseja analisar: uma LayerNorm afim
aprendida depende do modelo/cabeça/dataset do treino. O script não ajusta parâmetros
nem usa rótulos da validação para aprender a transformação.

## Arquivos de saída

Padrão: `resultados/embeddings_val_last_layer/`.

```text
MODELO/DATASET/raw.npy       # [N, D], float32, média temporal sem LN adicional
MODELO/DATASET/afim.npy      # [N, D], float32
MODELO/DATASET/nao_afim.npy  # [N, D], float32
MODELO/DATASET/index.csv    # uma linha por vetor, com áudio, MOS e status
MODELO/DATASET/request.json # proveniência e parâmetros
MODELO/DATASET/status.json
MODELO/DATASET/errors.json
pares_val.csv               # modelo, layernorm, dataset_a, dataset_b e caminhos
summary.json
```

São seis pares não direcionais por modelo e por LayerNorm: 144 linhas para os
12 backbones, ou 96 para os oito da lista. Cada dataset é extraído uma vez por
backbone; os pares apontam para os mesmos arquivos. A tabela não calcula distância
ou similaridade entre datasets.

Falhas de áudio permanecem como linhas NaN nas matrizes, identificadas por status
no `index.csv`. Filtre essas linhas antes de analisar. Falhas individuais, de dataset
ou de carregamento de modelo são registradas, e a execução tenta os demais.
O código de saída é 1 se algum trabalho ficar incompleto.

## Retomada e teste curto

Reexecute o mesmo comando para aproveitar linhas completas e tentar novamente
as falhas. Configuração ou metadados diferentes exigem outro `--output` para
evitar mistura de experimentos. Não execute dois processos no mesmo destino.

Teste curto, com saída separada:

```bash
python extract_embs/extract_val_embeddings.py --gpu 0 --models wavlm_large --datasets brspeech bvcc --limit 2 --local-files-only --output /tmp/calmos-val-embeddings-smoke
```

`--limit` é somente para verificação; omita para extrair toda a validação.
O conteúdo do áudio é considerado estável na retomada; se substituir áudios nos
mesmos caminhos, use um novo destino.
