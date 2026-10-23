# CONTEXT.md

> Decisões, armadilhas e padrões aprendidos entre sessões. ≠ `AGENTS.md` (operacional) e ≠ plan ativo (estado vivo).

## Decisões de arquitetura
- ReLU-KAN recebe `sigmoid(LayerNorm(x))`, com LayerNorm não-afim + sigmoide antes de cada camada interna, e padding 0 nos blocos de dataset, como a MLP (decisão de 2026-10-23; o −1 de 2026-10-22 foi revertido). A MLP continua só com LayerNorm. — A grade da ReLU-KAN só existe em `[-k/g, 1 + k/g]`; com LayerNorm puro ~40% das coordenadas do WavLM-large caíam fora (base e gradiente zero) e o treino não convergia, enquanto o mHuBERT funcionava por acaso. Spec: `specs/kan-input-sigmoid.md`. (2026-10-22)

## Armadilhas / gotchas
- A ReLU-KAN não satura fora da grade: a coordenada morre (contribuição e gradiente zero, inclusive dos limites da grade). Qualquer mudança na escala da entrada da KAN precisa checar a fração fora de `[-k/g, 1 + k/g]`.
- Na KAN, padding dentro da grade (ex.: 0) não é inerte: ativa bases e vira um viés por dataset do tamanho do sinal.
- A expansão 4x multiplica a primeira camada por 4 (MLP 1,05M → 4,20M; KAN g=5,k=2 7,38M → 29,4M). É o desenho, não um bug.

## Mapa mental do código
- Pipeline da cabeça: backbone congelado → camada escolhida → mean pooling → `LayerNorm` não-afim → (KAN: sigmoide) → `DatasetBlockExpansion` → cabeça.
- `src/train_sequence.py` força as opções do protocolo (blocos 4x, LN não-afim, padding 0) por cima dos YAMLs.

## Questões em aberto
- Temperatura da sigmoide (`sigmoid(x/T)`) para devolver resolução às caudas: não testada.
