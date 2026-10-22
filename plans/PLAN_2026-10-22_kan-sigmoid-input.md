# Plan — Sigmoide após o LayerNorm na entrada da ReLU-KAN
Data: 2026-10-22 · Escopo: colocar a entrada (e as camadas internas) da ReLU-KAN dentro da grade, só na KAN · Parent Linear: — (Linear desligado nesta sessão)
Specs relevantes: [specs/kan-input-sigmoid.md] · Feedback loops: unit, numérico, treino curto real
Branch: `feat/kan-sigmoid-input` a partir de `origin/feat/4times` (979c3ee). Modelo: Opus em tudo (pedido do usuário).

## Objetivos
- KAN recebe `sigmoid(LN(x))`, LN+sigmoide entre camadas, padding fora da grade.
- MLP intacta.

## Workstreams
- WS1 Modelo · dono: `src/models/torch_relu_kan.py` · dep: —
- WS2 Protocolo sequencial · dono: `src/train_sequence.py`, `src/utils/sequence_outputs.py` · dep: WS1 (regra de padding)
- WS3 Testes · dono: `test_kan_input_squash.py`, `test_dataset_blocks.py`, `test_train_sequence.py` · dep: WS1, WS2
- WS4 Validação real · dono: scratchpad (fora do repo) · dep: WS1 (extração de embeddings em paralelo)

## Itens
- [x] (WS1) 1 ReLUKAN — spec §Regras 1-3, 6
  - [x] 1.1 sigmoide após `input_layer_norm`, antes da expansão
  - [x] 1.2 LN não-afim + sigmoide antes de cada `ReLUKANLayer` a partir da 2ª
  - [x] 1.3 padding: default fora da grade; erro se configurado dentro
  - [x] 1.4 `_extra_state` com a versão da entrada (checkpoint antigo falha em strict)
  - [x] 1.5 debug print mostra também o vetor após a sigmoide
- [x] (WS2) 2 Protocolo — spec §Critérios
  - [x] 2.1 `load_plan`: padding por `model_type`
  - [x] 2.2 `SequenceOutputs.on_fit_start`: validação de padding por cabeça; `fill` e `normalization` corretos
  - [x] 2.3 `capture_head`: esperado = `full(padding)` + scatter da entrada roteada
  - [x] 2.4 `result_rows`: coluna `fill` com o valor real
- [x] (WS3) 3 Testes
  - [x] 3.1 novo arquivo de testes da KAN (faixa, padding inerte, gradiente zero no padding, entre camadas, erro de padding, checkpoint antigo, MLP inalterada)
  - [x] 3.2 ajustar testes existentes que esperam zeros fora da janela na KAN
- [ ] (WS4) 4 Validação real
  - [x] 4.1 embeddings WavLM-large do subset BVCC local
  - [ ] 4.2 treino de cabeça: KAN antiga × KAN nova × MLP, mesmo LR do config wavlm (1e-3)

- [x] (REVIEW) 5 Achados do revisor independente
  - [x] 5.1 checkpoint KAN 4x antigo rejeitado com mensagem errada (bloco falhava antes) — erro levantado antes dos filhos
  - [x] 5.2 `wavlm_large_dataset_blocks_lr5e-4_kan.yaml` com padding 0.0 na KAN — trocado por -1.0
  - [x] 5.3 `debug/kan_camada_0/fora_de_todas_bases` contava o padding — mascarado na camada 0
  - [x] 5.4 `strict=False` nao relaxava o checkpoint antigo — documentado; rejeicao sempre (mais seguro)
  - [x] 5.5 padding acima da grade aceito no modelo e recusado no protocolo — agora so abaixo
  - [x] 5.6 fill do results.csv sem padding configurado — helper unico `kan_block_padding`
  - [x] 5.7 camada oculta de largura 1 — ValueError
  - [ ] 5.8 nomenclatura `hist_normalizado` (LN antes da sigmoide) — mantido, so rotulo

## Feedback matrix
### WS1+WS2 — modelo e protocolo
- [auto] unit (arquivo novo + suítes existentes rodáveis) · limiar: verde, sem regressão vs baseline
- [auto] numérico: 0% da entrada fora da grade, padding com gradiente 0, MLP bit-idêntica · limiar: exato
- [auto] contagem de parâmetros inalterada · limiar: igual
### WS4 — comportamento
- [auto] treino curto real: KAN nova vs antiga · limiar: SRCC de validação maior e predições não constantes
- [gated] aceite humano: rodar o protocolo completo no cluster (fora desta máquina)
- fora: diff visual, qualidade-IA (não se aplica)

## Decisões da sessão
- Sigmoide sempre aplicada na KAN (também se `use_layer_norm: false`); entre camadas sempre LN+sigmoide.
- Padding default −1.0 para KAN (regra 3); MLP mantém 0.
- Sem push; commit local na branch.

## Execution log
- 2026-10-22: branch criada; extração de embeddings iniciada.
- Baseline feat/4times: test_dataset_blocks 16/19 (3 stub gerenciador), test_train_sequence 5/8 (3 scipy .statistic).
- Branch: test_kan_input_squash 8/8; test_dataset_blocks 16/19 (mesmos 3 stubs; com ROOT dos fontes originais o teste de checkpoint passa); test_train_sequence 9/9 com shim de scipy.
- MLP: saida bit-identica a feat/4times (mesmos pesos, mesma entrada).
- Review independente: 0 ALTA; 3 MEDIA + 4 BAIXA corrigidos (itens 5.x).
