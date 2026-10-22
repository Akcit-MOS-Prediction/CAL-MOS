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
- [ ] (WS1) 1 ReLUKAN — spec §Regras 1-3, 6
  - [ ] 1.1 sigmoide após `input_layer_norm`, antes da expansão
  - [ ] 1.2 LN não-afim + sigmoide antes de cada `ReLUKANLayer` a partir da 2ª
  - [ ] 1.3 padding: default fora da grade; erro se configurado dentro
  - [ ] 1.4 `_extra_state` com a versão da entrada (checkpoint antigo falha em strict)
  - [ ] 1.5 debug print mostra também o vetor após a sigmoide
- [ ] (WS2) 2 Protocolo — spec §Critérios
  - [ ] 2.1 `load_plan`: padding por `model_type`
  - [ ] 2.2 `SequenceOutputs.on_fit_start`: validação de padding por cabeça; `fill` e `normalization` corretos
  - [ ] 2.3 `capture_head`: esperado = `full(padding)` + scatter da entrada roteada
  - [ ] 2.4 `result_rows`: coluna `fill` com o valor real
- [ ] (WS3) 3 Testes
  - [ ] 3.1 novo arquivo de testes da KAN (faixa, padding inerte, gradiente zero no padding, entre camadas, erro de padding, checkpoint antigo, MLP inalterada)
  - [ ] 3.2 ajustar testes existentes que esperam zeros fora da janela na KAN
- [ ] (WS4) 4 Validação real
  - [ ] 4.1 embeddings WavLM-large do subset BVCC local
  - [ ] 4.2 treino de cabeça: KAN antiga × KAN nova × MLP, mesmo LR do config wavlm (1e-3)

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
