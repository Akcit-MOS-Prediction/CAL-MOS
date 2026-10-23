# Spec — entrada da ReLU-KAN dentro da grade (sigmoide após o LayerNorm)

Status: aprovada pelo Pedro em 2026-10-22 ("implemente a alternativa da sigmoid após a layernorm exatamente conforme sugerido, somente para a KAN").
Origem do diagnóstico: relatório "Revisão CAL-MOS feat/4times" (https://claude.ai/artifact/4qhdWVJ4W5zKRSWSci8579), seção "Entrada fora da grade da ReLU-KAN".

## Problema
A ReLU-KAN só responde dentro da grade fixa `[-k/g, 1 + k/g]` (g = `relukan_grid`, k = `relukan_k`).
O LayerNorm não-afim da branch `feat/4times` entrega coordenadas ~N(0, 1). Com WavLM-large e g=5, k=2,
~40% das coordenadas caem fora da grade: base zero, gradiente zero. A segunda camada da KAN não tem
normalização e satura quando a saída da primeira cresce. No layout 4x, o padding 0 está dentro da grade
e injeta um viés por dataset.

## Regras (invariantes)
1. **KAN** (`model_type: dynamic_kan`, classe `ReLUKAN`): entrada da cabeça = `sigmoid(LayerNorm(x))`.
   A sigmoide vem ANTES da expansão em blocos.
2. **KAN, entre camadas**: a entrada de toda `ReLUKANLayer` a partir da segunda = `sigmoid(layer_norm(h))`
   (LayerNorm não-afim funcional, sem parâmetros).
3. **KAN, padding dos blocos**: valor estritamente abaixo de `-k/g` (fora da grade), para que posições de
   padding tenham base, contribuição e gradiente exatamente zero. Default `-1.0`; se `-1.0` não estiver
   abaixo de `-k/g`, usar `-k/g - 0.5`. Padding configurado dentro da grade é erro (`ValueError`).
4. **MLP** (`model_type: dynamic`, classe `MLPBase`): inalterada. Só LayerNorm, sem sigmoide, padding 0.
5. Nenhuma mudança na contagem de parâmetros de nenhuma das cabeças.
6. Checkpoints de KAN anteriores à mudança não podem carregar em silêncio com o comportamento novo
   (`strict=True` deve falhar com mensagem clara).

## Critérios de aceite
- KAN: valores que entram em cada `ReLUKANLayer` estão em (0, 1), exceto o padding, que fica < `-k/g`.
- KAN: fração de coordenadas fora da grade = 0 na entrada (excluindo padding) para WavLM e mHuBERT.
- KAN: gradiente dos pesos da 1ª camada nas posições de padding = 0 exatamente.
- MLP: saída bit-idêntica à da `feat/4times` para os mesmos pesos e entradas.
- `train_sequence.py`: padding −1.0 (ou a regra 3) para `dynamic_kan`, 0.0 para `dynamic`.
- `SequenceOutputs`: validação da entrada da cabeça passa para as duas cabeças com a nova regra.
- Treino curto real (WavLM-large congelado, subset BVCC local, cabeça só): KAN nova converge
  (SRCC de validação claramente acima da KAN antiga, predições não constantes).

## Não-objetivos
- Corrigir os demais achados da revisão (scipy `.statistic`, imports `gerenciador`/`grid`, etc.).
- Mudar a MLP, a expansão em blocos em si, ou a contagem de parâmetros.
- Temperatura da sigmoide (`sigmoid(x/T)`): fora de escopo; T=1.

## Pontos de implementação
- `src/models/torch_relu_kan.py` (`ReLUKAN`): forward + padding + extra_state de versão.
- `src/models/dataset_blocks.py`: já aceita `padding_value`; sem mudança esperada.
- `src/train_sequence.py` (`load_plan`): padding por tipo de cabeça.
- `src/utils/sequence_outputs.py`: validação do hook e `fill` reportado.
- Testes: novo `test_kan_input_squash.py` (sem dependência de `gerenciador`/`grid`); ajustar testes que
  afirmam "zeros fora da janela" para a KAN.
