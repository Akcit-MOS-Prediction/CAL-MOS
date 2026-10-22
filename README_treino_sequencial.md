# Treino sequencial com quatro blocos

Entrada única: `src/train_sequence.py`. Protocolo: `config/sequential.yaml`.
Não depende de `grid.py`, `gerenciador.py` ou serviços de debug em segundo plano.
As filas antigas continuam compatíveis; resultados/checkpoints antigos não são removidos.

## Configuração e execução

Edite a lista `configs` em `config/sequential.yaml` com os YAMLs desejados.
Os caminhos nesse arquivo são relativos à pasta `config/`. Pode usar um config de modelo
ou config completo, desde que `model_type` seja `dynamic` (MLP) ou `dynamic_kan` (KAN).
O modelo deve ser suportado pela factory existente. Dimensão do backbone, arquitetura da
head, K/G, pooling, escolha de camada, LR, scheduler e épocas continuam vindo do config.
Um config do mHuBERT com D=768 produz 3072 dimensões; WavLM/W2V-BERT com D=1024 produz 4096.
Não se deve trocar `mlp_input_dim` para 4D: a expansão é feita dentro da head, só uma vez.

Exemplos a partir da raiz do projeto, no ambiente que contém os datasets:

```bash
python src/train_sequence.py --output resultados/sequencial/nova-rodada --gpu 5 --dry-run
python src/train_sequence.py --output resultados/sequencial/nova-rodada --gpu 5
python src/train_sequence.py --output resultados/sequencial/nova-rodada --gpu 5 --resume
```

Para substituir a lista do protocolo na linha de comando:

```bash
python src/train_sequence.py --config config/models/wavlm_large.yaml --output resultados/sequencial/wavlm-mlp --gpu 6
```

`--config` pode ser repetido. Os configs da lista são executados um após o outro na GPU
informada, não em paralelo. `--gpu` é o índice lógico visível no contêiner; `--gpu cpu` é
aceito para testes. Não há lançamento automático nem seleção silenciosa de GPUs.
`--dry-run` valida todos os caminhos/colunas dos quatro datasets e o primeiro áudio de cada
split sem carregar o backbone, iniciar treinamento ou criar a pasta de saída.

`overrides` no protocolo permite sobrescrever parâmetros comuns depois de combinar
dataset e modelo. Opções de modelo/treino prevalecem sobre os defaults do dataset.
Os caminhos dos splits vêm exclusivamente dos quatro YAMLs de dataset.
Os defaults atuais desses YAMLs usam `/workspace/datasets`, disponível nos contêineres.
O W&B é opcional (`wandb.enabled: true`); os CSVs são sempre gravados localmente.

### W&B e debug

O protocolo padrão habilita W&B. Cada etapa tem sua própria run, agrupada pela rodada.
`wandb_run.json` na pasta da tentativa guarda o ID/link; a run registra o config efetivo,
a linhagem do melhor checkpoint e, após o treino, os resultados dos testes daquela etapa.

Os painéis `debug/*` incluem o vetor completo desde o primeiro batch, a tabela com todas
as coordenadas, normas e gradientes por bloco, histogramas antes/depois da LayerNorm,
MOS real × predito na validação completa, gradientes por parâmetro e sinal de saída constante.
Para KAN, registram também a fração de entradas fora de TODAS as bases, mínimo/máximo da
entrada e limites do suporte, por camada, no primeiro batch de cada época.
As capturas de vetores/gradientes são amostrais (primeiro batch/primeiro update da época),
não estatísticas de todo o treino; as predições de validação são do split inteiro.
Além dos gráficos nativos interativos, uma imagem de visão geral é salva em PNG.
Arquivos CSV, JSON e NPZ de debug são enviados para Files da run e preservados localmente.

Os blocos zerados da ENTRADA são verificados exatamente. Isso não garante gradiente zero
nos pesos da KAN associados a esses blocos: a base ReLU-KAN pode ser ativa em x=0.
Esse comportamento é registrado, não alterado pelo debug.

## Protocolo fixo

1. Treino em BVCC.
2. Melhor checkpoint do BVCC → fine-tuning em SingMOS.
3. Melhor checkpoint de SingMOS → fine-tuning em TMHINT.
4. Melhor checkpoint de TMHINT → fine-tuning em BRSpeech.
5. Melhor checkpoint de BRSpeech → fine-tuning novamente em BVCC.

Cada etapa tem seu próprio melhor checkpoint, escolhido SOMENTE na validação conforme
`model_checkpoint.monitor/mode`, nunca pelo teste. Otimizador e scheduler reiniciam em cada
etapa; só os pesos são transferidos, com carregamento estrito.
O backbone permanece congelado/em eval; somente a head MLP/KAN aprende.

Depois do pooling configurado, aplica-se LayerNorm **não afim em D dimensões**, e somente
depois a expansão em quatro blocos contíguos, sem parâmetros treináveis no roteamento:

```text
dataset_id=1 BVCC:     [x, 0, 0, 0]
dataset_id=2 SingMOS:  [0, x, 0, 0]
dataset_id=3 TMHINT:   [0, 0, x, 0]
dataset_id=4 BRSpeech: [0, 0, 0, x]
```

A identidade é a do áudio/dataset avaliado, não a do checkpoint. Esse protocolo substitui
configurações antigas de expansão compacta, `block_width` ou fill=0,1, e rejeita
PowerTransformer. Não aplica nova normalização no vetor 4D nem altera a ativação da KAN.

## Dados salvos

O diretório `--output` contém `results.csv` combinado e uma pasta por config. Cada config
tem seu próprio `results.csv`, `manifest.json`, `progress.json` e cinco pastas numeradas:

```text
01-bvcc/       02-singmos/       03-tmhint/       04-brspeech/       05-bvcc/
```

Em cada etapa:

- `trained.json`: melhor checkpoint, checkpoint anterior e critério de seleção.
- `config.yaml`: configuração efetiva, já resolvida.
- `attempts/<id>/checkpoints/`: melhor checkpoint dessa tentativa.
- `attempts/<id>/epochs.csv`: MSE/SRCC de validação, LR, gradiente do primeiro update,
  mínimo/máximo das predições de validação e flag de saída constante por época.
- `attempts/<id>/head_inputs.csv`: vetor COMPLETO que entrou na primeira camada aprendida,
  primeira amostra do primeiro batch de cada época, com índice, bloco e dataset_id.
- `attempts/<id>/lightning/`: métricas padrão de treino/validação em CSV.
- `evaluations/<dataset>/predictions.csv`: nome do áudio, índice no metadata, MOS real,
  MOS predito e erro quadrático por amostra do teste completo.
- `evaluations/<dataset>/metrics.json`: métricas e hash do metadata utilizado.

Os CSVs usam **separador `;` e vírgula decimal**, UTF-8 com BOM. Nomes de arquivos e
identificadores não têm seus pontos substituídos. Durante treino o vetor real é conferido
contra a cópia normalizada no bloco correto, incluindo zeros exatos nos demais blocos.

Após cada etapa, o melhor checkpoint faz inferência no teste de todos os datasets já
vistos: 1 + 2 + 3 + 4 + 4 = 14 avaliações por config. Não há reamostragem do conjunto de
teste. A referência de cada dataset é sua avaliação de teste imediatamente após seu
PRIMEIRO treino/FT nessa cadeia:

- `esquecimento_srcc = SRCC_referencia - SRCC_atual`;
- `esquecimento_mse = MSE_atual - MSE_referencia`;
- `variacao_mse` repete a diferença acima, sem valor absoluto;
- também são salvos os deltas em relação ao checkpoint da etapa anterior.

Valores negativos indicam melhora. No retorno final a BVCC, a referência continua sendo
a etapa 1, não é sobrescrita. SRCC matematicamente indefinido fica vazio, com flag explícita;
não é convertido silenciosamente em zero. Assim o colapso não parece uma execução saudável.

## Falhas, retomada e limites

`--resume` reutiliza etapas treinadas e avaliações concluídas, validando a configuração,
a linhagem dos checkpoints e o hash do teste. Uma tentativa interrompida antes de gravar
`trained.json` é reiniciada a partir do melhor checkpoint da etapa ANTERIOR, em outra pasta
de tentativa; não é retomada no meio da época. Tentativas anteriores são preservadas.
Se a etapa foi treinada mas o teste falhou, a retomada faz só a avaliação pendente.

Uma falha nunca permite pular um elo da cadeia. Com `continue_on_error: true`, os próximos
configs independentes são tentados; a execução termina com código 1 se qualquer um falhar.
Não reutilize uma pasta de saída com um protocolo/config diferente. Há um lock contra duas
execuções no mesmo destino; após encerramento forçado, só remova `.sequence.lock` depois de
confirmar que o PID registrado não existe mais.

**A limpeza não é uma correção da arquitetura ReLU-KAN.** O colapso por sair da faixa das
bases ainda pode acontecer, especialmente com o LR da execução anterior. O código registra
e avisa quando a validação fica constante, mas não insere sigmoid/clipping/normalização
entre camadas nem muda LR silenciosamente. Alterações assim devem ser experimentos explícitos.

Teste rápido, sem baixar backbone e sem GPU:

```bash
PYTHONPATH=src python -m unittest test_train_sequence -v
```

O teste usa heads reais e Lightning, com backbone/áudio pequenos sintéticos. Confere a
cadeia completa para MLP/KAN, roteamento, checkpoints, backbone congelado, CSVs e retomada;
não é evidência de qualidade/convergência em áudio real.
