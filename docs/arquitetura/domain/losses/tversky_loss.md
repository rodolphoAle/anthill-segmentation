# Tversky Loss

## Objetivo

Este arquivo implementa a `TverskyLoss`, uma função de perda utilizada em segmentação semântica binária.

O objetivo da Tversky Loss é controlar o equilíbrio entre:

- falsos positivos (FP);
- falsos negativos (FN).

---

## Problema Resolvido

Em tarefas de segmentação, especialmente com datasets desbalanceados, o modelo pode:

- detectar regiões inexistentes;
- deixar de detectar regiões importantes.

No projeto, perder um formigueiro é mais crítico do que gerar um falso positivo.

Por isso, a Tversky Loss foi configurada para penalizar mais falsos negativos.

---

## Fórmula

:contentReference[oaicite:0]{index=0}

Onde:

| Termo | Significado |
|---|---|
| TP | True Positives |
| FP | False Positives |
| FN | False Negatives |

---

## Configuração Utilizada

:contentReference[oaicite:1]{index=1}

Essa configuração:

- reduz o peso dos falsos positivos;
- aumenta o peso dos falsos negativos;
- melhora o Recall do modelo.

---

## Principais Parâmetros

| Parâmetro | Descrição |
|---|---|
| `alpha` | Peso aplicado aos falsos positivos |
| `beta` | Peso aplicado aos falsos negativos |
| `smooth` | Evita divisão por zero |
| `ignore_index` | Pixels ignorados no cálculo |

---

## Fluxo de Execução

1. Converte logits em probabilidades usando Softmax.
2. Seleciona probabilidades da classe formigueiro.
3. Remove pixels ignorados.
4. Calcula:
   - True Positives;
   - False Positives;
   - False Negatives.
5. Calcula o índice Tversky.
6. Retorna a loss final.

---

## Entrada e Saída

| Entrada | Tipo | Descrição |
|---|---|---|
| `inputs` | `torch.Tensor` | Logits produzidos pelo modelo |
| `targets` | `torch.Tensor` | Máscara real do dataset |

| Saída | Tipo | Descrição |
|---|---|---|
| Loss | `torch.Tensor` | Valor escalar da Tversky Loss |

---

## Importância no Projeto

A Tversky Loss é importante porque aumenta a capacidade do modelo detectar formigueiros.

Ela reduz o impacto do desbalanceamento entre fundo e formigueiro, melhorando principalmente a métrica Recall.

---

## Relação com Outros Arquivos

| Arquivo | Relação |
|---|---|
| `combined_loss.py` | Utiliza a Tversky Loss na composição da loss total |
| `metrics.py` | Mede Recall, IoU e Dice |
| `training_service.py` | Usa a loss durante o treinamento |

---

## Resumo 

A Tversky Loss foi utilizada para melhorar a detecção de formigueiros.

Ela permite controlar o peso dos erros do modelo, priorizando a redução de falsos negativos e aumentando o Recall da segmentação.