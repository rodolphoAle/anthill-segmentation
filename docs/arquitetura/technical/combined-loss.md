# Combined Tversky Focal Loss

## Objetivo

Este arquivo define uma função de perda combinada para o treinamento do modelo de segmentação semântica.

A classe `CombinedTverskyFocalLoss` combina três funções de perda:

- Tversky Loss
- Focal Loss
- Lovász Hinge Loss

O objetivo é melhorar o aprendizado do modelo em um cenário de forte desbalanceamento entre as classes `fundo` e `formigueiro`.

---

## Problema Resolvido

Durante o treinamento, o dataset possui muito mais pixels de fundo do que pixels de formigueiro.

Esse desbalanceamento pode fazer o modelo aprender a prever apenas a classe majoritária, ou seja, o fundo.

A loss combinada busca reduzir esse problema usando diferentes estratégias de penalização.

---

## Estratégia Utilizada

### Tversky Loss

A Tversky Loss ajuda a reduzir falsos negativos.

No contexto do projeto, isso significa evitar que o modelo deixe de detectar regiões de formigueiro.

### Focal Loss

A Focal Loss reduz o peso de exemplos fáceis e aumenta o foco em exemplos difíceis.

Ela é útil quando existem muitos pixels de fundo e poucos pixels de formigueiro.

### Lovász Hinge Loss

A Lovász Hinge Loss é opcional e busca melhorar diretamente a métrica IoU.

Essa métrica é importante porque mede a sobreposição entre a máscara prevista e a máscara real.

---

## Fórmula da Loss

```text
total =
    tversky_weight * TverskyLoss
  + lovasz_weight  * LovaszLoss
  + focal_weight   * FocalLoss
```

```text
focal_weight =
    1 - tversky_weight - lovasz_weight
```

---

## Principais Parâmetros

| Parâmetro        | Descrição                                                 |
| ---------------- | --------------------------------------------------------- |
| `tversky_alpha`  | Peso aplicado aos falsos positivos                        |
| `tversky_beta`   | Peso aplicado aos falsos negativos                        |
| `tversky_weight` | Peso da Tversky Loss na loss total                        |
| `lovasz_weight`  | Peso da Lovász Loss na loss total                         |
| `focal_gamma`    | Controla o foco da Focal Loss em exemplos difíceis        |
| `class_weights`  | Pesos por classe usados para lidar com desbalanceamento   |
| `ignore_index`   | Valor da máscara que deve ser ignorado no cálculo da loss |

---

## Fluxo de Execução

1. Valida se a soma dos pesos de Tversky e Lovász não ultrapassa `1.0`.
2. Inicializa a Tversky Loss.
3. Inicializa a Focal Loss.
4. Inicializa a Lovász Loss apenas se `lovasz_weight > 0`.
5. Calcula automaticamente o peso restante da Focal Loss.
6. No método `forward`, calcula a loss total combinada.
7. Retorna um valor escalar usado no processo de backpropagation.

---

## Entrada e Saída

### Entrada

| Nome      | Tipo           | Descrição                                       |
| --------- | -------------- | ----------------------------------------------- |
| `inputs`  | `torch.Tensor` | Saída bruta do modelo, também chamada de logits |
| `targets` | `torch.Tensor` | Máscara real do dataset                         |

### Saída

| Tipo           | Descrição                       |
| -------------- | ------------------------------- |
| `torch.Tensor` | Valor escalar da loss combinada |

---

## Importância no Projeto

Essa classe é importante porque centraliza a estratégia de treinamento usada para melhorar a segmentação dos formigueiros.

Ela ajuda o modelo a:

* lidar com desbalanceamento de classes;
* reduzir falhas na detecção de formigueiros;
* melhorar o Recall;
* melhorar a métrica IoU;
* tornar o treinamento mais estável.

---

## Relação com Outros Arquivos

| Arquivo               | Relação                                       |
| --------------------- | --------------------------------------------- |
| `focal_loss.py`       | Implementa a Focal Loss usada na combinação   |
| `tversky_loss.py`     | Implementa a Tversky Loss usada na combinação |
| `lovasz_loss.py`      | Implementa a Lovász Hinge Loss opcional       |
| `training_service.py` | Usa a loss durante o treinamento do modelo    |
| `metrics.py`          | Avalia se a loss melhorou IoU, Dice e Recall  |

---

## Resumo

A `CombinedTverskyFocalLoss` foi criada para melhorar o treinamento do modelo em um dataset desbalanceado.

Ela combina diferentes funções de perda para atacar problemas específicos:

* a Focal Loss ajuda no desbalanceamento;
* a Tversky Loss melhora o Recall;
* a Lovász Loss contribui para melhorar o IoU.

Com isso, o modelo deixa de favorecer apenas o fundo e passa a aprender melhor a classe formigueiro.
