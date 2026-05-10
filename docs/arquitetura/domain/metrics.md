# Metrics

## Objetivo

Este arquivo `metrics.py` implementa estruturas responsáveis pelo cálculo e armazenamento das métricas utilizadas na avaliação do modelo de segmentação semântica.

As métricas são utilizadas durante:
- validação;
- avaliação final;
- análise de desempenho do modelo.

---

# Estrutura Principal

O arquivo possui duas classes principais:

| Classe | Responsabilidade |
|---|---|
| `ValidationMetrics` | Métricas simples de validação |
| `EvaluationMetrics` | Métricas completas de avaliação |

---

# ValidationMetrics

Classe utilizada para armazenar métricas agregadas da validação.

---

## Informações armazenadas

| Campo | Descrição |
|---|---|
| `total_images` | Total de imagens processadas |
| `anthill_detections` | Quantidade de detecções |
| `pixel_accuracy` | Acurácia global por pixel |
| `mean_iou` | Média de IoU |
| `mean_dice` | Média de Dice Score |

---

## Métricas por imagem

A classe também armazena métricas individuais:

| Campo | Descrição |
|---|---|
| `per_image_iou` | IoU individual |
| `per_image_dice` | Dice individual |

---

# EvaluationMetrics

Classe responsável pela avaliação completa do modelo.

Acumula:
- métricas de detecção;
- métricas de segmentação;
- métricas por pixel;
- métricas globais.

---

# Avaliação em Dois Níveis

O sistema avalia:

## 1. Detecção por imagem

Verifica se o modelo detectou presença de formigueiro.

---

## 2. Segmentação por pixel

Verifica qualidade da segmentação dos pixels.

---

# Métricas de Detecção

## TP — True Positive

Modelo detectou formigueiro corretamente.

---

## FP — False Positive

Modelo detectou formigueiro onde não existia.

---

## FN — False Negative

Modelo não detectou um formigueiro existente.

---

## TN — True Negative

Modelo corretamente identificou ausência de formigueiro.

---

# Métricas por Pixel

A classe também acumula:

| Campo | Descrição |
|---|---|
| `px_inter` | Interseção |
| `px_union` | União |
| `px_pred_sum` | Pixels previstos |
| `px_gt_sum` | Pixels reais |
| `px_correct` | Pixels corretos |
| `px_total` | Pixels válidos |

---

# update_image_level()

Atualiza métricas de detecção por imagem.

## Objetivo

Determinar:
- Precision;
- Recall;
- F1 Score.

---

# update_pixel_level()

Atualiza métricas de segmentação por pixel.

## Etapas

1. remove pixels ignorados;
2. compara predição e ground truth;
3. calcula interseção;
4. calcula união;
5. acumula estatísticas.

---

# Pixels Ignorados

Pixels com valor:

```text
255
````

são removidos do cálculo.

Esses pixels representam:

* bordas;
* regiões sem anotação;
* áreas inválidas.

---

# Métricas Calculadas

## Pixel Accuracy

PixelAccuracy = \frac{Pixels\ Corretos}{Pixels\ Totais}

---

## Precision

Precision = \frac{TP}{TP + FP}

---

## Recall

Recall = \frac{TP}{TP + FN}

---

## F1 Score

F1 = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}

---

## IoU

IoU = \frac{Interseccao}{Uniao}

---

## Dice Score

Dice = \frac{2 \cdot Interseccao}{Predicao + GroundTruth}

---

# report()

Gera relatório textual completo das métricas.

O relatório inclui:

* estatísticas gerais;
* métricas de detecção;
* métricas de segmentação;
* médias globais.

---

# Classes Utilizadas

| Classe      | Valor |
| ----------- | ----- |
| Fundo       | 0     |
| Formigueiro | 1     |
| Ignorado    | 255   |

Importadas de:

```python
mask_utils.py
```

---

# Fluxo Geral

```text
Predição
    ↓
Comparação com ground truth
    ↓
Atualização das métricas
    ↓
Acúmulo global
    ↓
Relatório final
```

---

# Entrada e Saída

## Entrada

| Entrada      | Tipo         |
| ------------ | ------------ |
| Ground Truth | `np.ndarray` |
| Predição     | `np.ndarray` |

---

## Saída

| Saída     | Descrição                           |
| --------- | ----------------------------------- |
| IoU       | Qualidade da segmentação            |
| Dice      | Similaridade                        |
| Precision | Taxa de acerto                      |
| Recall    | Sensibilidade                       |
| F1        | Equilíbrio entre precision e recall |

---

# Relação com Outros Arquivos

| Arquivo                 | Relação                   |
| ----------------------- | ------------------------- |
| `validation_service.py` | Usa métricas na validação |
| `prediction_service.py` | Gera máscaras previstas   |
| `mask_utils.py`         | Define classes e máscaras |
| `training_service.py`   | Avalia evolução do modelo |

---

# Importância no Projeto

As métricas implementadas neste arquivo foram fundamentais para:

* avaliar qualidade do modelo;
* medir segmentação;
* detectar overfitting;
* comparar experimentos;
* validar melhorias do pipeline.

---

# Resumo 

O `metrics.py` centraliza todas as métricas utilizadas no projeto.

Ele calcula:

* Pixel Accuracy;
* Precision;
* Recall;
* F1 Score;
* IoU;
* Dice Score.

Essas métricas permitiram avaliar a qualidade da detecção e segmentação dos formigueiros pela U-Net.


