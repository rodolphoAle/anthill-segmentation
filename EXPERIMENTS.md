# Training & Validation Experiments — UNet Anthill Segmentation

Histórico completo dos experimentos de treinamento e validação do modelo U-Net para detecção de formigueiros em imagens aéreas.

---

## Glossário de Siglas e Termos

| Sigla / Termo        | Significado                                                                                                  |
| -------------------- | ------------------------------------------------------------------------------------------------------------ |
| **U-Net**      | Arquitetura de rede neural convolucional encoder-decoder com skip connections, projetada para segmentação de imagens |
| **RGB**        | Red Green Blue — imagem colorida com 3 canais                                                                |
| **GT**         | Ground Truth — label anotada manualmente, usada como referência para avaliação                             |
| **mIoU**       | Mean Intersection over Union — média do IoU calculado sobre todas as classes (fundo + formigueiro)          |
| **IoU**        | Intersection over Union — sobreposição pixel a pixel: área da interseção ÷ área da união entre predição e GT |
| **Dice**       | Dice Coefficient (equivalente ao F1 de pixels) — `2×interseção / (pred + GT)`; mais sensível a regiões pequenas |
| **TP**         | True Positive — imagem COM formigueiro na GT e que o modelo **detectou**                                    |
| **FP**         | False Positive — imagem SEM formigueiro na GT mas que o modelo "detectou" (alarme falso)                    |
| **FN**         | False Negative — imagem COM formigueiro na GT mas que o modelo **não detectou** (formigueiro perdido)       |
| **TN**         | True Negative — imagem SEM formigueiro na GT e que o modelo corretamente não detectou                       |
| **Precision**  | `TP / (TP + FP)` — dos que o modelo disse "tem formigueiro", quantos realmente tinham?                      |
| **Recall**     | `TP / (TP + FN)` — dos que realmente tinham formigueiro, quantos o modelo encontrou?                        |
| **F1 Score**   | `2 × Precision × Recall / (Precision + Recall)` — média harmônica entre Precision e Recall                 |
| **LR**         | Learning Rate — taxa de aprendizado do otimizador Adam                                                       |
| **val_loss**   | Validation Loss — valor da função de perda no conjunto de validação (sem gradient update)                   |
| **train_loss** | Training Loss — valor da função de perda no conjunto de treino durante o forward pass                       |
| **BCE**        | Binary Cross-Entropy — função de perda para classificação binária por pixel                                  |
| **Focal Loss** | Variante da Cross-Entropy que reduz o peso de exemplos fáceis (γ > 0), focando o treino em casos difíceis  |
| **γ (gamma)**  | Parâmetro do Focal Loss — valores maiores aumentam o foco em exemplos difíceis                              |
| **Tversky Loss** | Função de perda derivada do Dice que permite controlar independentemente o peso de FP (α) e FN (β). Com β > α, penaliza mais os FN, otimizando diretamente o Recall |
| **α (alpha)**  | Peso dos FP no denominador da Tversky Loss — menor α = mais tolerante a alarmes falsos                      |
| **β (beta)**   | Peso dos FN no denominador da Tversky Loss — maior β = penaliza mais fortemente formigueiros não detectados |
| **pp**         | Pontos percentuais — diferença absoluta entre dois valores percentuais (ex: 49% → 75% = +26 pp)             |
| **CUDA**       | Compute Unified Device Architecture — plataforma de computação paralela em GPU NVIDIA                       |
| **VRAM**       | Video RAM — memória da GPU usada para armazenar pesos e ativações durante treino/inferência                  |
| **DataLoader** | Componente PyTorch que carrega batches de dados em paralelo durante treino e validação                       |
| **BatchNorm**  | Batch Normalization — normaliza ativações por batch; melhora estabilidade e velocidade de convergência      |
| **softmax**    | Função que converte logits em probabilidades somando 1.0 — usada na camada de saída da U-Net                |
| **argmax**     | Retorna o índice da maior probabilidade — forma mais simples de classificação (equivale a threshold 0.5)     |

---

## Modelo — Arquitetura U-Net

| Componente              | Detalhe                                             |
| ----------------------- | --------------------------------------------------- |
| Arquitetura             | U-Net (encoder-decoder com skip connections)        |
| Canais de entrada       | 3 (RGB)                                             |
| Classes de saída       | 2 (fundo, formigueiro)                              |
| Profundidade do encoder | 5 níveis (64 → 128 → 256 → 512 → 1024 filtros) |
| Decoder                 | Bilinear upsample + double conv em cada nível      |
| Camada de saída        | Conv2d 1×1 → 2 logits                             |
| Normalização          | Nenhuma (sem BatchNorm)                             |
| Ativação              | ReLU (inplace)                                      |

### Carregando os pesos

```python
import torch
from app.domain.unet import UNet

model = UNet(n_channels=3, n_classes=2)
model.load_state_dict(torch.load("best_model_params.pth", map_location="cuda"))
model.eval()
```

---

## Dataset

| Item                  | Detalhe                                                  |
| --------------------- | -------------------------------------------------------- |
| Fonte                 | Google Drive (streaming) + local (`data/`)             |
| Pares de treino       | ~2.466 pares RGB + máscara                              |
| Pares de validação  | 2.466 pares RGB + máscara                               |
| Formato das máscaras | PNG binário (0 = fundo, 1 = formigueiro, 255 = ignorar) |
| Tipo de imagem        | Ortofoto aérea RGB (drone/satélite)                    |

---

## Parâmetros Fixos (ambos os runs)

| Parâmetro                   | Valor                                          |
| ---------------------------- | ---------------------------------------------- |
| Batch size                   | 4                                              |
| Learning rate inicial        | 1e-3                                           |
| Otimizador                   | Adam                                           |
| Função de loss             | CrossEntropyLoss (ponderada, ignore_index=255) |
| Peso classe fundo            | 1.0                                            |
| Gradient clipping (max norm) | 1.0                                            |
| LR scheduler                 | ReduceLROnPlateau (mode=min, factor=0.5)       |
| Device                       | CUDA                                           |
| Workers (DataLoader)         | 4                                              |
| Preload dataset              | False (streaming do disco)                     |

---

## Run 01 — Baseline sem augmentations

**Data:** 2026-04-03
**Checkpoint:** `best_model_params.pth`
**Épocas configuradas:** 100 | **Épocas executadas:** ~73 (parado manualmente após convergência)

### Parâmetros específicos

| Parâmetro              | Valor     |
| ----------------------- | --------- |
| Peso classe formigueiro | 10.0      |
| Scheduler patience      | 3 épocas |
| Horizontal flip         | Off       |
| Vertical flip           | Off       |
| Rotação aleatória    | Off (0°) |
| Color jitter            | Off       |

### Curva de Loss

| Época       | train_loss       | val_loss                 | LR                    |
| ------------ | ---------------- | ------------------------ | --------------------- |
| 1            | 0.2785           | 0.1446                   | 1.00e-3               |
| 3            | 0.1442           | 0.1081                   | 1.00e-3               |
| 7            | 0.1261           | 0.1050                   | 1.00e-3               |
| 24           | 0.0935           | 0.0809                   | 5.00e-4               |
| 29           | 0.0856           | 0.0795                   | 2.50e-4               |
| **33** | **0.0842** | **0.0756** ← best | 2.50e-4               |
| 64           | 0.0756           | 0.0766                   | 1.95e-6               |
| 73           | 0.0755           | 0.0768                   | 4.88e-7 ← convergido |

**Melhor val_loss:** `0.0756` (época ~33)

### Resultados da Validação

**Diretório de saída:** `validation_results/` | **Threshold:** 0.1%

| Métrica            | Valor                    |
| ------------------- | ------------------------ |
| Pixel Accuracy      | 0.6588 (65.9%)           |
| **mIoU**      | **0.3718 (37.2%)** |
| **Mean Dice** | **0.4186 (41.9%)** |

### Análise

- val_loss baixo (0.075) mas métricas reais ruins → modelo **memorizou** o dataset de treino
- Ausência de augmentations permitiu decorar padrões específicos sem generalizar

---

## Run 02 — Com augmentations e hyperparâmetros ajustados

**Data:** 2026-04-03 → 2026-04-04
**Checkpoint:** `best_model_params.pth`
**Épocas configuradas:** 100 | **Épocas executadas:** 100 (completo)
**Final loss:** 0.1218 | **Final val_loss:** 0.1178

### Parâmetros alterados em relação ao Run 01

| Parâmetro              | Run 01 | Run 02         | Motivo                                     |
| ----------------------- | ------ | -------------- | ------------------------------------------ |
| Peso classe formigueiro | 10.0   | **6.0**  | Reduzir falsos positivos                   |
| Scheduler patience      | 3      | **5**    | Evitar redução prematura do LR           |
| Horizontal flip         | Off    | **On**   | Imagens aéreas sem orientação fixa      |
| Vertical flip           | Off    | **On**   | Idem                                       |
| Rotação aleatória    | 0°    | **15°** | Variação angular de drone                |
| Color jitter            | Off    | **On**   | Diferentes condições de voo/iluminação |

### Curva de Loss

| Época       | train_loss       | val_loss                 | LR                    |
| ------------ | ---------------- | ------------------------ | --------------------- |
| 1            | 0.3816           | 0.1939                   | 1.00e-3               |
| 3            | 0.1587           | 0.1480                   | 1.00e-3               |
| 7            | 0.1521           | 0.1441                   | 1.00e-3               |
| 21           | 0.1370           | 0.1307                   | 5.00e-4 ← LR reduziu |
| 22           | 0.1314           | 0.1246                   | 5.00e-4               |
| **26** | **0.1291** | **0.1227** ← best | 5.00e-4               |
| 100          | 0.1218           | 0.1178                   | —                    |

**Melhor val_loss:** `0.1227` (época 26)

### Resultados da Validação

**Diretório de saída:** `validation_results_run02/` | **Threshold:** 0.1%

| Métrica            | Valor                    |
| ------------------- | ------------------------ |
| Pixel Accuracy      | 0.6568 (65.7%)           |
| **mIoU**      | **0.3495 (35.0%)** |
| **Mean Dice** | **0.3950 (39.5%)** |

### Análise

- Augmentations tornaram o treino mais difícil → val_loss convergiu mais alto (0.12 vs 0.075), esperado
- Resultado ligeiramente inferior ao Run 01 — mais falsos positivos apesar da redução de peso da classe
- Problema parece ser estrutural: distribuição diferente entre treino e validação

---

## Run 02 + Filtros de Pós-processamento

**Data:** 2026-04-04
**Checkpoint:** `best_model_params.pth` (mesmo do Run 02 — sem retreinar)
**Diretório de saída:** `validation_results_run02_filtered/`

### Parâmetros de filtragem adicionados

| Parâmetro                       | Valor         | Descrição                                                                          |
| -------------------------------- | ------------- | ------------------------------------------------------------------------------------ |
| `anthill_confidence_threshold` | **0.7** | Pixel marcado como formigueiro somente se softmax ≥ 70% (substituiu argmax simples) |
| `min_anthill_region_px`        | **200** | Grupos de pixels conectados com menos de 200 pixels são descartados como ruído     |

### Resultados da Validação com Filtros

| Métrica            | Sem filtro (Run 02) | Com filtro       | Melhora          |
| ------------------- | ------------------- | ---------------- | ---------------- |
| Pixel Accuracy      | 0.6568              | **0.6601** | +0.003           |
| **mIoU**      | 0.3495              | **0.4326** | **+0.083** |
| **Mean Dice** | 0.3950              | **0.4739** | **+0.079** |

### Análise

- **mIoU +23.7% relativo** e **Mean Dice +19.97% relativo** apenas com pós-processamento, sem retreinar
- O threshold de confiança (0.7) filtrou predições fracas; o filtro de região (200px) removeu fragmentos isolados
- PixelAcc quase inalterada (esperado: o filtro remove regiões pequenas, que têm pouco peso no total de pixels)

---

## Run 03 — Focal Loss + class_weight reduzido

**Data:** 2026-04-04 → 2026-04-05
**Checkpoint:** `best_model_params.pth`
**Épocas configuradas:** 100 | **Épocas executadas:** ~48 (interrompido durante época 49)

### Parâmetros alterados em relação ao Run 02

| Parâmetro              | Run 02           | Run 03                       | Motivo                                                            |
| ----------------------- | ---------------- | ---------------------------- | ----------------------------------------------------------------- |
| Função de loss        | CrossEntropyLoss | **FocalLoss (γ=2.0)** | Foco nos exemplos difíceis (solo ambíguo perto de formigueiros) |
| Peso classe formigueiro | 6.0              | **4.0**                | Reduzir falsos positivos com o Focal Loss compensando             |

### Curva de Loss

| Época       | train_loss       | val_loss                 | LR                                    |
| ------------ | ---------------- | ------------------------ | ------------------------------------- |
| 1            | 0.1188           | 0.0778                   | 1.00e-3                               |
| 2            | 2.4146           | 0.0788                   | 1.00e-3 ← spike inicial Focal        |
| 9            | 0.0622           | 0.0678                   | 1.00e-3                               |
| 15           | 0.0605           | 0.0705                   | **5.00e-4** ← 1ª redução LR |
| 16           | 0.0586           | 0.0663                   | 5.00e-4                               |
| 28           | 0.0539           | 0.0602                   | 5.00e-4                               |
| 30           | 0.0542           | 0.0591                   | 5.00e-4                               |
| 36           | 0.0545           | 0.0596                   | **2.50e-4** ← 2ª redução LR |
| 39           | 0.0527           | 0.0576                   | 2.50e-4                               |
| **42** | **0.0521** | **0.0571** ← best | 2.50e-4                               |
| 48           | 0.0532           | 0.0589                   | **1.25e-4** ← 3ª redução LR |

**Melhor val_loss:** `0.0571` (época 42)

### Resultados da Validação (com filtros de pós-processamento)

**Diretório de saída:** `validation_results_run3/` | **Filtros ativos:** confiança ≥ 0.7 · min 200px · max 5.000px

| Métrica            | Valor                    |
| ------------------- | ------------------------ |
| Pixel Accuracy      | 0.6601 (66.0%)           |
| **mIoU**      | **0.4347 (43.5%)** |
| **Mean Dice** | **0.4754 (47.5%)** |

## Run 04 — CombinedTverskyFocalLoss (α=0.3, β=0.7 + Focal γ=2.0)

**Data:** 2026-04-05 → 2026-04-06
**Checkpoint:** `best_model_params.pth`
**Épocas configuradas:** 100 | **Épocas executadas:** 64 (interrompido manualmente — LR < 4e-6, platô confirmado)

### Motivação

O experimento de sensibilidade de parâmetros no Run 03 revelou que o modelo já detecta ~78% dos formigueiros, mas com probabilidade softmax entre 0.6–0.7 (abaixo do threshold padrão 0.7). Isso indica que o problema não é cegueira do modelo, mas **falta de polarização das probabilidades**: formigueiros verdadeiros ficam na faixa de ambiguidade enquanto o modelo não tem incentivo para empurrar essas predições para ≥ 0.85.

A **Tversky Loss** penaliza FN 2.3× mais que FP, forçando o modelo a aprender representações mais confiantes para a classe formigueiro:

$$TL = 1 - \frac{TP}{TP + \alpha \cdot FP + \beta \cdot FN}$$

> **⚠️ Incidente Run 04-a (Tversky pura):** A primeira tentativa com Tversky Loss pura colapsou na época 1 — `val_loss` travou em **0.3735** por 87 épocas com LR decaindo até 6e-8. Diagnóstico: com desbalanço extremo (~5-10% pixels de formigueiro), o termo `α×FP` domina o denominador no início do treino e o gradiente empurra `anthill_prob → 0` para todos os pixels. Fix: loss combinada ancorada pela Focal Loss.

### Parâmetros alterados em relação ao Run 03

| Parâmetro                    | Run 03            | Run 04                                       | Motivo                                                             |
| ----------------------------- | ----------------- | -------------------------------------------- | ------------------------------------------------------------------ |
| Função de loss              | FocalLoss (γ=2.0) | **CombinedTverskyFocal (50% Tversky + 50% Focal)** | Tversky pura colapsa com desbalanço extremo; Focal ancora o treino |
| Tversky α / β                | —                 | **α=0.3 · β=0.7**                      | FN penalizado 2.3× mais que FP; empurra Recall                   |
| Focal γ (dentro da combined) | 2.0 (autônoma)    | **2.0** (componente da combined)       | Mantém foco em casos difíceis                                     |
| class_weight_anthill         | 4.0               | **6.0**                                | Aplicado ao componente Focal da combined loss                      |
| anthill_confidence_threshold | 0.7               | **0.6**                                | Threshold mais baixo para capturar mais formigueiros               |
| min_anthill_region_px        | 200               | **100**                                | Reduz FN de formigueiros pequenos                                  |

### Parâmetros inalterados

| Parâmetro           | Valor |
| -------------------- | ----- |
| Batch size           | 4     |
| Learning rate        | 1e-3  |
| Otimizador           | Adam  |
| Scheduler patience   | 5     |
| Scheduler factor     | 0.5   |
| Grad clip max norm   | 1.0   |
| Augmentations        | Todas ativas (flip H/V, rot 15°, color jitter) |

### Curva de Loss

> **Nota:** A escala de val_loss **não é comparável** com os runs anteriores — a TverskyLoss contribui com valores em [0, 1] e a FocalLoss com valores ~0.06, resultando em combined_loss ~0.45 no início. Os valores de Run 01–03 eram somente Focal/CE.

| Época       | train_loss       | val_loss                  | LR                                     |
| ------------ | ---------------- | ------------------------- | -------------------------------------- |
| 10           | 0.4627           | **0.4650** ← best   | 1.00e-03                               |
| 14           | 0.4575           | 0.4657                    | 1.00e-03                               |
| 15           | 0.4536           | 0.4774                    | 1.00e-03                               |
| 16           | 0.4543           | 0.4777                    | **5.00e-04** ← 1ª redução LR  |
| 19           | 0.4445           | **0.4646** ← best   | 5.00e-04                               |
| **21** | **0.4435** | **0.4602** ← best   | 5.00e-04                               |
| 26           | 0.4442           | 0.4691                    | 5.00e-04                               |
| 27           | 0.4441           | 0.4687                    | **2.50e-04** ← 2ª redução LR  |
| **28** | **0.4400** | **0.4592** ← best   | 2.50e-04                               |
| 58           | 0.4360           | 0.4656                    | 7.81e-06                               |
| 63           | 0.4341           | 0.4642                    | 7.81e-06                               |
| 64           | 0.4348           | 0.4650                    | **3.91e-06** ← interrompido   |

**Melhor val_loss:** `0.4592` (época 28) — escala combinada, não comparável com runs anteriores

### Análise do treino

- **Sem colapso** ✓ — a CombinedTverskyFocalLoss resolveu o problema da Tversky pura; modelo aprendeu normalmente desde a época 1
- **Convergência rápida**: melhor checkpoint na época 28; a partir daí val_loss ficou travada em 0.463–0.469 com LR decaindo progressivamente
- **Platô confirmado na época ~30**: após o best em ep.28, val_loss não melhorou nas 36 épocas seguintes; scheduler reduziu LR de 2.5e-4 → 7.81e-6 (6 reduções extras) sem progresso
- **train_loss ainda decaindo** (0.4627 ep.10 → 0.4341 ep.63) enquanto val_loss estagnada → leve overfitting nos últimos epochs
- A loss combinada impede comparação direta com runs anteriores; somente métricas de detecção (TP/FP/FN via `evaluate_detections.py`) revelam se o objetivo de Recall ≥ 80% foi atingido

### Resultados da Validação (com filtros de pós-processamento)

*(a preencher — executar `run_validation.py` + `scripts/evaluate_detections.py`)*

---

## Comparação entre Runs

| Métrica               | Run 01 | Run 02 | Run 02 + Filtros      | Run 03 + Filtros | Melhor                   |
| ---------------------- | ------ | ------ | --------------------- | ---------------- | ------------------------ |
| Best val_loss (treino) | 0.0756 | 0.1227 | 0.1227 (mesmo modelo) | **0.0571** | **Run 03**         |
| Pixel Accuracy         | 0.6588 | 0.6568 | 0.6601                | **0.6601** | Run 02+F / Run 03+F      |
| **mIoU**         | 0.3718 | 0.3495 | 0.4326                | **0.4347** | **Run 03+Filtros** |
| **Mean Dice**    | 0.4186 | 0.3950 | 0.4739                | **0.4754** | **Run 03+Filtros** |

---

## Avaliação de Detecção vs Ground Truth

Métricas medidas pelo script `scripts/evaluate_detections.py`, comparando as máscaras de predição geradas por `run_validation.py` diretamente contra as labels RGB do set de validação.

> **Dataset de referência:** 2.466 imagens · **593 GT positivas** (contêm formigueiro) · **1.873 GT negativas**

### Detecção no nível de imagem

| Métrica            | Run 02 (t=0.7, min=200) | Run 03 (t=0.7, min=200) | Variação |
| ------------------- | ----------------------- | ----------------------- | ---------- |
| Pred positivas      | 11                      | 395                     | +384       |
| TP                  | 0                       | **296**           | +296       |
| FP (alarme falso)   | 11                      | 99                      | +88        |
| FN (perdido)        | 593                     | **297**           | −296      |
| TN                  | 1862                    | **1774**          | −88       |
| **Precision** | 0.0%                    | **74.9%**         | +74.9 pp   |
| **Recall**    | 0.0%                    | **49.9%**         | +49.9 pp   |
| **F1 Score**  | 0.0%                    | **59.9%**         | +59.9 pp   |

### Segmentação no nível de pixel

| Métrica            | Run 02 (t=0.7, min=200) | Run 03 (t=0.7, min=200) | Variação         |
| ------------------- | ----------------------- | ----------------------- | ------------------ |
| Pixel Accuracy      | 98.6%                   | **98.7%**         | +0.1 pp            |
| IoU — fundo        | 98.6%                   | **98.7%**         | +0.1 pp            |
| IoU — formigueiro  | 0.0%                    | **20.9%**         | +20.9 pp           |
| Dice — fundo       | 99.3%                   | **99.4%**         | +0.1 pp            |
| Dice — formigueiro | 0.0%                    | **34.5%**         | +34.5 pp           |
| **mIoU**      | 49.3%                   | **59.8%**         | **+10.5 pp** |
| **Mean Dice** | 49.6%                   | **66.9%**         | **+17.3 pp** |

### Análise

- **Run 02 com filtros detectou praticamente nada** (11 predições, 0 TP): o modelo Run 02 produz predições mais fracas e dispersas — o filtro de confiança 0.7 + regiões ≥ 200 px elimina quase tudo, deixando apenas 11 imagens com alguma detecção, todas falsos positivos.
- **Run 03 Precision = 74.9%**: quando o modelo detecta, 3 em cada 4 imagens realmente têm formigueiro — o filtro de confiança está funcionando bem para qualidade.
- **Run 03 Recall = 49.9%**: o modelo ainda perde ~metade dos formigueiros presentes no set de validação — principal gap a resolver em próximos runs.
- **IoU de Anthill (20.9%)** vs mIoU reportado pela validation_service (43.5%): a diferença se deve ao método de cálculo — `evaluate_detections.py` acumula pixels globalmente sobre todo o dataset (mais rigoroso), enquanto `validation_service` faz média de IoU por imagem e depois agrega.

---

### Experimento de Sensibilidade de Parâmetros — Run 03 (t=0.6, min=100px)

Sem retreinar o modelo, apenas reduzindo o threshold de confiança (0.7 → 0.6) e a região mínima (200 → 100 px):

#### Detecção no nível de imagem

| Métrica            | Run 03 (t=0.7, min=200) | Run 03 (t=0.6, min=100) | Variação            |
| ------------------- | ----------------------- | ----------------------- | --------------------- |
| Pred positivas      | 395                     | 893                     | +498                  |
| TP                  | 296                     | **460**           | **+164**        |
| FP (alarme falso)   | 99                      | 433                     | +334                  |
| FN (perdido)        | 297                     | **133**           | **−164**        |
| TN                  | 1774                    | 1440                    | −334                  |
| **Precision** | 74.9%                   | 51.5%                   | −23.4 pp              |
| **Recall**    | 49.9%                   | **77.6%**         | **+27.7 pp**    |
| **F1 Score**  | 59.9%                   | **61.9%**         | +2.0 pp               |

#### Segmentação no nível de pixel

| Métrica            | Run 03 (t=0.7, min=200) | Run 03 (t=0.6, min=100) | Variação         |
| ------------------- | ----------------------- | ----------------------- | ------------------ |
| IoU — formigueiro  | 20.9%                   | **23.9%**         | +3.0 pp            |
| Dice — formigueiro | 34.5%                   | **38.6%**         | +4.1 pp            |
| **mIoU**      | 59.8%                   | **61.2%**         | +1.4 pp            |
| **Mean Dice** | 66.9%                   | **68.9%**         | +2.0 pp            |

#### Análise do experimento

- **+27.7 pp de Recall sem retreinar** confirma que o modelo Run 03 já "vê" a maioria dos formigueiros — eles estão sendo preditos com probabilidade softmax entre 0.6 e 0.7, portanto abaixo do threshold padrão 0.7.
- **Precision caiu de 74.9% → 51.5%**: no limiar 0.6, o modelo passa a incluir regiões de solo avermelhado ou textura ambígua que parecem formigueiros a 60-70% de confiança — falsos positivos que o threshold 0.7 antes filtrava.
- **Conclusão crítica**: o modelo não é cego a formigueiros — ele os detecta, mas não consegue separá-los com confiança suficiente do ruído de fundo. O gap está na **falta de polarização das probabilidades**: formigueiros verdadeiros ficam em 0.6–0.7 enquanto o ideal seria ≥ 0.85–0.90.
- **Solução**: uma função de perda que penalize FN mais pesadamente que FP força o modelo a aprender probabilidades mais polarizadas para a classe formigueiro → **Tversky Loss** (Run 04).

---

## Diagnóstico

Os filtros de pós-processamento mostraram que grande parte dos falsos positivos era ruído de baixa confiança e fragmentos isolados — não uma falha fundamental do modelo.

### Hipóteses restantes

1. **Distribution shift** — imagens de voos, datas, sensores ou condições diferentes entre treino e validação. O modelo aprende padrões que não se transferem.
2. **Dataset pequeno** — poucos exemplos únicos de formigueiros dificultam o aprendizado de bordas e formas precisas, resultando em IoU moderado.

### Próximos passos recomendados

- [X] Filtro de confiança e região mínima — **implementado, +23% mIoU (Run 02→02+F)**
- [X] Focal Loss (γ=2.0) — **implementado no Run 03, melhor val_loss absoluto: 0.0571**
- [X] Filtro de tamanho máximo de região (max 5.000px) — **implementado**
- [X] Executar `scripts/evaluate_detections.py` contra os resultados de cada run — **concluído; Run 03 com t=0.7/min=200: Recall 49.9%, Precision 74.9%, F1 59.9%**
- [X] Verificar quantas imagens do set de validação realmente contêm formigueiro — **593 GT positivas de 2.466 total (24.1%)**
- [X] Experimento de sensibilidade de parâmetros (t=0.6, min=100) — **Recall +27.7 pp sem retreinar; confirmou lacuna de polarização de probabilidades**
- [X] **Run 04 — CombinedTverskyFocalLoss** — **treinado até ep.64; best val_loss=0.4592 (ep.28); platô confirmado; interrompido**
- [ ] Executar `run_validation.py` + `scripts/evaluate_detections.py` com o checkpoint Run 04 para medir Recall/Precision/F1 reais
- [ ] Adicionar BatchNorm à arquitetura U-Net para melhorar estabilidade e generalização
- [ ] Avaliar uso de loss combinada (Tversky + Focal) se Run 04 não atingir Precision-alvo
