# Training & Validation Experiments - UNet Anthill Segmentation

Histórico completo de treinamento e validação do modelo U-Net para detecção de formigueiros em imagens aéreas.

---

## Glossário

| Termo                       | Significado                                                                                  |
| --------------------------- | -------------------------------------------------------------------------------------------- |
| **GT**                | Ground Truth - label anotada manualmente                                                    |
| **mIoU**              | Mean Intersection over Union - média do IoU sobre todas as classes                         |
| **IoU**               | Interseção ÷ União entre predição e GT, por pixel                                      |
| **Dice**              | `2×interseção / (pred + GT)` - equivalente ao F1 de pixels                             |
| **TP / FP / FN / TN** | True/False Positive/Negative no nível de imagem                                             |
| **Precision**         | `TP / (TP + FP)` - dos detectados, quantos eram reais                                     |
| **Recall**            | `TP / (TP + FN)` - dos reais, quantos foram detectados                                    |
| **F1 Score**          | `2 × Precision × Recall / (Precision + Recall)`                                          |
| **Focal Loss**        | Variante da CE que down-pesa exemplos fáceis (γ > 0)                                       |
| **Tversky Loss**      | `1 − TP / (TP + α·FP + β·FN)` - controla peso de FP (α) e FN (β) independentemente |
| **BatchNorm**         | Batch Normalization - normaliza ativações por batch; estabiliza gradiente                 |
| **ConvTranspose2d**   | Upsampling aprendido (vs bilinear fixo)                                                      |
| **pp**                | Pontos percentuais - diferença absoluta entre dois valores %                               |
| **LR**                | Learning Rate - taxa de aprendizado do Adam                                                 |

---

## Setup Geral

### Arquitetura U-Net

| Componente                   | Detalhe                                             |
| ---------------------------- | --------------------------------------------------- |
| Tipo                         | Encoder-decoder com skip connections                |
| Entrada                      | 3 canais (RGB)                                      |
| Saída                       | 2 classes (fundo, formigueiro)                      |
| Profundidade                 | 5 níveis (64 → 128 → 256 → 512 → 1024 filtros) |
| Decoder (Runs 01–04)        | `nn.Upsample(bilinear)` + double conv             |
| Decoder (Run 05+)            | `nn.ConvTranspose2d` + double conv                |
| Normalização (Runs 01–04) | Nenhuma                                             |
| Normalização (Run 05+)     | BatchNorm2d após cada Conv2d                       |
| Ativação                   | ReLU (inplace)                                      |
| Saída final                 | Conv2d 1×1 → 2 logits                             |

```python
import torch
from app.domain.unet import UNet

model = UNet(n_channels=3, n_classes=2)
model.load_state_dict(torch.load("best_model_params.pth", map_location="cuda"))
model.eval()
```

### Dataset

| Item                  | Detalhe                                                             |
| --------------------- | ------------------------------------------------------------------- |
| Fonte                 | Google Drive (streaming) + local (`data/`)                        |
| Pares de treino       | ~2.466 pares RGB + máscara                                         |
| Pares de validação  | 2.466 pares RGB + máscara (593 GT positivas · 1.873 GT negativas) |
| Formato das máscaras | PNG binário (0 = fundo · 1 = formigueiro · 255 = ignorar)        |
| Tipo de imagem        | Ortofoto aérea RGB (drone/satélite)                               |

### Parâmetros fixos em todos os runs

| Parâmetro               | Valor                                                        |
| ------------------------ | ------------------------------------------------------------ |
| Batch size               | 4                                                            |
| Otimizador               | Adam                                                         |
| Learning rate inicial    | 1e-3                                                         |
| LR scheduler             | ReduceLROnPlateau (mode=min, patience variável, factor=0.5) |
| Gradient clipping        | max_norm = 1.0                                               |
| Device                   | CUDA                                                         |
| Preload dataset          | False                                                        |
| Augmentações (Run 01)  | Nenhuma                                                      |
| Augmentações (Run 02+) | Flip H/V · Rotação 15° · Color jitter                   |

---

## Evolução dos Hiperparâmetros por Run

| Parâmetro                     | Run 01       | Run 02       | Run 03         | Run 04                | Run 05                    |
| ------------------------------ | ------------ | ------------ | -------------- | --------------------- | ------------------------- |
| **Data**                 | 04-03        | 04-03→04    | 04-04→05      | 04-05→06             | 04-08→09                 |
| **Épocas executadas**   | ~73          | 100          | ~48            | 64                    | 100                       |
| **Função de loss**     | CrossEntropy | CrossEntropy | Focal (γ=2.0) | Tversky+Focal (50/50) | Tversky+CE (50/50)        |
| **class_weight_anthill** | 10.0         | 6.0          | 4.0            | 6.0                   | 4.0                       |
| **Tversky α / β**      | -           | -           | -             | 0.3 / 0.7             | 0.3 / 0.6                 |
| **focal_loss_gamma**     | 0.0          | 0.0          | 2.0            | 2.0                   | 0.0                       |
| **Scheduler patience**   | 3            | 5            | 5              | 5                     | 5                         |
| **BatchNorm**            | Não         | Não         | Não           | Não                  | **Sim**             |
| **Upsampling decoder**   | Bilinear     | Bilinear     | Bilinear       | Bilinear              | **ConvTranspose2d** |
| **confidence_threshold** | -           | -           | 0.7            | 0.6                   | 0.6                       |
| **min_region_px**        | -           | -           | 200            | 100                   | 100                       |
| **max_region_px**        | -           | -           | 5.000          | 5.000                 | 5.000                     |
| **num_workers**          | 4            | 4            | 4              | 4                     | 5                         |
| **Augmentações**       | Off          | On           | On             | On                    | On                        |
| **Melhor val_loss**      | 0.0756       | 0.1227       | 0.0571         | 0.4592¹              | 0.2045¹                  |
| **Época do best**       | ~33          | 26           | 42             | 28                    | 87                        |

> ¹ Escala Tversky+CE diferente - não comparável com Runs 01–03.

---

## Runs - Detalhamento

### Run 01 - Baseline sem augmentations

**Objetivo:** estabelecer linha de base sem regularização.

**Curva de loss:**

| Época       | train_loss       | val_loss                 | LR                    |
| ------------ | ---------------- | ------------------------ | --------------------- |
| 1            | 0.2785           | 0.1446                   | 1.00e-3               |
| 7            | 0.1261           | 0.1050                   | 1.00e-3               |
| 24           | 0.0935           | 0.0809                   | 5.00e-4               |
| **33** | **0.0842** | **0.0756** ← best | 2.50e-4               |
| 73           | 0.0755           | 0.0768                   | 4.88e-7 ← convergido |

**Análise:** val_loss baixo (0.075) mas métricas reais ruins - modelo **memorizou** o dataset. Ausência de augmentations permitiu decorar padrões específicos sem generalizar.

---

### Run 02 - Augmentations + ajuste de hiperparâmetros

**Objetivo:** introduzir augmentations e reduzir overfitting.

**Curva de loss:**

| Época       | train_loss       | val_loss                 | LR                       |
| ------------ | ---------------- | ------------------------ | ------------------------ |
| 1            | 0.3816           | 0.1939                   | 1.00e-3                  |
| 7            | 0.1521           | 0.1441                   | 1.00e-3                  |
| 21           | 0.1370           | 0.1307                   | 5.00e-4 ← 1ª redução |
| **26** | **0.1291** | **0.1227** ← best | 5.00e-4                  |
| 100          | 0.1218           | 0.1178                   | -                       |

**Pós-processamento (adicionado sem retreinar):** `confidence_threshold=0.7` · `min_region_px=200`

**Análise:** augmentations tornaram o treino mais difícil (val_loss convergiu mais alto, esperado). A adição dos filtros de pós-processamento gerou ganho expressivo em mIoU (+23.7% relativo) sem novo treinamento - confirmando que o modelo produzia muitas predições de baixa confiança.

---

### Run 03 - Focal Loss (γ=2.0)

**Objetivo:** forçar foco nos exemplos difíceis (solo ambíguo perto de formigueiros), reduzindo falsos positivos triviais.

**Curva de loss:**

| Época       | train_loss       | val_loss                 | LR                       |
| ------------ | ---------------- | ------------------------ | ------------------------ |
| 1            | 0.1188           | 0.0778                   | 1.00e-3                  |
| 2            | 2.4146           | 0.0788                   | 1.00e-3 ← spike inicial |
| 9            | 0.0622           | 0.0678                   | 1.00e-3                  |
| 15           | 0.0605           | 0.0705                   | 5.00e-4 ← 1ª redução |
| **42** | **0.0521** | **0.0571** ← best | 2.50e-4                  |
| 48           | 0.0532           | 0.0589                   | 1.25e-4 ← interrompido  |

**Experimento de sensibilidade de threshold (sem retreinar):**

| Configuração     | Precision | Recall | F1    | IoU anthill |
| ------------------ | --------- | ------ | ----- | ----------- |
| t=0.7 · min=200px | 74.9%     | 49.9%  | 59.9% | 20.9%       |
| t=0.6 · min=100px | 51.5%     | 77.6%  | 61.9% | 23.9%       |

**Análise:** val_loss melhor de todos os runs (0.0571). Sensibilidade ao threshold revelou que o modelo já "via" os formigueiros - eles eram preditos com softmax 0.6–0.7, abaixo do threshold 0.7. **Lacuna de polarização** identificada: o modelo não tinha incentivo para empurrar probabilidades de formigueiro para ≥ 0.85.

---

### Run 04 - CombinedTverskyFocalLoss (α=0.3, β=0.7)

**Objetivo:** penalizar FN 2.3× mais que FP via Tversky Loss para forçar probabilidades mais polarizadas.

$$
TL = 1 - \frac{TP}{TP + \alpha \cdot FP + \beta \cdot FN}
$$

> **⚠️ Incidente Run 04-a (Tversky pura):** primeira tentativa colapsou - `val_loss` travou em 0.3735 por 87 épocas. Com desbalanceamento extremo (~5–10% pixels de formigueiro), o termo `α×FP` domina o denominador e o gradiente empurra `anthill_prob → 0`. **Fix:** loss combinada com Focal como âncora.

**Curva de loss** *(escala Tversky+Focal - não comparável com Runs 01–03):*

| Época       | train_loss       | val_loss                 | LR                       |
| ------------ | ---------------- | ------------------------ | ------------------------ |
| 10           | 0.4627           | 0.4650                   | 1.00e-3                  |
| 16           | 0.4543           | 0.4777                   | 5.00e-4 ← 1ª redução |
| **28** | **0.4400** | **0.4592** ← best | 2.50e-4                  |
| 58           | 0.4360           | 0.4656                   | 7.81e-6                  |
| 64           | 0.4348           | 0.4650                   | 3.91e-6 ← interrompido  |

**Análise:** sem colapso ✓, mas **platô precoce na época 28**. val_loss ficou travada em 0.463–0.469 por 36 épocas enquanto train_loss continuou caindo - leve overfitting. Hipótese: ausência de BatchNorm gerava gradientes instáveis impedindo aprendizado mais profundo.

*Validação numérica: a preencher (checkpoint não avaliado formalmente).*

---

### Run 05 - BatchNorm + ConvTranspose2d + Tversky β=0.6

**Objetivo:** resolver instabilidade de gradiente (BatchNorm) e melhorar reconstrução de bordas (ConvTranspose2d).

**Mudanças arquiteturais:**

| Componente     | Run 04                    | Run 05                             |
| -------------- | ------------------------- | ---------------------------------- |
| Normalização | Nenhuma                   | BatchNorm2d após cada Conv2d      |
| Upsampling     | `nn.Upsample(bilinear)` | `nn.ConvTranspose2d` (aprendido) |

**Curva de loss** *(escala Tversky+CE - não comparável com Runs 01–03):*

| Época       | train_loss       | val_loss                 | LR                       |
| ------------ | ---------------- | ------------------------ | ------------------------ |
| 1            | 0.4390           | 0.4652                   | 1.00e-3                  |
| 9            | 0.3449           | 0.2563                   | 1.00e-3                  |
| 16           | 0.2687           | 0.2466                   | 5.00e-4 ← 1ª redução |
| 29           | 0.2115           | 0.2207                   | 2.50e-4                  |
| 57           | 0.1918           | 0.2122                   | 1.56e-5                  |
| **87** | **0.1820** | **0.2045** ← best | 9.77e-7                  |
| 99           | 0.1800           | 0.2720                   | 2.44e-7                  |

**Análise:** convergência contínua até ep.87 - nunca entrou em platô prolongado como no Run 04. Overfitting leve nas últimas épocas (ep.99 val_loss > best). Checkpoint da época 87 é o ideal.

---

## Resultados Consolidados

### Métricas de segmentação (validation_service - média por imagem)

| Run              | Filtros                              | Pixel Acc        | mIoU             | Mean Dice        |
| ---------------- | ------------------------------------ | ---------------- | ---------------- | ---------------- |
| Run 01           | t=0.1%                               | 0.6588           | 0.3718           | 0.4186           |
| Run 02           | sem filtro                           | 0.6568           | 0.3495           | 0.3950           |
| Run 02           | t=0.7 · min=200px                   | 0.6601           | 0.4326           | 0.4739           |
| Run 03           | t=0.7 · min=200px · max=5000px     | 0.6601           | 0.4347           | 0.4754           |
| **Run 05** | **t=0.6 · min=100px · max=5000px** | **0.6605** | **0.4401** | **0.4819** |

### Métricas de detecção por imagem (evaluate_detections - acumulado global)

Dataset: 2.466 imagens · 593 GT positivas · 1.873 GT negativas

| Configuração                   | TP            | FP           | FN           | TN              | Precision       | Recall          | F1              |
| -------------------------------- | ------------- | ------------ | ------------ | --------------- | --------------- | --------------- | --------------- |
| Run 02 · t=0.7 · min=200px     | 0             | 11           | 593          | 1.862           | 0.0%            | 0.0%            | 0.0%            |
| Run 03 · t=0.7 · min=200px     | 296           | 99           | 297          | 1.774           | 74.9%           | 49.9%           | 59.9%           |
| Run 03 · t=0.6 · min=100px     | 460           | 433          | 133          | 1.440           | 51.5%           | 77.6%           | 61.9%           |
| **Run 05 · t=0.6 · min=100px** | **495** | **87** | **98** | **1.786** | **85.1%** | **83.5%** | **84.3%** |

### Métricas de segmentação por pixel (evaluate_detections - acumulado global)

| Configuração                   | Pixel Acc       | IoU fundo       | IoU formigueiro | Dice formigueiro | mIoU            | Mean Dice       |
| -------------------------------- | --------------- | --------------- | --------------- | ---------------- | --------------- | --------------- |
| Run 02 · t=0.7 · min=200px     | 98.6%           | 98.6%           | 0.0%            | 0.0%             | 49.3%           | 49.6%           |
| Run 03 · t=0.7 · min=200px     | 98.7%           | 98.7%           | 20.9%           | 34.5%            | 59.8%           | 66.9%           |
| Run 03 · t=0.6 · min=100px     | -              | -              | 23.9%           | 38.6%            | 61.2%           | 68.9%           |
| **Run 05 · t=0.6 · min=100px** | **98.9%** | **98.9%** | **35.2%** | **52.1%**  | **67.1%** | **75.8%** |

> **Nota sobre diferença de mIoU:** `validation_service` reporta média de IoU *por imagem depois agrega* (~0.44 Run 05), enquanto `evaluate_detections` acumula pixels *globalmente* (~0.67 Run 05). O segundo é mais rigoroso para datasets desbalanceados.

---

## Diagnóstico e Aprendizados

| Problema identificado                          | Evidência                                                       | Solução aplicada                     | Run      |
| ---------------------------------------------- | ---------------------------------------------------------------- | -------------------------------------- | -------- |
| Memorização (overfitting severo)             | val_loss 0.075 mas métricas ruins                               | Augmentations                          | Run 02   |
| Predições de baixa confiança (ruído)       | mIoU +23.7% só com filtros, sem retreinar                       | Filtro de confiança + região mínima | Run 02+F |
| Foco em exemplos triviais (solo limpo)         | Focal Loss reduziu val_loss 0.12 → 0.057                        | Focal Loss γ=2.0                      | Run 03   |
| Lacuna de polarização de probabilidades      | Formigueiros preditos com softmax 0.6–0.7 (abaixo do threshold) | Tversky Loss (penaliza FN)             | Run 04   |
| Tversky pura colapsa com desbalanceamento      | val_loss travado em 0.37 por 87 épocas                          | Loss combinada Tversky+Focal/CE        | Run 04   |
| Platô precoce (ep.28) e gradientes instáveis | val_loss estagnado enquanto train_loss caía                     | BatchNorm2d + ConvTranspose2d          | Run 05   |

---

## Próximos Passos

- [X] Filtros de confiança e região mínima - **+23% mIoU relativo (Run 02 → Run 02+F)**
- [X] Focal Loss (γ=2.0) - **melhor val_loss absoluto: 0.0571 (Run 03)**
- [X] Filtro de região máxima (max 5.000px) - **implementado**
- [X] Experimento de sensibilidade de parâmetros - **Recall +27.7 pp sem retreinar; lacuna de polarização confirmada**
- [X] Run 04 - CombinedTverskyFocalLoss - **platô ep.28; interrompido ep.64**
- [X] BatchNorm + ConvTranspose2d - **implementado no Run 05**
- [X] Run 05 - **100 épocas completas; F1=84.3%; objetivo F1 ≥ 80% atingido**
- [ ] Run 06 - investigar se aumentar dataset ou data augmentation mais agressiva melhora IoU anthill (35.2% ainda é o gap principal)
