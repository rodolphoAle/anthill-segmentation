# 4. Training & Validation Experiments - UNet Anthill Segmentation

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
| Profundidade                 | 5 níveis (64 → 128 → 256 → 256 → 1024 filtros) |
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

| Parâmetro                     | Run 01       | Run 02       | Run 03         | Run 04                | Run 05                    | Run 06                         | Run 07                                    | Run 08                                    | Run 09                                         | Run 10                                         | Run 11                                                  |
| ------------------------------ | ------------ | ------------ | -------------- | --------------------- | ------------------------- | ------------------------------ | ----------------------------------------- | ----------------------------------------- | ---------------------------------------------- | ---------------------------------------------- | ------------------------------------------------------- |
| **Data**                 | 04-03        | 04-03→04    | 04-04→05      | 04-05→06             | 04-08→09                 | 04-09→10                      | 04-16 (interrompido)                     | 04-17→18                                 | 04-18 (interrompido)                          | 04-24→04-25                                   | 04-25                                                   |
| **Épocas executadas**   | ~73          | 100          | ~48            | 64                    | 100                       | 100                            | ~50 / 100                                | 100                                       | 51 / 100 (interrompido)                       | **100**                                        | **50**                                                  |
| **Função de loss**     | CrossEntropy | CrossEntropy | Focal (γ=2.0) | Tversky+Focal (50/50) | Tversky+CE (50/50)        | Tversky+CE (70/30)             | Tversky+CE (70/30)                        | Tversky+Focal (70/30)                     | Tversky+Focal (85/15)                          | Tversky+Focal (85/15)                          | **Tversky+Focal+Lovász (50/20/30)**                     |
| **class_weight_anthill** | 10.0         | 6.0          | 4.0            | 6.0                   | 4.0                       | 4.0                            | 4.0                                       | 4.0                                       | 4.0                                            | 4.0                                            | 4.0                                                     |
| **Tversky α / β**      | -           | -           | -             | 0.3 / 0.7             | 0.3 / 0.6                 | 0.3 / 0.8                      | 0.3 / 0.8                                 | 0.3 / 0.8                                 | 0.1 / 0.9                                      | 0.1 / 0.9                                      | **0.3 / 0.7**                                           |
| **focal_loss_gamma**     | 0.0          | 0.0          | 2.0            | 2.0                   | 0.0                       | 0.0                            | 0.0                                       | 2.0                                       | 2.0                                            | 2.0                                            | 2.0                                                     |
| **Scheduler**            | ReduceLR     | ReduceLR     | ReduceLR       | ReduceLR              | ReduceLR                  | ReduceLR                       | **CosineAnnealingLR**                    | **CosineAnnealingLR**                    | **CosineAnnealingLR**                          | **CosineAnnealingLR**                          | **CosineAnnealingLR (T_max=50)**                        |
| **BatchNorm**            | Não         | Não         | Não           | Não                  | **Sim**             | **Sim**                  | **Sim**                                 | **Sim**                                 | **Sim**                                        | **Sim**                                        | **Sim**                                                 |
| **Upsampling decoder**   | Bilinear     | Bilinear     | Bilinear       | Bilinear              | **ConvTranspose2d** | **ConvTranspose2d**      | **ConvTranspose2d**                     | **ConvTranspose2d**                     | **ConvTranspose2d**                            | **ConvTranspose2d**                            | **ConvTranspose2d**                                     |
| **Oversampling**         | -           | -           | -             | -                    | -                        | **3:1 WeightedRandomSampler** | **3:1 WeightedRandomSampler**            | **Removido** (shuffle=True)              | **Removido** (shuffle=True)                   | **Removido** (shuffle=True)                   | **Removido** (shuffle=True)                            |
| **Copy-Paste Aug**       | -           | -           | -             | -                    | -                        | -                             | -                                        | **Sim** (p=0.5, padding=5px)              | **Sim** (p=0.1, padding=5px)                  | **Sim** (p=0.1, padding=5px)                  | **Sim** (p=0.4, gauss σ=1.5px, size 30-5000)            |
| **Augmentações Extras** | -           | -           | -             | -                    | -                        | -                             | -                                        | **RandomRotate90 + ElasticTransform**     | **ElasticTransform (a=25, s=4), no Rotate90** | **ElasticTransform (a=25, s=4), no Rotate90** | **ElasticTransform (a=25, s=4) + flip/rot90 do doador** |
| **Filtro tile (treino)** | -           | -           | -             | -                    | -                        | -                             | -                                        | -                                         | -                                             | -                                             | **Drop tiles >70% ignore (1.713 → 8.149)**             |
| **confidence_threshold** | -           | -           | 0.7            | 0.6                   | 0.6                       | 0.5                            | 0.5                                       | **0.4** (testado pós-treino)             | 0.5                                            | 0.5                                            | 0.5                                                     |
| **min_region_px**        | -           | -           | 200            | 100                   | 100                       | 100                            | 100                                       | 100                                       | 100                                            | 100                                            | 100                                                     |
| **max_region_px**        | -           | -           | 5.000          | 5.000                 | 5.000                     | 5.000                          | 5.000                                     | 5.000                                     | 5.000                                          | 5.000                                          | 5.000                                                   |
| **num_workers**          | 4            | 4            | 4              | 4                     | 5                         | 5                              | 5                                         | 5                                         | 5                                              | 5                                              | 5                                                       |
| **Augmentações base**  | Off          | On           | On             | On                    | On                        | On                             | On                                        | On                                        | On                                             | On                                             | On                                                      |
| **Melhor val_loss**      | 0.0756       | 0.1227       | 0.0571         | 0.4592¹              | 0.2045¹                  | 0.2770¹                       | **0.4441¹** ← fracasso     | 0.2708¹                                  | 0.2314¹                                        | **0.2002¹**                                    | **0.6553²**                                             |
| **Época do best**       | ~33          | 26           | 42             | 28                    | 87                        | 36                             | 33                                        | 59                                        | 30                                             | **93**                                         | **44**                                                  |

> ¹ Escala Tversky+CE diferente - não comparável com Runs 01–03.
> ² Escala Tversky+Focal+Lovász diferente — Lovász Hinge Loss adiciona um termo numericamente maior; não comparável com Runs 01–10.

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

### Run 06 - WeightedRandomSampler + Tversky β=0.8

**Objetivo:** maximizar Recall (meta ≥ 95%) aceitando Precision ≥ 75%. Oversampling 3:1 aumenta exposição a exemplos positivos; β=0.8 penaliza FN mais agressivamente que o Run 05.

**Mudanças em relação ao Run 05:**

| Parâmetro              | Run 05 | Run 06                         |
| ---------------------- | ------ | ------------------------------ |
| Tversky β              | 0.6    | **0.8**                        |
| tversky_loss_weight    | 0.5    | **0.7**                        |
| confidence_threshold   | 0.6    | **0.5**                        |
| Oversampling           |       | **3:1 WeightedRandomSampler**  |

**Dataset de treino:** 9.862 pares (oversampler com `num_samples=9.862`, `replacement=True`). Validação: 2.466 pares (inalterada).

**Curva de loss** *(escala Tversky+CE 70/30  β=0.8 eleva baseline da loss vs Run 05):*

| Época        | train_loss       | val_loss                 | LR                       |
| ------------ | ---------------- | ------------------------ | ------------------------ |
| 1            | 0.4799           | 0.5814                   | 1.00e-3                  |
| ~10          | ~0.42            | 0.4259                   | 1.00e-3                  |
| 21           | 0.2741           | 0.4421                   | 5.00e-4 ← 1ª redução  |
| 26           | 0.2548           | 0.3926                   | 2.50e-4                  |
| 31           | 0.2488           | 0.3526                   | 2.50e-4                  |
| **36**       | **0.2402**       | **0.2770** ← best        | 2.50e-4                  |
| 72           | 0.2260           | 0.3420                   | 3.91e-6                  |
| 100          | 0.2249           | 0.3944                   | 2.44e-7 ← convergido   |

**Análise:** best checkpoint salvo na época 36; após isso val_loss oscilou amplamente (0.27–0.49) sem melhora  LR decaiu para 2.44e-7 indicando plateau persistente. A alta variância da val_loss é efeito direto do WeightedRandomSampler: cada época expõe o modelo a uma composição upsampled diferente, tornando a curva mais ruidosa. O best val_loss (0.2770) é numericamente superior ao do Run 05 (0.2045)  esperado, pois β=0.8 eleva a penalidade Tversky para FN e tversky_loss_weight=0.7 amplifica esse componente na loss combinada.

**Resultados de validação:**

| Métrica (validation_service)    | Valor  |
| ------------------------------- | ------ |
| Pixel Accuracy                  | 0.6607 |
| mIoU                            | 0.4400 |
| Mean Dice                       | 0.4818 |

| Métrica (evaluate_detections)   | Run 05 | Run 06 | Δ      |
| ------------------------------- | ------ | ------ | ------ |
| TP                              | 495    | 481    | -14    |
| FP                              | 87     | 79     | -8     |
| FN                              | 98     | **112**| **+14**|
| TN                              | 1.786  | 1.794  | +8     |
| Precision                       | 85.1%  | 85.9%  | +0.8pp |
| Recall                          | 83.5%  | **81.1%** | **-2.4pp** |
| F1                              | 84.3%  | 83.4%  | -0.9pp |
| IoU anthill                     | 35.2%  | 34.3%  | -0.9pp |
| Dice anthill                    | 52.1%  | 51.1%  | -1.0pp |

**Diagnóstico:** objetivo de ↑Recall **não atingido**  Recall caiu 2.4 pp. Hipóteses: (1) best checkpoint na época 36 é muito precoce  o modelo ainda não havia convergido para aproveitar o oversampling; (2) o WeightedRandomSampler inflacionou a val_loss variância, fazendo o ReduceLROnPlateau reduzir a LR de 1e-3 para 2.50e-4 já na época 21, congelando o aprendizado cedo; (3) β=0.8 + weight=0.7 combinados podem ter tornado o gradiente Tversky instável antes da epoch de best. A leve melhora de Precision (+0.8 pp) sugere que o threshold=0.5 não foi o problema  o modelo ficou mais conservador, não mais agressivo.

---

### Run 07 - CosineAnnealingLR (tentativa de corrigir Run 06)  FRACASSO

**Objetivo:** resolver o problema do Run 06 (LR congelada precocemente) substituindo `ReduceLROnPlateau` por `CosineAnnealingLR`, um scheduler com calendário fixo imune à variância da val_loss causada pelo WeightedRandomSampler.

**Mudanças em relação ao Run 06:**

| Parâmetro           | Run 06               | Run 07                      |
| ------------------- | -------------------- | --------------------------- |
| Scheduler           | ReduceLROnPlateau    | **CosineAnnealingLR**       |
| T_max               | -                   | 100 (épocas totais)         |
| eta_min             | -                   | 1e-6                        |
| Resto               | Idêntico ao Run 06   | Idêntico ao Run 06          |

**Curva de loss** *(interrompido na época 50):*

| Época   | train_loss | val_loss           | LR        |
| ------- | ---------- | ------------------ | --------- |
| 1       | 0.4835     | 0.5983             | 1.00e-3   |
| 9       | 0.3495     | 0.5201             | 9.80e-4   |
| 12      | 0.3289     | **0.4871**         | 9.65e-4   |
| 17      | 0.3233     | **0.4846**         | 9.30e-4   |
| 26      | 0.3054     | **0.4787**         | 8.42e-4   |
| **33**  | **0.2750** | **0.4441** ← best | 7.55e-4   |
| 47      | 0.2553     | 0.4820             | 5.48e-4   |
| 50      | 0.2500     | 0.4744             | 5.32e-4 ← interrompido |

**Comparação Run 06 vs Run 07 (best checkpoint):**

| Métrica         | Run 06 (ep.36) | Run 07 (ep.33) | Δ          |
| --------------- | -------------- | -------------- | ---------- |
| Best val_loss   | **0.2770**     | **0.4441**     | **+60.3%** |
| Train loss      | 0.2402         | 0.2750         | +14.5%     |
| LR no best      | 2.50e-4        | 7.55e-4        | -         |

**Diagnóstico  por que piorou tanto?**

1. **CosineAnnealingLR não resolveu o problema raiz**: o scheduler não era o gargalo. A verdadeira causa é a **combinação WeightedRandomSampler (3:1) + β=0.8 + weight=0.7**. O sampler **repete** os mesmos 593 exemplos positivos 3× sem adicionar variabilidade real  modelo decora ao invés de generalizar.

2. **Val_loss=0.4441 (Run 07) vs 0.2770 (Run 06)**: o Run 07 é 60% pior que o Run 06. Isso indica que a **sequência de batches** gerada pelo WeightedRandomSampler na época 33 do Run 07 foi particularmente desfavorável. Como o sampler é estocástico (usa `replacement=True`), cada run tem exposição diferente aos dados upsampled.

3. **Hipótese de overfitting precoce**: ambos os runs (06 e 07) salvaram o best nas épocas 33-36 (de 100 totais)  um terço do treinamento. Isso sugere que o oversampling 3:1 está **saturando o modelo cedo demais**, antes de explorar padrões mais sutis.

4. **IoU anthill = 34.3% (Run 06) permanece o gap principal**: aumentar Recall exige melhor segmentação pixel-a-pixel, não apenas ajuste de scheduler. O problema é **qualidade/variabilidade do dataset de treino**.

**Conclusão:** o fracasso do Run 07 prova que **scheduler não é o gargalo**  o problema real é o dataset limitado (593 positivas) e o oversampling que repete dados ao invés de criar variabilidade. WeightedRandomSampler descartado como abordagem.

---

### Run 08 - Copy-Paste Augmentation + Augmentações Agressivas (solução pós-Run 07)

**Objetivo:** substituir WeightedRandomSampler por **variabilidade real** através de (1) Copy-Paste Augmentation que recombina formigueiros reais com fundos diferentes, e (2) augmentações geométricas agressivas (RandomRotate90, ElasticTransform) para criar deformações realistas.

**Mudanças em relação ao Run 07:**

| Parâmetro                  | Run 07                       | Run 08                                                      |
| -------------------------- | ---------------------------- | ----------------------------------------------------------- |
| **WeightedRandomSampler**  | 3:1 (repete dados)           | **Removido**  train_loader com shuffle=True                |
| **Copy-Paste Augmentation**|                             | **Ativo** (p=0.5, extracts connected components, 5px pad)   |
| **RandomRotate90**         |                             | **Adicionado** (p=0.5 para 90°/180°/270°)                  |
| **ElasticTransform**       |                             | **Adicionado** (alpha=50, sigma=5, p=0.3)                   |
| **Focal Loss gamma**       | 0.0                          | **2.0** (reativado)                                         |
| **Tversky loss weight**    | 0.7                          | **0.7** (mantido)                                           |
| **Scheduler**              | CosineAnnealingLR            | **CosineAnnealingLR** (T_max=100, eta_min=1e-6)             |
| **Resto**                  |                             | Tversky α=0.3 / β=0.8, class_weight=4.0                     |

**Implementação técnica:**

1. **Copy-Paste Pipeline** (`app/infrastructure/segmentation_dataset.py`):
   - `_build_positive_index()`: indexa tiles com formigueiros no `__init__`
   - `_extract_anthill_region()`: usa `scipy.ndimage.label` para detectar connected components, extrai região com padding de 5px
   - `_apply_copy_paste()`: seleciona tile negativa, cola formigueiro extraído em posição aleatória com alpha masking, pinta máscara de vermelho (255,0,0) na região colada
   - Aplicado **ANTES** das augmentações joint (flip/rotate) para que o formigueiro colado também seja transformado

2. **Augmentações Agressivas** (`app/service/data_service.py`):
   ```python
   v2.RandomChoice([
       v2.RandomRotation([90]),
       v2.RandomRotation([180]),
       v2.RandomRotation([270])
   ], p=[0.5])  # RandomRotate90
   
   v2.ElasticTransform(
       alpha=settings.aug_elastic_alpha,    # 50.0
       sigma=settings.aug_elastic_sigma      # 5.0
   )  # aplicado com p=0.3
   ```

3. **Configurações** (`app/core/config.py`):
   - `aug_copy_paste: bool = True`
   - `aug_copy_paste_prob: float = 0.5`
   - `aug_random_rotate_90: bool = True`
   - `aug_elastic_transform: bool = True`
   - `use_cosine_scheduler: bool = True`

**Curva de loss** *(100 épocas completas):*

| Época       | train_loss       | val_loss                 | LR                       |
| ------------ | ---------------- | ------------------------ | ------------------------ |
| 1            | 0.4912           | 0.5324                   | 1.00e-3                  |
| 11           | 0.3156           | 0.3756                   | 9.87e-4                  |
| 21           | 0.2561           | 0.3205                   | 9.51e-4                  |
| 31           | 0.2224           | 0.2954                   | 8.91e-4                  |
| 41           | 0.1922           | 0.2860                   | 8.09e-4                  |
| 51           | 0.1716           | 0.2784                   | 7.07e-4                  |
| **59**       | **0.1547**       | **0.2708** ← best        | 6.36e-4                  |
| 68           | 0.1408           | 0.2973                   | 5.16e-4 ← overfitting   |
| 78           | 0.1287           | 0.3156                   | 3.61e-4                  |
| 88           | 0.1189           | 0.3298                   | 2.01e-4                  |
| 100          | 0.1115           | 0.3424                   | 1.00e-6 ← fim do cosine |

**Análise da curva:**

- **Overfitting detectado após época 59**: val_loss subiu de 0.2708 → 0.3424 (+26%) enquanto train_loss continuou caindo de 0.1547 → 0.1115 (-28%)
- **CosineAnnealingLR funcionou conforme esperado**: LR decaiu suavemente de 1e-3 → 1e-6 ao longo de 100 épocas (vs congelamento precoce do ReduceLROnPlateau nos Runs 06/07)
- **Best checkpoint salvo na época 59**: val_loss=0.2708 é 2.2% melhor que Run 06 (0.2770) e **39% melhor** que Run 07 (0.4441)
- **Hipótese do overfitting**: copy-paste + ElasticTransform podem ter criado contextos artificiais demais  modelo decorou combinações específicas ao invés de generalizar padrões de formigueiros

**Resultados de validação (threshold=0.4  mais agressivo que padrão 0.5):**

| Métrica (evaluate_detections)   | Valor       |
| ------------------------------- | ----------- |
| TP                              | 406         |
| FP                              | 44          |
| FN                              | 187         |
| TN                              | 1.829       |
| **Precision**                   | **90.2%**   |
| **Recall**                      | **68.5%**   |
| **F1 Score**                    | **77.9%**   |
| IoU anthill                     | 34.2%       |
| Dice anthill                    | 51.0%       |
| Pixel Accuracy                  | 98.9%       |

**Comparação crítica com baselines:**

| Métrica       | Run 05 (t=0.6) | Run 06 (t=0.5) | Run 08 (t=0.4) | Δ vs Run 05 | Δ vs Run 06 |
| ------------- | -------------- | -------------- | -------------- | ----------- | ----------- |
| **Precision** | 85.1%          | 85.9%          | **90.2%**      | **+5.1pp**  | **+4.3pp**  |
| **Recall**    | **83.5%**      | **81.1%**      | **68.5%**      | **-15.0pp** | **-12.6pp** |
| **F1**        | **84.3%**      | **83.4%**      | **77.9%**      | **-6.4pp**  | **-5.5pp**  |
| FN (perdidos) | 98             | 112            | **187**        | +89         | +75         |
| FP (alarmes)  | 87             | 79             | **44**         | -43         | -35         |
| IoU anthill   | 35.2%          | 34.3%          | 34.2%          | -1.0pp      | -0.1pp      |

**Diagnóstico  por que o Run 08 falhou?**

1. **Recall despencou -15pp vs Run 05**: 187 FN (formigueiros perdidos) vs 98 do Run 05  modelo ficou **excessivamente conservador** mesmo com threshold=0.4 (mais agressivo que o 0.6 do Run 05).

2. **Precision subiu +5pp mas é vitória de Pirro**: FP caiu de 87 → 44 porque o modelo **detecta menos**, não porque detecta melhor. Um classificador que nunca prediz "positivo" tem Precision=0 e Recall=0.

3. **Overfitting após época 59 é sintoma, não causa**: o problema não é treinar 100 épocas  é que as augmentações criaram **padrões artificiais** que não existem na validação:
   - Copy-paste cola formigueiros com bordas nítidas de 5px padding → modelo aprende a procurar "formigueiros recortados"
   - ElasticTransform (alpha=50) cria deformações extremas → modelo aprende texturas distorcidas que não aparecem em imagens reais
   - RandomRotate90 (90°/180°/270°) + RandomRotation(15°) gera orientações que podem não existir no dataset de validação

4. **IoU anthill=34.2% estagnado**: copy-paste não melhorou segmentação pixel-a-pixel das bordas  apenas criou mais exemplos de "formigueiros mal colados".

5. **Threshold=0.4 não recuperou Recall**: no Run 05, threshold=0.6 → Recall=83.5%. Se o modelo do Run 08 fosse equivalente, threshold=0.4 deveria elevar Recall para ~90%. O fato de Recall=68.5% com t=0.4 prova que o modelo **não está aprendendo padrões reais** de formigueiros.

**Lições aprendidas:**

- **Copy-paste não é silver bullet**: criar dados sintéticos só funciona se as combinações forem realistas. Colar formigueiros aleatoriamente em fundos arbitrários pode ensinar o modelo a reconhecer "artefatos de colagem" ao invés de formigueiros.
  
- **ElasticTransform muito agressivo**: alpha=50 pode estar criando deformações irrealistas. Reduzir para alpha=20-30 em próximo run.

- **CosineAnnealingLR validado**: scheduler funcionou perfeitamente  problema não é otimização, é qualidade dos dados de treino.

- **Augmentações devem preservar realismo**: rotações de 90° assumem que formigueiros aparecem em qualquer orientação. Se o dataset de validação tem predominância de orientações específicas (e.g., sempre "apontando norte"), o modelo treinado com rotate90 vai performar mal.

**Conclusão:** Run 08 prova que **augmentações sintéticas mal calibradas pioram generalização**. A queda de 15pp no Recall (-18% relativo) indica que o modelo aprendeu padrões que não existem em cenários reais. Próximo run deve: (1) reduzir agressividade das augmentações (ElasticTransform alpha=30, remover rotate90 completo), (2) testar copy-paste com blending suave (gaussian blur nas bordas) ao invés de alpha masking binário, ou (3) abandonar copy-paste e focar em **coletar mais dados reais** rotulados.

---

### Run 09 - Tversky beta=0.9 + aug mais realistas (interrompido)

**Objetivo:** maximizar Recall com Tversky beta=0.9 e reduzir agressividade das augmentacoes, mantendo CosineAnnealingLR.

**Mudancas em relacao ao Run 08:**

| Parâmetro               | Run 08                                | Run 09                                             |
| ----------------------- | ------------------------------------- | -------------------------------------------------- |
| **Tversky alpha/beta**  | 0.3 / 0.8                             | **0.1 / 0.9**                                      |
| **Tversky loss weight** | 0.7                                   | **0.85**                                           |
| **RandomRotate90**      | On                                    | **Off**                                            |
| **ElasticTransform**    | alpha=50, sigma=5                     | **alpha=25, sigma=4**                              |
| **Copy-Paste**          | On (p=0.5)                            | **On (p=0.1)**                                     |
| **Scheduler**           | CosineAnnealingLR (T_max=100)         | **CosineAnnealingLR (T_max=100)**                  |

**Curva de loss** *(interrompido no inicio da epoca 52):*

| Epoca | train_loss | val_loss | LR       |
| ----- | ---------- | -------- | -------- |
| 1     | 0.5945     | 0.6230   | 1.00e-3  |
| 7     | 0.4318     | 0.3106   | 9.88e-4  |
| 14    | 0.3765     | 0.2404   | 9.52e-4  |
| **30** | **0.3010** | **0.2314** ← best | 7.94e-4  |
| 36    | 0.2998     | 0.2431   | 7.13e-4  |
| 41    | 0.2770     | 0.2392   | 6.40e-4  |
| 45    | 0.2700     | 0.2451   | 5.79e-4  |
| 51    | 0.2511     | 0.2452   | 4.85e-4  |

**Analise da curva:**

- **Best checkpoint salvo na epoca 30**: val_loss=0.2314 (melhor do run ate o stop).
- **Sem melhora consistente apos epoca 30**: val_loss oscilou entre ~0.24 e ~0.39.
- **Treino interrompido no inicio da epoca 52**; ultimo resumo completo foi a epoca 51.

**Resultados de validacao (evaluate_detections):**

| Metrica | Valor |
| ------- | ----- |
| TP | 516 |
| FP | 171 |
| FN | 77 |
| TN | 1702 |
| Precision | 75.1% |
| Recall | 87.0% |
| F1 Score | 80.6% |
| Pixel Accuracy | 98.7% |
| IoU anthill | 29.1% |
| Dice anthill | 45.0% |
| Mean IoU | 63.9% |
| Mean Dice | 72.2% |

**Configuracao de validacao:** threshold=0.5, min_region_px=100, pred_dir=output/validation_results_run9, save_dir=output/evalutaion_run9

---

### Run 10 - Run 09 completo (100 épocas)

**Objetivo:** completar o treinamento do Run 09 que foi interrompido na época 52, verificando se o modelo continua melhorando em épocas mais tardias com CosineAnnealingLR.

**Mudanças em relação ao Run 09:** nenhuma  configuração idêntica. O Run 10 é uma nova execução do zero com os mesmos hiperparâmetros.

**Curva de loss** *(100 épocas completas):*

| Época       | train_loss       | val_loss                  | LR                       |
| ------------ | ---------------- | ------------------------- | ------------------------ |
| 1            | 0.5691           | 0.6534                    | 1.00e-3                  |
| 6            | 0.4162           | 0.2407                    | 9.91e-4                  |
| 17           | 0.3340           | 0.2135                    | 9.30e-4                  |
| 31           | 0.2861           | 0.2231                    | 7.81e-4                  |
| 50           | 0.2479           | 0.3767                    | 5.01e-4                  |
| 65           | 0.2225           | 0.2040                    | 2.74e-4                  |
| **93** | **0.1909** | **0.2002** ← best   | 1.30e-5                  |
| 97           | 0.1992           | 0.2051                    | 3.22e-6                  |
| 100          | 0.1933           | 0.2540                    | 1.00e-6 ← fim do cosine |

**Análise da curva:**

- **Best checkpoint na época 93**: val_loss=0.2002  completar as 100 épocas permitiu encontrar um checkpoint 13.5% melhor que o Run 09 (0.2314 na ep.30)
- **val_loss extremamente volátil**: oscilou entre 0.2002 e 0.5246 ao longo do treino  padrão consistente com Runs 08-09, causado pelas augmentações (copy-paste + ElasticTransform) que geram batches de validação com distribuição instável
- **Sem melhora após época 93**: val_loss subiu de 0.2002 → 0.2540 nas últimas 7 épocas enquanto train_loss continuou caindo (0.1909 → 0.1933), indicando leve overfitting no final do ciclo cosine

**Resultados de validação (validation_service):**

| Métrica          | Valor  |
| ---------------- | ------ |
| Pixel Accuracy   | 0.6601 |
| mIoU             | 0.4398 |
| Mean Dice        | 0.4817 |

**Resultados de validação (evaluate_detections):**

| Métrica (evaluate_detections) | Valor     |
| ----------------------------- | --------- |
| TP                            | 505       |
| FP                            | 105       |
| FN                            | 88        |
| TN                            | 1.768     |
| **Precision**                 | **82.8%** |
| **Recall**                    | **85.2%** |
| **F1 Score**                  | **84.0%** |
| Pixel Accuracy                | 98.8%     |
| IoU anthill                   | 30.3%     |
| Dice anthill                  | 46.5%     |
| Mean IoU                      | 64.5%     |
| Mean Dice                     | 72.9%     |

**Configuração de validação:** threshold=0.5, min_region_px=100, pred_dir=output/validation_run10, save_dir=output/evaluation_run10

**Comparação com baselines:**

| Métrica       | Run 05 (t=0.6) | Run 09 (t=0.5) | Run 10 (t=0.5) | Δ vs Run 05 | Δ vs Run 09  |
| ------------- | -------------- | -------------- | -------------- | ----------- | ------------ |
| **Precision** | **85.1%**      | 75.1%          | 82.8%          | -2.3pp      | **+7.7pp**   |
| **Recall**    | 83.5%          | **87.0%**      | **85.2%**      | **+1.7pp**  | -1.8pp       |
| **F1**        | **84.3%**      | 80.6%          | 84.0%          | -0.3pp      | **+3.4pp**   |
| FN (perdidos) | 98             | 77             | 88             | -10         | +11          |
| FP (alarmes)  | 87             | 171            | **105**        | -18         | **-66**      |
| IoU anthill   | **35.2%**      | 29.1%          | 30.3%          | -4.9pp      | +1.2pp       |
| Dice anthill  | **52.1%**      | 45.0%          | 46.5%          | -5.6pp      | +1.5pp       |

---

### Run 11 - Tversky+Focal+Lovász + filtro de tile + copy-paste melhorado

**Objetivo:** recuperar IoU anthill ≥ 38% mantendo F1 ≥ 84%, atacando três frentes em paralelo: (1) rebalanceamento da loss com Lovász Hinge para otimizar IoU diretamente; (2) filtro de tiles dominados por área ignore no treino; (3) melhorias qualitativas no copy-paste (bordas gaussianas, filtro de tamanho do doador, flip/rot90 do doador).

**Mudanças em relação ao Run 10:**

| #  | Categoria       | Parâmetro                  | Run 10                  | Run 11                                              |
| -- | --------------- | -------------------------- | ----------------------- | --------------------------------------------------- |
| 1  | Loss            | Tversky α / β              | 0.1 / 0.9               | **0.3 / 0.7**                                       |
| 2  | Loss            | tversky_loss_weight        | 0.85                    | **0.5**                                             |
| 3  | Loss            | Lovász Hinge Loss          | —                       | **Sim** (peso 0.3)                                  |
| 4  | Loss            | Combinação efetiva         | 0.85·Tversky+0.15·Focal | **0.5·Tversky+0.2·Focal+0.3·Lovász**                |
| 5  | Augmentação     | aug_copy_paste_prob        | 0.1                     | **0.4**                                             |
| 6  | Augmentação     | Bordas suaves (P1)         | Alpha binário           | **Alpha gaussiano** (σ=1.5px)                       |
| 7  | Augmentação     | Limites de tamanho (P2)    | Sem filtro              | **30 ≤ doador ≤ 5.000 px**                          |
| 8  | Augmentação     | Flip + rot90 doador (P3)   | —                       | **Flip H/V + rot90 aleatórios**                     |
| 9  | Dataset         | Filtro de tile no treino   | —                       | **Drop tiles >70% ignore (1.713 / 9.862 = 17.4%)** |
| 10 | Treino          | num_epochs                 | 100                     | **50**                                              |

**Curva de loss** *(escala Tversky+Focal+Lovász — não comparável com Runs 01–10):*

| Época       | train_loss       | val_loss                  | LR                       |
| ------------ | ---------------- | ------------------------- | ------------------------ |
| 1            | 0.8092           | 0.7996                    | 1.00e-3                  |
| 8            | 0.7858           | 0.7843                    | 9.38e-4                  |
| 16           | 0.7033           | 0.7094                    | 7.68e-4                  |
| 28           | 0.6160           | 0.6992                    | 4.07e-4                  |
| 36           | 0.5864           | 0.6676                    | 1.82e-4                  |
| **44** | **0.5526** | **0.6553** ← best   | 3.61e-5                  |
| 48           | 0.5419           | 0.6612                    | 4.94e-6                  |
| 50           | 0.5489           | 0.6687                    | 1.00e-6 ← fim do cosine |

**Análise da curva:**

- **Best checkpoint na época 44** — val_loss=0.6553, dentro da janela de refinamento final do cosine (epochs 35-48)
- **Curva muito mais estável que Run 10** — sem spikes acima de 0.85 (no Run 10 oscilava entre 0.20 e 0.52); efeito visível das melhorias #6-9 (copy-paste limpo) e #9 (filtro de tile)
- **Train_loss decrescente do início ao fim**: 0.8092 → 0.5489 (-32%) sem saturação
- **Train-val gap final**: 0.5489 vs 0.6687 = 0.120 (overfitting leve, dentro do esperado em treinos com augmentação forte)
- **Tempo total**: 6h 24min para 50 epochs (~7.7 min/epoch — mais rápido que Run 10 graças ao filtro de tile)

**Resultados de validação (validation_service):**

| Métrica          | Valor  |
| ---------------- | ------ |
| Pixel Accuracy   | 0.6603 |
| mIoU             | 0.4392 |
| Mean Dice        | 0.4800 |

**Resultados de validação (evaluate_detections):**

| Métrica (evaluate_detections) | Valor      |
| ----------------------------- | ---------- |
| TP                            | 422        |
| FP                            | **60**     |
| FN                            | 171        |
| TN                            | **1.813**  |
| **Precision**                 | **87.6%**  |
| **Recall**                    | **71.2%**  |
| **F1 Score**                  | **78.5%**  |
| Pixel Accuracy                | 98.9%      |
| IoU anthill                   | 28.5%      |
| Dice anthill                  | 44.3%      |
| Mean IoU                      | 63.7%      |
| Mean Dice                     | 71.9%      |

**Configuração de validação:** threshold=0.5, min_region_px=100, pred_dir=output/validation_run11, save_dir=output/evaluation_run11

**Comparação com baselines:**

| Métrica       | Run 05 (t=0.6) | Run 10 (t=0.5) | Run 11 (t=0.5) | Δ vs Run 05  | Δ vs Run 10  |
| ------------- | -------------- | -------------- | -------------- | ------------ | ------------ |
| **Precision** | 85.1%          | 82.8%          | **87.6%**      | **+2.5pp**   | **+4.8pp**   |
| **Recall**    | **83.5%**      | **85.2%**      | 71.2%          | **-12.3pp**  | **-14.0pp**  |
| **F1**        | **84.3%**      | **84.0%**      | 78.5%          | **-5.8pp**   | **-5.5pp**   |
| FN (perdidos) | 98             | 88             | **171**        | +73          | +83          |
| FP (alarmes)  | 87             | 105            | **60**         | -27          | -45          |
| IoU anthill   | **35.2%**      | **30.3%**      | 28.5%          | **-6.7pp**   | **-1.8pp**   |
| Dice anthill  | **52.1%**      | **46.5%**      | 44.3%          | **-7.8pp**   | **-2.2pp**   |

**Diagnóstico — metas não atingidas:**

| Critério        | Alvo    | Resultado | Δ          |
| --------------- | ------- | --------- | ---------- |
| IoU anthill     | ≥ 38%   | **28.5%** | **-9.5 pp** |
| F1              | ≥ 84%   | **78.5%** | **-5.5 pp** |
| Recall          | ≥ 84%   | **71.2%** | **-12.8 pp** |
| Precision       | ≥ 84%   | **87.6%** | **+3.6 pp** ✓ |
| val_loss        | ≤ 0.20  | 0.6553²   | n/a (escala diferente) |

**Observações:**

1. **Inversão do trade-off Precision/Recall vs Run 10:** Run 11 entregou a maior Precision do projeto (87.6%) e o menor número de FP de todos os runs (60), mas o Recall caiu para 71.2% — o pior valor desde o Run 03. O modelo ficou **mais conservador**, detectando menos formigueiros porém com mais confiança.

2. **IoU anthill regrediu apesar da Lovász:** 28.5% é o **menor** IoU desde o Run 03 (excluindo runs colapsados). A hipótese de que Lovász Hinge recuperaria a precisão de máscara não se confirmou no agregado pixel-a-pixel — o ganho ocorreu apenas no nível de imagem (Precision).

3. **FN dobrou em relação ao Run 10:** de 88 → 171 (+94%). 171 / 593 GT positivos = **28.8% das imagens com formigueiro foram perdidas**, enquanto Runs 05/06/10 ficavam em 16-19%.

4. **Curva de loss tecnicamente saudável:** decrescente, com best na ep.44 (zona típica do cosine), train-val gap controlado. A trajetória interna do treino foi a mais limpa de todos os runs com Tversky.

5. **Filtro de tile entregou speedup esperado:** ~17% redução de tempo por epoch (de ~9 min/epoch no Run 10 para ~7.7 min/epoch).

**Diagnóstico:**

- **Completar 100 épocas melhorou F1 +3.4 pp vs Run 09** (84.0% vs 80.6%)  a lição principal é que com CosineAnnealingLR o best checkpoint pode aparecer muito tarde (ep.93/100)
- **Precision +7.7 pp vs Run 09** (82.8% vs 75.1%)  FP caiu de 171 → 105 (-38%); as épocas adicionais calibraram melhor a fronteira de decisão
- **F1 quase idêntico ao Run 05** (84.0% vs 84.3%, -0.3 pp): Run 10 tem melhor Recall (+1.7 pp) mas pior Precision (-2.3 pp) e pior IoU anthill (-4.9 pp)
- **IoU anthill = 30.3% vs 35.2% (Run 05)**: segmentação pixel-a-pixel piorou com as augmentações mais agressivas  o modelo detecta mais imagens corretas (Recall), mas segmenta menos precisamente dentro de cada imagem
- **val_loss mais baixo de todos os Runs Tversky+Focal** (0.2002 vs 0.2314 do Run 09 e 0.2708 do Run 08)

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
| Run 06     | t=0.5 · min=100px · max=5000px     | 0.6607           | 0.4400           | 0.4818           |
| Run 10     | t=0.5 · min=100px · max=5000px     | 0.6601           | 0.4398           | 0.4817           |
| Run 11     | t=0.5 · min=100px · max=5000px     | 0.6603           | 0.4392           | 0.4800           |

### Métricas de detecção por imagem (evaluate_detections - acumulado global)

Dataset: 2.466 imagens · 593 GT positivas · 1.873 GT negativas

| Configuração                   | TP            | FP           | FN           | TN              | Precision       | Recall          | F1              |
| -------------------------------- | ------------- | ------------ | ------------ | --------------- | --------------- | --------------- | --------------- |
| Run 02 · t=0.7 · min=200px     | 0             | 11           | 593          | 1.862           | 0.0%            | 0.0%            | 0.0%            |
| Run 03 · t=0.7 · min=200px     | 296           | 99           | 297          | 1.774           | 74.9%           | 49.9%           | 59.9%           |
| Run 03 · t=0.6 · min=100px     | 460           | 433          | 133          | 1.440           | 51.5%           | 77.6%           | 61.9%           |
| **Run 05 · t=0.6 · min=100px** | **495** | **87** | **98** | **1.786** | **85.1%** | **83.5%** | **84.3%** |
| Run 06 · t=0.5 · min=100px     | 481     | 79     | 112    | 1.794     | 85.9%     | 81.1%     | 83.4%     |
| Run 08 · t=0.4 · min=100px     | 406           | 44           | 187          | 1.829           | 90.2%           | 68.5%           | 77.9%           |
| Run 09 · t=0.5 · min=100px     | 516           | 171          | 77           | 1.702           | 75.1%           | 87.0%           | 80.6%           |
| **Run 10 · t=0.5 · min=100px** | **505** | **105** | **88** | **1.768** | **82.8%** | **85.2%** | **84.0%** |
| Run 11 · t=0.5 · min=100px     | 422           | 60           | 171          | 1.813           | 87.6%           | 71.2%           | 78.5%           |

### Métricas de segmentação por pixel (evaluate_detections - acumulado global)

| Configuração                   | Pixel Acc       | IoU fundo       | IoU formigueiro | Dice formigueiro | mIoU            | Mean Dice       |
| -------------------------------- | --------------- | --------------- | --------------- | ---------------- | --------------- | --------------- |
| Run 02 · t=0.7 · min=200px     | 98.6%           | 98.6%           | 0.0%            | 0.0%             | 49.3%           | 49.6%           |
| Run 03 · t=0.7 · min=200px     | 98.7%           | 98.7%           | 20.9%           | 34.5%            | 59.8%           | 66.9%           |
| Run 03 · t=0.6 · min=100px     | -              | -              | 23.9%           | 38.6%            | 61.2%           | 68.9%           |
| **Run 05 · t=0.6 · min=100px** | **98.9%** | **98.9%** | **35.2%** | **52.1%**  | **67.1%** | **75.8%** |
| Run 06 · t=0.5 · min=100px     | 98.9%     | 98.9%     | 34.3%     | 51.1%      | 66.6%     | 75.3%     |
| Run 08 · t=0.4 · min=100px     | 98.9%           | 98.9%           | 34.2%           | 51.0%            | 66.6%           | 75.2%           |
| Run 09 · t=0.5 · min=100px     | 98.7%           | 98.7%           | 29.1%           | 45.0%            | 63.9%           | 72.2%           |
| Run 10 · t=0.5 · min=100px     | 98.8%           | 98.8%           | 30.3%           | 46.5%            | 64.5%           | 72.9%           |
| Run 11 · t=0.5 · min=100px     | 98.9%           | 98.9%           | 28.5%           | 44.3%            | 63.7%           | 71.9%           |

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
| WeightedRandomSampler inflaciona variância da val_loss | best checkpoint na ep.36; ReduceLROnPlateau congelou LR cedo; Recall caiu -2.4 pp vs Run 05 | CosineAnnealingLR (testado no Run 07) | Run 06   |
| WeightedRandomSampler repete dados ao invés de criar variabilidade | Run 07 (CosineAnnealing) val_loss=0.4441 vs Run 06 (ReduceLR) val_loss=0.2770  piorou 60%; ambos best na ep.33-36 | **Descartado**: oversampling só repete 593 exemplos sem generalização; necessário aumentar dataset ou augmentations diferenciais | Run 07   |
| Dataset limitado (593 GT positivas) + IoU anthill=34% | 112 FN (formigueiros perdidos); segmentação pixel-a-pixel imprecisa | Aumentar variabilidade real: augmentations diferenciais para positivos ou pseudo-labeling | Run 06/07|
| Copy-paste + augmentações agressivas criam padrões irrealistas | Recall despencou -15pp vs Run 05 (68.5% vs 83.5%); 187 FN vs 98; threshold=0.4 não recuperou Recall; modelo excessivamente conservador | **Falhou**: ElasticTransform alpha=50 muito agressivo; copy-paste com bordas nítidas ensina "artefatos de colagem"; RandomRotate90 pode criar orientações inexistentes na validação; reduzir alpha → 20-30, adicionar gaussian blur nas bordas do copy-paste, ou **coletar dados reais** | Run 08   |
| Tversky beta=0.9 + aug mais realistas estabilizou val_loss cedo | Best val_loss=0.2314 na epoca 30; avaliacao: Precision 75.1%, Recall 87.0%, F1 80.6 | Meta de Recall 99% nao atingida; testar threshold menor e rever pos-processamento | Run 09   |
| Run 09 interrompido na ep.52; best checkpoint pode aparecer muito tarde com CosineAnnealingLR | Run 10 (mesmos params, 100 épocas) encontrou best na ep.93  13.5% melhor val_loss (0.2002 vs 0.2314); F1=84.0% quase igual ao Run 05 (84.3%) | IoU anthill ainda 4.9 pp abaixo do Run 05 (30.3% vs 35.2%)  segmentação pixel-a-pixel regrediu com augmentações agressivas | Run 10   |
| Lovász Hinge + Tversky β=0.7 + filtro de tile + copy-paste melhorado (P1+P2+P3) **não atingiu** as metas planejadas | Best val_loss=0.6553 na ep.44 (curva tecnicamente saudável); Precision=87.6% (maior do projeto), FP=60 (menor do projeto), mas Recall=71.2% (-14 pp vs Run 10), F1=78.5% e IoU anthill=28.5% | Inversão do trade-off: modelo ficou conservador (FN dobrou para 171); Lovász não recuperou IoU como hipotetizado; metas IoU≥38% e F1≥84% não atingidas | Run 11   |

---

## Próximos Passos

- [X] Filtros de confiança e região mínima - **+23% mIoU relativo (Run 02 → Run 02+F)**
- [X] Focal Loss (γ=2.0) - **melhor val_loss absoluto: 0.0571 (Run 03)**
- [X] Filtro de região máxima (max 5.000px) - **implementado**
- [X] Experimento de sensibilidade de parâmetros - **Recall +27.7 pp sem retreinar; lacuna de polarização confirmada**
- [X] Run 04 - CombinedTverskyFocalLoss - **platô ep.28; interrompido ep.64**
- [X] BatchNorm + ConvTranspose2d - **implementado no Run 05**
- [X] Run 05 - **100 épocas completas; F1=84.3%; objetivo F1 ≥ 80% atingido**
- [X] Run 06 - WeightedRandomSampler 3:1 + Tversky β=0.8 + threshold=0.5  **Precision=85.9% / Recall=81.1% / F1=83.4%  objetivo Recall ≥ 95% NÃO atingido**
- [X] Run 07 - CosineAnnealingLR para corrigir scheduler do Run 06  **FRACASSO: val_loss=0.4441 (+60% vs Run 06); WeightedRandomSampler descartado**
- [X] Run 08 - Copy-Paste Augmentation + RandomRotate90 + ElasticTransform  **FRACASSO: Recall=68.5% (-15pp vs Run 05); FN=187 (+91% vs Run 05); augmentações irrealistas pioraram generalização**
- [X] Run 09 - Tversky β=0.9 + aug mais realistas  **interrompido na epoca 52; best val_loss=0.2314 (ep.30); Precision=75.1% / Recall=87.0% / F1=80.6%**
- [X] Run 10 - Run 09 completo (100 épocas)  **best val_loss=0.2002 (ep.93); Precision=82.8% / Recall=85.2% / F1=84.0%; IoU anthill=30.3%**
- [X] Run 11 - Rebalanceamento de loss (Tversky β=0.7 + Lovász Hinge) + filtro de tile + copy-paste com bordas gaussianas, filtro de tamanho e flip/rot90 do doador  **metas NÃO atingidas: Precision=87.6% (maior do projeto), Recall=71.2%, F1=78.5%, IoU anthill=28.5% (alvo era ≥ 38%); FN=171 (dobrou vs Run 10)**

---

## Plano Run 11  Recuperar IoU sem perder Recall

**Contexto:** Run 10 atingiu F1=84.0% (igualando Run 05) mas com **IoU anthill regressando 4.9 pp** (35.2% → 30.3%). A causa raiz é a combinação Tversky β=0.9 + class imbalance extremo (anthills <1% dos pixels): o modelo aprende a *detectar* formigueiros mas não a *delinear* suas bordas com precisão. FP > FN (105 vs 88) confirma que o modelo está "trigger-happy" demais.

**Hipótese central:** três frentes simultâneas atacam o platô do Run 10  (1) **rebalanceamento da loss** suavizando β=0.9→0.7 e adicionando **Lovász Hinge Loss** (otimiza IoU diretamente, enquanto Tversky/Focal otimizam classificação por pixel apenas), recuperando a precisão de máscara perdida; (2) **limpeza do dataset de treino** descartando tiles ≥70% pixels ignore, reduzindo o número de gradientes "vazios" por epoch; (3) **melhoria qualitativa da copy-paste augmentation**, eliminando os artefatos de colagem que o modelo aprendia como atalho no Run 10.

### Mudanças em relação ao Run 10

| #  | Categoria       | Parâmetro                  | Run 10                      | Run 11                              | Justificativa                                                  |
| -- | --------------- | -------------------------- | --------------------------- | ----------------------------------- | -------------------------------------------------------------- |
| 1  | Loss            | Tversky α                  | 0.1                         | **0.3**                             | Devolver penalidade de FP para reduzir alarmes falsos          |
| 2  | Loss            | Tversky β                  | 0.9                         | **0.7**                             | β=0.9 era exagero  Run 05 com 0.6 tinha IoU=35.2%             |
| 3  | Loss            | tversky_loss_weight        | 0.85                        | **0.5**                             | Reduzir peso de Tversky para abrir espaço para Lovász          |
| 4  | Loss            | Lovász Hinge Loss          |                            | **Sim** (peso 0.3)                  | Otimiza IoU diretamente (não via proxy)                        |
| 5  | Loss            | Combinação efetiva         | 0.85·Tversky + 0.15·Focal   | **0.5·Tversky + 0.2·Focal + 0.3·Lovász** | Loss tripla balanceada                                    |
| 6  | Augmentação     | aug_copy_paste_prob        | 0.1                         | **0.4**                             | 0.1 só gerava ~60 sintéticos/epoch; 0.4 gera ~240              |
| 7  | Augmentação     | **Bordas suaves (P1)**     | Alpha binário               | **Alpha gaussiano** (σ=1.5px)       | Elimina silhueta recortada; modelo deixa de aprender o atalho de borda |
| 8  | Augmentação     | **Limites de tamanho (P2)** | Sem filtro                  | **30 ≤ doador ≤ 5.000 px**          | Rejeita ruído de 1-2px e mask errors gigantes (>10% da tile)   |
| 9  | Augmentação     | **Flip + rot90 doador (P3)** |                            | **Flip H/V + rot90 aleatórios**     | ~600 doadores → ~4.800 variantes sem custo de I/O              |
| 10 | Dataset         | Filtro de tile no treino   |                            | **Drop tiles >70% ignore**          | Tiles com 70-95% borda branca não dão sinal  só desperdiçam compute |
| 11 | Treino          | num_epochs                 | 100                         | **50**                              | Run 10 atingiu best na ep.93; com loss melhor deve convergir antes |
|   | Resto           | Arquitetura / scheduler    |                            | Idêntico ao Run 10                  | Mesma U-Net, CosineAnnealingLR, val dataset intacto            |

> ⚠️ **Comparabilidade preservada:** o filtro de tiles (#10) é aplicado **apenas no dataset de treino**. O val_dataset permanece com as mesmas 2.466 imagens dos Runs 05–10, garantindo comparação direta de métricas.

#### Detalhamento das melhorias de copy-paste (#7, #8, #9)

**Diagnóstico do Run 10:** previews da augmentação revelaram três falhas qualitativas que explicam parte do platô em IoU:

1. **Silhueta recortada (bordas duras)**  alpha binário criava transição abrupta de 1 pixel entre doador e fundo. O modelo identificava esse padrão e o usava como atalho ("se há borda nítida vermelha-com-pixel-vizinho-aleatório, então é formigueiro"), inflando Recall mas degradando IoU porque a fronteira aprendida é o artefato de colagem, não a fronteira real do formigueiro.

2. **Doadores anômalos**  sem filtro de tamanho, componentes conexos de 1-2 pixels (ruído de label) e mask errors espalhados por crop rows (>5.000 px) eram colados. O segundo caso é especialmente nocivo: ensina ao modelo um "shape prior" errado.

3. **Diversidade limitada**  os ~600 tiles positivos eram amostrados sempre na mesma orientação. Com p=0.4 e 50 epochs, cada doador apareceria ~33× sem variação geométrica → memorização ao invés de generalização.

**Soluções implementadas em [app/infrastructure/segmentation_dataset.py](app/infrastructure/segmentation_dataset.py):**

| Item | Local                                                                | Mudança                                                                     |
| ---- | -------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| P1   | `_apply_copy_paste()` linhas 217+                                    | `ndimage.gaussian_filter(mask, σ=1.5)` antes do blend RGB; label permanece binário |
| P2   | `_extract_anthill_region()` linhas 199+                              | Filtra `valid_components` por área ∈ [30, 5000] px antes do `random.choice` |
| P3   | `_apply_copy_paste()` início do bloco                                | `np.fliplr` (p=0.5) + `np.flipud` (p=0.5) + `np.rot90(k∈{0,1,2,3})` no crop |

**Princípio chave de P1:** o blending gaussiano afeta **somente o RGB visível**; a máscara de label permanece binária (hard ground truth). Isso preserva a nitidez do sinal de supervisão enquanto remove o artefato visual de borda  o modelo é forçado a aprender a fronteira *semântica* do formigueiro, não a fronteira *de colagem*.

### Critérios de sucesso

| Métrica         | Run 05 (referência) | Run 10 (atual) | Run 11 (alvo) |
| --------------- | ------------------- | -------------- | ------------- |
| IoU anthill     | 35.2%               | 30.3%          | **≥ 38%**     |
| F1              | 84.3%               | 84.0%          | **≥ 84%**     |
| Recall          | 83.5%               | 85.2%          | ≥ 84%         |
| Precision       | 85.1%               | 82.8%          | ≥ 84%         |
| Best val_loss   | (n/a)               | 0.2002         | **≤ 0.20**    |

Se Run 11 atingir IoU ≥ 38% e F1 ≥ 84%, **encerra a campanha de melhoria de loss/aug** e foca em qualidade do dataset.

### Próximos passos pós-Run 11

- Se Run 11 falhar: **subamostrar negativos** para 1:2 com positivos (parte da Fase 2 não aplicada agora)  muda distribuição de classes mais agressivamente que apenas o filtro de ignore
- Independente do resultado: implementar **TTA (Test-Time Augmentation)** no [app/service/validation_service.py](app/service/validation_service.py)  flip H/V + média dá +1-2 pp IoU sem retreino
- Wire-up do **patch-based training** (já implementado em `model/segmentation_dataset.py`, commit 82a9b1d) no pipeline principal  possivelmente Run 12

---

## Plano de Ação para ↑Recall ≥ 90%

**Meta:** Recall ≥ 90% mantendo Precision ≥ 75% (F1 ~ 82%)

**Gargalos identificados:**
- Dataset limitado: 593 GT positivas (24%) vs 1.873 negativas (76%)
- IoU anthill = 34.3% → segmentação pixel-a-pixel imprecisa (bordas erradas, fragmentação)
- 112 FN (formigueiros perdidos) → modelo ignora casos pequenos/difíceis
- WeightedRandomSampler **não funciona**: repete dados ao invés de criar variabilidade

---

### Fase 1: Quick Wins (custo zero, 10 min)

#### 1A. Testar Run 05 com threshold reduzido
```bash
# Run 05 (melhor checkpoint): β=0.6, weight=0.5, threshold=0.6 → Recall=83.5%
# Reduzir threshold aceita predições menos confiantes

# Testar threshold=0.5
python run_validation.py --weights best_model_params_run5.pth --output-dir output/validation_run5_t05
python run_evaluate.py --pred-dir output/validation_run5_t05 --save-dir output/evaluation_run5_t05

# Se Recall < 90%, testar threshold=0.4
python run_validation.py --weights best_model_params_run5.pth --output-dir output/validation_run5_t04
python run_evaluate.py --pred-dir output/validation_run5_t04 --save-dir output/evaluation_run5_t04
```
**Expectativa:** Recall 88-92%, Precision cai para 75-80%  
**Critério de sucesso:** Se Recall ≥ 90%, **PARAR AQUI** ✅

#### 1B. Testar Run 05 com min_region_px=50 (metade do atual)
```bash
# Atual: min_region_px=100 pode estar filtrando formigueiros pequenos
python run_validation.py --weights best_model_params_run5.pth \
  --output-dir output/validation_run5_min50
# (editar config.py para min_region_px=50 temporariamente)
```
**Expectativa:** +3-5% Recall capturando formigueiros menores

---

### Fase 2: Augmentations Diferenciais (1-2h implementação, 14h GPU)

**Problema**: WeightedRandomSampler repete os mesmos 593 exemplos positivos 3× sem criar variabilidade.  
**Solução**: Aplicar augmentations **mais agressivas** apenas em tiles com formigueiros, criando variações reais.

#### 2A. Implementar dual augmentation pipeline

**Adicionar em `app/service/data_service.py`:**
```python
import albumentations as A

def create_train_transforms_positive():
    """Augmentations FORTES para tiles com formigueiro."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),              # ← 90° completo (vs 15° atual)
        A.Rotate(limit=45, p=0.7),            # ← até 45°
        A.ElasticTransform(alpha=50, sigma=5, p=0.3),  # ← deforma bordas
        A.GridDistortion(p=0.3),              # ← distorce perspectiva
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, p=0.8),
    ])

def create_train_transforms_negative():
    """Augmentations LEVES para tiles sem formigueiro."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),            # ← mantém 15°
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, p=0.5),
    ])

# Modificar create_dataloaders() para aplicar transform condicional
# baseado em has_anthill(index)
```

**Efeito esperado:** Cria 4-6 variações únicas de cada formigueiro (rotações, distorções, perspectivas) → modelo aprende padrões ao invés de decorar tiles específicos.

#### 2B. Configuração do Run 08
```python
# config.py para Run 08
tversky_beta = 0.6              # ← voltar para Run 05 (0.8 foi instável)
tversky_loss_weight = 0.5       # ← voltar para Run 05
use_cosine_scheduler = False    # ← voltar para ReduceLROnPlateau
scheduler_patience = 7          # ← aumentar de 5 para tolerar variância
# REMOVER WeightedRandomSampler (comentar código em data_service.py)
```

**Justificativa:** Augmentations diferenciais substituem o oversampling, criando variabilidade **sem repetir dados**.

---

### Fase 3: Alternativas se Fase 2 falhar (semana seguinte)

#### 3A. Testar Focal Loss pura com class_weight elevado
```python
# config.py
tversky_alpha = 0.0       # desativa Tversky
tversky_beta = 0.0
focal_loss_gamma = 2.5    # vs 2.0 do Run 03
class_weight_anthill = 10.0  # vs 4.0 atual
```
**Razão:** Run 03 (Focal γ=2.0, weight=4) teve melhor val_loss absoluto (0.0571). Aumentar γ e weight pode empurrar Recall.

#### 3B. Mixup para anthill tiles
```python
# Alpha blending de 2 tiles com formigueiro
img_mixed = 0.7 * img1 + 0.3 * img2
mask_mixed = torch.clamp(mask1 + mask2, 0, 1)
```
**Efeito:** Modelo aprende padrões de formigueiros em contextos variados.

#### 3C. Pseudo-labeling com Run 05 (custo: revisão manual)
```bash
# 1. Rodar inferência em imagens não rotuladas
# 2. Filtrar predições de alta confiança (softmax > 0.85)
# 3. Revisar manualmente top-100
# 4. Adicionar ao dataset de treino
```
**Ganho esperado:** +100-200 exemplos positivos validados

#### 3D. Coletar mais dados reais
Rotular 200-300 novos tiles com formigueiros (processo manual, ~20h de trabalho).

---

## Cronograma Executivo

| Fase          | Ação                                   | Tempo          | Probabilidade ↑Recall ≥ 90% |
| ------------- | -------------------------------------- | -------------- | ---------------------------- |
| **Fase 1A**   | Testar threshold=0.5 e 0.4 no Run 05   | 10 min         | **60%** ← começar aqui      |
| **Fase 1B**   | Testar min_region_px=50                | 5 min          | 30%                          |
| **Fase 2**    | Augmentations diferenciais (Run 08)    | 2h + 14h GPU   | **80%**                     |
| **Fase 3A**   | Focal Loss γ=2.5, weight=10            | 14h GPU        | 40%                          |
| **Fase 3B**   | Mixup                                  | 3h + 14h GPU   | 50%                          |
| **Fase 3C**   | Pseudo-labeling                        | 8h revisão     | 70%                          |
| **Fase 3D**   | Coletar dados reais                    | ~20h rotulagem | 90%                          |
