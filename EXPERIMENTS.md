# Training & Validation Experiments — UNet Anthill Segmentation

Histórico completo dos experimentos de treinamento e validação do modelo U-Net para detecção de formigueiros em imagens aéreas.

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

| Métrica                   | Valor                    |
| -------------------------- | ------------------------ |
| Pixel Accuracy             | 0.6588 (65.9%)           |
| **mIoU**             | **0.3718 (37.2%)** |
| **Mean Dice**        | **0.4186 (41.9%)** |

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

---

## Comparação entre Runs

| Métrica               | Run 01       | Run 02       | Run 02 + Filtros      | Run 03 + Filtros | Melhor                            |
| ---------------------- | ------------ | ------------ | --------------------- | ---------------- | --------------------------------- |
| Best val_loss (treino) | 0.0756       | 0.1227       | 0.1227 (mesmo modelo) | **0.0571** | **Run 03**                  |
| Pixel Accuracy         | 0.6588       | 0.6568       | 0.6601                | **0.6601** | Run 02+F / Run 03+F               |
| **mIoU**         | 0.3718       | 0.3495       | 0.4326                | **0.4347** | **Run 03+Filtros**          |
| **Mean Dice**    | 0.4186       | 0.3950       | 0.4739                | **0.4754** | **Run 03+Filtros**          |

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
- [ ] Executar `scripts/evaluate_detections.py` contra os resultados de cada run para medir TP/FP/FN/TN reais vs ground-truth do set de validação
- [ ] Adicionar BatchNorm à arquitetura U-Net para melhorar estabilidade e generalização
- [ ] Verificar quantas imagens do set de validação **realmente contêm formigueiro** nas labels (ground truth)
- [ ] Avaliar uso de Dice Loss ou loss combinada (BCE + Dice) para melhorar sobreposição de regiões pequenas
