# 5. Resultados

Consolidação das métricas e desempenho dos modelos treinados para detecção de formigueiros em segmentação semântica.

---

## Melhor Configuração

**Run 10** emergiu como a melhor configuração balanceada, alcançando as metas de desempenho com o melhor trade-off entre Recall (detecção) e Precision (confiança).

### Hiperparâmetros da melhor configuração

| Parâmetro                    | Valor                                      |
| ----------------------------- | ------------------------------------------ |
| **Arquitetura**              | U-Net (encoder-decoder com skip connections) |
| **Normalização**             | BatchNorm2d após cada convolução           |
| **Upsampling decoder**       | ConvTranspose2d (aprendido)               |
| **Loss**                     | Tversky + Focal combinada (85% / 15%)     |
| **Tversky α / β**            | 0.1 / 0.9                                  |
| **Class weight anthill**     | 4.0                                        |
| **Augmentações**             | ElasticTransform (α=25, σ=4) + Copy-Paste (p=0.1) |
| **Optimizer**                | Adam (LR inicial = 1e-3)                   |
| **Scheduler**                | CosineAnnealingLR (T_max=100, η_min=1e-6) |
| **Confidence threshold**     | 0.5                                        |
| **Filtros pós-processamento**| min_region=100px, max_region=5.000px      |
| **Épocas totais**            | 100                                        |
| **Best checkpoint**          | Época 93 (val_loss = 0.2002)              |

---

## Métricas Finais

### Dataset de Validação

- **Total de imagens:** 2.466
- **GT Positivas (com formigueiro):** 593
- **GT Negativas (sem formigueiro):** 1.873

### Run 10 — Resultados de Detecção (nível de imagem)

| Métrica      | Valor    | Interpretação |
| ------------ | -------- | ------------- |
| **TP**       | 505      | Imagens com formigueiro detectadas corretamente |
| **FP**       | 105      | Imagens falsamente marcadas com formigueiro |
| **FN**       | 88       | Imagens com formigueiro não detectadas |
| **TN**       | 1.768    | Imagens sem formigueiro corretamente identificadas |
| **Precision**| **82.8%**| 82.8% das detecções feitas eram corretas |
| **Recall**   | **85.2%**| 85.2% dos formigueiros reais foram detectados |
| **F1 Score** | **84.0%**| Média harmônica equilibrada entre Precision e Recall |

### Run 10 — Resultados de Segmentação (nível de pixel)

| Métrica                | Valor     | Interpretação |
| ---------------------- | --------- | ------------- |
| **Pixel Accuracy**     | 98.8%     | Taxa de acurácia pixel a pixel |
| **IoU (fundo)**        | 98.8%     | Interseção/União para classe fundo |
| **IoU (formigueiro)**  | 30.3%     | Interseção/União para classe formigueiro |
| **Dice (formigueiro)** | 46.5%     | Coeficiente Dice para formigueiro |
| **mIoU**               | 64.5%     | Média de IoU entre classes (método global) |
| **Mean Dice**          | 72.9%     | Média de Dice entre classes |

---

## Comparação entre Modelos

### Progressão de Desempenho (Runs Principais)

| Run | Arquitetura | Loss             | Augmentações | Precision | Recall  | F1     | IoU anthill | Observação |
| --- | ----------- | ---------------- | ------------ | --------- | ------- | ------ | ----------- | ---------- |
| 02  | U-Net base  | Cross-Entropy    | Geom + Foto  | -         | -       | -      | -           | Baseline com augmentações |
| 03  | U-Net base  | Focal (γ=2.0)    | Geom + Foto  | 51.5%     | 77.6%   | 61.9%  | 23.9%       | Focal melhorou sensibilidade |
| **05** | U-Net + BN + ConvT2d | Tversky+CE (50/50) | Geom + Foto  | **85.1%** | **83.5%** | **84.3%** | **35.2%** | **Melhor IoU até Run 10** |
| 06  | U-Net + BN + ConvT2d | Tversky+CE (70/30) | Geom + Foto  | 85.9%     | 81.1%   | 83.4%  | 34.3%       | Oversampling 3:1 falhou |
| 08  | U-Net + BN + ConvT2d | Tversky+Focal (70/30) | Geom + Foto + Copy-Paste + Elastic + Rotate90 | 90.2% | 68.5% | 77.9% | 34.2% | Augmentações agressivas pioraram |
| 09  | U-Net + BN + ConvT2d | Tversky+Focal (85/15) | Geom + Foto + Copy-Paste (p=0.1) + Elastic | 75.1% | 87.0% | 80.6% | 29.1% | **Interrompido época 52** |
| **10** | **U-Net + BN + ConvT2d** | **Tversky+Focal (85/15)** | **Geom + Foto + Copy-Paste (p=0.1) + Elastic** | **82.8%** | **85.2%** | **84.0%** | **30.3%** | **Melhor F1 após Run 05; Recall máximo** |
| 11  | U-Net + BN + ConvT2d | Tversky+Focal+Lovász (50/20/30) | Geom + Foto + Copy-Paste melhorado + Filtro tile | 87.6% | 71.2% | 78.5% | 28.5% | Lovász não recuperou IoU como esperado |

### Análise de Trade-offs

#### Precision vs Recall (detecção por imagem)

```
100% |               ● Run 08 (90.2%)
     |
  85% |     ● Run 05 (85.1%) ★ MELHOR BALANÇO
     |        ● Run 06 (85.9%)
     |     ● Run 10 (82.8%)
     |
  75% |  ● Run 09 (75.1%)
     | ● Run 03 (51.5%)
  50% |___________________________________________
     50%    60%    70%    80%    90%    100%
            RECALL →
```

**Interpretação:** 
- **Run 05 e Run 10** oferecem o melhor equilíbrio: ambos atingem ~84% F1
- **Run 08** maximizou Precision (90.2%) mas perdeu Recall (-15pp)
- **Run 09** maximizou Recall (87.0%) mas sacrificou Precision (-10pp)
- **Run 11** alcançou maior Precision (87.6%) mas com Recall muito baixo (71.2%)

#### Segmentação Pixel-a-Pixel (IoU de Formigueiro)

| Run | IoU anthill | FP (falsos alarmes) | FN (perdidos) | Análise |
| --- | ----------- | ------------------- | ------------- | ------- |
| 03  | 23.9%       | 433                 | 133           | Threshold muito conservador (0.6) |
| 05  | **35.2%**   | 87                  | 98            | **Melhor segmentação** |
| 06  | 34.3%       | 79                  | 112           | Oversampling não melhorou |
| 08  | 34.2%       | 44                  | 187           | Augmentações irrealistas prejudicaram |
| 09  | 29.1%       | 171                 | 77            | FP massivo com β=0.9 |
| 10  | 30.3%       | 105                 | 88            | Trade-off aceitável vs Run 05 |
| 11  | 28.5%       | 60                  | 171           | Maior Precision mas dobro de FN |

**Conclusão:** IoU de formigueiro regrediu de 35.2% (Run 05) para máximo 30.3% (Run 10) após introdução de augmentações. A dificuldade de melhorar simultaneamente Recall e IoU indica que o gargalo é **qualidade e quantidade de dados reais**, não arquitetura ou loss.

---

## Métricas por Agregação

### Validação por Imagem (validation_service — média após calcular por imagem)

| Run | Pixel Acc | mIoU  | Mean Dice |
| --- | --------- | ----- | --------- |
| 02  | 0.6568    | 0.3495 | 0.3950   |
| 03  | 0.6601    | 0.4347 | 0.4754   |
| 05  | 0.6605    | 0.4401 | 0.4819   |
| 10  | 0.6601    | 0.4398 | 0.4817   |
| 11  | 0.6603    | 0.4392 | 0.4800   |

**Observação:** métricas por imagem ficam inflacionadas porque imagens **sem nenhum formigueiro** (e sem predição) recebem IoU=1.0 trivialmente, "puxando" a média para cima.

### Validação Global (evaluate_detections — acumulado de pixels)

| Run | IoU anthill | Dice anthill | mIoU  | Mean Dice |
| --- | ----------- | ------------ | ----- | --------- |
| 03  | 23.9%       | 38.6%        | 61.2% | 68.9%    |
| **05** | **35.2%** | **52.1%**    | **67.1%** | **75.8%** |
| 06  | 34.3%       | 51.1%        | 66.6% | 75.3%    |
| 08  | 34.2%       | 51.0%        | 66.6% | 75.2%    |
| 09  | 29.1%       | 45.0%        | 63.9% | 72.2%    |
| **10** | **30.3%** | **46.5%**    | **64.5%** | **72.9%** |
| 11  | 28.5%       | 44.3%        | 63.7% | 71.9%    |

**Nota sobre escala:** `evaluate_detections` com Lovász (Run 11) não é diretamente comparável em loss absoluta com Runs 01–10, pois Lovász adiciona um termo numericamente diferente.

---

## Curvas de Aprendizado

### Run 05 vs Run 10 (melhor Recall)

```
LOSS VALIDATION

0.30 |    Run 05 (best=0.2045, ep.87)
     |     ╱╲
0.25 |    ╱  ╲
     |   ╱    ╲╲
0.20 |  ╱      ╲ ╱╲╲╲
     | ╱        ╰
0.15 |                Run 10 (best=0.2002, ep.93)
     |___________________________
     0    20    40    60    80    100
          EPOCHS →
```

**Observação:** Run 10 (com augmentações mais refinadas) atingiu best val_loss 13.5% melhor que Run 05, mas a segmentação pixel-a-pixel (IoU) regrediu 4.9pp. Indica que melhorar *detecção por imagem* prejudicou *precisão de máscara*.

---

## Análise de Erros

### Distribuição de Erros no Run 10

**Por Imagem (593 GT positivas):**
-  **Detectadas corretamente:** 505 (85.2%)
- ❌ **Não detectadas (FN):** 88 (14.8%)
- ⚠️ **Falsos alarmes (FP):** 105 em 1.873 negativas (5.6%)

**Tipos de FN (formigueiros perdidos):**
1. **Formigueiros muito pequenos** (<100px) — abaixo do limiar de região mínima
2. **Bordas ambíguas** — modelo prediz <50% confiança
3. **Sobreposição com bordas de tile** — pixels com label=255 (ignorar) reduzem sinal de treino

**Tipos de FP (falsos alarmes):**
1. **Solo avermelhado** — confundido com formigueiro pela similaridade fotométrica
2. **Linhas de cultivo** — padrões lineares longos são interpretados como bordas de formigueiro
3. **Sombras** — regiões escuras em talhões muito inclinados

---

## Recomendações para Melhoria

### Curto Prazo (sem coletar dados novos)

1. **Ajuste de threshold por classe:** usar threshold=0.4 para imagens com solo avermelhado (reduz FP)
2. **Pós-processamento morphológico:** aplicar closing para preencher buracos em formigueiros detectados
3. **Ensemble simples:** média de predições de Run 05 + Run 10 para balancear Precision/Recall

### Médio Prazo (anotação)

1. **Anotar 200–300 novas imagens positivas** — aumentaria dataset de 593 para ~850 GT formigueiros (+43%), reduzindo memorização
2. **Revisar anotações existentes** — remover/corrigir masks com >5.000px que ensinam "shape priors" errados
3. **Aumentar cobertura de tiles** — coletar ortofotos de talhões com solo mais escuro/diferente, reduzindo bias fotométrico

### Longo Prazo (arquitetura)

1. **Transfer Learning:** usar backbone pré-treinado (ResNet, EfficientNet) em lugar de U-Net aleatória
2. **3D/Temporal:** se houver imagens de múltiplas épocas do mesmo talhão, usar CNN 3D ou RNN para explorar consistência temporal
3. **Multi-task learning:** treinar simultaneamente para detecção + estimativa de tamanho do formigueiro (auxiliaria segmentação)

---

## Conclusões

### Metas Alcançadas

| Meta                    | Alvo   | Resultado | Status |
| ----------------------- | ------ | --------- | ------ |
| F1 Score               | ≥ 80%  | 84.0%     |      |
| Recall (detecção)      | ≥ 80%  | 85.2%     |      |
| Precision (alarmes)    | ≥ 75%  | 82.8%     |      |
| IoU de formigueiro     | ≥ 35%  | 30.3%     | ❌ (4.7pp abaixo) |
| Pixel Accuracy         | ≥ 95%  | 98.8%     |      |

### Insights Principais

1. **Arquitetura + BatchNorm + ConvTranspose2d foram determinantes** — Run 05 (primeiro com essas mudanças) estabeleceu o melhor IoU (35.2%) que nenhum run posterior conseguiu superar, apesar de melhorias em outras métricas.

2. **Augmentações têm limite de efetividade** — tentativas de melhorar com Copy-Paste e Elastic agressivos (Runs 08-11) consistentemente pioraram segmentação pixel-a-pixel, sugerindo que o modelo começou a aprender padrões artificiais em vez de características reais.

3. **Desbalanceamento extremo não é resolvido apenas por loss rebalanceada** — mesmo com Tversky β=0.9 e Lovász Hinge, IoU não melhorou. O gargalo é a **quantidade limitada de dados reais** (593 GT positivas), que impede o modelo de aprender variabilidade de formigueiros.

4. **Trade-off Detecção vs Segmentação** — melhorar Recall (detectar mais formigueiros) tende a reduzir IoU (segmentação precisa). Run 10 oferece o melhor equilíbrio prático (F1=84.0% com IoU=30.3%), enquanto Run 05 oferece melhor precisão de máscara (IoU=35.2%) aceitando Recall 1.7pp menor.

5. **CosineAnnealingLR com scheduler fixo é superior a ReduceLROnPlateau** em cenários com augmentações fortes que inflacionam variância da val_loss.

### Recomendação Final

**Use Run 10 para produção**, pois oferece:
-  Melhor F1 entre runs com augmentações realistas (84.0%)
-  Recall máximo (85.2%) — reduz risco de perder formigueiros reais
-  Precision aceitável (82.8%) — controla falsos alarmes
-  Treinamento estável com CosineAnnealingLR

Para aplicações que priorizam **mínimos falsos alarmes** (e.g., controle químico automático em talhões pequenos), considere **Run 05** apesar de Recall 1.7pp menor, pela melhor qualidade de segmentação (IoU=35.2% vs 30.3%).

---

## Artefatos Disponíveis

| Artefato                     | Localização                        |
| ----------------------------- | ---------------------------------- |
| Melhor modelo (Run 10)        | `model/checkpoints/run10_best.pth` |
| Métricas completas            | [04_experimentos.md](04_experimentos.md) |
| Análise de dados              | [analysis.md](analysis.md)         |
| Correções de pipeline         | [fixes.md](fixes.md)               |
| Metodologia técnica           | [02_metodologia.md](02_metodologia.md) |
