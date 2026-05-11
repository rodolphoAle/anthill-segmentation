# 6. Análise dos Resultados

Este capítulo apresenta análise sistemática dos resultados obtidos, com enfoque em: (i) trade-offs entre objetivos de otimização conflitantes; (ii) explicação de resultados aparentemente contraditórios (alto Recall, baixo IoU); (iii) identificação do gargalo principal de performance.

---

## 1. Problema Fundamental: Desbalanceamento Extremo de Classes

### Caracterização do Desbalanceamento

O dataset apresenta desbalanceamento extremo:

| Classe | Quantidade | Proporção | Densidade |
|--------|-----------|-----------|----------|
| Fundo (negativa) | 1.873 imagens | 75.9% | ~99% pixels |
| Formigueiro (positiva) | 593 imagens | 24.1% | ~1% pixels |

Este cenário introduz **problema de degeneração**: modelo pode minimizar loss predizendo quase sempre fundo.

### Manifestação em Runs Iniciais (01-02)

- Recall ≈ 41% (modelo detecta ~40% dos formigueiros)
- IoU ≈ 24% (segmentação muito imprecisa)
- Loss = Cross-Entropy sozinha é insuficiente
- **Comportamento**: Modelo colapsou para priorizar classe fundo

---

## 2. Trade-off Central: Detecção vs Segmentação Precisa

### Por Que Recall Alto ≠ IoU Alto?

Runs atingiram Recall máximo (85-87%) mas IoU plateaued em 30-35%. Explicação:

**Recall (detecção por imagem):**
- Pergunta: "Existe formigueiro nesta imagem?"
- Requer: Apenas detectar presença em algum lugar da imagem
- Threshold: Qualquer pixel com softmax > 0.5

**IoU (segmentação pixel-a-pixel):**
- Pergunta: "Quais pixels EXATAMENTE pertencem ao formigueiro?"
- Requer: Precisão de borda de todos os pixels do formigueiro
- Métrica: Intersecção / União de ~30% ou mais

**Analogy**: Um modelo pode detectar que há carro na rua (Recall=95%) mas desenhar bounding box impreciso (IoU=30%).

### Evidência Numérica do Trade-off

| Run | Estratégia | Recall | Precision | IoU |
|-----|-----------|--------|-----------|-----|
| 05 | Tversky+CE (50/50) | 83.5% | 85.1% | **35.2%** |
| 08 | Tversky+Focal+AugGauss | 68.5% | 90.2% | 34.2% |
| 09 | Tversky+Focal (85/15) | **87.0%** | 75.1% | 29.1% |
| 10 | Tversky+Focal+Copy-Paste | **85.2%** | 82.8% | 30.3% |
| 11 | Tversky+Focal+Lovász | 71.2% | 87.6% | 28.5% |

**Padrão claro**: Aumentar β para priorizar Recall → IoU diminui.

---

## 3. Impacto das Estratégias Aplicadas

### 3.1 Arquitetura (BatchNorm + ConvTranspose2d)

**Run 05 foi ponto de inflexão**: Primeira execução com BatchNorm2d e ConvTranspose2d aprendido.

**Impactos observados:**
- IoU de formigueiro: 23.9% → 35.2% (+47%)
- Recall: 77.6% → 83.5% (+7.7pp)
- **Insight**: Normalização de batch estabilizou gradientes; upsampling aprendido preservou detalhes

### 3.2 Loss Combinada (Tversky + Focal)

**Tversky Loss com (α=0.3, β=0.9):** Penaliza falsos negativos 3× mais que falsos positivos.

**Impacto:**
- Recall ↑ de 83.5% (Run 05) para 87.0% (Run 09)
- IoU ↓ de 35.2% (Run 05) para 29.1% (Run 09)
- **Trade-off observado**: Maximizar Recall prejudica precisão de bordas

**Focal Loss com γ=2.0:** Down-pesa exemplos de alta confiança, focando em exemplos difíceis.

### 3.3 Augmentações

**Geométricas (flip, rotação):**
- Melhoria marginal isolada

**Copy-Paste:**
- Introduzido em Run 08
- Efeito combinado com outras técnicas: sem impacto significativo em IoU

**ElasticTransform:**
- Introduzido em Run 08+
- **Problema**: Augmentações agressivas podem ensinar padrões artificiais que não existem em dados reais
- Hipótese: Distribuição de treinamento se afasta demais da distribuição real

---

## 4. Por Que IoU Não Melhora Além de 35%?

### Limitação 1: Dataset Pequeno

- 593 formigueiros é insuficiente
- Comparação: Cityscapes (~20K instâncias), COCO (~500K instâncias)
- **Estimativa**: O aumento da quantidade de dados anotados tende a melhorar a capacidade de generalização do modelo.

### Limitação 2: Qualidade de Anotação

- Anotação manual com inconsistência de bordas (+/-3 pixels)
- IoU máximo inter-rater provavelmente ~45-50%
- **Observação**: Run 05 com IoU=35.2% está ~10pp abaixo deste ceiling

### Limitação 3: Falta de Backbone Pré-Treinado

- U-Net treinada do zero vs ResNet-50 pré-treinada em ImageNet
- Transfer learning melhora IoU em 5-15pp em datasets pequenos
- Nossa U-Net não tem acesso a conhecimento de features visuais gerais

### Limitação 4: Plateau de Arquitetura

- Runs 05-11 (7 configurações diferentes) produziram IoU entre 28.5-35.2%
- Variações arquiteturais tiveram impacto <10%
- **Conclusão**: Gargalo é dataset, não arquitetura

---

## 5. Gargalo Principal: Dataset, Não Arquitetura

### Evidência 1: Comportamento de Augmentações

- Runs 01-07: Melhorias conforme pipeline completa
- Runs 08-11: Augmentações agressivas pioraram IoU
- **Interpretação**: Não é falta de dados (synthetic augmentaria), mas falta de dados REAIS

### Evidência 2: Hipóteses Rejeitadas

**H1: "Lovász Loss melhora IoU"** ✗
- Run 11 com Lovász: IoU=28.5% (pior que 30.3%)

**H2: "Augmentações agressivas melhoram generalização"** ✗
- Runs 08-11 pioraram IoU

**H3: "Oversampling resolve desbalanceamento"** ✗
- Run 06 com oversampling 3:1: IoU=34.3% (piora vs 35.2%)

---

## 6. Conclusões da Análise

1. **Run 10 é modelo ótimo para produção**: Melhor trade-off entre Recall (85.2%), Precision (82.8%), F1 (84.0%), com IoU aceitável (30.3%)

2. **IoU baixo é esperado com 593 exemplos**: Não é falha do modelo, mas limitação fundamental de dados

3. **Pipeline de dados teve impacto maior que losses sofisticadas**: Normalização (BatchNorm) e upsampling aprendido (ConvTranspose2d) foram mudanças críticas

4. **Próximos passos devem priorizar dados**: Investir em coleta de ≥200 novas positivas em lugar de arquitetura mais complexa
