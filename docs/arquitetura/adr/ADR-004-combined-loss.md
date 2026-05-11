# ADR-004 — Uso de Loss Combinada (Focal + Tversky + Lovász)

## Status

Aceito

---

## Contexto

Segmentação semântica com forte desbalanceamento de classes apresenta conflito entre objetivos:

- **Cross-Entropy**: Perde efetividade com classes minoritárias
- **Focal Loss (Lin et al., 2017)**: Reduz peso de exemplos fáceis, mas não otimiza IoU diretamente
- **Tversky Loss (Salehi et al., 2017)**: Generaliza Dice com controle independente de FP (α) e FN (β), permitindo maximização de Recall
- **Lovász Hinge (Berman et al., 2018)**: Proximal direto para métrica IoU, mas sensível a desbalanceamento extremo

Testes exploratórios demonstraram:

- Focal Loss isolada: Recall melhorado, porém IoU de formigueiro limitado a ~24%
- Tversky Loss isolada (β=0.9): Recall máximo (~87%), com IoU degradado a ~29%
- Lovász Hinge isolada: IoU de formigueiro máximo (~35%), com Recall reduzido (~68%)

**Conclusão**: Otimização multi-objetivo requer combinação ponderada de losses complementares.

---

## Decisão

Foi implementada **CombinedTverskyFocalLoss**, que combina três componentes:

```
Total Loss = 
    tversky_weight × TverskyLoss
  + lovasz_weight  × LovaszLoss  
  + focal_weight   × FocalLoss
```

### Pesos Configurados

| Componente      | Peso | Objetivo                                  |
| --------------- | ---- | ----------------------------------------- |
| Tversky         | 0.5  | Maximizar Recall (prioridade alta)        |
| Lovász          | 0.3  | Otimizar IoU                              |
| Focal           | 0.2  | Lidar com desbalanceamento                |

### Validação de Pesos

```python
if tversky_weight + lovasz_weight > 1.0:
    raise ValueError("Soma de pesos não pode exceder 1.0")

focal_weight = 1.0 - tversky_weight - lovasz_weight
```

Garante que soma sempre = 1.0 e focal_weight sempre ≥ 0.

---

## Consequências

### Consequências Positivas

✓ **Otimização multi-objetivo**: Ataca desbalanceamento, detecção e precisão de bordas simultaneamente

✓ **Flexibilidade de ajuste**: Pesos ajustáveis permitem priorização conforme objetivo (Recall vs. IoU)

✓ **Estabilidade numérica**: Combinação linear reduz variância comparado a losses individuais

✓ **Performance comprovada**: Runs 08-11 demonstraram F1-Score ~78-84% com IoU de formigueiro ~30%

✓ **Compatibilidade com regularização**: Convive com class weighting e gradient clipping sem instabilidades

### Consequências Negativas

✗ **Superfície de busca complexa**: 5 hiperparâmetros tuneáveis (α, β, γ, w_tversky, w_lovasz) resultam em espaço exponencial de configurações

✗ **Overhead computacional**: Três termos de loss computados por backward pass (~10-15% overhead)

✗ **Sensibilidade a pesos**: Variação de ±0.1 nos pesos ocasiona mudanças de ~2-4 pp em métricas

✗ **Generalização não garantida**: Pesos otimizados para dataset atual podem não transferir para geographia/season diferentes

---

## Estratégia de Ajuste

Para um novo dataset, recomenda-se:

1. Iniciar com pesos atuais (0.5, 0.3, 0.2)
2. Monitorar: Recall, Precision, IoU, F1-Score
3. Se Recall muito baixo: aumentar `tversky_weight`
4. Se IoU ruim: aumentar `lovasz_weight`
5. Se desbalanceamento evidente: aumentar `focal_weight`

---

## Implementação

**Arquivo**: [app/domain/losses/combined_loss.py](../../app/domain/losses/combined_loss.py)

```python
criterion = CombinedTverskyFocalLoss(
    tversky_alpha=0.3,      # Penalidade para falsos positivos
    tversky_beta=0.7,       # Penalidade para falsos negativos (β > α para Recall)
    tversky_weight=0.5,     # Fração da loss (~50%)
    lovasz_weight=0.3,      # Fração da loss (~30%)
    focal_gamma=2.0,        # Fator de foco (exemplos difíceis)
    class_weights=tensor([1.0, 4.0]),  # Ponderação por classe
    ignore_index=255,       # Pixels a ignorar (mascaras inválidas)
)
```

---

## Referências

- Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal Loss for Dense Object Detection. In *ICCV*, pp. 2980–2988.
- Salehi, S. S. M., Erdogmus, D., & Gholipour, A. (2017). Tversky Loss Function for Image Segmentation Using 3D Fully Convolutional Deep Networks. In *MICCAI 2017 Workshop*.
- Berman, M., Rannen Triki, A., & Blaschko, M. B. (2018). The Lovász Hinge Loss for Semantic Segmentation. In *ICML*, pp. 438–447.
- Implementação: [app/domain/losses/combined_loss.py](../../app/domain/losses/combined_loss.py)
- Treinamento: [app/service/training_service.py](../../app/service/training_service.py)
- Validação: [docs/artigo/relatorio/04_experimentos.md](../artigo/relatorio/04_experimentos.md)
