# ADR-004 — Uso de Loss Combinada (Focal + Tversky + Lovász)

## Status

Aceito

---

## Contexto

Nenhuma loss única consegue otimizar simultaneamente:

- desbalanceamento de classes;
- maximização de Recall;
- otimização de IoU.

Testes com losses individuais mostraram:

- **Focal Loss sozinha**: bom Recall, mas IoU não otimizado
- **Tversky Loss sozinha**: melhor Recall, porém ainda sem alinhamento com IoU
- **Lovász Loss sozinha**: IoU otimizada, porém menos enfoque em desbalanceamento

**Necessidade**: combinar complementos para atacar múltiplos problemas simultaneamente.

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

### Positivas

 **Multi-objetivo**: Ataca múltiplos problemas simultaneamente

 **Flexibilidade**: Pesos ajustáveis para diferentes cenários

 **Estabilidade**: Combinação reduz extremos individuais

 **Resultados comprovados**: Treinamento mais estável com perda menor

 **Recall significativamente melhorado**: Formigueiros detectados com alta confiabilidade

### Negativas

 **Complexidade**: Mais hiperparâmetros para ajustar

 **Custo computacional**: Cálculo de 3 losses simultaneamente

 **Sensibilidade**: Pequenas mudanças nos pesos afetam significativamente resultado

 **Necessidade de tuning**: Requer experimentação para novo dataset

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

**Arquivo**: `app/domain/losses/combined_loss.py`

```python
criterion = CombinedTverskyFocalLoss(
    tversky_alpha=0.4,      # Peso para falsos positivos
    tversky_beta=0.6,       # Peso para falsos negativos
    tversky_weight=0.5,     # Fração da loss
    lovasz_weight=0.3,      # Fração da loss
    focal_gamma=2.0,        # Foco em exemplos difíceis
    class_weights=weights,  # Pesos por classe
    ignore_index=255,       # Pixels a ignorar
)
```

---

## Referências

- Implementação: `app/domain/losses/combined_loss.py`
- Uso em treinamento: `app/service/training_service.py`
- Avaliação de métricas: `app/domain/metrics.py`
