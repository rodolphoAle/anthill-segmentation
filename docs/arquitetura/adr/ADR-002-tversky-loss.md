# ADR-002 — Uso de Tversky Loss para Priorizar Recall

## Status

Aceito

---

## Contexto

Durante o desenvolvimento, identificou-se que a Focal Loss sozinha não era suficiente para:

- maximizar Recall da classe formigueiro;
- reduzir falsos negativos (regiões de formigueiro não detectadas);
- penalizar corretamente a perda de informação positiva.

Em segmentação de formigueiros, **não detectar um formigueiro é mais prejudicial que falsos positivos**, pois:

- falso positivo: marca fundo como formigueiro (erro de comissão);
- falso negativo: deixa formigueiro sem detecção (erro de omissão crítico).

---

## Decisão

Foi adotada **Tversky Loss** como segundo componente da loss combinada.

### Características da Tversky Loss

A Tversky Loss é uma generalização do Dice Loss que permite **ponderar diferentes tipos de erro**:

```
TverskyLoss = 1 - TP / (TP + α*FP + β*FN)
```

Onde:

- **TP**: Verdadeiros Positivos
- **FP**: Falsos Positivos
- **FN**: Falsos Negativos
- **α**: peso para falsos positivos (0.4)
- **β**: peso para falsos negativos (0.6)

### Objetivo

Como `β > α`:

- Falsos negativos são penalizados **mais severamente**;
- Modelo aprende a prioritizar Recall;
- Detecção de formigueiros é maximizada.

---

## Consequências

### Positivas

 **Maximiza Recall**: Reduz falsos negativos significativamente

 **Alinhado com objetivo**: Melhor detectar formigueiros, aceitando alguns falsos positivos

 **Métrica IoU melhorada**: Calcula interseção/união, alinhado com avaliação

 **Complementa Focal Loss**: Juntas lidam melhor com desbalanceamento

### Negativas

❌ **Menos ênfase em precisão**: Falsos positivos podem aumentar

❌ **Requer calibração de α e β**: Ajuste fino necessário

❌ **Sensível ao desbalanceamento**: Precisa ser combinada com outras losses

---

## Referências

- Salehi et al., 2017. "Tversky loss function for image segmentation using 3D fully convolutional deep networks"
- Implementação: `app/domain/losses/tversky_loss.py`
- Uso: `app/domain/losses/combined_loss.py`
