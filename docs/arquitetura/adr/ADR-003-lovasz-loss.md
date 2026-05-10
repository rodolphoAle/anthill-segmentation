# ADR-003 — Uso de Lovász Hinge Loss para Otimizar IoU Direto

## Status

Aceito

---

## Contexto

As losses anteriores (Focal e Tversky) otimizam métricas **por pixel**:

- Focal Loss: probabilidade de classe;
- Tversky Loss: sobreposição local.

Porém, a métrica de **avaliação final é IoU (Intersection over Union)**:

```
IoU = |Predição ∩ Máscara Real| / |Predição ∪ Máscara Real|
```

Problema: **otimizar métricas por pixel não garante IoU otimizado**.

Descolamento entre loss de treinamento e métrica de avaliação pode levar a:

- modelo converge para loss baixa mas IoU ruim;
- desalinhamento entre treinamento e teste.

---

## Decisão

Foi adotada **Lovász Hinge Loss** como terceiro componente opcional da loss combinada.

### Características da Lovász Hinge Loss

A Lovász Hinge Loss aproxima a perda com a extensão de Lovász da função IoU:

1. Calcula **margem** entre classes (formigueiro - fundo)
2. Ordena erros do maior para menor
3. Calcula **extensão Lovász** dos erros ordenados
4. Otimiza **diretamente a métrica IoU**

### Objetivo

- Alinhar treinamento com métrica de avaliação;
- Garantir que loss baixa corresponde a IoU alta;
- Melhoria na qualidade das máscaras preditas.

---

## Consequências

### Positivas

 **Otimização direta de IoU**: Loss alinhada com métrica de avaliação

 **Melhor qualidade de segmentação**: Máscaras mais precisas

 **Reduz descolamento train/test**: Convergência mais confiável

 **Complementa outras losses**: Refina bordas e sobreposição

### Negativas

 **Mais complexa computacionalmente**: Ordenação e extensão Lovász

 **Menos estável no início do treinamento**: Recomenda-se uso gradual

 **Requer weight tuning**: Balanço com outras componentes

---

## Decisão de Implementação

A Lovász Loss foi implementada como **componente opcional**:

```python
if lovasz_weight > 0:
    # calcula Lovász
    lovasz_loss = LovaszHingeLoss(...)
else:
    # usa apenas Focal + Tversky
    lovasz_loss = 0
```

Isso permite **experimentação** sem obrigatoriedade.

---

## Referências

- Berman et al., 2018. "The Lovász-Softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks"
- Implementação: `app/domain/losses/lovasz_loss.py`
- Uso: `app/domain/losses/combined_loss.py`
