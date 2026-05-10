# ADR-009 — Normalização com ImageNet Standard para Consistência

## Status

Aceito

---

## Contexto

Redes neurais treinadas em ImageNet utilizaram normalização específica:

```
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

Esta normalização é padrão de facto em visão computacional:

-  Transfer learning: Modelos pré-treinados esperam esta normalização
-  Comparabilidade: Comunidade usa mesmo padrão
-  Robustez: Normalização testada em milhões de imagens

Usar normalização diferente:

- ❌ Incompatível com modelos pré-treinados
- ❌ Requer re-treinamento completo
- ❌ Desalinha com benchmark da comunidade

---

## Decisão

Foi adotada **normalização padrão ImageNet** em todo o pipeline:

### Aplicado em

1. **Treinamento**: `SegmentationDataset`
2. **Validação**: Mesma normalização
3. **Inferência**: `PredictionService`
4. **Pré-treinamento**: Futuros modelos transfer learning

### Transformações

```python
transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
```

---

## Consequências

### Positivas

 **Transfer learning preparado**: Compatível com modelos pré-treinados

 **Consistência**: Mesma normalização em todo pipeline

 **Padrão industrial**: Alinhado com comunidade

 **Benchmarking**: Comparação válida com outros trabalhos

 **Documentação**: Facilita reprodução por outros

### Negativas

❌ **Possível desalinhamento**: Se dados diferem significativamente de ImageNet

❌ **Falta de exploração**: Normalização customizada poderia ser melhor

---

## Validação

Verificar que normalização está aplicada:

```python
# Treinamento
train_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Inferência
predict_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Verificar consistência
assert train_transform == predict_transform
```

---

## Implementação

**Arquivo**: `app/infrastructure/segmentation_dataset.py`

```python
self._transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])
```

**Arquivo**: `app/service/prediction_service.py`

```python
self._transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])
```

---

## Futuro: Transfer Learning

Se adotar modelo pré-treinado em ImageNet:

```python
# Compatível com essa normalização
model = torchvision.models.resnet50(pretrained=True)
```

A normalização já está alinhada!

---

## Referências

- ImageNet normalization: Standard de facto
- Implementação: `app/infrastructure/segmentation_dataset.py`
- Uso: `app/service/prediction_service.py`
