# ADR-006 — Filtragem de Regiões por Tamanho em Pós-Processamento

## Status

Aceito

---

## Contexto

Após inferência, o modelo gera máscaras binárias com alguns ruídos:

- **Falsos positivos pequenos**: Pixels isolados preditos como formigueiro
- **Artefatos**: Pequenas regiões espúrias
- **Ruído de fundo**: Regiões inválidas

Testes iniciais mostravam:

- ❌ Muitos falsos positivos pequenos;
- ❌ Artefatos de bordas;
- ❌ Diminuição de precisão.

Removê-los manualmente é impraticável, necessitando **filtro automático**.

---

## Decisão

Foi implementado **filtro de regiões por tamanho** em pós-processamento:

```python
def filter_regions(mask):
    # Identifica componentes conectados
    labeled_array, num_features = ndimage_label(mask)
    
    for region_id in range(1, num_features + 1):
        region_size = (labeled_array == region_id).sum()
        
        # Remove se muito pequeno
        if region_size < MIN_SIZE:
            mask[labeled_array == region_id] = 0
        
        # Remove se muito grande (falso positivo)
        elif MAX_SIZE > 0 and region_size > MAX_SIZE:
            mask[labeled_array == region_id] = 0
    
    return mask
```

### Parâmetros

| Parâmetro | Descrição |
| --- | --- |
| `MIN_ANTHILL_REGION_PX` | Tamanho mínimo (ex: 50 pixels) |
| `MAX_ANTHILL_REGION_PX` | Tamanho máximo (ex: 10000 pixels) |

---

## Consequências

### Positivas

 **Reduz falsos positivos**: Remove artefatos pequenos

 **Melhora precisão**: Predições mais confiáveis

 **Simples de implementar**: Pós-processamento rápido

 **Flexível**: Parâmetros ajustáveis por contexto

 **Robustez**: Formigueiros realistas não são removidos

### Negativas

❌ **Pode remover formigueiros pequenos**: Se MIN muito alto

❌ **Requer calibração**: Depende de características do dataset

❌ **Híper-realista**: Pode parecer "trucagem" se desajustado

---

## Calibração de Parâmetros

### Processo Recomendado

1. **Calcular estatísticas do dataset**:
   ```python
   # Medir tamanhos reais de formigueiros
   sizes = measure_anthill_sizes(label_dataset)
   min_size = percentile(sizes, 5)   # 5º percentil
   max_size = percentile(sizes, 95)  # 95º percentil
   ```

2. **Ajustar com margem**:
   ```
   MIN = min_size * 0.8  # 20% abaixo
   MAX = max_size * 1.2  # 20% acima
   ```

3. **Validar em conjunto de teste**:
   - Manter formigueiros reais?
   - Remover ruído suficiente?

---

## Integração em Pipeline

1. **Treinamento**: Sem filtro (deixa modelo aprender tudo)
2. **Validação**: Com filtro (avalia qualidade real)
3. **Inferência**: Com filtro (garante confiabilidade)

---

## Implementação

**Arquivo**: `app/service/prediction_service.py`

```python
if settings.use_region_filter:
    prediction = self._filter_regions(prediction)
```

**Configuração**: `app/core/config.py`

```python
USE_REGION_FILTER = True
MIN_ANTHILL_REGION_PX = 50
MAX_ANTHILL_REGION_PX = 10000
```

---

## Referências

- Implementação: `app/service/prediction_service.py#_filter_regions`
- Configuração: `app/core/config.py`
