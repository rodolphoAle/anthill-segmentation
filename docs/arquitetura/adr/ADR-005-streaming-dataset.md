# ADR-005 — Uso de Streaming Dataset para Escalabilidade

## Status

Aceito

---

## Contexto

O dataset de treinamento possui:

- Grande volume de imagens (centenas/milhares);
- Tamanho total pode exceder capacidade de armazenamento local;
- Necessidade de processar dados remotos (Google Drive, cloud);
- Restrição de espaço em disco em ambiente de containerização;
- Escalabilidade para futuros datasets maiores.

Carregamento tradicional (dataset completo em disco):

- ❌ Consome muito espaço;
- ❌ Dificulta testes de novo dataset;
- ❌ Não é escalável para produção.

---

## Decisão

Foi implementado **StreamingDataset**, que carrega imagens sob demanda:

1. **Baixa sob demanda**: Imagens baixadas apenas quando necessárias
2. **Processa em memória**: Usa BytesIO para não salvar em disco
3. **Descarta após uso**: Libera memória após processamento

### Arquitetura

```
DataLoader
    ↓
StreamingDataset.__getitem__()
    ↓
fetch_image_from_remote()
    ↓
BytesIO (processamento em memória)
    ↓
Tensor
    ↓
retorna ao DataLoader
```

### Suportado

- Google Drive (via `GoogleDriveClient`)
- URLs diretas
- Qualquer storage remoto via API

---

## Consequências

### Positivas

 **Menor uso de disco**: Sem necessidade armazenar dataset completo

 **Escalabilidade**: Suporta datasets arbitrariamente grandes

 **Pipeline leve**: Imagens descartadas após uso

 **Flexibilidade**: Funciona com dados remotos

 **Produção-ready**: Padrão usado em deployments modernos

### Negativas

❌ **Dependência de rede**: Requer conexão estável

❌ **Latência**: Primeira leitura lenta (download + processamento)

❌ **Custo de banda**: Downloads repetidos podem ser caros

❌ **Complexidade**: Mais código que dataset local

---

## Mecanismos de Otimização

### Pré-carregamento (Preload)

```python
StreamingDataset(..., preload=True)
```

- Carrega todas as imagens em RAM na inicialização
- Trade-off: mais memória, menos latência
- Recomendado para datasets pequenos/médios

### Cache em Memória

- LRU cache evita re-downloads repetidos
- Balanceamento automático de memória

---

## Comparação com SegmentationDataset

| Aspecto | SegmentationDataset | StreamingDataset |
| --- | --- | --- |
| Armazenamento | Local em disco | Remoto (on-demand) |
| Uso de disco | Alto | Mínimo |
| Escalabilidade | Limitada | Ilimitada |
| Latência | Baixa | Média (primeira leitura) |
| Produção | Limitado | Recomendado |

---

## Implementação

**Arquivo**: `app/infrastructure/streaming_dataset.py`

**Uso**:

```python
dataset = StreamingDataset(
    image_urls=[...],
    label_urls=[...],
    augmentations=augmentation_transforms,
    preload=False,  # False para streaming, True para preload
)

loader = DataLoader(dataset, batch_size=16, num_workers=4)
```

---

## Referências

- Implementação: `app/infrastructure/streaming_dataset.py`
- Cliente Google Drive: `app/infrastructure/google_drive_client.py`
- Uso em pipeline: `app/service/data_service.py`
