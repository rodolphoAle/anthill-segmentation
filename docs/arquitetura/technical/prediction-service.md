# Prediction Service

## Objetivo

Este arquivo implementa o `PredictionService`, responsável pela inferência do modelo de segmentação.

O serviço é responsável por:

- receber uma imagem em bytes;
- aplicar o mesmo pré-processamento usado no treinamento;
- executar inferência com a U-Net;
- gerar uma máscara binária de formigueiro;
- aplicar filtro por tamanho de região;
- retornar a máscara como array ou PNG.

---

## Responsabilidades

O `PredictionService` realiza:

- carregamento e conversão de imagens em bytes;
- pré-processamento com normalização ImageNet;
- execução de inferência no modelo;
- conversão de logits em probabilidades;
- aplicação de threshold de confiança;
- filtragem de regiões por tamanho;
- exportação de resultados em diferentes formatos.

---

## Fluxo de Execução

```text
Imagem em bytes
    ↓
Conversão para RGB
    ↓
Pré-processamento (normalização)
    ↓
Adição de dimensão de batch
    ↓
Inferência no modelo
    ↓
Conversão logits → probabilidades
    ↓
Extração classe formigueiro
    ↓
Aplicação de threshold
    ↓
Filtragem de regiões
    ↓
Retorno máscara binária
```

---

## Pré-processamento

O serviço aplica as mesmas transformações usadas no treinamento:

```python
transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])
```

Isso garante consistência entre treinamento e inferência.

---

## Inferência

### Processo

1. Carrega modelo em modo `eval()`;
2. Desativa gradientes com `torch.no_grad()`;
3. Executa forward pass;
4. Aplica Softmax para obter probabilidades;
5. Extrai probabilidade da classe formigueiro (classe 1);
6. Compara com threshold de confiança.

### Threshold de Confiança

Apenas pixels com probabilidade >= `settings.anthill_confidence_threshold` são classificados como formigueiro.

---

## Filtragem de Regiões

O serviço remove regiões fora do intervalo de tamanho esperado.

### Objetivo

- Eliminar ruídos pequenos (falsos positivos);
- Remover regiões anormalmente grandes (falsos positivos);
- Manter formigueiros realistas.

### Parâmetros

| Parâmetro                  | Descrição                                    |
| -------------------------- | -------------------------------------------- |
| `min_anthill_region_px`    | Tamanho mínimo em pixels de uma região válida |
| `max_anthill_region_px`    | Tamanho máximo em pixels de uma região válida |
| `use_region_filter`        | Se deve usar o filtro de regiões             |

### Funcionamento

1. Identifica componentes conectados na máscara;
2. Para cada região detectada:
   - Remove se menor que `min_anthill_region_px`;
   - Remove se maior que `max_anthill_region_px` (se configurado).

---

## Principais Métodos

| Método           | Descrição                                       |
| ---------------- | ----------------------------------------------- |
| `__init__`       | Inicializa modelo e define transformações      |
| `_sync_predict`  | Executa predição de forma síncrona             |
| `_filter_regions`| Filtra regiões por tamanho                      |
| `predict`        | Executa predição assíncrona                     |
| `predict_to_png` | Executa predição e retorna máscara em PNG      |

---

## Operações Assíncronas

O serviço utiliza `asyncio.to_thread` para executar inferência em thread separada.

### Objetivo

Evitar bloqueio do loop assíncrono durante o processamento.

### Benefício

Permite que a aplicação web manipule múltiplas requisições simultaneamente.

---

## Entrada e Saída

### Método `predict`

| Entrada       | Tipo    | Descrição               |
| ------------- | ------- | ----------------------- |
| `image_bytes` | `bytes` | Imagem em formato binário |

| Saída      | Tipo         | Descrição                        |
| ---------- | ------------ | -------------------------------- |
| prediction | `np.ndarray` | Máscara binária (0 ou 1)        |

### Método `predict_to_png`

| Entrada       | Tipo    | Descrição               |
| ------------- | ------- | ----------------------- |
| `image_bytes` | `bytes` | Imagem em formato binário |

| Saída      | Tipo    | Descrição                      |
| ---------- | ------- | ------------------------------ |
| png_bytes  | `bytes` | Máscara em PNG (0 ou 255)      |

---

## Conversão de Saída

### Array numpy

Valores são 0 ou 1 (binários).

### PNG

Valores são convertidos para 0 ou 255 para melhor visualização.

---

## Importância no Projeto

O `PredictionService` é essencial para:

- disponibilizar o modelo treinado para uso;
- garantir consistência de pré-processamento;
- fornecer interface simples para inferência;
- suportar múltiplas requisições via async;
- garantir qualidade das predições via filtros.

---

## Relação com Outros Arquivos

| Arquivo               | Relação                           |
| --------------------- | --------------------------------- |
| `unet.py`             | Define a arquitetura do modelo    |
| `training_service.py` | Treina o modelo usado na predição |
| `config.py`           | Define configurações de predição  |
| `main.py`             | Integra o serviço na API          |

---

## Configurações Importantes

As seguintes configurações devem ser definidas em `settings`:

| Configuração                  | Descrição                           |
| ----------------------------- | ----------------------------------- |
| `anthill_confidence_threshold` | Threshold de confiança para predição |
| `min_anthill_region_px`       | Tamanho mínimo de região             |
| `max_anthill_region_px`       | Tamanho máximo de região             |
| `use_region_filter`           | Ativar/desativar filtro de regiões  |

---

## Resumo

O `PredictionService` é responsável por transformar o modelo treinado em um serviço de inferência pronto para produção.

Ele garante:

- aplicação correta das transformações;
- execução eficiente da inferência;
- filtragem de resultados ruidosos;
- suporte a múltiplas requisições assíncronas;
- compatibilidade com diferentes formatos de saída.
