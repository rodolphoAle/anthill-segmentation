````md id="xj6m8u"
# Validation Service

## Objetivo

Este arquivo implementa o `ValidationService`, responsável pela validação do modelo de segmentação semântica.

O serviço executa inferência sobre o conjunto de validação e calcula métricas utilizadas para avaliar a qualidade da segmentação dos formigueiros.

---

# Responsabilidades

O `ValidationService` é responsável por:

- buscar imagens de validação no armazenamento remoto;
- baixar imagens e máscaras em memória;
- executar inferência com a U-Net;
- aplicar pós-processamento;
- calcular métricas;
- salvar resultados relevantes;
- gerar estatísticas da validação.

---

# Fluxo Geral

```text
Google Drive
    ↓
Download em memória
    ↓
Pré-processamento
    ↓
Inferência U-Net
    ↓
Threshold
    ↓
Filtro de regiões
    ↓
Cálculo de métricas
    ↓
Salvamento dos resultados
````

---

# Estrutura Principal

## ValidationService

Classe principal responsável pela validação do modelo.

---

# Pré-processamento

O serviço utiliza o mesmo pré-processamento aplicado no treinamento.

## Normalização

```text id="7zcfbw"
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

Isso mantém consistência entre:

* treino;
* validação;
* inferência.

---

# Streaming Remoto

As imagens são baixadas diretamente do armazenamento remoto.

O pipeline funciona totalmente em memória:

```text
download → processamento → descarte
```

Sem necessidade de salvar o dataset localmente.

---

# Inferência

A inferência ocorre em:

```python id="ck9k8s"
_predict_sync()
```

Etapas:

1. conversão da imagem para tensor;
2. forward da U-Net;
3. aplicação de Softmax;
4. geração da máscara binária.

---

# Threshold de Confiança

A máscara binária é gerada usando:

```python id="r0jkcb"
settings.anthill_confidence_threshold
```

Pixels acima do threshold são classificados como:

```text id="fdygo0"
1 = formigueiro
0 = fundo
```

---

# Pós-processamento

O serviço aplica filtro de regiões conectadas.

Objetivo:

* remover ruídos pequenos;
* remover falsos positivos grandes.

---

# Componentes Conectados

O filtro utiliza:

```python id="s7vh1n"
scipy.ndimage.label
```

Cada região prevista é analisada individualmente.

---

# Regras de Filtragem

## Região pequena

```text
size < min_anthill_region_px
```

Considerada ruído.

---

## Região grande

```text
size > max_anthill_region_px
```

Considerada falso positivo.

---

# Métricas Calculadas

## Pixel Accuracy

Accuracy = \frac{Pixels\ Corretos}{Pixels\ Totais}

---

## IoU

IoU = \frac{Interseccao}{Uniao}

---

## Dice Score

Dice = \frac{2 \cdot Interseccao}{Predicao + GroundTruth}

---

# Classes Utilizadas

| Classe      | Valor |
| ----------- | ----- |
| Fundo       | 0     |
| Formigueiro | 1     |
| Ignorado    | 255   |

---

# Pixels Ignorados

Pixels com valor:

```text id="y8r8df"
255
```

não participam do cálculo das métricas.

---

# Salvamento de Resultados

Resultados são salvos quando o percentual previsto de formigueiro ultrapassa:

```python id="v1e5vt"
settings.anthill_save_threshold
```

Arquivos salvos:

| Arquivo     | Descrição        |
| ----------- | ---------------- |
| `_rgb.png`  | imagem original  |
| `_mask.png` | máscara prevista |

---

# Matching de Arquivos

O serviço relaciona:

* imagem RGB;
* máscara correspondente.

O relacionamento ocorre pelo prefixo do nome do arquivo.

---

# Principais Métodos

| Método                  | Responsabilidade             |
| ----------------------- | ---------------------------- |
| `_resolve_subfolder_id` | Localiza pastas remotas      |
| `_predict_sync`         | Executa inferência           |
| `_filter_small_regions` | Remove regiões inválidas     |
| `_iou`                  | Calcula IoU                  |
| `_dice`                 | Calcula Dice                 |
| `_match_pairs`          | Relaciona imagens e máscaras |
| `run`                   | Executa validação completa   |

---

# Entrada e Saída

## Entrada

| Entrada      | Descrição             |
| ------------ | --------------------- |
| Imagens RGB  | conjunto de validação |
| Máscaras RGB | ground truth          |

---

## Saída

| Saída              | Descrição         |
| ------------------ | ----------------- |
| Pixel Accuracy     | acurácia global   |
| Mean IoU           | média de IoU      |
| Mean Dice          | média de Dice     |
| Máscaras previstas | resultados salvos |

---

# Relação com Outros Arquivos

| Arquivo                 | Relação                    |
| ----------------------- | -------------------------- |
| `training_service.py`   | Treina modelo validado     |
| `mask_utils.py`         | Conversão de máscaras      |
| `metrics.py`            | Estrutura das métricas     |
| `config.py`             | Configurações da validação |
| `prediction_service.py` | Serviço de inferência      |

---

# Importância no Projeto

O `ValidationService` é responsável por medir a qualidade da segmentação produzida pela U-Net.

Ele permite:

* avaliar desempenho do modelo;
* detectar overfitting;
* acompanhar evolução do treinamento;
* salvar exemplos relevantes da segmentação.

---

# Resumo 

O `ValidationService` executa a validação completa do modelo utilizando imagens remotas.

O serviço:

* baixa imagens em memória;
* executa inferência;
* aplica pós-processamento;
* calcula métricas;
* salva resultados importantes.

Ele foi fundamental para avaliar a capacidade da U-Net em detectar formigueiros corretamente.

```
```
