# Mask Utils

## Objetivo

Este arquivo implementa utilitários responsáveis pela conversão e manipulação das máscaras RGB utilizadas na segmentação semântica de formigueiros.

As máscaras anotadas manualmente utilizam cores específicas para representar cada classe.

---

# Mapeamento de Classes

As cores da máscara são convertidas para valores numéricos utilizados pela U-Net.

| Cor | Classe | Valor |
|---|---|---|
| Preto | Fundo | 0 |
| Vermelho | Formigueiro | 1 |
| Branco | Ignorado | 255 |

---

# Objetivo da Conversão

As redes neurais não trabalham diretamente com cores RGB.

Por isso, as máscaras precisam ser convertidas para arrays numéricos contendo IDs de classes.

---

# Fluxo Geral

```text
Máscara RGB
    ↓
Separação dos canais
    ↓
Identificação das cores
    ↓
Conversão para classes
    ↓
Array NumPy
````

---

# Classes Utilizadas

## LABEL_BACKGROUND

```python id="5c95ea"
LABEL_BACKGROUND = 0
```

Representa pixels de fundo.

---

## LABEL_ANTHILL

```python id="v3x4d4"
LABEL_ANTHILL = 1
```

Representa pixels de formigueiro.

---

## LABEL_IGNORE

```python id="ntz8b5"
LABEL_IGNORE = 255
```

Representa pixels ignorados durante:

* treinamento;
* validação;
* cálculo de métricas.

---

# Conversão das Máscaras

## decode_rgb_mask()

Converte máscara RGB para array `uint8`.

### Funcionamento

1. converte imagem para RGB;
2. separa canais R, G e B;
3. identifica pixels vermelhos;
4. identifica pixels pretos;
5. gera array de classes.

---

## Critério para Formigueiro

```text id="pqwyaj"
R > 150
G < 100
B < 100
```

---

## Critério para Fundo

```text id="98g0h5"
R < 50
G < 50
B < 50
```

---

# Conversão para PyTorch

## decode_rgb_mask_to_int64()

Converte máscara RGB para:

```python id="nfhyr7"
np.int64
```

Esse formato é necessário para:

* CrossEntropy;
* Focal Loss;
* Tversky Loss;
* Lovász Loss.

---

# Verificação de Formigueiros

## has_anthill_pixels()

Verifica se existe ao menos um pixel de formigueiro na máscara.

### Uso

Utilizado para:

* identificar imagens positivas;
* augmentations;
* balanceamento do dataset.

---

# Máscara Binária

## get_anthill_binary_mask()

Extrai uma máscara binária contendo apenas:

| Valor | Significado |
| ----- | ----------- |
| 0     | Fundo       |
| 1     | Formigueiro |

---

# Pixels Ignorados

## compute_ignore_pixel_pct()

Calcula a proporção de pixels ignorados.

Pixels ignorados são regiões brancas da máscara.

### Objetivo

Permitir:

* filtragem de tiles inválidos;
* remoção de regiões sem anotação;
* melhoria da qualidade do dataset.

---

# Principais Métodos

| Método                     | Função                           |
| -------------------------- | -------------------------------- |
| `decode_rgb_mask`          | Converte RGB para classes        |
| `decode_rgb_mask_to_int64` | Conversão compatível com PyTorch |
| `has_anthill_pixels`       | Detecta presença de formigueiro  |
| `get_anthill_binary_mask`  | Gera máscara binária             |
| `compute_ignore_pixel_pct` | Mede pixels ignorados            |

---

# Entrada e Saída

## Entrada

| Entrada     | Tipo                        |
| ----------- | --------------------------- |
| Máscara RGB | `PIL.Image` ou `np.ndarray` |

---

## Saída

| Saída               | Tipo         |
| ------------------- | ------------ |
| Máscara de classes  | `np.ndarray` |
| Máscara binária     | `np.ndarray` |
| Percentual ignorado | `float`      |

---

# Importância no Projeto

O `mask_utils.py` é fundamental para garantir consistência entre:

* treinamento;
* validação;
* inferência.

Ele centraliza toda a lógica de interpretação das máscaras RGB.

---

# Relação com Outros Arquivos

| Arquivo                   | Relação                                  |
| ------------------------- | ---------------------------------------- |
| `segmentation_dataset.py` | Conversão das máscaras locais            |
| `streaming_dataset.py`    | Conversão das máscaras remotas           |
| `validation_service.py`   | Decodificação do ground truth            |
| `prediction_service.py`   | Comparação entre máscara prevista e real |

---

# Benefícios da Centralização

Centralizar essa lógica evita:

* inconsistência entre datasets;
* diferenças entre treino e validação;
* interpretação incorreta das máscaras.

---

# Resumo 

O `mask_utils.py` é responsável por converter máscaras RGB em classes numéricas usadas pela rede neural.

Ele:

* interpreta as cores das máscaras;
* identifica formigueiros;
* gera máscaras binárias;
* calcula pixels ignorados;
* garante padronização do pipeline inteiro.

