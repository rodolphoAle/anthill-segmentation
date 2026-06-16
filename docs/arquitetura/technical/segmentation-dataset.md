# Segmentation Dataset

## Objetivo

Este arquivo implementa o `SegmentationDataset`, responsável pelo carregamento das imagens e máscaras utilizadas no treinamento da rede U-Net.

O dataset prepara os dados para segmentação semântica binária de formigueiros.

---

## Responsabilidades

O dataset é responsável por:

- carregar imagens RGB;
- carregar máscaras de segmentação;
- relacionar imagens e máscaras;
- aplicar augmentations;
- aplicar normalização;
- converter máscaras RGB para classes;
- retornar tensores para treinamento.

---

## Estrutura Esperada

```text
dataset/
├── rgb/
│   ├── img_001.png
│   ├── img_002.png
│   └── ...
│
└── labels/
    ├── img_001.png
    ├── img_002.png
    └── ...
```

As máscaras devem possuir o mesmo nome das imagens RGB.

---

# Fluxo Geral

```text
Imagem RGB
    ↓
Carregamento
    ↓
Augmentations
    ↓
Normalização
    ↓
Conversão para Tensor
    ↓
Retorno ao DataLoader
```

---

# Principais Componentes

## Carregamento das imagens

O dataset carrega:

* imagem RGB;
* máscara correspondente.

O carregamento pode ocorrer:

* diretamente do disco;
* pela memória RAM quando `preload=True`.

---

## Pré-carregamento em RAM

Quando habilitado:

```python
preload=True
```

Todas as imagens são carregadas na memória durante a inicialização.

### Benefícios

* reduz leituras no disco;
* diminui gargalos de I/O;
* melhora velocidade do treinamento.

---

# Augmentations

## Transformações sincronizadas

As transformações são aplicadas simultaneamente na imagem e máscara.

Isso evita desalinhamento entre:

* imagem de entrada;
* máscara real.

---

## Copy-Paste Augmentation

O dataset pode inserir formigueiros artificialmente em imagens negativas.

### Objetivo

Aumentar diversidade do dataset e reduzir desbalanceamento.

### Funcionamento

1. Seleciona uma imagem sem formigueiro;
2. Escolhe um formigueiro de outra imagem;
3. Copia e cola o objeto na imagem atual.

---

## Anthill Duplicate

Duplica formigueiros dentro da própria imagem.

### Objetivo

Aumentar quantidade de exemplos positivos.

### Funcionamento

1. Detecta regiões de formigueiro;
2. Cria cópias rotacionadas;
3. Cola em regiões vazias da mesma imagem.

---

# Conversão das Máscaras

As máscaras RGB são convertidas para classes numéricas.

| Cor      | Classe      |
| -------- | ----------- |
| Preto    | Fundo       |
| Vermelho | Formigueiro |
| Branco   | Ignorado    |

---

# Normalização

As imagens utilizam normalização padrão ImageNet.

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

---

# Filtro de Pixels Ignorados

O dataset pode remover imagens com excesso de pixels ignorados.

## Objetivo

Evitar treinamento com regiões inválidas.

---

# Principais Métodos

| Método                  | Função                           |
| ----------------------- | -------------------------------- |
| `_match_pairs`          | Relaciona imagens e máscaras     |
| `_preload_to_ram`       | Carrega imagens na RAM           |
| `_build_positive_index` | Indexa imagens com formigueiro   |
| `_load_pair`            | Carrega imagem e máscara         |
| `has_anthill`           | Verifica presença de formigueiro |
| `__getitem__`           | Retorna amostra do dataset       |

---

# Entrada e Saída

## Entrada

| Entrada     | Descrição              |
| ----------- | ---------------------- |
| Imagem RGB  | Imagem aérea           |
| Máscara RGB | Máscara de segmentação |

---

## Saída

| Saída        | Tipo           |
| ------------ | -------------- |
| image_tensor | `torch.Tensor` |
| mask_tensor  | `torch.Tensor` |
| filename     | `str`          |

---

# Importância no Projeto

O `SegmentationDataset` é um dos componentes centrais do pipeline.

Ele garante:

* carregamento correto dos dados;
* alinhamento entre imagem e máscara;
* aplicação das augmentations;
* preparação dos tensores usados pela U-Net.

---

# Relação com Outros Arquivos

| Arquivo                | Relação                          |
| ---------------------- | -------------------------------- |
| `mask_utils.py`        | Conversão de máscaras RGB        |
| `augmentations.py`     | Aplicação de augmentations       |
| `training_service.py`  | Utiliza o dataset no treinamento |
| `data_service.py`      | Cria DataLoaders                 |
| `streaming_dataset.py` | Variante streaming do dataset    |

---

# Resumo

O `SegmentationDataset` é responsável pela preparação completa dos dados utilizados pela rede neural.

Ele realiza:

* carregamento das imagens;
* aplicação de augmentations;
* balanceamento artificial;
* normalização;
* conversão das máscaras;
* geração dos tensores usados no treinamento.

Esse componente é essencial para garantir qualidade e estabilidade no treinamento do modelo.
