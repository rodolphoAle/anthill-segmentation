````md id="3y3r3s"
# Streaming Segmentation Dataset

## Objetivo

Este arquivo implementa o `StreamingSegmentationDataset`, responsável por carregar imagens e máscaras diretamente de um armazenamento remoto.

O dataset funciona totalmente em memória, sem salvar arquivos no disco local.

---

# Principais Características

- download sob demanda;
- processamento em memória;
- ausência de escrita em disco;
- suporte a augmentations;
- conversão automática para tensores PyTorch.

---

# Fluxo Geral

```text
Google Drive / Armazenamento Remoto
            ↓
Download em memória
            ↓
Decodificação da imagem
            ↓
Augmentations
            ↓
Normalização
            ↓
Conversão para Tensor
            ↓
Treinamento da U-Net
````

---

# Estrutura dos Dados

Cada amostra contém:

```python id="sv8h8t"
(rgb_meta, label_meta)
```

Onde:

| Campo  | Descrição                       |
| ------ | ------------------------------- |
| `id`   | Identificador remoto do arquivo |
| `name` | Nome do arquivo                 |

---

# Download Sob Demanda

As imagens não permanecem armazenadas localmente.

Cada chamada de:

```python id="7j0rk7"
__getitem__()
```

realiza:

1. download da imagem RGB;
2. download da máscara;
3. processamento em memória;
4. descarte automático após uso.

---

# Processamento em Memória

As imagens são carregadas usando:

```python id="w2dfr0"
io.BytesIO
```

Isso evita:

* uso excessivo de disco;
* armazenamento permanente do dataset;
* necessidade de grandes volumes locais.

---

# Decodificação das Imagens

As imagens RGB são convertidas utilizando:

```python id="q53m7n"
Image.open(...).convert("RGB")
```

As máscaras também permanecem em RGB para preservar o canal vermelho utilizado na anotação dos formigueiros.

---

# Augmentations

O dataset suporta transformações sincronizadas entre:

* imagem RGB;
* máscara de segmentação.

Exemplos:

* flip;
* rotação;
* resize;
* crop.

---

# Normalização

As imagens utilizam normalização padrão ImageNet.

```text id="s4x7op"
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

A normalização é aplicada após as augmentations.

---

# Conversão das Máscaras

As máscaras RGB são convertidas para classes numéricas utilizando:

```python id="zq5fr8"
decode_rgb_mask_to_int64()
```

---

# Classes Utilizadas

| Classe      | Valor |
| ----------- | ----- |
| Fundo       | 0     |
| Formigueiro | 1     |
| Ignorado    | 255   |

---

# Conversão para Tensor

Após o processamento:

* imagem → `float32`
* máscara → `long`

Compatível com:

* CrossEntropy;
* Focal Loss;
* Tversky Loss;
* Lovász Loss.

---

# Interface do Dataset

## `__len__()`

Retorna quantidade total de amostras.

---

## `__getitem__()`

Executa:

* download remoto;
* decodificação;
* augmentations;
* normalização;
* conversão para tensor.

---

# Entrada e Saída

## Entrada

| Entrada        | Descrição    |
| -------------- | ------------ |
| RGB remoto     | imagem aérea |
| Máscara remota | ground truth |

---

## Saída

| Saída          | Tipo           |
| -------------- | -------------- |
| `image_tensor` | `torch.Tensor` |
| `mask_tensor`  | `torch.Tensor` |
| `filename`     | `str`          |

---

# Vantagens do Streaming

## Menor uso de disco

Nenhum arquivo precisa ser salvo localmente.

---

## Escalabilidade

Permite trabalhar com datasets grandes sem ocupar armazenamento local.

---

## Processamento dinâmico

As imagens são carregadas apenas quando necessárias.

---

# Limitações

## Dependência da rede

A velocidade depende do download remoto.

---

## Maior latência

Downloads podem aumentar o tempo entre batches.

---

# Relação com Outros Arquivos

| Arquivo                   | Relação                          |
| ------------------------- | -------------------------------- |
| `google_drive_client.py`  | Responsável pelo download remoto |
| `mask_utils.py`           | Conversão das máscaras RGB       |
| `training_service.py`     | Usa dataset durante treinamento  |
| `validation_service.py`   | Usa dataset durante validação    |
| `segmentation_dataset.py` | Versão local do dataset          |

---

# Importância no Projeto

O `StreamingSegmentationDataset` permitiu executar treinamento e validação sem necessidade de armazenar permanentemente o dataset localmente.

Isso tornou o pipeline:

* mais leve;
* mais flexível;
* mais escalável.

---

# Resumo 

O `StreamingSegmentationDataset` é responsável por carregar imagens remotamente sob demanda.

O serviço:

* baixa imagens em memória;
* aplica augmentations;
* normaliza dados;
* converte máscaras;
* gera tensores para treinamento da U-Net.

Seu principal benefício foi permitir processamento de grandes datasets sem necessidade de armazenamento local permanente.

```
```
