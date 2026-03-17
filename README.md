# UNet Segmentation Pipeline

Detecção automática de formigueiros em imagens aéreas utilizando uma rede U-Net treinada com segmentação semântica binária (formigueiro / não formigueiro).

---

## Como o projeto funciona

### Visão geral

O pipeline acessa as imagens **diretamente do Google Drive via API** — nenhuma imagem é baixada em massa para o disco. Cada par (imagem RGB + máscara de label) é buscado em memória (`BytesIO`), processado e descartado antes do próximo par ser carregado.

A estrutura de pastas esperada no Google Drive é:

```
<pasta raiz>
├ treino/
│   ├ rgb/         ← imagens RGB de treinamento
│   └ labels/      ← máscaras de segmentação (.png)
└ validacao/
    ├ rgb/         ← imagens RGB de validação
    └ labels/      ← máscaras de segmentação (.png)
```

---

### Modo Treinamento (`run_training.py`)

1. Os **metadados** (IDs e nomes) de todos os arquivos das pastas `treino/rgb` e `treino/labels` são listados via API do Drive.
2. Um `StreamingSegmentationDataset` é criado — ele guarda apenas os IDs; as imagens são baixadas sob demanda a cada batch.
3. A U-Net é treinada com `CrossEntropyLoss` e otimizador `Adam`, com augmentações (flip horizontal, rotação aleatória).
4. Ao fim de cada época, uma passagem de validação é executada com as imagens de `validacao/`.
5. Os pesos do modelo são salvos em disco ao final (`u_net.pth`).

**Nada é escrito em disco durante o treinamento** (somente o arquivo `.pth` ao final).

---

### Modo Validação (`run_validation.py`)

1. Os pesos salvos (`u_net.pth`) são carregados.
2. Cada par de `validacao/rgb` + `validacao/labels` é baixado em memória, um a um.
3. A inferência é executada na GPU e as seguintes métricas são calculadas por imagem:
   - **Cobertura de formigueiro (%)** — percentual de pixels classificados como formigueiro
   - **IoU** (Intersection over Union)
   - **Dice Score**
4. Se a cobertura de formigueiro ultrapassar o threshold (`UNET_ANTHILL_SAVE_THRESHOLD`, padrão `40%`), o par é salvo em disco na pasta `validation_results/`:
   - `<nome>_rgb.png` — imagem original
   - `<nome>_mask.png` — máscara predita
5. Ao final, as métricas agregadas são exibidas no terminal:
   - Pixel Accuracy global
   - mIoU médio
   - Dice médio
   - Total de detecções salvas

**Somente imagens com formigueiro detectado acima do threshold são salvas em disco.**

---

### Arquitetura do projeto

```
app/
├ core/
│   ├ config.py              ← configurações via variáveis de ambiente
│   ├ exceptions.py          ← exceções de domínio
│   └ logging_config.py      ← configuração de logs (loguru)
├ domain/
│   ├ protocols.py           ← contratos (StorageClientProtocol)
│   └ unet.py                ← arquitetura da U-Net
├ infrastructure/
│   ├ google_drive_client.py ← cliente assíncrono do Google Drive
│   ├ segmentation_dataset.py   ← dataset para arquivos locais
│   └ streaming_dataset.py      ← dataset streaming (sem disco)
├ service/
│   ├ data_service.py        ← criação de DataLoaders (streaming)
│   ├ training_service.py    ← loop de treino e avaliação
│   └ validation_service.py  ← validação com métricas e salvamento
└ main.py                    ← dispatcher (train / validate)

run_training.py    ← ponto de entrada para treinamento
run_validation.py  ← ponto de entrada para validação
```

---

## Como executar

### Pré-requisitos

- [Docker](https://docs.docker.com/get-docker/) instalado
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) instalado (para uso de GPU)
- Arquivo `credentials.json` de uma Service Account do Google com acesso ao Drive na raiz do projeto
- Arquivo `.env` configurado (veja a seção [Configuração](#configuração))

---

### 1. Build da imagem

```bash
docker compose build
```

> Só é necessário rebuildar quando o `Dockerfile` ou o `requirements.txt` mudar. Alterações no código Python são refletidas instantaneamente pelo volume mount.

---

### 2. Subir o container em background

```bash
docker compose up -d
```

O container fica rodando em modo idle (`sleep infinity`) aguardando você entrar.

---

### 3. Entrar no container

```bash
docker exec -it unet-segmentation-pipeline bash
```

---

### 4. Treinar o modelo

Dentro do container:

```bash
python run_training.py
```

Os pesos serão salvos em `u_net.pth` (mapeado para o seu disco local pelo volume mount).

---

### 5. Validar o modelo

Dentro do container:

```bash
python run_validation.py
```

As imagens com formigueiro detectado são salvas em `validation_results/` (também visível no seu disco local).

---

### 6. Parar o container

```bash
# Fora do container
docker compose down
```

---

## Configuração

Todas as variáveis são definidas no arquivo `.env` na raiz do projeto:

| Variável | Padrão | Descrição |
|---|---|---|
| `UNET_BASE_FOLDER_ID` | — | ID da pasta raiz no Google Drive |
| `UNET_GOOGLE_CREDENTIALS_PATH` | `credentials.json` | Caminho para o JSON da service account |
| `UNET_MODEL_SAVE_PATH` | `u_net.pth` | Arquivo onde os pesos são salvos |
| `UNET_NUM_EPOCHS` | `20` | Número de épocas de treinamento |
| `UNET_BATCH_SIZE` | `4` | Tamanho do batch |
| `UNET_LEARNING_RATE` | `0.001` | Taxa de aprendizado |
| `UNET_N_CLASSES` | `2` | Número de classes (binário: 0=fundo, 1=formigueiro) |
| `UNET_ANTHILL_SAVE_THRESHOLD` | `40.0` | % mínimo de pixels de formigueiro para salvar a imagem |
| `UNET_VALIDATION_OUTPUT_DIR` | `validation_results` | Pasta de saída das detecções |
| `UNET_DEBUG` | `false` | Ativa logs detalhados |

Você pode sobrescrever qualquer variável na hora da execução:

```bash
UNET_NUM_EPOCHS=50 python run_training.py
UNET_ANTHILL_SAVE_THRESHOLD=80.0 python run_validation.py
```
