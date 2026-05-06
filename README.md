# UNet Segmentation Pipeline

Detecção automática de formigueiros em imagens aéreas utilizando uma rede U-Net com segmentação semântica binária (fundo vs formigueiro).

---

## Visão Geral

O pipeline utiliza GPU automaticamente quando disponível, e executa:

- Treinamento com patch training (256x256)
- Inferência com threshold configurável
- Pós-processamento com filtro de regiões
- Avaliação com métricas (IoU, Dice, Pixel Accuracy)

As imagens são consumidas diretamente do Google Drive via streaming, sem download em massa para disco.

---

## Estrutura esperada no Google Drive

```
<pasta raiz>
├ treino/
│   ├ rgb/
│   └ labels/
└ validacao/
    ├ rgb/
    └ labels/
```

---

## Melhorias implementadas

### Patch Training

- Patches de 256x256
- Seleção baseada em presença mínima de formigueiros
- Redução do desbalanceamento de classes

### Métricas corrigidas

- Máscaras RGB decodificadas corretamente
- Pixels com valor 255 ignorados nas métricas
- Cálculo correto de IoU e Dice

### Treinamento otimizado

- Salvamento do modelo baseado no melhor IoU
- Não dependente apenas da função de perda

### Inferência aprimorada

- Separação entre threshold de segmentação e de detecção
- Filtro de regiões para remoção de ruído

---

## Configuração (.env)

```env
# Application 
UNET_APP_NAME="UNet Segmentation Pipeline"
UNET_DEBUG=false

# Google Drive 
UNET_GOOGLE_CREDENTIALS_PATH=credentials.json
UNET_BASE_FOLDER_ID=<id-da-pasta>

# Model 
UNET_MODEL_SAVE_PATH=u_net.pth
UNET_N_CHANNELS=3
UNET_N_CLASSES=2

# Training 
UNET_BATCH_SIZE=2
UNET_LEARNING_RATE=0.001
UNET_NUM_EPOCHS=20
UNET_NUM_WORKERS=2

# Data 
UNET_DATA_MODE=online
UNET_LOCAL_DATA_DIR=data

# Output
UNET_ANTHILL_SAVE_THRESHOLD=40.0

# Segmentation Threshold
UNET_ANTHILL_CONFIDENCE_THRESHOLD=0.40

# Region Filter
UNET_USE_REGION_FILTER=true
UNET_MIN_ANTHILL_REGION_PX=5
UNET_MAX_ANTHILL_REGION_PX=5000
```

---

## Execução

### 1. Build

```bash
docker compose build
```

### 2. Subir container

```bash
docker compose up -d
```

### 3. Acessar container

```bash
docker exec -it unet-segmentation-pipeline bash
```

---

## 4. Validação do dataset

```bash
python validate_dataset.py --local-dir ./data/
```

Valida:

- Shape das imagens (3, 256, 256)
- Normalização ImageNet
- Distribuição de classes
- Funcionamento do backward

---

## 5. Treinamento

```bash
python run_training.py
```

O modelo:

- Treina com patch training
- Realiza validação por época
- Salva o melhor modelo com base em IoU

---

## 6. Validação do modelo

```bash
python run_validation.py
```

Saída:

- Pixel Accuracy
- Mean IoU
- Mean Dice
- Número de detecções

---

## Métricas

| Métrica           | Descrição                                        |
| ----------------- | ------------------------------------------------ |
| IoU               | Sobreposição entre predição e ground truth       |
| Dice              | Similar ao IoU, mais sensível a regiões pequenas |
| Pixel Accuracy    | Percentual de pixels corretamente classificados  |
| Anthill Detection | Percentual de área classificada como formigueiro |

---

## Observações

- Pixel Accuracy pode ser inflada devido ao desbalanceamento de classes
- As métricas mais relevantes são:
  - IoU da classe formigueiro
  - Dice

---

## Resultados esperados

Após as melhorias:

```
IoU (formigueiro): 0.45 – 0.60+
Dice:              0.60 – 0.75+
```

---

## Arquitetura

```
app/
├ core/
├ domain/
├ infrastructure/
├ service/
└ main.py
```
