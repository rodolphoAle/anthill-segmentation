# Segmentação Semântica de Formigueiros em Imagens Aéreas via U-Net

## Resumo Executivo

Este repositório contém a implementação completa de um sistema de segmentação semântica para detecção e delineamento de formigueiros em ortofotografias aéreas de alta resolução. O projeto utiliza arquitetura U-Net com loss combinada (Tversky + Focal + Lovász) para otimizar múltiplos objetivos em contexto de forte desbalanceamento de classes.

## Objetivo de Pesquisa

**Questão central**: Qual é o impacto relativo do **pipeline de dados** e da **seleção de funções de perda** na performance de segmentação semântica sob cenários com desbalanceamento extremo?

**Hypótese**: Melhorias no pipeline de dados produzem ganho de performance superior à otimização isolada de funções de perda.

---

## Funcionalidades Principais

- **Treinamento** com augmentações configuráveis (flip, rotação, elastic transform, copy-paste)
- **Inferência** com threshold de confiança ajustável e filtragem por tamanho de componente
- **Avaliação** com métricas completas (IoU por classe, Dice, Precision, Recall, F1)
- **Aceleração GPU** via CUDA com fallback automático para CPU
- **Fontes de dados** flexíveis: disco local ou streaming de Google Drive

---

## Project Structure

```
.
├ app/                              # Main application package
│   ├ main.py                       # Async CLI entry point (train / validate)
│   ├ core/                         # Configuration & cross-cutting concerns
│   │   ├ config.py                 #   Pydantic BaseSettings (all parameters)
│   │   ├ exceptions.py             #   Domain exception hierarchy
│   │   └ logging_config.py         #   Structured logging (loguru)
│   ├ domain/                       # Business logic & model definition
│   │   ├ unet.py                   #   UNet architecture (GroupNorm)
│   │   ├ protocols.py              #   StorageClientProtocol (DI contract)
│   │   ├ mask_utils.py             #   RGB mask decode/encode utilities
│   │   ├ metrics.py                #   ValidationMetrics & EvaluationMetrics
│   │   └ losses/                   #   Loss functions module
│   │       ├ focal_loss.py         #     Focal Loss (hard example mining)
│   │       ├ tversky_loss.py       #     Tversky Loss (Recall optimisation)
│   │       ├ lovasz_loss.py        #     Lovász Hinge (direct IoU surrogate)
│   │       └ combined_loss.py      #     Weighted combination of all three
│   ├ infrastructure/               # Data loading & external integrations
│   │   ├ google_drive_client.py    #   Async Google Drive wrapper
│   │   ├ augmentations.py          #   Transform builders & augmentation strategies
│   │   ├ segmentation_dataset.py   #   Local-file PyTorch Dataset
│   │   └ streaming_dataset.py      #   Zero-disk streaming Dataset
│   ├ service/                      # Application services (orchestration)
│   │   ├ data_service.py           #   Dataset download & DataLoader creation
│   │   ├ training_service.py       #   Training loop & model persistence
│   │   ├ validation_service.py     #   Streaming validation & metric computation
│   │   └ prediction_service.py     #   Single-image prediction
│   └ visualization/                # Plotting helpers
│       └ plotting.py               #   Matplotlib side-by-side panels
├ scripts/                          # CLI scripts & utilities
│   ├ run_training.py               #   Training CLI (also a sub-command)
│   ├ run_validation.py             #   Validation CLI (also a sub-command)
│   ├ run_evaluate.py               #   Evaluation CLI (also a sub-command)
│   ├ check_label_validity.py       #   Validate label masks
│   ├ debug_nan_batch.py            #   Inspect single batch for NaN/Inf
│   └ visualize_copy_paste.py       #   Preview augmentations
├ Dockerfile                        # CUDA 12.4 + Python 3.11
├ docker-compose.yml                # GPU passthrough + hot-reload
├ requirements.txt                  # Python dependencies
└ docs/                             # Project documentation
```

---

## Architecture

The project follows a **layered architecture** with strict separation of concerns:

```
CLI (python -m app.main <command>)  →  Service Layer  →  Domain Layer  →  Infrastructure Layer
```

| Layer | Responsibility |
|---|---|
| **Core** | Configuration, exceptions, logging |
| **Domain** | UNet model, loss functions, protocols, metrics, mask utilities |
| **Service** | Training orchestration, data pipeline, validation, prediction |
| **Infrastructure** | Datasets, Google Drive client, augmentations |

**Key design patterns:**
- **Dependency Inversion**: Services depend on `StorageClientProtocol`, not concrete implementations
- **Single Responsibility**: Losses, augmentations, metrics, mask decoding are each in dedicated modules
- **Composition over Inheritance**: Complex behaviour built from composable functions

---

## Quick Start

### Docker (recommended)

```bash
docker compose build
docker compose up -d
docker exec -it unet-segmentation-pipeline bash

# Inside the container:
python -m app.main train --epochs 50 --lr 0.0005
python -m app.main validate --device cuda
python -m app.main evaluate --pred-dir output/validation_results --save-dir output/evaluation
```

### Local

```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

python -m app.main train
python -m app.main validate
python -m app.main evaluate --pred-dir output/validation_results
```

You can also invoke each script directly:

```bash
python scripts/run_training.py --epochs 50
python scripts/run_validation.py --device cpu
python scripts/run_evaluate.py --pred-dir output/validation_results
```

---

## Configuration

All parameters are configured via environment variables (prefix `UNET_`) or a `.env` file.
See [app/core/config.py](app/core/config.py) for the full list with documentation.

Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `UNET_DATA_MODE` | `local` | `local` or `online` (Google Drive streaming) |
| `UNET_BATCH_SIZE` | `2` | Images per gradient update |
| `UNET_LEARNING_RATE` | `0.001` | Adam optimiser LR |
| `UNET_NUM_EPOCHS` | `20` | Training epochs |
| `UNET_DEVICE` | `cuda` | `cuda`, `cpu`, or `auto` |
| `UNET_ANTHILL_CONFIDENCE_THRESHOLD` | `0.40` | Minimum softmax probability for anthill class |
| `UNET_USE_REGION_FILTER` | `true` | Enable connected-component size filtering |

---

## Loss Function

The training uses a **triple-component loss**:

$$\mathcal{L} = w_T \cdot \text{Tversky} + w_L \cdot \text{Lovász} + w_F \cdot \text{Focal}$$

- **Focal Loss**: Anchors early training, down-weights easy examples
- **Tversky Loss**: Explicitly optimises Recall ($\beta > \alpha$)
- **Lovász Hinge**: Direct IoU surrogate for boundary precision

---

## Google Drive Structure

```
<root folder>/
├ treino/
│   ├ rgb/
│   └ labels/
└ validacao/
    ├ rgb/
    └ labels/
```

---

## License

This project is open source. See [LICENSE](LICENSE) for details.

