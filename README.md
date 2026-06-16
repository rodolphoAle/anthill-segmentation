# UNet Segmentation Pipeline

Automatic detection of anthills (*formigueiros*) in aerial imagery using a U-Net architecture with binary semantic segmentation (background vs. anthill).

---

## Overview

- **Training** with configurable augmentations (flip, rotation, elastic, copy-paste, anthill duplication)
- **Inference** with adjustable confidence threshold and connected-component filtering
- **Evaluation** with full metrics (IoU, Dice, Pixel Accuracy, Precision, Recall, F1)
- **GPU acceleration** via CUDA with automatic fallback to CPU
- **Data sources**: local disk or streaming from Google Drive

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

### Prerequisites

- **Python 3.11** — or **Docker** (see below), which bundles everything.
- **PyTorch** — GPU (CUDA 12.4) recommended for training; CPU works for inference and small runs.
- A **dataset** in the expected layout — see [Data Modes](#data-modes-local-vs-online). No dataset or
  trained weights ship with this repo (see [Data & Model Availability](#data--model-availability)).

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

## Data Modes: Local vs Online

The pipeline reads data in one of two modes, selected with `UNET_DATA_MODE` (or `--data-mode`).

### Local mode (recommended to get started)

`UNET_DATA_MODE=local` — reads image/mask tiles straight from disk. **No credentials needed.**

Expected layout under `UNET_LOCAL_DATA_DIR` (default `data/`):

```
data/
├ training/
│   ├ rgb/rgb/        # training images (.png / .jpg / .jpeg / .tif)
│   └ labels/labels/  # training masks  (.png, RGB-encoded)
└ validation/
    ├ rgb/rgb/        # validation images
    └ labels/labels/  # validation masks
```

Images and masks are paired by filename stem. The subpaths are configurable via
`UNET_TRAIN_RGB_SUBDIR`, `UNET_TRAIN_LABELS_SUBDIR`, `UNET_VAL_RGB_SUBDIR`, `UNET_VAL_LABELS_SUBDIR`.
A 5-epoch smoke test is available in [quick_start.sh](quick_start.sh).

### Online mode (streaming from Google Drive)

`UNET_DATA_MODE=online` — streams tiles from Google Drive on demand (zero local disk), via
[app/infrastructure/google_drive_client.py](app/infrastructure/google_drive_client.py).

> **Note:** the authors' own Drive folder is **private and is not shared**. Online mode is optional —
> to use it you must supply **your own** Google service account and **your own** Drive folder.

Setup:

1. In the [Google Cloud Console](https://console.cloud.google.com/), create a project and enable the **Google Drive API**.
2. Create a **service account**, download its **JSON key**, and save it as `credentials.json` in the project root (it is git-ignored).
3. In Google Drive, create a root folder with the structure below and **share it with the service account's email** (the `client_email` in the JSON) as at least *Viewer*.
4. Set in your `.env`: `UNET_DATA_MODE=online`, `UNET_GOOGLE_CREDENTIALS_PATH=credentials.json`, and `UNET_BASE_FOLDER_ID=<id from the folder URL>`.

Expected Drive folder structure:

```
<root folder>/
├ treino/
│   ├ rgb/      # training images
│   └ labels/   # training masks
└ validacao/
    ├ rgb/      # validation images
    └ labels/   # validation masks
```

---

## Data & Model Availability

This repository contains **code and documentation only**. To keep it lightweight and respect data
ownership, it does **not** include:

- **The dataset** — the anthill ortophoto tiles are not redistributed here. Bring your own data in
  the [layout above](#data-modes-local-vs-online), or use online mode with your own Drive.
- **Trained weights** (`*.pth`) — git-ignored due to size. Reproduce them by training from scratch
  (`python -m app.main train ...`); the best checkpoint is written to `best_model_params.pth`.

Full write-up lives in [docs/](docs/): architecture decisions in [docs/arquitetura/](docs/arquitetura/)
and the article/report in [docs/artigo/](docs/artigo/).

---

## License

See [LICENSE](LICENSE) for details.

