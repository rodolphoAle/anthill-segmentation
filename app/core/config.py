"""Centralized application configuration via Pydantic BaseSettings.

All tunable parameters live here as typed fields with documented defaults.
Values can be overridden in three ways (highest priority wins):

  1. CLI flag         --  python run_training.py --epochs 50
  2. Environment var  --  UNET_NUM_EPOCHS=50 python run_training.py
  3. .env file        --  set in the .env file next to this repo
  4. Default below    --  the value defined in this class

The ``settings`` singleton is constructed once at import time and must be
treated as **read-only** throughout the application lifecycle.

Where each group is consumed
----------------------------
* General / Google Drive / Model / Data / Pipeline
      → app/main.py, app/service/data_service.py
* Training  basic
      → app/service/training_service.py  (start_training, _train_loop)
* Training  loss & optimisation
      → app/service/training_service.py  (start_training)
* Training  augmentations
      → app/service/data_service.py      (create_train_transforms)
"""

from __future__ import annotations

from pydantic_settings import BaseSettings  # pyright: ignore[reportMissingImports]


class Settings(BaseSettings):
    """Immutable application settings loaded from environment / .env."""

    # ── General ───────────────────────────────────────────────────────────────
    app_name: str = "UNet Segmentation Pipeline"
    debug: bool = False

    # ── Google Drive ──────────────────────────────────────────────────────────
    # Path to the service-account JSON key used to authenticate with Drive.
    google_credentials_path: str = "credentials.json"
    # ID of the root Drive folder that contains "treino/" and "validacao/".
    base_folder_id: str = "1slS6V7OWBaBny7v94K3Vx9eGHp7lph91"

    # ── Model ─────────────────────────────────────────────────────────────────
    # Path where the last-epoch weights are written at the end of training.
    model_save_path: str = "u_net.pth"
    # Path where the best val_loss checkpoint is saved during training.
    best_model_params_path: str = "best_model_params.pth"
    # Number of input image channels (3 = RGB).
    n_channels: int = 3
    # Number of output segmentation classes (2 = background + anthill).
    n_classes: int = 2

    # ── Training  basic ──────────────────────────────────────────────────────

    # batch_size: how many image-mask pairs are processed together in one
    # gradient update.
    #   ↑ larger  → smoother loss curve, more stable gradients, more VRAM
    #   ↓ smaller → noisier gradients (can help escape local minima), less VRAM
    #   Applied in: DataLoader(batch_size=...) inside data_service.py
    batch_size: int = 2

    # learning_rate: step size for the Adam optimiser.
    #   ↑ larger  → faster convergence but risk of overshooting / loss oscillation
    #   ↓ smaller → slower but stable; use if val_loss bounces between epochs
    #   Applied in: optim.Adam(lr=learning_rate) inside training_service.py
    learning_rate: float = 0.001

    # num_epochs: total passes over the full training dataset.
    #   ↑ larger  → more learning time; monitor val_loss to detect overfitting
    #   ↓ smaller → faster experiment cycle
    #   Applied in: _train_loop outer loop inside training_service.py
    num_epochs: int = 20

    # num_workers: subprocess workers that load batches from disk in parallel.
    #   0   → main process only (use for debugging DataLoader issues)
    #   1-4 → parallel I/O; recommended when data_mode="local"
    #   Applied in: DataLoader(num_workers=...) inside data_service.py
    num_workers: int = 2

    # device: compute backend.
    #   "auto" → CUDA if available, otherwise CPU
    #   "cuda" → force GPU (fails if no CUDA)
    #   "cpu"  → force CPU (very slow for training, useful for debugging)
    #   Applied in: TrainingService._resolve_device() inside training_service.py
    device: str = "auto"

    # ── Training  loss & optimisation ────────────────────────────────────────

    # class_weight_background: CrossEntropyLoss weight for the background class.
    # Usually kept at 1.0 as the reference weight.
    #   Applied in: nn.CrossEntropyLoss(weight=[bg, anthill]) in training_service.py
    class_weight_background: float = 1.0

    # class_weight_anthill: CrossEntropyLoss weight for the anthill class.
    # The anthill class is rare (heavily imbalanced), so it needs a higher weight.
    #   ↑ larger  → model penalised more for missing anthill pixels
    #               → better recall, but may produce more false positives
    #   ↓ smaller → safer against false positives, but may miss small anthills
    #   Applied in: nn.CrossEntropyLoss(weight=[bg, anthill]) in training_service.py
    class_weight_anthill: float = 6.0

    # focal_loss_gamma: gamma parameter for Focal Loss.
    # Focal Loss down-weights easy examples (plain soil correctly classified)
    # so the model focuses on hard cases (ambiguous soil near anthills).
    #   0.0  → equivalent to standard CrossEntropyLoss (also disables Focal Loss
    #           in favour of Tversky when tversky_alpha/beta > 0)
    #   1.0  → mild focus on hard examples
    #   2.0  → standard Focal Loss (recommended for imbalanced segmentation)
    #   Applied in: training_service.py when tversky_alpha=0 and gamma > 0
    focal_loss_gamma: float = 2.0

    # tversky_alpha / tversky_beta: weights for FP and FN in Tversky Loss.
    # TL = 1 - TP / (TP + alpha*FP + beta*FN)
    #   alpha=0.5, beta=0.5 → equivalent to Dice Loss
    #   alpha=0.3, beta=0.7 → FN penalised 2.3× more than FP; pushes Recall up
    #   alpha=0.0, beta=0.0 → disables Tversky Loss (falls back to Focal or CE)
    # NOTE: Tversky Loss works on soft probabilities and does not use
    #       class_weight_anthill  the beta parameter replaces that role.
    #   Applied in: training_service.py when both alpha and beta > 0 (takes
    #               priority over focal_loss_gamma)
    tversky_alpha: float = 0.3   # UNET_TVERSKY_ALPHA
    tversky_beta: float = 0.7    # UNET_TVERSKY_BETA

    # tversky_loss_weight: fraction of the combined loss assigned to Tversky.
    # The remaining (1 - tversky_loss_weight) goes to Focal Loss.
    # Using a combined loss prevents mode collapse: Focal anchors early training
    # while Tversky pushes for higher Recall once the model is stable.
    #   0.5 → equal weighting (recommended starting point)
    #   0.7 → stronger Recall push (use if Recall is still insufficient after run)
    #   0.0 → disables Tversky (pure Focal Loss  same as focal_loss_gamma > 0 path)
    #   Applied in: CombinedTverskyFocalLoss when tversky_alpha > 0 and beta > 0
    tversky_loss_weight: float = 0.5   # UNET_TVERSKY_LOSS_WEIGHT

    # lovasz_loss_weight: fraction of the combined loss assigned to Lovász Hinge.
    # Lovász directly optimises IoU (Jaccard index) via a surrogate that is
    # convex on the simplex  unlike Tversky/Focal which optimise pixel
    # classification with an IoU side-effect.  Adding Lovász is the standard
    # technique to recover IoU when Tversky pushes Recall too aggressively.
    #   0.0  → disabled (only Tversky+Focal active, original Run 10 behaviour)
    #   0.3  → balanced triple loss: 0.5·Tversky + (1-0.5-0.3)·Focal + 0.3·Lovász
    #   0.5+ → IoU-dominant; risk of mode collapse early in training
    # NOTE: When > 0, the focal weight becomes (1 - tversky_loss_weight - lovasz_loss_weight).
    #   Applied in: training_service.py CombinedTverskyFocalLoss when > 0
    lovasz_loss_weight: float = 0.3   # UNET_LOVASZ_LOSS_WEIGHT

    # grad_clip_max_norm: maximum L2 norm allowed for the full gradient vector.
    # Prevents exploding gradients (critical for UNets without BatchNorm).
    #   ↑ larger  → allows bigger gradient steps; faster early phases
    #   ↓ smaller → more conservative updates; safer but potentially slower
    #   Applied in: clip_grad_norm_(params, max_norm=...) in training_service.py
    grad_clip_max_norm: float = 1.0

    # scheduler_factor: multiplier applied to the LR when a plateau is detected.
    # new_lr = current_lr * scheduler_factor
    #   ↑ closer to 1.0 (e.g. 0.9) → gentle reduction, stays near original LR
    #   ↓ closer to 0.0 (e.g. 0.1) → aggressive cut, effectively stops learning
    #   Applied in: ReduceLROnPlateau(factor=...) in training_service.py
    scheduler_factor: float = 0.5

    # scheduler_patience: epochs with no val_loss improvement before reducing LR.
    #   ↑ larger  → gives more time before reducing LR; good for noisy datasets
    #   ↓ smaller → reacts faster to plateaus; risk of reducing LR too early
    #   Applied in: ReduceLROnPlateau(patience=...) in training_service.py
    #   Ignored when use_cosine_scheduler=True.
    scheduler_patience: int = 5

    # use_cosine_scheduler: swap ReduceLROnPlateau for CosineAnnealingLR.
    # CosineAnnealingLR follows a fixed cosine decay over num_epochs, making it
    # immune to noisy val_loss (e.g. caused by WeightedRandomSampler).  This
    # prevents the scheduler from freezing the LR too early on a noisy curve.
    #   True  → CosineAnnealingLR (T_max=num_epochs, eta_min=cosine_eta_min)
    #   False → original ReduceLROnPlateau behaviour
    #   Applied in: training_service.py start_training
    use_cosine_scheduler: bool = True   # UNET_USE_COSINE_SCHEDULER

    # cosine_eta_min: lower bound LR at the end of the cosine cycle.
    #   Keeps a tiny update rate alive even at the last epoch.
    #   Applied in: CosineAnnealingLR(eta_min=...) when use_cosine_scheduler=True
    cosine_eta_min: float = 1e-6   # UNET_COSINE_ETA_MIN

    # ── Training  data augmentations ─────────────────────────────────────────
    # All augmentations are applied jointly to the image AND the mask so spatial
    # correspondence is preserved.  Applied in: create_train_transforms() in
    # data_service.py.  Augmentations do NOT affect the validation set.

    # aug_horizontal_flip: randomly mirror image left↔right with p=0.5.
    #   True  → strongly recommended for aerial imagery (no fixed orientation)
    aug_horizontal_flip: bool = True

    # aug_vertical_flip: randomly mirror image top↔bottom with p=0.5.
    #   True  → also recommended for aerial imagery
    aug_vertical_flip: bool = True

    # aug_rotation_degrees: max angle for RandomRotation in degrees.
    #   ↑ larger (→ 90) → more orientational variety; can push label pixels
    #                      outside the tile at large angles (handled by ignore)
    #   0               → disables rotation augmentation
    aug_rotation_degrees: int = 15

    # aug_color_jitter: apply random brightness / contrast / saturation shifts.
    # Does NOT affect the mask (applied to image only).
    #   True  → useful when images come from different flights, sensors, or seasons
    aug_color_jitter: bool = True

    # aug_color_jitter_brightness: max brightness deviation as fraction of original.
    #   0.0 → no change  |  0.5 → ±50%  |  keep below 0.3 for aerial imagery
    aug_color_jitter_brightness: float = 0.2

    # aug_color_jitter_contrast: max contrast deviation as fraction of original.
    #   0.0 → no change  |  0.5 → ±50%  |  keep below 0.3 for aerial imagery
    aug_color_jitter_contrast: float = 0.2

    # aug_color_jitter_saturation: max saturation deviation as fraction of original.
    #   0.0 → no change  |  keep low (0.1) to avoid unnatural colours
    aug_color_jitter_saturation: float = 0.1

    # aug_random_rotate_90: randomly rotate by 90°, 180°, or 270° (p=0.5).
    # Complementary to aug_rotation_degrees which handles small angles.
    # Particularly useful for aerial imagery where orientation is arbitrary.
    aug_random_rotate_90: bool = False

    # aug_elastic_transform: apply ElasticTransform to deform object borders.
    # Creates realistic shape variations of anthill boundaries, improving
    # generalisation with a limited number of positive examples.
    aug_elastic_transform: bool = True

    # aug_elastic_alpha: intensity of the elastic displacement field.
    #   ↑ larger → stronger deformation  |  50-80 is a good range for 256px tiles
    aug_elastic_alpha: float = 25.0

    # aug_elastic_sigma: smoothness of the displacement field.
    #   ↑ larger → smoother, more global deformation  |  5-7 is typical
    aug_elastic_sigma: float = 4.0

    # aug_copy_paste: paste random anthill regions from positive tiles onto
    # negative tiles during training.  Creates genuinely new training examples
    # instead of merely repeating existing positive tiles via oversampling.
    # DISABLED for Run 12: visual previews showed unrealistic compositions
    # (anthills pasted on dense vegetation, oversized blobs, context mismatch).
    # Replaced by aug_anthill_duplicate (intra-tile rotation), which preserves
    # the natural context (lighting, soil type, surrounding vegetation).
    aug_copy_paste: bool = False

    # aug_copy_paste_prob: probability of applying copy-paste to a negative tile.
    #   0.5 → half of negative tiles get a pasted anthill each epoch
    aug_copy_paste_prob: float = 0.4

    # aug_anthill_duplicate: when a tile contains anthills in the label, extract
    # them, rotate (90/180/270 + optional flip), and paste copies onto empty
    # regions of the SAME tile.  Unlike copy-paste, the anthill remains in its
    # natural context (no cross-image artefacts).  Multiplies positive pixel
    # count per epoch without introducing artificial colour/lighting mismatches.
    aug_anthill_duplicate: bool = True

    # aug_anthill_duplicate_prob: probability of applying duplication to a
    # positive tile (one that has anthills in the label).
    #   0.7 → 70% of positive tiles get 1-N rotated copies appended each epoch
    aug_anthill_duplicate_prob: float = 0.7

    # aug_anthill_duplicate_max_copies: maximum number of rotated copies pasted
    # onto a single tile.  Actual count is sampled from 1..max each call.
    #   2 → 1 or 2 extra copies (default)
    #   3 → up to 3 (use if dataset has many small isolated anthills)
    aug_anthill_duplicate_max_copies: int = 2

    # ── Data ──────────────────────────────────────────────────────────────────
    # data_mode: where images come from.
    #   "local"  → reads from local_data_dir on disk (fast, no internet needed)
    #   "online" → streams from Google Drive on-demand (no local storage needed)
    data_mode: str = "local"
    local_data_dir: str = "data"
    # Subdirectory paths relative to local_data_dir (local mode only).
    train_rgb_subdir: str = "training/rgb/rgb"
    train_labels_subdir: str = "training/labels/labels"
    val_rgb_subdir: str = "validation/rgb/rgb"
    val_labels_subdir: str = "validation/labels/labels"
    # preload_dataset: load ALL images into RAM at startup.
    #   True  → eliminates disk I/O during training; only viable if dataset fits in RAM
    #   False → reads from disk per batch (default, works for any dataset size)
    preload_dataset: bool = False

    # max_ignore_pixel_pct: drop training tiles whose label has more than this
    # fraction of ignore pixels (white = unlabelled border padding).  Many tiles
    # at dataset borders are 70-95% white and contribute zero supervision signal
    # per epoch  every batch step on them wastes compute.
    #   0.7  → drop tiles where >70% pixels are ignore (recommended)
    #   1.0  → disable filter (keep all tiles, original behaviour)
    # Applied in: SegmentationDataset._match_pairs()  train set only, never val.
    max_ignore_pixel_pct: float = 0.7

    # ── Pipeline ──────────────────────────────────────────────────────────────
    # pipeline_mode: what to run when the container starts.
    #   "train"    → train model, save weights to model_save_path
    #   "validate" → load weights, run metrics, save detections to validation_output_dir
    pipeline_mode: str = "train"
    validation_output_dir: str = "output/validation_results"
    # anthill_save_threshold: minimum % of pixels predicted as anthill to save
    # the image to validation_output_dir.
    #   ↑ larger  → only saves tiles with large anthill regions (fewer, more confident)
    #   ↓ smaller → saves tiles with even tiny anthill detections (more, noisier)
    anthill_save_threshold: float = 40.0

    # anthill_confidence_threshold: minimum softmax probability (0.5–1.0) required
    # to classify a pixel as anthill. Values above 0.5 make the model less trigger-happy.
    #   0.40 → more sensitive; improves recall for small anthills
    #   0.5  → same as argmax (default behaviour  accept any majority vote)
    #   0.7  → only mark pixel as anthill if model is ≥70% confident
    #   0.9  → very conservative; reduces false positives significantly
    anthill_confidence_threshold: float = 0.40

    # min_anthill_region_px: minimum number of connected pixels to keep as a valid
    # anthill detection. Isolated fragments smaller than this are removed (set to
    # background) after the confidence threshold is applied.
    #   ↑ larger → fewer, bigger detections (removes scattered noise)
    #   ↓ smaller (→ 1) → keeps every pixel cluster, even single-pixel noise
    min_anthill_region_px: int = 5

    # max_anthill_region_px: maximum number of connected pixels allowed for a
    # region to be kept as a valid anthill detection. Very large regions are
    # almost always false positives caused by reddish soil, crop furrows, or
    # field boundaries being mistaken for anthills.
    #
    # In a 256×256 tile, a formigueiro with ~70px diameter ≈ π*35²≈3.850px.
    # Blobs following sulco/road patterns are typically >8.000px.
    #
    #   ↑ larger (→ 65536) → tolerates bigger detections (fewer filtered out)
    #   ↓ smaller           → stricter; removes large false-positive blobs
    #   0                   → disables upper-bound filter
    max_anthill_region_px: int = 5000

    # use_region_filter: whether to apply the connected-component size filter
    # (min_anthill_region_px / max_anthill_region_px) after the confidence
    # threshold during validation.
    #   True  → filter active (default): removes noise fragments and oversized blobs
    #   False → raw confidence-threshold output, no region post-processing
    #           (useful to evaluate what the model produces without any cleanup)
    use_region_filter: bool = True

    model_config = {"env_file": ".env", "env_prefix": "UNET_"}


settings = Settings()