#!/usr/bin/env bash
#  UNet Segmentation Pipeline  interactive launcher 
# Drop a usage banner and then exec whatever command was passed
# (defaults to bash so the user can pick what to run).

cat <<'EOF'
╔══════════════════════════════════════════════════════════════════╗
║         UNet Segmentation Pipeline  available commands          ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  TRAIN (streams images, trains model, saves weights)             ║
║              ║
║  python -m app.main                                              ║
║    → uses UNET_PIPELINE_MODE=train (default)                     ║
║                                                                  ║
║  VALIDATE (loads saved weights, streams val images,              ║
║            saves anthill detections to validation_results/)      ║
║              ║
║  UNET_PIPELINE_MODE=validate python -m app.main                  ║
║                                                                  ║
║  HOT-RELOAD (auto-restarts on code change  dev only)            ║
║              ║
║  watchfiles "python -m app.main" ./app                           ║
║  UNET_PIPELINE_MODE=validate watchfiles "python -m app.main" ./app
║                                                                  ║
║  Useful env-var overrides (prefix UNET_):                        ║
║    UNET_NUM_EPOCHS=30  UNET_BATCH_SIZE=8  UNET_DEVICE=cpu        ║
║    UNET_VALIDATION_OUTPUT_DIR=my_results                         ║
╚══════════════════════════════════════════════════════════════════╝

EOF

exec "$@"
