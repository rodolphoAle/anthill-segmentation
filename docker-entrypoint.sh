#!/usr/bin/env bash
#  UNet Segmentation Pipeline  interactive launcher 
# Drop a usage banner and then exec whatever command was passed
# (defaults to bash so the user can pick what to run).

cat <<'EOF'
╔══════════════════════════════════════════════════════════════════╗
║         UNet Segmentation Pipeline — available commands          ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  TRAIN                                                           ║
║  python -m app.main train [--epochs 50] [--lr 0.0005]            ║
║                                                                  ║
║  VALIDATE                                                        ║
║  python -m app.main validate [--device cpu] [--output-dir DIR]   ║
║                                                                  ║
║  EVALUATE                                                        ║
║  python -m app.main evaluate --pred-dir DIR [--save-dir DIR]     ║
║                                                                  ║
║  HOT-RELOAD (dev only)                                           ║
║  watchfiles "python -m app.main train" ./app                     ║
║                                                                  ║
║  Each sub-command accepts --help for a full list of flags.        ║
║  All flags override UNET_* env-vars for the current run only.    ║
╚══════════════════════════════════════════════════════════════════╝

EOF

exec "$@"
