#!/bin/bash

echo "Starting quick training…"

# Train with default settings (5 epochs, local data)
python -m app.main train \
  --epochs 5 \
  --batch-size 4 \
  --data-mode local || exit 1

echo "Training complete!"