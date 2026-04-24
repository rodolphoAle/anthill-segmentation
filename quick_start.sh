#!/bin/bash

echo "Iniciando..."

# Verifica ambiente
python setup_training.py --check || exit 1

# Valida dataset
python validate_fixes.py || exit 1

# Verifica dados
python setup_training.py --check-dataset || exit 1

# Treina modelo
python train_with_monitoring.py \
  --epochs 5 \
  --batch-size 4 \
  --local-data \
  --data-dir ./data/ || exit 1

echo "Treinamento finalizado!"