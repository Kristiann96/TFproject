#!/bin/bash

# Simple Training Script (without venv management)
# ----------------------------------------------

# Configuration
OUTPUT_PREFIX="1year_"
MUNICIPALITIES=30
MIN_SEQUENCES=1000
LOG_FILE="training_$(date +%Y%m%d_%H%M%S).log"

echo "Starting training script at $(date)"
echo "====================================="

# Run training with parameters
echo "Starting model training..."
python main.py train \
    --municipalities $MUNICIPALITIES \
    --min-sequences $MIN_SEQUENCES \
    --output-prefix $OUTPUT_PREFIX \
    --convert-to-js \
    2>&1 | tee $LOG_FILE

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "====================================="
    echo "Training completed successfully!"
    
    # Run a quick test prediction
    echo "Running a test prediction for years 2025-2026..."
    python main.py predict --start-year 2025 --end-year 2026
else
    echo "====================================="
    echo "ERROR: Training failed. See log file for details: $LOG_FILE"
    exit 1
fi

echo "Training script finished at $(date)"