#!/bin/bash

# Population Growth Model Training Script
# ---------------------------------------

# Set script to exit on error
set -e

# Configuration
VENV_DIR="venv"                   # Virtual environment directory
OUTPUT_PREFIX="v1_"               # Prefix for output files
MUNICIPALITIES=30                 # Number of municipalities to use (more diverse data)
MIN_SEQUENCES=1000                # Minimum training sequences
LOG_FILE="training_$(date +%Y%m%d_%H%M%S).log"  # Log file with timestamp

echo "Starting training script at $(date)"
echo "====================================="
echo "Log file: $LOG_FILE"

# Create or activate virtual environment
if [ -d "$VENV_DIR" ]; then
    echo "Activating virtual environment..."
    source $VENV_DIR/bin/activate
else
    echo "Creating virtual environment..."
    python -m venv $VENV_DIR
    source $VENV_DIR/bin/activate
    
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install tensorflow pandas numpy scikit-learn matplotlib seaborn tensorflowjs
fi

# Check for GPU availability
if python -c "import tensorflow as tf; print('GPU available:', len(tf.config.list_physical_devices('GPU')) > 0)"; then
    echo "GPU is available for training"
else
    echo "WARNING: No GPU detected. Training may be slow."
    echo "Continue anyway? (y/n)"
    read -r response
    if [[ "$response" != "y" ]]; then
        echo "Training aborted."
        exit 1
    fi
fi

# Run training with parameters
echo "Starting model training..."
python main.py train \
    --municipalities $MUNICIPALITIES \
    --min-sequences $MIN_SEQUENCES \
    --output-prefix $OUTPUT_PREFIX \
    --convert-to-js \
    2>&1 | tee $LOG_FILE

# Check for success
if [ $? -eq 0 ]; then
    echo "====================================="
    echo "Training completed successfully!"
    echo "Model saved as: population_growth_model.h5"
    echo "TensorFlow.js model saved in: population_growth_model_tfjs"
    echo "Log file: $LOG_FILE"
    
    # Run a quick prediction to test the model
    echo "Running a test prediction for years 2025-2026..."
    python main.py predict --start-year 2025 --end-year 2026
else
    echo "====================================="
    echo "ERROR: Training failed. See log file for details: $LOG_FILE"
    exit 1
fi

echo "Training script finished at $(date)"