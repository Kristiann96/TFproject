"""
Configuration module for Population Growth Prediction model
Centralizes all configuration parameters
"""
import os

class ModelConfig:
    # Basic configuration
    SEQ_LENGTH = 1  # SIMPLIFIED: Use only 1 year for input sequence (was 8)
    BATCH_SIZE = 64
    EPOCHS = 150
    EARLY_STOPPING_PATIENCE = 25
    RANDOM_SEED = 42
    
    # Model architecture - SIMPLIFIED for single year input
    LSTM_UNITS = [64, 32]  # Reduced complexity (was [256, 128, 64])
    DENSE_UNITS = [32, 16, 1]  # Reduced complexity
    DROPOUT_RATES = [0.2, 0.2, 0.2]  # Adjusted dropout
    
    # File paths
    MODEL_PATH = 'population_growth_model.h5'
    ENHANCED_MODEL_PATH = 'population_growth_model_enhanced.h5'
    TFJS_MODEL_DIR = 'population_growth_model_tfjs'
    
    # Prediction configuration
    START_YEAR = 2025
    END_YEAR = 2030
    PREDICTION_OUTPUT_DIR = "Predictions"
    
    # Data files
    FEATURE_LIST_FILE = 'model_features.csv'
    COMPLETE_GRUNNKRETSER_FILE = 'complete_grunnkretser.csv'
    
    # Ensure required directories exist
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        os.makedirs(cls.PREDICTION_OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.TFJS_MODEL_DIR, exist_ok=True)
        os.makedirs('web_deployment', exist_ok=True)