"""
Utility module for Population Growth Prediction model
Contains utility functions and decorators
"""
import os
import json
import time
import pickle
import functools
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K

def timing_decorator(func):
    """Decorator to measure execution time of functions"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} completed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def load_model_safely(model_path):
    """Load model with proper error handling"""
    try:
        model = tf.keras.models.load_model(model_path)
        return model, None
    except Exception as e:
        return None, str(e)

def check_gpu_availability():
    """Check if GPU is available for training"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Set memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPUs: {gpus}")
            print("GPU is available for training")
            return True
        except RuntimeError as e:
            print(f"Error setting memory growth: {e}")
            return False
    else:
        print("No GPU available, using CPU")
        return False

def save_model_metadata(feature_cols, model_info=None, output_dir='web_deployment'):
    """Save model metadata for deployment"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the list of feature names
    feature_df = pd.DataFrame({'feature': feature_cols})
    feature_df.to_csv(f'{output_dir}/features.csv', index=False)
    
    # Create model metadata
    if model_info is None:
        model_info = {}
        
    metadata = {
        'model_name': 'Population Growth Prediction Model',
        'feature_count': len(feature_cols),
        'created_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
        'features': feature_cols,
        **model_info
    }
    
    # Save metadata as JSON
    with open(f'{output_dir}/model_info.json', 'w') as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Saved model metadata to '{output_dir}/features.csv' and '{output_dir}/model_info.json'")

def save_scalers(feature_scaler, target_scaler, output_dir='web_deployment'):
    """Save scalers for later use"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the scalers
    with open(f'{output_dir}/feature_scaler.pkl', 'wb') as f:
        pickle.dump(feature_scaler, f)
        
    with open(f'{output_dir}/target_scaler.pkl', 'wb') as f:
        pickle.dump(target_scaler, f)
        
    print(f"Saved scalers to '{output_dir}/feature_scaler.pkl' and '{output_dir}/target_scaler.pkl'")

def load_scalers(input_dir='web_deployment'):
    """Load saved scalers"""
    feature_scaler_path = f'{input_dir}/feature_scaler.pkl'
    target_scaler_path = f'{input_dir}/target_scaler.pkl'
    
    if not os.path.exists(feature_scaler_path) or not os.path.exists(target_scaler_path):
        print(f"Scaler files not found in '{input_dir}'")
        return None, None
    
    try:
        with open(feature_scaler_path, 'rb') as f:
            feature_scaler = pickle.load(f)
            
        with open(target_scaler_path, 'rb') as f:
            target_scaler = pickle.load(f)
            
        print(f"Loaded scalers from '{input_dir}'")
        return feature_scaler, target_scaler
    
    except Exception as e:
        print(f"Error loading scalers: {e}")
        return None, None

def convert_to_tfjs(model_path, output_dir, feature_cols=None, scaler_params=None):
    """Convert TensorFlow model to TensorFlow.js format"""
    try:
        # Check if tensorflowjs module is available
        import tensorflowjs as tfjs
        
        print(f"Converting model {model_path} to TensorFlow.js format")
        
        # Load model
        model, error = load_model_safely(model_path)
        if model is None:
            print(f"Failed to load model: {error}")
            return False
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert model
        tfjs.converters.save_keras_model(model, output_dir)
        print(f"Model converted and saved to '{output_dir}'")
        
        # Save feature list
        if feature_cols is not None:
            with open(os.path.join(output_dir, 'feature_list.json'), 'w') as f:
                json.dump(feature_cols, f)
            print(f"Feature list saved to '{output_dir}/feature_list.json'")
        
        # Save scaler parameters
        if scaler_params is not None:
            with open(os.path.join(output_dir, 'scaler_params.json'), 'w') as f:
                json.dump(scaler_params, f)
            print(f"Scaler parameters saved to '{output_dir}/scaler_params.json'")
            
        return True
        
    except ImportError:
        print("Warning: tensorflowjs package not found. TensorFlow.js model not saved.")
        print("To install, run: pip install tensorflowjs")
        return False
    except Exception as e:
        print(f"Error converting model to TensorFlow.js: {e}")
        return False