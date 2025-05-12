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
def save_model_for_tfjs(model, model_dir):
    """Save model specifically for TensorFlow.js conversion"""
    import os
    
    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Save in SavedModel format (better for conversion)
    saved_model_path = os.path.join(model_dir, "saved_model")
    model.save(saved_model_path, save_format='tf')
    
    # Now convert to TensorFlow.js format
    try:
        import tensorflowjs as tfjs
        tfjs_dir = os.path.join(model_dir, "tfjs")
        os.makedirs(tfjs_dir, exist_ok=True)
        tfjs.converters.convert_tf_saved_model(
            saved_model_path,
            tfjs_dir
        )
        print(f"Successfully converted model to TensorFlow.js format in {tfjs_dir}")
        return True
    except Exception as e:
        print(f"Error converting to TensorFlow.js: {e}")
        return False
def convert_to_tfjs_saved_model(model_path, output_dir, feature_cols=None, scaler_params=None):
    """Convert TensorFlow model to TensorFlow.js format"""
    try:
        # Import the tensorflowjs module
        import tensorflowjs as tfjs
        import tensorflow as tf
        
        print(f"Converting model {model_path} to TensorFlow.js format")
        
        # Define tfjs output directory
        tfjs_dir = os.path.join(output_dir, "tfjs")
        os.makedirs(tfjs_dir, exist_ok=True)
        
        # Method 1: Direct conversion from HDF5 file
        try:
            print("Attempting direct conversion from HDF5...")
            tfjs.converters.convert_keras_model_to_tfjs(
                model_path,  # Original .h5 file
                tfjs_dir
            )
            print(f"Direct conversion successful, model saved to '{tfjs_dir}'")
        except Exception as e:
            print(f"Direct conversion failed: {e}")
            
            # Method 2: Load and then convert
            try:
                print("Trying load-and-convert method...")
                model = tf.keras.models.load_model(model_path)
                tfjs.converters.save_keras_model(model, tfjs_dir)
                print(f"Load-and-convert successful, model saved to '{tfjs_dir}'")
            except Exception as e2:
                print(f"Load-and-convert failed: {e2}")
                
                # Method 3: Export to saved_model first
                try:
                    print("Trying saved_model export method...")
                    model = tf.keras.models.load_model(model_path)
                    
                    # Export to SavedModel format
                    saved_model_path = os.path.join(output_dir, "saved_model")
                    model.export(saved_model_path)
                    
                    # Convert from SavedModel
                    tfjs.converters.convert_tf_saved_model(
                        saved_model_path,
                        tfjs_dir
                    )
                    print(f"SavedModel export method successful, model saved to '{tfjs_dir}'")
                except Exception as e3:
                    print(f"SavedModel export method failed: {e3}")
                    raise Exception("All conversion methods failed")
        
        # Save feature list
        if feature_cols is not None:
            with open(os.path.join(tfjs_dir, 'model_metadata.json'), 'w') as f:
                json.dump({
                    "input_shape": [1, len(feature_cols)],  # Single year, 57 features
                    "feature_names": feature_cols
                }, f, indent=2)
            print(f"Feature metadata saved to '{tfjs_dir}/model_metadata.json'")
        
        # Save scaler parameters
        if scaler_params is not None:
            with open(os.path.join(tfjs_dir, 'scaler_params.json'), 'w') as f:
                json.dump(scaler_params, f, indent=2)
            print(f"Scaler parameters saved to '{tfjs_dir}/scaler_params.json'")
            
        return True
        
    except ImportError:
        print("Warning: tensorflowjs package not found. TensorFlow.js model not saved.")
        print("To install, run: pip install tensorflowjs")
        return False
    except Exception as e:
        print(f"Error converting model to TensorFlow.js: {e}")
        return False
    
def simple_convert_to_tfjs(model_path, output_dir, feature_cols=None, scaler_params=None):
    """Simplified conversion to TensorFlow.js format"""
    try:
        import tensorflowjs as tfjs
        import tensorflow as tf
        
        print(f"Converting model {model_path} to TensorFlow.js format")
        
        # Define tfjs output directory
        tfjs_dir = os.path.join(output_dir, "tfjs")
        os.makedirs(tfjs_dir, exist_ok=True)
        
        # First try direct conversion from .keras file if available
        keras_path = model_path.replace('.h5', '.keras')
        if os.path.exists(keras_path):
            print(f"Using native Keras format file: {keras_path}")
            
            # Load the model first
            model = tf.keras.models.load_model(keras_path)
            
            # Then save directly to tfjs format
            tfjs.converters.save_keras_model(model, tfjs_dir)
            print(f"Model converted and saved to '{tfjs_dir}'")
        else:
            print(f"Keras format file not found, using HDF5: {model_path}")
            # Load the model first
            model = tf.keras.models.load_model(model_path)
            
            # Then save directly to tfjs format
            tfjs.converters.save_keras_model(model, tfjs_dir)
            print(f"Model converted and saved to '{tfjs_dir}'")
        
        # Save feature list and metadata
        if feature_cols is not None:
            with open(os.path.join(tfjs_dir, 'model_metadata.json'), 'w') as f:
                json.dump({
                    "input_shape": [1, len(feature_cols)],
                    "feature_names": feature_cols
                }, f, indent=2)
            print(f"Feature metadata saved to '{tfjs_dir}/model_metadata.json'")
        
        # Save scaler parameters
        if scaler_params is not None:
            with open(os.path.join(tfjs_dir, 'scaler_params.json'), 'w') as f:
                json.dump(scaler_params, f, indent=2)
            print(f"Scaler parameters saved to '{tfjs_dir}/scaler_params.json'")
            
        return True
    except Exception as e:
        print(f"Error converting model to TensorFlow.js: {e}")
        return False