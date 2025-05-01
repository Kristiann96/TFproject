# First, install the required packages
# pip install tensorflowjs tensorflow

import tensorflow as tf
import tensorflowjs as tfjs
import pandas as pd
import json
import os

# Load your trained model
model_path = 'population_growth_model.h5'
model = tf.keras.models.load_model(model_path)

# Create output directory for the TF.js model
output_dir = 'tfjs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Convert the model to TensorFlow.js format
tfjs.converters.save_keras_model(model, output_dir)
print(f"Model converted and saved to {output_dir}")

# Save the feature scaler parameters
if os.path.exists('model_features.csv'):
    features_df = pd.read_csv('model_features.csv')
    feature_list = features_df['feature'].tolist()
    
    # Save feature list as JSON for JavaScript
    with open(os.path.join(output_dir, 'feature_list.json'), 'w') as f:
        json.dump(feature_list, f)
    print(f"Feature list saved to {output_dir}/feature_list.json")

# If you have saved scalers, you could also save their parameters
# This example assumes you're using RobustScaler and have saved its parameters
try:
    import joblib
    import numpy as np
    
    feature_scaler = joblib.load('feature_scaler.joblib')
    target_scaler = joblib.load('target_scaler.joblib')
    
    # Save scaler parameters
    scaler_params = {
        'feature_scaler': {
            'center_': feature_scaler.center_.tolist() if hasattr(feature_scaler, 'center_') else None,
            'scale_': feature_scaler.scale_.tolist() if hasattr(feature_scaler, 'scale_') else None,
        },
        'target_scaler': {
            'center_': target_scaler.center_.tolist() if hasattr(target_scaler, 'center_') else None,
            'scale_': target_scaler.scale_.tolist() if hasattr(target_scaler, 'scale_') else None,
        }
    }
    
    with open(os.path.join(output_dir, 'scaler_params.json'), 'w') as f:
        json.dump(scaler_params, f)
    print(f"Scaler parameters saved to {output_dir}/scaler_params.json")
except Exception as e:
    print(f"Couldn't save scaler parameters: {e}")
    # Generate dummy scaler parameters if real ones aren't available
    print("Creating dummy scaler parameters")
    feature_count = model.input_shape[-1]
    scaler_params = {
        'feature_scaler': {
            'center_': [0] * feature_count,
            'scale_': [1] * feature_count,
        },
        'target_scaler': {
            'center_': [0],
            'scale_': [1],
        }
    }
    with open(os.path.join(output_dir, 'scaler_params.json'), 'w') as f:
        json.dump(scaler_params, f)
    print(f"Dummy scaler parameters saved to {output_dir}/scaler_params.json")

print("Model conversion complete!")