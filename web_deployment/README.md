# Population Growth Prediction Model

This directory contains files for deploying the Norwegian population growth prediction model to web applications.

## Files:
- model/: TensorFlow.js model files
- features.csv: List of features used by the model
- model_info.json: Metadata about the model
- feature_scaler.pkl: Scikit-learn scaler for input features
- target_scaler.pkl: Scikit-learn scaler for prediction output

## Usage:
1. Load the TensorFlow.js model in your web application
2. Preprocess input data using the same features and scaling
3. Make predictions using sequences of length 8
4. Scale the output predictions back to original units

For questions, refer to the original project documentation.
