"""
Model module for Population Growth Prediction model
Simplified version that only uses one year as input
"""
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
import pandas as pd

from config import ModelConfig

class PopulationGrowthModel:
    def __init__(self, config=None):
        self.config = config or ModelConfig()
        self.model = None
        self.feature_scaler = None
        self.target_scaler = None
        
        # Set random seeds for reproducibility
        np.random.seed(self.config.RANDOM_SEED)
        tf.random.set_seed(self.config.RANDOM_SEED)
    
    def build(self, input_shape):
        """Build simplified model architecture that works with single year inputs"""
        print(f"Building model with input shape {input_shape}")
        
        # For single year inputs, we don't need sequence modeling with LSTM
        # We can use a simpler dense neural network
        model = tf.keras.Sequential([
            # Input layer
            tf.keras.layers.InputLayer(input_shape=input_shape),
            
            # First hidden layer
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            # Second hidden layer
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            
            # Third hidden layer
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.1),
            
            # Output layer
            tf.keras.layers.Dense(1)
        ])
        
        # Use a lower initial learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        # Use explicit loss and metric objects for better compatibility
        model.compile(
            optimizer=optimizer, 
            loss=tf.keras.losses.MeanSquaredError(), 
            metrics=[tf.keras.metrics.MeanAbsoluteError()]
        )
        model.summary()
        
        self.model = model
        return model
    
    def check_for_invalid_values(self, data, name):
        """Check for NaN or infinite values in data"""
        has_nan = np.isnan(data).any()
        has_inf = np.isinf(data).any()
        if has_nan or has_inf:
            print(f"WARNING: {name} contains {'NaN' if has_nan else ''} {'and Inf' if has_inf else ''} values")
            # Replace NaN with 0 and Inf with large numbers
            cleaned_data = np.nan_to_num(data, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
            return cleaned_data
        return data
    
    def scale_data(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Scale the data for better model performance using a robust approach"""
        print("Scaling the data...")
        
        # First check for invalid values
        X_train = self.check_for_invalid_values(X_train, "X_train")
        X_val = self.check_for_invalid_values(X_val, "X_val")
        X_test = self.check_for_invalid_values(X_test, "X_test")
        y_train = self.check_for_invalid_values(y_train, "y_train")
        y_val = self.check_for_invalid_values(y_val, "y_val")
        y_test = self.check_for_invalid_values(y_test, "y_test")
        
        # Use RobustScaler which is less sensitive to outliers
        self.feature_scaler = RobustScaler()
        
        # Fit on training data
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        
        # Transform validation and test data
        X_val_scaled = self.feature_scaler.transform(X_val)
        X_test_scaled = self.feature_scaler.transform(X_test)
        
        # Scale the target variable
        self.target_scaler = RobustScaler()
        y_train_scaled = self.target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = self.target_scaler.transform(y_val.reshape(-1, 1)).flatten()
        y_test_scaled = self.target_scaler.transform(y_test.reshape(-1, 1)).flatten()
        
        # Final check for any remaining invalid values
        X_train_scaled = self.check_for_invalid_values(X_train_scaled, "X_train_scaled")
        X_val_scaled = self.check_for_invalid_values(X_val_scaled, "X_val_scaled")
        X_test_scaled = self.check_for_invalid_values(X_test_scaled, "X_test_scaled")
        y_train_scaled = self.check_for_invalid_values(y_train_scaled, "y_train_scaled")
        y_val_scaled = self.check_for_invalid_values(y_val_scaled, "y_val_scaled")
        y_test_scaled = self.check_for_invalid_values(y_test_scaled, "y_test_scaled")
        
        return (X_train_scaled, X_val_scaled, X_test_scaled, 
                y_train_scaled, y_val_scaled, y_test_scaled)
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train the model with early stopping"""
        print("Training model...")
        
        if self.model is None:
            self.build(X_train.shape[1:])
        
        # Add callbacks
        callbacks = [
            # Early stopping with increased patience
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            # Model checkpoint to save the best model
            tf.keras.callbacks.ModelCheckpoint(
                self.config.ENHANCED_MODEL_PATH,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            # Reduce learning rate on plateau
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.7,
                patience=8,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        # Train model
        print(f"Training model for up to {self.config.EPOCHS} epochs with batch size {self.config.BATCH_SIZE}...")
        history = self.model.fit(
            X_train, y_train,
            epochs=self.config.EPOCHS,
            batch_size=self.config.BATCH_SIZE,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Save model in Keras 3 native format
        keras_model_path = self.config.MODEL_PATH.replace('.h5', '.keras')
        self.model.save(keras_model_path)
        print(f"Saved model to '{keras_model_path}'")
        
        # Also save as HDF5 for backward compatibility
        try:
            self.model.save(self.config.MODEL_PATH, save_format='h5')
            print(f"Also saved model in legacy HDF5 format to '{self.config.MODEL_PATH}'")
        except Exception as e:
            print(f"Note: Could not save in legacy HDF5 format: {e}")
        
        return history
    
    def predict(self, X):
        """Make predictions for single year inputs"""
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet")
        
        # Make predictions
        predictions = self.model.predict(X, verbose=0)
        
        return predictions
    
    def load_model(self, model_path=None):
        """Load a saved model"""
        if model_path is None:
            model_path = self.config.MODEL_PATH
            
        print(f"Loading model from {model_path}...")
        
        try:
            # First try: direct loading
            self.model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully.")
            return True
        except Exception as e:
            print(f"Standard model loading failed: {e}")
            
            try:
                # Second try: with custom objects
                self.model = tf.keras.models.load_model(
                    model_path,
                    custom_objects={
                        'MeanSquaredError': tf.keras.losses.MeanSquaredError,
                        'MeanAbsoluteError': tf.keras.metrics.MeanAbsoluteError
                    }
                )
                print("Model loaded with custom objects.")
                return True
            except Exception as e2:
                print(f"Custom objects loading failed: {e2}")
                
                # Third try: rebuild model and load weights
                print("Rebuilding model and loading weights only...")
                
                try:
                    # Try to load feature list to get input shape
                    if os.path.exists(self.config.FEATURE_LIST_FILE):
                        features_df = pd.read_csv(self.config.FEATURE_LIST_FILE)
                        feature_list = features_df['feature'].tolist()
                        input_shape = (len(feature_list),)  # Single year input shape
                    else:
                        input_shape = (57,)  # Default single year input shape
                    
                    # Build the model
                    self.build(input_shape)
                    
                    # Try to load weights
                    self.model.load_weights(model_path)
                    print("Model weights loaded successfully.")
                    return True
                except Exception as e3:
                    print(f"Weight loading failed: {e3}")
                    print("Failed to load model.")
                    return False