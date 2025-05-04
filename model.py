"""
Model module for Population Growth Prediction model
Contains the model class and related functions
"""
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import RobustScaler

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
        """Build model architecture"""
        print(f"Building model with input shape {input_shape}")
        
        model = tf.keras.Sequential([
            # First LSTM layer with increased units
            tf.keras.layers.LSTM(
                self.config.LSTM_UNITS[0], 
                return_sequences=True, 
                input_shape=(input_shape[1], input_shape[2])
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(self.config.DROPOUT_RATES[0]),
            
            # Second LSTM layer with more units
            tf.keras.layers.LSTM(
                self.config.LSTM_UNITS[1], 
                return_sequences=True
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(self.config.DROPOUT_RATES[1]),
            
            # Third LSTM layer
            tf.keras.layers.LSTM(self.config.LSTM_UNITS[2]),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(self.config.DROPOUT_RATES[2]),
            
            # Deeper dense layers
            tf.keras.layers.Dense(self.config.DENSE_UNITS[0], activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(self.config.DROPOUT_RATES[3]),
            
            tf.keras.layers.Dense(self.config.DENSE_UNITS[1], activation='relu'),
            tf.keras.layers.BatchNormalization(),
            
            # Output layer
            tf.keras.layers.Dense(self.config.DENSE_UNITS[2])
        ])
        
        # Use a lower initial learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
        
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
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
        
        # Reshape for scaling
        X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
        X_val_reshaped = X_val.reshape(X_val.shape[0], -1)
        X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
        
        # Fit on training data only
        X_train_scaled = self.feature_scaler.fit_transform(X_train_reshaped)
        
        # Transform validation and test data
        X_val_scaled = self.feature_scaler.transform(X_val_reshaped)
        X_test_scaled = self.feature_scaler.transform(X_test_reshaped)
        
        # Reshape back to 3D for LSTM
        X_train_scaled = X_train_scaled.reshape(X_train.shape)
        X_val_scaled = X_val_scaled.reshape(X_val.shape)
        X_test_scaled = X_test_scaled.reshape(X_test.shape)
        
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
            self.build(X_train.shape)
        
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
        
        # Save model
        self.model.save(self.config.MODEL_PATH)
        print(f"Saved model to '{self.config.MODEL_PATH}'")
        
        return history
    
    def predict(self, X):
        """Make predictions with proper reshaping"""
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet")
        
        # Reshape input for LSTM if needed
        if len(X.shape) == 2:  # If X is 2D
            # For sequence length n, we need n identical frames as a starting point
            X = np.array([X] * self.config.SEQ_LENGTH)
            X = np.transpose(X, (1, 0, 2))
        
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
                        'mse': tf.keras.losses.mean_squared_error,
                        'mae': tf.keras.metrics.mean_absolute_error
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
                        input_shape = (self.config.SEQ_LENGTH, len(feature_list))
                    else:
                        input_shape = (self.config.SEQ_LENGTH, 57)  # Default from training
                    
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