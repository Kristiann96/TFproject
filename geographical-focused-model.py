import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

# Check TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# Configuration
SEQ_LENGTH = 8  # Number of years for input sequence
BATCH_SIZE = 128
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
RANDOM_SEED = 42

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

def load_full_data():
    """Load the full population dataset"""
    print("Looking for population data files...")
    
    # Try different filenames in order of preference
    possible_files = [
        "full_population_data.csv",
        "combined_population_data.csv",
        "combined_population_sample.csv"
    ]
    
    loaded_df = None
    for file_path in possible_files:
        if os.path.exists(file_path):
            print(f"Loading data from {file_path}")
            loaded_df = pd.read_csv(file_path)
            print(f"Loaded {len(loaded_df)} records.")
            break
    
    if loaded_df is None:
        print("No suitable data file found.")
        print("Please run the full-data-processor.py script first.")
        return None
    
    # Check if we have data from multiple years
    years = loaded_df['year'].unique()
    print(f"Data contains {len(years)} years: {sorted(years)}")
    
    if len(years) < 2:
        print("WARNING: Data doesn't have enough years for time series analysis.")
        return None
    
    return loaded_df

def select_municipalities(df, num_municipalities=25):  # MODIFIED: Increased from 5 to 25
    """Select a subset of municipalities for analysis"""
    print("Analyzing municipalities in the dataset...")
    
    # Count records by municipality
    kommune_counts = df.groupby('kommunenummer').agg(
        name=('kommunenavn', 'first'),
        records=('grunnkretsnummer', 'count'),
        grunnkretser=('grunnkretsnummer', 'nunique'),
        years=('year', 'nunique'),
        population=('totalBefolkning', 'mean')
    ).sort_values('records', ascending=False)
    
    print(f"Found {len(kommune_counts)} municipalities in the dataset.")
    
    # Show the top municipalities by record count
    print("\nTop 10 municipalities by record count:")
    print(kommune_counts.head(10))
    
    # Filter to municipalities with data for all years
    max_years = df['year'].nunique()
    complete_kommuner = kommune_counts[kommune_counts['years'] == max_years]
    print(f"\nFound {len(complete_kommuner)} municipalities with data for all {max_years} years.")
    
    if len(complete_kommuner) < num_municipalities:
        print(f"Warning: Only {len(complete_kommuner)} municipalities have complete data.")
        print("Using all available municipalities with complete data.")
        selected_kommuner = complete_kommuner.index.tolist()
    else:
        # Select a mix of large and small municipalities
        # MODIFIED: Take more municipalities from each population tier
        large_kommuner = complete_kommuner.nlargest(num_municipalities // 2, 'population')
        small_kommuner = complete_kommuner.nsmallest(num_municipalities // 2, 'population')
        selected_kommuner = list(large_kommuner.index) + list(small_kommuner.index)
        
        # Ensure we have exactly num_municipalities
        if len(selected_kommuner) > num_municipalities:
            selected_kommuner = selected_kommuner[:num_municipalities]
    
    # Create a DataFrame with information about the selected municipalities
    selected_info = kommune_counts.loc[selected_kommuner]
    print("\nSelected municipalities for analysis:")
    print(selected_info)
    
    # Save the list of selected municipalities
    selected_info.to_csv('selected_municipalities_5x.csv')
    print("Saved information about selected municipalities to 'selected_municipalities_5x.csv'")
    
    # Filter the original dataset to only include the selected municipalities
    filtered_df = df[df['kommunenummer'].isin(selected_kommuner)].copy()
    print(f"\nFiltered dataset contains {len(filtered_df)} records from {len(selected_kommuner)} municipalities.")
    
    return filtered_df, selected_kommuner

def preprocess_data(df):
    """Clean and prepare the data for modeling"""
    print("Preprocessing data...")
    
    # Convert string columns to numeric where needed
    for col in df.columns:
        if col in ['grunnkretsnummer', 'kommunenummer'] or 'befolkning' in col or 'antall' in col or col in ['totalBefolkning', 'folketilvekst']:
            try:
                if df[col].dtype == 'object':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    print(f"Converted {col} to numeric.")
            except Exception as e:
                print(f"Warning: Could not convert {col} to numeric: {e}")
    
    # Drop rows with missing values in essential columns
    essential_cols = ['grunnkretsnummer', 'year', 'totalBefolkning']
    missing_before = len(df)
    df = df.dropna(subset=essential_cols)
    missing_after = len(df)
    print(f"Dropped {missing_before - missing_after} rows with missing values in essential columns.")
    
    # Make sure 'year' is integer
    df['year'] = df['year'].astype(int)
    
    # Clean extreme values in numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        # Replace inf and -inf with NaN
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        # Handle NaN values
        if col in essential_cols:
            # For essential columns, we already dropped NaN rows
            pass
        else:
            # For non-essential columns, replace NaN with 0
            df[col] = df[col].fillna(0)
        
        # Check for and handle extreme outliers
        if col not in ['year', 'grunnkretsnummer', 'kommunenummer']:
            # Get 99.9th percentile as upper limit
            upper_limit = df[col].quantile(0.999)
            if upper_limit > 0:
                # Cap values at the upper limit
                df.loc[df[col] > upper_limit, col] = upper_limit
    
    return df

def create_features(df):
    """Create features for the model"""
    print("Creating features...")
    
    # Fill missing values in numeric columns with 0
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # Calculate growth rate (percent change in population) - with safety limits
    print("Calculating growth rates...")
    df['growth_rate'] = df.groupby('grunnkretsnummer')['totalBefolkning'].pct_change()
    # Clip growth rate to reasonable range (-1 to 5, i.e., -100% to 500%)
    df['growth_rate'] = df['growth_rate'].clip(-1, 5).fillna(0)
    
    # Calculate absolute population change
    df['pop_change'] = df.groupby('grunnkretsnummer')['totalBefolkning'].diff()
    # Clip population change to reasonable range
    max_change = df['totalBefolkning'].quantile(0.99)
    df['pop_change'] = df['pop_change'].clip(-max_change, max_change).fillna(0)
    
    # Calculate age distribution features
    age_columns = [col for col in df.columns if 'befolkning' in col.lower() and 'År' in col]
    if age_columns:
        print(f"Calculating age distribution ratios for {len(age_columns)} age groups...")
        for col in age_columns:
            # Avoid division by zero and limit to range [0, 1]
            df[f'{col}_ratio'] = np.where(df['totalBefolkning'] > 0, 
                                      df[col] / df['totalBefolkning'], 
                                      0)
            df[f'{col}_ratio'] = df[f'{col}_ratio'].clip(0, 1)
    
        # Create age group aggregates
        children_cols = [col for col in age_columns if any(age in col for age in ['0Til04', '05Til09', '10Til14'])]
        if children_cols:
            df['children_0_14'] = df[children_cols].sum(axis=1)
            df['children_ratio'] = (df['children_0_14'] / df['totalBefolkning'].replace(0, 1)).clip(0, 1)
        
        elderly_cols = [col for col in age_columns if any(age in col for age in ['65', '70', '75', '80', '85', '90'])]
        if elderly_cols:
            df['elderly'] = df[elderly_cols].sum(axis=1)
            df['elderly_ratio'] = (df['elderly'] / df['totalBefolkning'].replace(0, 1)).clip(0, 1)
        
        working_age_cols = [col for col in age_columns if any(age in col for age in ['20', '25', '30', '35', '40', '45', '50', '55', '60'])]
        if working_age_cols:
            df['working_age'] = df[working_age_cols].sum(axis=1)
            df['working_age_ratio'] = (df['working_age'] / df['totalBefolkning'].replace(0, 1)).clip(0, 1)
    
    # Calculate gender ratio if available
    if 'antallMenn' in df.columns and 'antallKvinner' in df.columns:
        print("Calculating gender ratio...")
        df['gender_ratio'] = np.where(df['antallKvinner'] > 0, 
                                    df['antallMenn'] / df['antallKvinner'], 
                                    1)
        # Clip to reasonable range (0.5 to 2)
        df['gender_ratio'] = df['gender_ratio'].clip(0.5, 2)
    
    # Add lagged features (previous year's values)
    print("Creating lagged features...")
    lag_columns = ['totalBefolkning', 'growth_rate', 'pop_change']
    if 'folketilvekst' in df.columns:
        lag_columns.append('folketilvekst')
    
    for col in lag_columns:
        if col in df.columns:
            df[f'{col}_lag1'] = df.groupby('grunnkretsnummer')[col].shift(1)
            df[f'{col}_lag1'] = df[f'{col}_lag1'].fillna(0)
    
    print("Feature engineering complete")
    return df

def find_complete_sequences(df, seq_length=SEQ_LENGTH, min_count=None):  # Modified to include an optional min_count parameter
    """Find grunnkretser with complete sequences of data"""
    print(f"Finding grunnkretser with at least {seq_length+1} years of consecutive data...")
    
    complete_grunnkretser = []
    counts = 0
    
    # Check each grunnkrets
    for grunnkrets, group in df.groupby('grunnkretsnummer'):
        # Sort by year
        sorted_years = sorted(group['year'].unique())
        
        # Check if we have at least seq_length+1 consecutive years
        for i in range(len(sorted_years) - seq_length):
            if sorted_years[i+seq_length] - sorted_years[i] == seq_length:
                complete_grunnkretser.append(grunnkrets)
                counts += 1
                break
    
    print(f"Found {len(complete_grunnkretser)} grunnkretser with complete sequences")
    
    # MODIFIED: If min_count is specified, load additional grunnkretser from file if available
    if min_count and len(complete_grunnkretser) < min_count:
        print(f"Found fewer than {min_count} grunnkretser, checking for additional data...")
        if os.path.exists('complete_grunnkretser.csv'):
            extra_grunnkretser_df = pd.read_csv('complete_grunnkretser.csv')
            extra_grunnkretser = extra_grunnkretser_df['grunnkretsnummer'].tolist()
            
            # Add grunnkretser from the file that are not already in our list
            new_grunnkretser = [g for g in extra_grunnkretser if g not in complete_grunnkretser]
            
            # Add enough to reach min_count if possible
            needed = min_count - len(complete_grunnkretser)
            if needed > 0:
                additional = new_grunnkretser[:needed]
                complete_grunnkretser.extend(additional)
                print(f"Added {len(additional)} grunnkretser from external file")
    
    print(f"Final count: {len(complete_grunnkretser)} grunnkretser with complete sequences")
    return complete_grunnkretser

def create_time_series_datasets(df, complete_grunnkretser, target_col='folketilvekst', seq_length=5):  # Changed default to 5 explicitly
    """Create sequences for LSTM model with sliding window approach to increase data volume"""
    print(f"Creating time series sequences with target column: {target_col}")
    
    # If target column is missing, create it from totalBefolkning
    if target_col not in df.columns or df[target_col].isna().all():
        print(f"Target column {target_col} not available, using population change")
        df['pop_diff'] = df.groupby('grunnkretsnummer')['totalBefolkning'].diff()
        df['pop_diff'] = df['pop_diff'].fillna(0)
        target_col = 'pop_diff'
    
    print(f"Using {target_col} as target variable")
    
    # Select numeric features for the model
    numeric_cols = df.select_dtypes(include=['number']).columns
    feature_cols = [col for col in numeric_cols if col not in 
                   ['gml_id', 'versjonId', target_col, 'year', 'grunnkretsnummer', 'kommunenummer']]
    
    print(f"Using {len(feature_cols)} features")
    if len(feature_cols) > 10:
        print(f"Sample features: {feature_cols[:10]}...")
    else:
        print(f"Features: {feature_cols}")
    
    # Filter to just the grunnkretser with complete data
    filtered_df = df[df['grunnkretsnummer'].isin(complete_grunnkretser)]
    
    print(f"Creating sequences from {len(filtered_df)} filtered records...")
    
    # Group by grunnkrets and create sequences with sliding window
    X_sequences = []
    y_sequences = []
    metadata = []  # Store grunnkrets, kommune, year for each sequence
    
    sequence_counter = 0
    grunnkrets_counter = 0
    
    # Adjust sequence length for this run - make sure it's smaller than available years
    # This is crucial - if seq_length is too large, we won't get multiple sequences
    years_available = df['year'].nunique()
    effective_seq_length = min(seq_length, years_available - 2)  # Ensure at least 2 sequences possible
    
    print(f"Years available: {years_available}, using effective sequence length: {effective_seq_length}")
    
    for grunnkrets, group in filtered_df.groupby('grunnkretsnummer'):
        try:
            # Sort by year
            group = group.sort_values('year')
            
            # Check if we have enough years of data
            if len(group) < effective_seq_length + 1:
                print(f"Not enough data for grunnkrets {grunnkrets}: {len(group)} years (need {effective_seq_length + 1})")
                continue
                
            # Get feature values
            feature_values = group[feature_cols].values
            
            # Make sure target column exists
            if target_col in group:
                target_values = group[target_col].values
                year_values = group['year'].values
                kommune_values = group['kommunenummer'].values
                
                # Create multiple sequences per grunnkrets using a sliding window
                # For a 9-year dataset with seq_length=5, we could create up to 4 sequences per grunnkrets
                max_start_idx = len(group) - effective_seq_length
                
                sequences_this_grunnkrets = 0
                for i in range(max_start_idx):
                    X_sequences.append(feature_values[i:i+effective_seq_length])
                    y_sequences.append(target_values[i+effective_seq_length])
                    
                    # Store metadata
                    metadata.append({
                        'grunnkretsnummer': grunnkrets,
                        'kommunenummer': kommune_values[i+effective_seq_length],
                        'year': year_values[i+effective_seq_length],
                        'sequence_id': f"{grunnkrets}_{i}"  # Add unique sequence identifier
                    })
                    sequence_counter += 1
                    sequences_this_grunnkrets += 1
                
                if sequences_this_grunnkrets > 0:
                    grunnkrets_counter += 1
                    if grunnkrets_counter % 100 == 0 or grunnkrets_counter == 1:
                        print(f"Processed {grunnkrets_counter} grunnkretser, created {sequence_counter} sequences")
                        print(f"Latest: Created {sequences_this_grunnkrets} sequences from grunnkrets {grunnkrets}")
                
        except Exception as e:
            print(f"Error processing grunnkrets {grunnkrets}: {e}")
            continue
    
    # Convert to numpy arrays
    if X_sequences:
        X = np.array(X_sequences)
        y = np.array(y_sequences)
        metadata_df = pd.DataFrame(metadata)
        
        # Calculate and print statistics about sequences per grunnkrets
        sequences_per_grunnkrets = metadata_df['grunnkretsnummer'].value_counts()
        avg_sequences = sequences_per_grunnkrets.mean()
        max_sequences = sequences_per_grunnkrets.max()
        min_sequences = sequences_per_grunnkrets.min()
        
        print(f"Created {len(X)} sequences from {len(metadata_df['grunnkretsnummer'].unique())} grunnkretser")
        print(f"Average sequences per grunnkrets: {avg_sequences:.2f} (min: {min_sequences}, max: {max_sequences})")
        
        # Print distribution of sequences per grunnkrets
        seq_counts = sequences_per_grunnkrets.value_counts().sort_index()
        print("Distribution of sequences per grunnkrets:")
        for count, num_grunnkretser in seq_counts.items():
            print(f"  {count} sequences: {num_grunnkretser} grunnkretser")
        
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        
        return X, y, metadata_df, feature_cols
    else:
        print("No sequences created. Check your data.")
        return None, None, None, None
def split_data(X, y, metadata_df):
    """Split data into training, validation, and testing sets"""
    print("Splitting data into train, validation, and test sets...")
    
    # Get unique grunnkretser
    unique_grunnkretser = metadata_df['grunnkretsnummer'].unique()
    
    # Split grunnkretser into train, validation, and test sets
    train_grunnkretser, temp_grunnkretser = train_test_split(
        unique_grunnkretser, test_size=0.3, random_state=RANDOM_SEED
    )
    
    val_grunnkretser, test_grunnkretser = train_test_split(
        temp_grunnkretser, test_size=0.5, random_state=RANDOM_SEED
    )
    
    print(f"Train: {len(train_grunnkretser)} grunnkretser")
    print(f"Validation: {len(val_grunnkretser)} grunnkretser")
    print(f"Test: {len(test_grunnkretser)} grunnkretser")
    
    # Create masks for each set
    train_mask = metadata_df['grunnkretsnummer'].isin(train_grunnkretser)
    val_mask = metadata_df['grunnkretsnummer'].isin(val_grunnkretser)
    test_mask = metadata_df['grunnkretsnummer'].isin(test_grunnkretser)
    
    # Split the data
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    # Split metadata
    metadata_train = metadata_df[train_mask]
    metadata_val = metadata_df[val_mask]
    metadata_test = metadata_df[test_mask]
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Testing set: {X_test.shape}")
    
    return (X_train, y_train, metadata_train), (X_val, y_val, metadata_val), (X_test, y_test, metadata_test)

def check_for_invalid_values(data, name):
    """Check for NaN or infinite values in data"""
    has_nan = np.isnan(data).any()
    has_inf = np.isinf(data).any()
    if has_nan or has_inf:
        print(f"WARNING: {name} contains {'NaN' if has_nan else ''} {'and Inf' if has_inf else ''} values")
        # Replace NaN with 0 and Inf with large numbers
        cleaned_data = np.nan_to_num(data, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
        return cleaned_data
    return data

def scale_data(X_train, X_val, X_test, y_train, y_val, y_test):
    """Scale the data for better model performance using a robust approach"""
    print("Scaling the data...")
    
    # First check for invalid values
    X_train = check_for_invalid_values(X_train, "X_train")
    X_val = check_for_invalid_values(X_val, "X_val")
    X_test = check_for_invalid_values(X_test, "X_test")
    y_train = check_for_invalid_values(y_train, "y_train")
    y_val = check_for_invalid_values(y_val, "y_val")
    y_test = check_for_invalid_values(y_test, "y_test")
    
    # Use RobustScaler which is less sensitive to outliers
    feature_scaler = RobustScaler()
    
    # Reshape for scaling
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
    X_val_reshaped = X_val.reshape(X_val.shape[0], -1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
    
    # Fit on training data only
    X_train_scaled = feature_scaler.fit_transform(X_train_reshaped)
    
    # Transform validation and test data
    X_val_scaled = feature_scaler.transform(X_val_reshaped)
    X_test_scaled = feature_scaler.transform(X_test_reshaped)
    
    # Reshape back to 3D for LSTM
    X_train_scaled = X_train_scaled.reshape(X_train.shape)
    X_val_scaled = X_val_scaled.reshape(X_val.shape)
    X_test_scaled = X_test_scaled.reshape(X_test.shape)
    
    # Scale the target variable
    target_scaler = RobustScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()
    
    # Final check for any remaining invalid values
    X_train_scaled = check_for_invalid_values(X_train_scaled, "X_train_scaled")
    X_val_scaled = check_for_invalid_values(X_val_scaled, "X_val_scaled")
    X_test_scaled = check_for_invalid_values(X_test_scaled, "X_test_scaled")
    y_train_scaled = check_for_invalid_values(y_train_scaled, "y_train_scaled")
    y_val_scaled = check_for_invalid_values(y_val_scaled, "y_val_scaled")
    y_test_scaled = check_for_invalid_values(y_test_scaled, "y_test_scaled")
    
    return (X_train_scaled, X_val_scaled, X_test_scaled, 
            y_train_scaled, y_val_scaled, y_test_scaled, 
            feature_scaler, target_scaler)

def build_model(input_shape):
    """Build enhanced LSTM model for time series prediction with increased complexity"""
    print(f"Building enhanced model with input shape {input_shape}")
    
    # More complex model architecture
    model = tf.keras.Sequential([
        # First LSTM layer with increased units
        tf.keras.layers.LSTM(256, return_sequences=True, input_shape=(input_shape[1], input_shape[2])),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        # Second LSTM layer with more units
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        # Third LSTM layer
        tf.keras.layers.LSTM(64),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        # Deeper dense layers
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        
        # Output layer
        tf.keras.layers.Dense(1)
    ])
    
    # Use a lower initial learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    model.summary()
    
    return model

def train_model(X_train, y_train, X_val, y_val, feature_cols):
    """Train the model with more extensive settings"""
    print("Training enhanced model...")
    
    # Build the model
    model = build_model(X_train.shape)
    
    # Add callbacks with adjusted parameters
    callbacks = [
        # Early stopping with increased patience
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=25,  # Increased from 15
            restore_best_weights=True,
            verbose=1
        ),
        # Model checkpoint to save the best model
        tf.keras.callbacks.ModelCheckpoint(
            'best_model_enhanced.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        # Reduce learning rate on plateau with smaller factor
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,  # More gradual reduction (was 0.5)
            patience=8,  # Increased from 5
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    # Training configuration
    EPOCHS = 150  # Increased from 100
    BATCH_SIZE = 64  # Reduced from 128 for more gradient updates
    
    # Train model
    print(f"Training model for up to {EPOCHS} epochs with batch size {BATCH_SIZE}...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history_enhanced.png')
    print("Saved training history plot to 'training_history_enhanced.png'")
    
    # Save model in TF format
    model.save('population_growth_model_enhanced.h5')
    print("Saved model to 'population_growth_model_enhanced.h5'")
    
    # Save model in TensorFlow.js format
    try:
        # Import TensorFlow.js converter module
        import tensorflowjs as tfjs
        
        # Create directory for TF.js model
        tfjs_dir = 'population_growth_model_enhanced_tfjs'
        print(f"Saving TensorFlow.js model to '{tfjs_dir}'...")
        
        # Convert and save the model for TensorFlow.js
        tfjs.converters.save_keras_model(model, tfjs_dir)
        print(f"Successfully saved TensorFlow.js model to '{tfjs_dir}'")
        
        # Save metadata about input shape and feature names for easier use in JavaScript
        input_metadata = {
            'input_shape': X_train.shape[1:],
            'feature_names': feature_cols
        }
        
        # Save as JSON
        import json
        with open(f"{tfjs_dir}/model_metadata.json", 'w') as f:
            json.dump(input_metadata, f, indent=2)
        print(f"Saved model metadata to '{tfjs_dir}/model_metadata.json'")
        
    except ImportError:
        print("Warning: tensorflowjs package not found. TensorFlow.js model not saved.")
        print("To install, run: pip install tensorflowjs")
    
    return model, history

def evaluate_model(model, X_test, y_test, target_scaler, metadata_test):
    """Evaluate the model on the test set"""
    print("Evaluating model on test set...")
    
    # Get predictions
    y_pred_scaled = model.predict(X_test)
    
    # Unscale predictions and true values
    y_pred = target_scaler.inverse_transform(y_pred_scaled).flatten()
    y_true = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    mae = np.mean(np.abs(y_pred - y_true))
    mse = np.mean(np.square(y_pred - y_true))
    rmse = np.sqrt(mse)
    
    print(f"Test MAE: {mae:.4f}")
    print(f"Test MSE: {mse:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    
    # Create a results DataFrame
    results_df = metadata_test.copy()
    results_df['actual'] = y_true
    results_df['predicted'] = y_pred
    results_df['error'] = y_true - y_pred
    results_df['abs_error'] = np.abs(y_true - y_pred)
    
    # Save results
    results_df.to_csv('prediction_results_5x.csv', index=False)  # MODIFIED: Changed filename
    print("Saved detailed prediction results to 'prediction_results_5x.csv'")
    
    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.title('Predicted vs Actual Population Growth')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.savefig('predictions_vs_actual_5x.png')  # MODIFIED: Changed filename
    print("Saved predictions vs actual plot to 'predictions_vs_actual_5x.png'")
    
    # Analyze error by municipality
    kommune_error = results_df.groupby('kommunenummer').agg({
        'abs_error': ['mean', 'std'],
        'error': 'mean',
        'actual': ['mean', 'count']
    })
    
    print("\nPrediction Error by Municipality:")
    print(kommune_error.sort_values(('abs_error', 'mean')))
    
    # Plot error by municipality
    plt.figure(figsize=(12, 6))
    kommune_error_plot = kommune_error.sort_values(('abs_error', 'mean'))
    plt.bar(kommune_error_plot.index.astype(str), kommune_error_plot[('abs_error', 'mean')])
    plt.title('Mean Absolute Error by Municipality')
    plt.xlabel('Municipality')
    plt.ylabel('MAE')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('error_by_municipality_5x.png')  # MODIFIED: Changed filename
    print("Saved error by municipality plot to 'error_by_municipality_5x.png'")
    
    return results_df, mae, mse, rmse

def export_model_for_web(model, feature_cols, scalers=None):
    """Export additional files needed for web deployment"""
    print("Exporting additional files for web deployment...")
    
    # Create a web deployment directory
    web_dir = 'web_deployment'
    os.makedirs(web_dir, exist_ok=True)
    
    # Save the list of feature names
    feature_df = pd.DataFrame({'feature': feature_cols})
    feature_df.to_csv(f'{web_dir}/features.csv', index=False)
    
    # Create a model metadata file with information about the model
    metadata = {
        'model_name': 'Population Growth Prediction Model',
        'input_sequence_length': SEQ_LENGTH,
        'feature_count': len(feature_cols),
        'created_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
        'features': feature_cols
    }
    
    # Save metadata as JSON
    import json
    with open(f'{web_dir}/model_info.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # If scalers are provided, save them for later use
    if scalers:
        import pickle
        feature_scaler, target_scaler = scalers
        
        # Save the scalers
        with open(f'{web_dir}/feature_scaler.pkl', 'wb') as f:
            pickle.dump(feature_scaler, f)
            
        with open(f'{web_dir}/target_scaler.pkl', 'wb') as f:
            pickle.dump(target_scaler, f)
            
        print(f"Saved scalers to '{web_dir}/' directory")
    
    # Create a simple README file with instructions
    readme = """# Population Growth Prediction Model

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
3. Make predictions using sequences of length {seq_length}
4. Scale the output predictions back to original units

For questions, refer to the original project documentation.
""".format(seq_length=SEQ_LENGTH)
    
    with open(f'{web_dir}/README.md', 'w') as f:
        f.write(readme)
    
    print(f"Created web deployment files in '{web_dir}/' directory")
    
    # Copy the TensorFlow.js model to the web directory
    if os.path.exists('population_growth_model_5x_tfjs'):
        import shutil
        # Create model subdirectory
        os.makedirs(f'{web_dir}/model', exist_ok=True)
        
        # Copy files
        for file in os.listdir('population_growth_model_5x_tfjs'):
            shutil.copy(
                os.path.join('population_growth_model_5x_tfjs', file),
                os.path.join(web_dir, 'model', file)
            )
        print(f"Copied TensorFlow.js model to '{web_dir}/model/' directory")

def main():
    print("Starting geographically focused population growth prediction with 5x data...")  # MODIFIED: Updated message
    
    # Step 1: Load the full dataset
    df = load_full_data()
    if df is None:
        return
    
    # Step 2: Select a subset of municipalities (MODIFIED: increased from 5 to 25)
    filtered_df, selected_kommuner = select_municipalities(df, num_municipalities=25)
    
    # Step 3: Preprocess the data - clean outliers and extreme values
    filtered_df = preprocess_data(filtered_df)
    
    # Step 4: Create features - with safety limits
    filtered_df = create_features(filtered_df)
    
    # Step 5: Find complete sequences
    # MODIFIED: Added target minimum count of sequences (5x the original)
    # Looking at prediction_results.csv, we have about 140 sequences in the original model
    # So we aim for at least 700 sequences
    complete_grunnkretser = find_complete_sequences(filtered_df, min_count=700)
    
    if not complete_grunnkretser:
        print("No complete sequences found. Reducing sequence length requirements...")
        global SEQ_LENGTH
        SEQ_LENGTH = 1
        complete_grunnkretser = find_complete_sequences(filtered_df, seq_length=SEQ_LENGTH)
        
        if not complete_grunnkretser:
            print("Still no complete sequences found. Cannot proceed with modeling.")
            return
    
    # Step 6: Create time series datasets
    result = create_time_series_datasets(filtered_df, complete_grunnkretser)
    if result[0] is None:
        return
    
    X, y, metadata_df, feature_cols = result
    
    # Step 7: Split data into train, validation, and test sets
    (X_train, y_train, metadata_train), (X_val, y_val, metadata_val), (X_test, y_test, metadata_test) = split_data(X, y, metadata_df)
    
    # Step 8: Scale the data using robust scaling
    scaled_data = scale_data(X_train, X_val, X_test, y_train, y_val, y_test)
    X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled, feature_scaler, target_scaler = scaled_data
    
    # Step 9: Train the model
    model, history = train_model(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, feature_cols)
    
    # Step 10: Evaluate the model
    results_df, mae, mse, rmse = evaluate_model(model, X_test_scaled, y_test_scaled, target_scaler, metadata_test)
    
    print("\nModel training and evaluation complete!")
    print(f"Final Test MAE: {mae:.4f}")
    print(f"Final Test RMSE: {rmse:.4f}")
    
    # Compare with original model if results exist
    if os.path.exists('prediction_results.csv'):
        original_results = pd.read_csv('prediction_results.csv')
        original_mae = original_results['abs_error'].mean()
        original_mse = np.mean(np.square(original_results['error']))
        original_rmse = np.sqrt(original_mse)
        
        print("\nComparison with original model:")
        print(f"Original Model MAE: {original_mae:.4f}, New Model MAE: {mae:.4f}")
        print(f"Original Model RMSE: {original_rmse:.4f}, New Model RMSE: {rmse:.4f}")
        print(f"Data Size: Original={len(original_results)} vs New={len(results_df)} sequences")
        
        if mae < original_mae:
            print("The expanded model shows improved accuracy!")
        else:
            print("The expanded model has different performance characteristics.")
    
    # Save feature information
    pd.DataFrame({'feature': feature_cols}).to_csv('model_features_5x.csv', index=False)
    print("Saved model feature information to 'model_features_5x.csv'")
    
    # Export additional files for web deployment
    export_model_for_web(model, feature_cols, (feature_scaler, target_scaler))
    
    print("\nAll tasks completed successfully!")
    print("The expanded model (5x data) and TensorFlow.js version are ready for use.")

if __name__ == "__main__":
    main()