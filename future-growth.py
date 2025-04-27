import tensorflow as tf
import numpy as np
import pandas as pd
import os
import copy
from datetime import datetime

# Configuration
START_YEAR = 2025
END_YEAR = 2030
MODEL_PATH = 'population_growth_model.h5'
SEQ_LENGTH = 2  # This should match what you used for training

def rebuild_model(input_shape=(2, 57)):
    """Rebuild the model with the same architecture as the original"""
    print(f"Rebuilding model with input shape {input_shape}...")
    
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def load_model_and_data():
    """Load the trained model and the most recent data"""
    print("Loading trained model and recent data...")
    
    # Check if the model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file {MODEL_PATH} not found.")
        return None, None, None
    
    # Load the feature list
    if os.path.exists('model_features.csv'):
        features_df = pd.read_csv('model_features.csv')
        feature_list = features_df['feature'].tolist()
        print(f"Loaded {len(feature_list)} features from model_features.csv")
    else:
        print("Warning: model_features.csv not found. Using default features.")
        feature_list = None
    
    # Try to load the model in different ways
    model = None
    
    try:
        # First try: direct loading
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Standard model loading failed: {e}")
        
        try:
            # Second try: with custom objects
            model = tf.keras.models.load_model(
                MODEL_PATH,
                custom_objects={
                    'mse': tf.keras.losses.mean_squared_error,
                    'mae': tf.keras.metrics.mean_absolute_error
                }
            )
            print("Model loaded with custom objects.")
        except Exception as e2:
            print(f"Custom objects loading failed: {e2}")
            
            # Third try: rebuild model and load weights
            print("Rebuilding model and loading weights only...")
            # Get the input shape from the features
            if feature_list:
                input_shape = (SEQ_LENGTH, len(feature_list))
            else:
                input_shape = (SEQ_LENGTH, 57)  # Default from your training
            
            model = rebuild_model(input_shape)
            
            try:
                # Try to load weights
                model.load_weights(MODEL_PATH)
                print("Model weights loaded successfully.")
            except Exception as e3:
                print(f"Weight loading failed: {e3}")
                
                print("Creating a new model without weights...")
                model = rebuild_model(input_shape)
                print("WARNING: Using an untrained model. Predictions will not be accurate.")
    
    if model is None:
        return None, None, None
    
    # Load the most recent data (2024) as the starting point
    latest_gml_file = "Befolkingsdata_formatert/Befolkning_0000_Norge_25833_BefolkningPaGrunnkretsniva2024_GML.gml"
    print(f"Looking for data file: {latest_gml_file}")
    
    if not os.path.exists(latest_gml_file):
        print(f"Error: Latest data file {latest_gml_file} not found.")
        # Try to find any GML file
        import glob
        gml_files = glob.glob("Befolkingsdata_formatert/*.gml")
        if gml_files:
            latest_gml_file = sorted(gml_files)[-1]
            print(f"Using alternative file: {latest_gml_file}")
        else:
            print("No GML files found.")
            return model, None, feature_list
    
    # Load the CSV data instead of parsing GML (simpler approach)
    csv_file = latest_gml_file.replace('.gml', '.csv')
    if os.path.exists(csv_file):
        print(f"Loading data from CSV: {csv_file}")
        latest_data = pd.read_csv(csv_file)
    else:
        print(f"Creating a simple DataFrame for testing...")
        # Create a simple DataFrame with basic columns
        latest_data = pd.DataFrame({
            'grunnkretsnummer': range(1000, 1100),
            'grunnkretsnavn': [f"Test Area {i}" for i in range(100)],
            'kommunenummer': [301] * 100,  # Oslo
            'kommunenavn': ['Oslo'] * 100,
            'totalBefolkning': np.random.randint(100, 5000, 100),
            'folketilvekst': np.random.randint(-50, 100, 100),
            'antallMenn': np.random.randint(50, 2500, 100),
            'antallKvinner': np.random.randint(50, 2500, 100),
            'year': [2024] * 100
        })
    
    print(f"Loaded data with {len(latest_data)} records")
    return model, latest_data, feature_list

def create_features(df, prev_df=None):
    """Create model features from raw data"""
    print("Creating features for prediction...")
    
    # Make a copy to avoid modifying the original
    df_features = df.copy()
    
    # Fill missing values
    numeric_cols = df_features.select_dtypes(include=['number']).columns
    df_features[numeric_cols] = df_features[numeric_cols].fillna(0)
    
    # If we have previous year data, calculate year-over-year features
    if prev_df is not None:
        # Create a lookup dictionary for faster retrieval
        prev_data = {}
        for _, row in prev_df.iterrows():
            prev_data[row['grunnkretsnummer']] = row
        
        # Calculate growth rate and other lagged features
        for idx, row in df_features.iterrows():
            grunnkrets = row['grunnkretsnummer']
            if grunnkrets in prev_data:
                prev_row = prev_data[grunnkrets]
                prev_pop = prev_row['totalBefolkning']
                curr_pop = row['totalBefolkning']
                
                # Only calculate if both values are valid
                if prev_pop and curr_pop and prev_pop > 0:
                    df_features.at[idx, 'growth_rate'] = (curr_pop - prev_pop) / prev_pop
                    df_features.at[idx, 'pop_change'] = curr_pop - prev_pop
                    df_features.at[idx, 'totalBefolkning_lag1'] = prev_pop
                    
                    if 'folketilvekst' in prev_row and prev_row['folketilvekst'] is not None:
                        df_features.at[idx, 'folketilvekst_lag1'] = prev_row['folketilvekst']
                else:
                    df_features.at[idx, 'growth_rate'] = 0
                    df_features.at[idx, 'pop_change'] = 0
                    df_features.at[idx, 'totalBefolkning_lag1'] = 0
                    df_features.at[idx, 'folketilvekst_lag1'] = 0
            else:
                # Default values if no previous data
                df_features.at[idx, 'growth_rate'] = 0
                df_features.at[idx, 'pop_change'] = 0
                df_features.at[idx, 'totalBefolkning_lag1'] = 0
                df_features.at[idx, 'folketilvekst_lag1'] = 0
    else:
        # If no previous data, set default values
        df_features['growth_rate'] = 0
        df_features['pop_change'] = 0
        df_features['totalBefolkning_lag1'] = 0
        df_features['folketilvekst_lag1'] = 0
    
    # Calculate age distribution ratios if age columns exist
    age_columns = [col for col in df_features.columns if 'befolkning' in col.lower() and 'År' in col]
    for col in age_columns:
        # Avoid division by zero
        df_features[f'{col}_ratio'] = np.where(df_features['totalBefolkning'] > 0, 
                                          df_features[col] / df_features['totalBefolkning'], 
                                          0)
    
    # Calculate gender ratio if gender columns exist
    if 'antallMenn' in df_features.columns and 'antallKvinner' in df_features.columns:
        df_features['gender_ratio'] = np.where(df_features['antallKvinner'] > 0, 
                                          df_features['antallMenn'] / df_features['antallKvinner'], 
                                          1)
    
    # Create aggregated age groups if age columns exist
    if age_columns:
        children_cols = [col for col in age_columns if any(age in col for age in ['0Til04', '05Til09', '10Til14'])]
        if children_cols:
            df_features['children_0_14'] = df_features[children_cols].sum(axis=1)
            df_features['children_ratio'] = df_features['children_0_14'] / df_features['totalBefolkning'].replace(0, 1)
        
        elderly_cols = [col for col in age_columns if any(age in col for age in ['65', '70', '75', '80', '85', '90'])]
        if elderly_cols:
            df_features['elderly'] = df_features[elderly_cols].sum(axis=1)
            df_features['elderly_ratio'] = df_features['elderly'] / df_features['totalBefolkning'].replace(0, 1)
        
        working_age_cols = [col for col in age_columns if any(age in col for age in ['20', '25', '30', '35', '40', '45', '50', '55', '60'])]
        if working_age_cols:
            df_features['working_age'] = df_features[working_age_cols].sum(axis=1)
            df_features['working_age_ratio'] = df_features['working_age'] / df_features['totalBefolkning'].replace(0, 1)
    
    return df_features

def prepare_prediction_data(current_data, previous_data, feature_list):
    """Prepare data for prediction"""
    # Create features from current data
    features_df = create_features(current_data, previous_data)
    
    # Select only the features needed for the model
    if feature_list:
        # Filter to only features we have available
        available_features = [f for f in feature_list if f in features_df.columns]
        if len(available_features) < len(feature_list):
            print(f"Warning: Only {len(available_features)} of {len(feature_list)} features available")
        
        X = features_df[available_features].values
    else:
        # Use all numeric features except target and identifier columns
        exclude_cols = ['folketilvekst', 'grunnkretsnummer', 'year', 'kommunenummer']
        numeric_cols = features_df.select_dtypes(include=['number']).columns
        cols_to_use = [col for col in numeric_cols if col not in exclude_cols]
        X = features_df[cols_to_use].values
        print(f"Using {len(cols_to_use)} features: {cols_to_use[:5]}...")
    
    # Get a mapping from index to grunnkretsnummer for later use
    grunnkrets_mapping = features_df['grunnkretsnummer'].to_dict()
    
    return X, grunnkrets_mapping, features_df

def make_predictions(model, X, grunnkrets_mapping, features_df, year):
    """Make predictions for a future year"""
    print(f"Making predictions for year {year}...")
    
    # Reshape input for LSTM if needed
    if len(X.shape) == 2:  # If X is 2D
        # For sequence length 2, we need 2 identical frames as a starting point
        X = np.array([X, X])
        X = np.transpose(X, (1, 0, 2))
    
    # Make predictions
    predictions = model.predict(X, verbose=0)
    
    # Create a DataFrame with predictions
    results = pd.DataFrame({
        'grunnkretsnummer': list(grunnkrets_mapping.values()),
        'predicted_growth': predictions.flatten()
    })
    
    # Merge with the original data
    results = pd.merge(features_df, results, on='grunnkretsnummer')
    
    # Calculate new total population
    results['new_totalBefolkning'] = results['totalBefolkning'] + results['predicted_growth']
    
    # Ensure no negative populations
    results['new_totalBefolkning'] = results['new_totalBefolkning'].apply(lambda x: max(0, round(x)))
    
    # Update age distributions proportionally if age columns exist
    age_columns = [col for col in results.columns if 'befolkning' in col.lower() and 'År' in col]
    if age_columns:
        for col in age_columns:
            ratio_col = f'{col}_ratio'
            if ratio_col in results.columns:
                results[f'new_{col}'] = (results[ratio_col] * results['new_totalBefolkning']).round().astype(int)
            else:
                # If ratio not available, try to maintain the same proportion
                results[f'new_{col}'] = (results[col] / results['totalBefolkning'].replace(0, 1) * results['new_totalBefolkning']).round().astype(int)
    
    # Update gender counts based on gender ratio
    if 'gender_ratio' in results.columns:
        # Use gender ratio to split new population
        total_new_pop = results['new_totalBefolkning']
        results['new_antallKvinner'] = (total_new_pop / (results['gender_ratio'] + 1)).round().astype(int)
        results['new_antallMenn'] = (total_new_pop - results['new_antallKvinner']).astype(int)
    else:
        # If gender columns exist, maintain same gender proportion
        if 'antallMenn' in results.columns and 'antallKvinner' in results.columns:
            results['new_antallMenn'] = (results['antallMenn'] / results['totalBefolkning'].replace(0, 1) * results['new_totalBefolkning']).round().astype(int)
            results['new_antallKvinner'] = (results['new_totalBefolkning'] - results['new_antallMenn']).astype(int)
    
    # Set year
    results['new_year'] = year
    
    return results

def save_prediction_results(prediction_results, year, output_dir="Predictions"):
    """Save prediction results to CSV"""
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a simplified DataFrame with key columns
    output_df = pd.DataFrame({
        'grunnkretsnummer': prediction_results['grunnkretsnummer'],
        'grunnkretsnavn': prediction_results['grunnkretsnavn'] if 'grunnkretsnavn' in prediction_results.columns else '',
        'kommunenummer': prediction_results['kommunenummer'],
        'kommunenavn': prediction_results['kommunenavn'] if 'kommunenavn' in prediction_results.columns else '',
        'year': year,
        'totalBefolkning': prediction_results['new_totalBefolkning'].astype(int),
        'folketilvekst': prediction_results['predicted_growth'].round().astype(int),
        'antallMenn': prediction_results['new_antallMenn'].astype(int) if 'new_antallMenn' in prediction_results.columns else 0,
        'antallKvinner': prediction_results['new_antallKvinner'].astype(int) if 'new_antallKvinner' in prediction_results.columns else 0
    })
    
    # Add age columns if they exist
    age_columns = [col for col in prediction_results.columns if col.startswith('new_befolkning')]
    for col in age_columns:
        original_col = col[4:]  # Remove 'new_' prefix
        output_df[original_col] = prediction_results[col].astype(int)
    
    # Save to CSV
    csv_file = f"{output_dir}/Befolkning_0000_Norge_25833_BefolkningPaGrunnkretsniva{year}.csv"
    output_df.to_csv(csv_file, index=False)
    print(f"Saved predictions to {csv_file}")
    
    return output_df

def main():
    """Main function to create future predictions"""
    print(f"Generating population predictions for years {START_YEAR}-{END_YEAR}...")
    
    # Step 1: Load model and data
    model, latest_data, feature_list = load_model_and_data()
    if model is None:
        print("Failed to load model. Aborting.")
        return
    
    if latest_data is None:
        print("Failed to load data. Aborting.")
        return
    
    # Output directory for predicted files
    output_dir = "Predictions"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save summary of starting data
    print(f"\nStarting prediction process with data from year {int(latest_data['year'].iloc[0])}:")
    total_pop = latest_data['totalBefolkning'].sum()
    avg_pop = latest_data['totalBefolkning'].mean()
    print(f"Total population: {total_pop:,}")
    print(f"Average population per grunnkrets: {avg_pop:.1f}")
    print(f"Number of grunnkretser: {len(latest_data)}")
    
    # For tracking data across years
    current_year_data = latest_data
    previous_year_data = None
    
    # Create a summary DataFrame to track changes
    summary = []
    
    # Step 2: Make predictions for each year
    for year in range(START_YEAR, END_YEAR + 1):
        print(f"\n===== Processing Year {year} =====")
        
        # Prepare data for prediction
        X, grunnkrets_mapping, features_df = prepare_prediction_data(
            current_year_data, previous_year_data, feature_list
        )
        
        # Make predictions
        prediction_results = make_predictions(model, X, grunnkrets_mapping, features_df, year)
        
        # Save prediction results
        year_results = save_prediction_results(prediction_results, year, output_dir)
        
        # Update summary
        total_pop = year_results['totalBefolkning'].sum()
        growth = total_pop - current_year_data['totalBefolkning'].sum()
        summary.append({
            'year': year,
            'total_population': total_pop,
            'population_growth': growth,
            'growth_percentage': growth / current_year_data['totalBefolkning'].sum() * 100,
        })
        
        # For the next year's prediction, use this year's prediction as the base
        previous_year_data = current_year_data
        
        # Update current year data with predictions
        current_year_data = prediction_results.copy()
        current_year_data['year'] = year
        current_year_data['totalBefolkning'] = current_year_data['new_totalBefolkning']
        current_year_data['folketilvekst'] = current_year_data['predicted_growth']
        
        # Update gender counts
        if 'new_antallMenn' in current_year_data.columns:
            current_year_data['antallMenn'] = current_year_data['new_antallMenn']
        if 'new_antallKvinner' in current_year_data.columns:
            current_year_data['antallKvinner'] = current_year_data['new_antallKvinner']
        
        # Update age distributions
        age_columns = [col for col in current_year_data.columns if col.startswith('new_befolkning')]
        for col in age_columns:
            original_col = col[4:]  # Remove 'new_' prefix
            current_year_data[original_col] = current_year_data[col]
    
    # Save summary
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(f"{output_dir}/population_prediction_summary.csv", index=False)
    
    # Print summary
    print("\n===== Prediction Summary =====")
    for row in summary:
        print(f"Year {row['year']}: Population {row['total_population']:,.0f} (Growth: {row['population_growth']:,.0f}, {row['growth_percentage']:.2f}%)")
    
    print("\nPrediction process complete!")
    print(f"Prediction files for years {START_YEAR}-{END_YEAR} have been saved to the '{output_dir}' directory.")

if __name__ == "__main__":
    main()