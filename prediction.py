"""
Prediction module for Population Growth Prediction model
Contains functions for making future predictions
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime

from config import ModelConfig
from features import create_features
from model import PopulationGrowthModel
from utils import load_scalers

class PopulationPredictor:
    def __init__(self, config=None):
        self.config = config or ModelConfig()
        self.model = PopulationGrowthModel(self.config)
        self.feature_scaler = None
        self.target_scaler = None
        
    def load_model_and_scalers(self):
        """Load the trained model and scalers"""
        print("Loading trained model and scalers...")
        
        # Load model
        model_loaded = self.model.load_model()
        if not model_loaded:
            return False
            
        # Load scalers
        self.feature_scaler, self.target_scaler = load_scalers()
        if self.feature_scaler is None or self.target_scaler is None:
            print("Warning: Scalers not loaded. Predictions may not be accurate.")
            
        return True
        
    def load_latest_data(self):
        """Load the most recent population data"""
        print("Loading latest population data...")
        
        # Try to find latest data file (2024)
        latest_csv_file = "Befolkingsdata_formatert/Befolkning_0000_Norge_25833_BefolkningPaGrunnkretsniva2024.csv"
        
        if not os.path.exists(latest_csv_file):
            # Try to find any population CSV file
            import glob
            csv_files = glob.glob("Befolkingsdata_formatert/*.csv")
            if csv_files:
                latest_csv_file = sorted(csv_files)[-1]
                print(f"Using alternative file: {latest_csv_file}")
            else:
                print("No CSV files found. Creating test data...")
                # Create a simple test DataFrame
                latest_data = self.create_test_data()
                return latest_data
                
        # Load data from CSV
        try:
            latest_data = pd.read_csv(latest_csv_file)
            print(f"Loaded {len(latest_data)} records from {latest_csv_file}")
            return latest_data
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Creating test data instead...")
            return self.create_test_data()
    
    def create_test_data(self):
        """Create a simple test DataFrame for testing predictions"""
        print("Creating sample test data...")
        
        # Create a simple DataFrame with basic columns
        test_data = pd.DataFrame({
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
        
        print(f"Created test data with {len(test_data)} records")
        return test_data
    
    def prepare_data_for_prediction(self, current_data, previous_data=None):
        """Prepare the data for prediction"""
        print("Preparing data for prediction...")
        
        # Create features
        features_df = create_features(current_data)
        
        # Load feature list
        if os.path.exists(self.config.FEATURE_LIST_FILE):
            features_list_df = pd.read_csv(self.config.FEATURE_LIST_FILE)
            feature_list = features_list_df['feature'].tolist()
            
            # Filter to only features we have available
            available_features = [f for f in feature_list if f in features_df.columns]
            if len(available_features) < len(feature_list):
                print(f"Warning: Only {len(available_features)} of {len(feature_list)} features available")
                
            X = features_df[available_features].values
        else:
            # Use all numeric features except target and identifier columns
            print("Feature list file not found. Using all numeric features.")
            exclude_cols = ['folketilvekst', 'grunnkretsnummer', 'year', 'kommunenummer']
            numeric_cols = features_df.select_dtypes(include=['number']).columns
            cols_to_use = [col for col in numeric_cols if col not in exclude_cols]
            X = features_df[cols_to_use].values
            
        # Get a mapping from index to grunnkretsnummer for later use
        grunnkrets_mapping = features_df['grunnkretsnummer'].to_dict()
        
        return X, grunnkrets_mapping, features_df
    
    def make_predictions(self, X, grunnkrets_mapping, features_df, year):
        """Make predictions for a future year"""
        print(f"Making predictions for year {year}...")
        
        # Scale X if scaler is available
        if self.feature_scaler is not None:
            X_reshaped = X.reshape(X.shape[0], -1)
            X_scaled = self.feature_scaler.transform(X_reshaped)
            X_scaled = X_scaled.reshape(X.shape)
        else:
            X_scaled = X
        
        # Reshape input for LSTM if needed
        if len(X_scaled.shape) == 2:  # If X is 2D
            # For sequence length 2, we need 2 identical frames as a starting point
            seq_length = self.config.SEQ_LENGTH
            X_scaled = np.array([X_scaled] * seq_length)
            X_scaled = np.transpose(X_scaled, (1, 0, 2))
        
        # Make predictions
        predictions = self.model.model.predict(X_scaled, verbose=0)
        
        # Unscale predictions if scaler is available
        if self.target_scaler is not None:
            predictions = self.target_scaler.inverse_transform(predictions)
        
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
    
    def save_prediction_results(self, prediction_results, year):
        """Save prediction results to CSV"""
        # Ensure output directory exists
        output_dir = self.config.PREDICTION_OUTPUT_DIR
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
    
    def generate_predictions(self, start_year=None, end_year=None):
        """Generate predictions for a range of years"""
        if start_year is None:
            start_year = self.config.START_YEAR
        if end_year is None:
            end_year = self.config.END_YEAR
            
        print(f"Generating population predictions for years {start_year}-{end_year}...")
        
        # Load model and scalers
        if not self.load_model_and_scalers():
            print("Failed to load model. Aborting.")
            return
        
        # Load latest data
        current_year_data = self.load_latest_data()
        if current_year_data is None or len(current_year_data) == 0:
            print("Failed to load data. Aborting.")
            return
        
        # Save summary of starting data
        print(f"\nStarting prediction process with data from year {int(current_year_data['year'].iloc[0])}:")
        total_pop = current_year_data['totalBefolkning'].sum()
        avg_pop = current_year_data['totalBefolkning'].mean()
        print(f"Total population: {total_pop:,}")
        print(f"Average population per grunnkrets: {avg_pop:.1f}")
        print(f"Number of grunnkretser: {len(current_year_data)}")
        
        # For tracking data across years
        previous_year_data = None
        
        # Create a summary DataFrame to track changes
        summary = []
        
        # Make predictions for each year
        for year in range(start_year, end_year + 1):
            print(f"\n===== Processing Year {year} =====")
            
            # Prepare data for prediction
            X, grunnkrets_mapping, features_df = self.prepare_data_for_prediction(
                current_year_data, previous_year_data
            )
            
            # Make predictions
            prediction_results = self.make_predictions(X, grunnkrets_mapping, features_df, year)
            
            # Save prediction results
            year_results = self.save_prediction_results(prediction_results, year)
            
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
        summary_df.to_csv(f"{self.config.PREDICTION_OUTPUT_DIR}/population_prediction_summary.csv", index=False)
        
        # Print summary
        print("\n===== Prediction Summary =====")
        for row in summary:
            print(f"Year {row['year']}: Population {row['total_population']:,.0f} (Growth: {row['population_growth']:,.0f}, {row['growth_percentage']:.2f}%)")
        
        print("\nPrediction process complete!")
        print(f"Prediction files for years {start_year}-{end_year} have been saved to the '{self.config.PREDICTION_OUTPUT_DIR}' directory.")
        
        return summary_df