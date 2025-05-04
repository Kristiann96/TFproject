"""
Data pipeline module for Population Growth Prediction model
Simplified to use only 1 input year for predictions
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import ModelConfig
from features import create_features, get_feature_list

class DataPipeline:
    def __init__(self, config=None):
        self.config = config or ModelConfig()
        
    def load_data(self):
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
            print("WARNING: Data needs at least 2 years (current year and target year).")
            return None
        
        return loaded_df
    
    def select_municipalities(self, df, num_municipalities=25):
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
        selected_info.to_csv('selected_municipalities.csv')
        print("Saved information about selected municipalities to 'selected_municipalities.csv'")
        
        # Filter the original dataset to only include the selected municipalities
        filtered_df = df[df['kommunenummer'].isin(selected_kommuner)].copy()
        print(f"\nFiltered dataset contains {len(filtered_df)} records from {len(selected_kommuner)} municipalities.")
        
        return filtered_df, selected_kommuner
    
    def preprocess_data(self, df):
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
        
        # Apply feature engineering
        df = create_features(df)
        
        return df
    
    def find_complete_sequences(self, df, seq_length=None, min_count=None):
        """Find grunnkretser with target year data available (simplified from multi-year sequences)"""
        # Always use sequence length of 1 for simplified model
        seq_length = 1
            
        print(f"Finding grunnkretser with at least two consecutive years of data...")
        
        complete_grunnkretser = []
        
        # Get all unique years
        all_years = sorted(df['year'].unique())
        
        # We need at least one year as input and another year as target
        if len(all_years) < 2:
            print("Not enough years in dataset to create input-target pairs")
            return []
        
        # For each grunnkrets, check if it has data for at least 2 consecutive years
        for grunnkrets, group in df.groupby('grunnkretsnummer'):
            years_present = set(group['year'].unique())
            
            has_consecutive_years = False
            for i in range(len(all_years)-1):
                if all_years[i] in years_present and all_years[i+1] in years_present:
                    has_consecutive_years = True
                    break
                    
            if has_consecutive_years:
                complete_grunnkretser.append(grunnkrets)
        
        print(f"Found {len(complete_grunnkretser)} grunnkretser with consecutive years of data")
        
        # If min_count is specified, load additional grunnkretser from file if available
        if min_count and len(complete_grunnkretser) < min_count:
            print(f"Found fewer than {min_count} grunnkretser, checking for additional data...")
            if os.path.exists(self.config.COMPLETE_GRUNNKRETSER_FILE):
                extra_grunnkretser_df = pd.read_csv(self.config.COMPLETE_GRUNNKRETSER_FILE)
                extra_grunnkretser = extra_grunnkretser_df['grunnkretsnummer'].tolist()
                
                # Add grunnkretser from the file that are not already in our list
                new_grunnkretser = [g for g in extra_grunnkretser if g not in complete_grunnkretser]
                
                # Add enough to reach min_count if possible
                needed = min_count - len(complete_grunnkretser)
                if needed > 0:
                    additional = new_grunnkretser[:needed]
                    complete_grunnkretser.extend(additional)
                    print(f"Added {len(additional)} grunnkretser from external file")
        
        print(f"Final count: {len(complete_grunnkretser)} grunnkretser")
        return complete_grunnkretser
    
    def create_time_series_datasets(self, df, complete_grunnkretser, target_col='folketilvekst', seq_length=None):
        """Create simplified input-target pairs (using just one year to predict the next)"""
        # Override seq_length to always be 1
        seq_length = 1
            
        print(f"Creating input-target pairs with target column: {target_col}")
        
        # If target column is missing, create it from totalBefolkning
        if target_col not in df.columns or df[target_col].isna().all():
            print(f"Target column {target_col} not available, using population change")
            df['pop_diff'] = df.groupby('grunnkretsnummer')['totalBefolkning'].diff()
            df['pop_diff'] = df['pop_diff'].fillna(0)
            target_col = 'pop_diff'
        
        print(f"Using {target_col} as target variable")
        
        # Get feature columns
        feature_cols = get_feature_list(df)
        
        print(f"Using {len(feature_cols)} features")
        if len(feature_cols) > 10:
            print(f"Sample features: {feature_cols[:10]}...")
        else:
            print(f"Features: {feature_cols}")
        
        # Filter to just the grunnkretser with complete data
        filtered_df = df[df['grunnkretsnummer'].isin(complete_grunnkretser)]
        
        print(f"Creating input-target pairs from {len(filtered_df)} filtered records...")
        
        # Group by grunnkrets and create pairs
        X_data = []  # Single year inputs
        y_data = []  # Target values
        metadata = []  # Store metadata
        
        # Get all years in sorted order
        all_years = sorted(df['year'].unique())
        
        # For each grunnkrets
        for grunnkrets, group in filtered_df.groupby('grunnkretsnummer'):
            group = group.sort_values('year')
            
            # For all consecutive year pairs
            for i in range(len(all_years)-1):
                year = all_years[i]
                next_year = all_years[i+1]
                
                # Get data for current year and next year
                current_data = group[group['year'] == year]
                next_data = group[group['year'] == next_year]
                
                # Skip if either year is missing
                if current_data.empty or next_data.empty:
                    continue
                
                # Get features for current year
                feature_values = current_data[feature_cols].values[0]  # Just one row
                
                # Get target value for next year
                target_value = next_data[target_col].values[0]
                
                # Store data
                X_data.append(feature_values)
                y_data.append(target_value)
                
                # Store metadata
                metadata.append({
                    'grunnkretsnummer': grunnkrets,
                    'kommunenummer': current_data['kommunenummer'].values[0],
                    'year': next_year,  # Year of the target
                    'sequence_id': f"{grunnkrets}_{year}"
                })
        
        # Convert to numpy arrays
        if X_data:
            X = np.array(X_data)
            y = np.array(y_data)
            metadata_df = pd.DataFrame(metadata)
            
            print(f"Created {len(X)} input-target pairs from {len(metadata_df['grunnkretsnummer'].unique())} grunnkretser")
            print(f"X shape: {X.shape}, y shape: {y.shape}")
            
            return X, y, metadata_df, feature_cols
        else:
            print("No input-target pairs created. Check your data.")
            return None, None, None, None
    
    def split_data(self, X, y, metadata_df):
        """Split data into training, validation, and testing sets"""
        print("Splitting data into train, validation, and test sets...")
        
        # Get unique grunnkretser
        unique_grunnkretser = metadata_df['grunnkretsnummer'].unique()
        
        # Split grunnkretser into train, validation, and test sets
        train_grunnkretser, temp_grunnkretser = train_test_split(
            unique_grunnkretser, test_size=0.3, random_state=self.config.RANDOM_SEED
        )
        
        val_grunnkretser, test_grunnkretser = train_test_split(
            temp_grunnkretser, test_size=0.5, random_state=self.config.RANDOM_SEED
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