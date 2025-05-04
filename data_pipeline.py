"""
Data pipeline module for Population Growth Prediction model
Handles data loading, preprocessing, and sequence creation
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
            print("WARNING: Data doesn't have enough years for time series analysis.")
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
        """Find grunnkretser with complete sequences of data"""
        if seq_length is None:
            seq_length = self.config.SEQ_LENGTH
            
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
        
        print(f"Final count: {len(complete_grunnkretser)} grunnkretser with complete sequences")
        return complete_grunnkretser
    
    def create_time_series_datasets(self, df, complete_grunnkretser, target_col='folketilvekst', seq_length=None):
        """Create sequences for LSTM model with sliding window approach"""
        if seq_length is None:
            seq_length = self.config.SEQ_LENGTH
            
        print(f"Creating time series sequences with target column: {target_col}")
        
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
        
        print(f"Creating sequences from {len(filtered_df)} filtered records...")
        
        # Group by grunnkrets and create sequences with sliding window
        X_sequences = []
        y_sequences = []
        metadata = []  # Store grunnkrets, kommune, year for each sequence
        
        sequence_counter = 0
        grunnkrets_counter = 0
        
        # Adjust sequence length based on available years
        years_available = df['year'].nunique()
        effective_seq_length = min(seq_length, years_available - 2)  # Ensure at least 2 sequences possible
        
        print(f"Years available: {years_available}, using effective sequence length: {effective_seq_length}")
        
        for grunnkrets, group in filtered_df.groupby('grunnkretsnummer'):
            try:
                # Sort by year
                group = group.sort_values('year')
                
                # Check if we have enough years of data
                if len(group) < effective_seq_length + 1:
                    continue
                    
                # Get feature values
                feature_values = group[feature_cols].values
                
                # Make sure target column exists
                if target_col in group:
                    target_values = group[target_col].values
                    year_values = group['year'].values
                    kommune_values = group['kommunenummer'].values
                    
                    # Create multiple sequences per grunnkrets using a sliding window
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
                
            except Exception as e:
                print(f"Error processing grunnkrets {grunnkrets}: {e}")
                continue
        
        # Convert to numpy arrays
        if X_sequences:
            X = np.array(X_sequences)
            y = np.array(y_sequences)
            metadata_df = pd.DataFrame(metadata)
            
            print(f"Created {len(X)} sequences from {len(metadata_df['grunnkretsnummer'].unique())} grunnkretser")
            print(f"X shape: {X.shape}, y shape: {y.shape}")
            
            return X, y, metadata_df, feature_cols
        else:
            print("No sequences created. Check your data.")
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