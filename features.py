"""
Feature engineering module for Population Growth Prediction model
Contains functions for creating and manipulating features
"""
import numpy as np
import pandas as pd

def create_demographic_features(df):
    """Create demographic features like age ratios, gender ratio, etc."""
    # Fill missing values in numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
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
        df['gender_ratio'] = np.where(df['antallKvinner'] > 0, 
                                    df['antallMenn'] / df['antallKvinner'], 
                                    1)
        # Clip to reasonable range (0.5 to 2)
        df['gender_ratio'] = df['gender_ratio'].clip(0.5, 2)
    
    return df

def create_time_features(df, group_col='grunnkretsnummer'):
    """Create time-based features like growth rates and lagged variables"""
    print("Creating time-based features...")
    
    # Calculate growth rate (percent change in population) - with safety limits
    df['growth_rate'] = df.groupby(group_col)['totalBefolkning'].pct_change()
    # Clip growth rate to reasonable range (-1 to 5, i.e., -100% to 500%)
    df['growth_rate'] = df['growth_rate'].clip(-1, 5).fillna(0)
    
    # Calculate absolute population change
    df['pop_change'] = df.groupby(group_col)['totalBefolkning'].diff()
    # Clip population change to reasonable range
    max_change = df['totalBefolkning'].quantile(0.99)
    df['pop_change'] = df['pop_change'].clip(-max_change, max_change).fillna(0)
    
    # Add lagged features (previous year's values)
    lag_columns = ['totalBefolkning', 'growth_rate', 'pop_change']
    if 'folketilvekst' in df.columns:
        lag_columns.append('folketilvekst')
    
    for col in lag_columns:
        if col in df.columns:
            df[f'{col}_lag1'] = df.groupby(group_col)[col].shift(1)
            df[f'{col}_lag1'] = df[f'{col}_lag1'].fillna(0)
    
    return df

def get_feature_list(df):
    """Extract the list of numeric features suitable for modeling"""
    # Select numeric features for the model
    numeric_cols = df.select_dtypes(include=['number']).columns
    exclude_cols = ['gml_id', 'versjonId', 'folketilvekst', 'year', 
                    'grunnkretsnummer', 'kommunenummer']
    
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    return feature_cols

def create_features(df):
    """Combined function to create all features"""
    print("Creating all features...")
    
    # Apply demographic features
    df = create_demographic_features(df)
    
    # Apply time-based features
    df = create_time_features(df)
    
    print("Feature engineering complete")
    return df