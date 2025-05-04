"""
Feature engineering module for Population Growth Prediction model
Contains functions for creating and manipulating features
Simplified to focus on single-year features without sequence dependencies
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

def create_static_features(df):
    """Create static features that don't depend on time sequences"""
    # Add population density-related features if area information is available
    if 'areal' in df.columns and df['areal'].sum() > 0:
        df['population_density'] = df['totalBefolkning'] / df['areal'].replace(0, 1)
        df['population_density'] = df['population_density'].clip(0, df['population_density'].quantile(0.99))
    
    # Add percentage of males and females
    if 'antallMenn' in df.columns and 'antallKvinner' in df.columns:
        df['male_percentage'] = df['antallMenn'] / df['totalBefolkning'].replace(0, 1)
        df['female_percentage'] = df['antallKvinner'] / df['totalBefolkning'].replace(0, 1)
    
    # Add kommune-level aggregations as features
    if 'kommunenummer' in df.columns:
        kommune_pop = df.groupby('kommunenummer')['totalBefolkning'].transform('sum')
        df['kommune_population'] = kommune_pop
        df['percentage_of_kommune'] = df['totalBefolkning'] / kommune_pop.replace(0, 1)
    
    return df

def calculate_current_growth_rate(df):
    """Calculate current growth rates using whatever data is available"""
    # Calculate simple growth rate if 'folketilvekst' is available
    if 'folketilvekst' in df.columns:
        df['current_growth_rate'] = np.where(df['totalBefolkning'] > 0,
                                         df['folketilvekst'] / df['totalBefolkning'],
                                         0)
        df['current_growth_rate'] = df['current_growth_rate'].clip(-1, 5)
    
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
    print("Creating features for single-year input...")
    
    # Apply demographic features
    df = create_demographic_features(df)
    
    # Apply static features
    df = create_static_features(df)
    
    # Calculate current growth rates
    df = calculate_current_growth_rate(df)
    
    print("Feature engineering complete")
    return df