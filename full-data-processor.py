import pandas as pd
import os
import glob
from osgeo import ogr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def find_gml_files():
    """Find all population GML files in the project directory"""
    print("Looking for GML files...")
    
    # Look in the TFproject folder first
    patterns = [
        "TFproject/Befolkingsdata_formatert/*.gml",
        "**/Befolkingsdata_formatert/*.gml",
    ]
    
    found_files = []
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        if files:
            found_files.extend(files)
            print(f"Found {len(files)} GML files matching pattern '{pattern}'")
    
    if found_files:
        found_files.sort()  # Sort by filename to ensure year order
        print(f"Found {len(found_files)} total GML files:")
        for file in found_files:
            print(f"  - {file}")
    else:
        print("No GML files found.")
    
    return found_files

def load_gml_data(file_path):
    """Load GML file and extract population data"""
    print(f"Loading data from {file_path}")
    
    driver = ogr.GetDriverByName('GML')
    data_source = driver.Open(file_path, 0)
    
    if data_source is None:
        print(f"Could not open {file_path}")
        return None
    
    layer = data_source.GetLayer(0)  # BefolkningPåGrunnkrets layer
    
    if layer is None:
        print(f"No valid layer found in {file_path}")
        return None
    
    # Get the layer definition to see available fields
    layer_defn = layer.GetLayerDefn()
    field_names = [layer_defn.GetFieldDefn(i).GetName() for i in range(layer_defn.GetFieldCount())]
    print(f"Available fields: {field_names[:10]}... (total: {len(field_names)})")
    
    # Create dataframe to store the data
    data = []
    feature_count = 0
    
    for feature in layer:
        feature_count += 1
        try:
            # Build a row with all available fields
            row = {}
            for field in field_names:
                row[field] = feature.GetField(field)
            
            data.append(row)
            
            # Print progress
            if feature_count % 5000 == 0:
                print(f"Processed {feature_count} features...")
                
        except Exception as e:
            print(f"Error processing feature: {e}")
            continue
    
    print(f"Loaded {len(data)} records from {file_path}")
    return pd.DataFrame(data)

def extract_year_from_filename(filename):
    """Extract year from filename pattern"""
    import re
    match = re.search(r'(\d{4})_GML', filename)
    if match:
        return int(match.group(1))
    return None

def process_all_gml_files():
    """Process all GML files and combine into a single dataset"""
    # Find all GML files
    gml_files = find_gml_files()
    
    if not gml_files:
        print("No GML files found. Cannot proceed.")
        return None
    
    # Process each file
    all_data = []
    
    for file_path in gml_files:
        year = extract_year_from_filename(file_path)
        if year is None:
            print(f"Could not extract year from {file_path}, skipping...")
            continue
        
        df = load_gml_data(file_path)
        if df is not None and not df.empty:
            # Add year column
            df['year'] = year
            all_data.append(df)
            print(f"Successfully processed data for year {year}, shape: {df.shape}")
        else:
            print(f"Failed to load data for {file_path}")
    
    if not all_data:
        print("No data was successfully loaded.")
        return None
    
    # Combine all years into one dataframe
    print("Combining data from all years...")
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Combined data shape: {combined_df.shape}")
    
    # Save the full dataset
    output_file = "full_population_data.csv"
    print(f"Saving full dataset to {output_file}...")
    combined_df.to_csv(output_file, index=False)
    print(f"Full dataset saved successfully with {len(combined_df)} records.")
    
    return combined_df

def create_annual_summary(df):
    """Create a summary of the data by year"""
    print("Creating annual summary...")
    
    if 'year' not in df.columns:
        print("Year column not found in the data.")
        return
    
    # Convert datatypes if needed
    if df['totalBefolkning'].dtype == 'object':
        df['totalBefolkning'] = pd.to_numeric(df['totalBefolkning'], errors='coerce')
    
    yearly_summary = df.groupby('year').agg(
        total_population=('totalBefolkning', 'sum'),
        avg_population=('totalBefolkning', 'mean'),
        num_grunnkretser=('grunnkretsnummer', 'nunique'),
        num_records=('grunnkretsnummer', 'count')
    ).reset_index()
    
    print("\nAnnual Summary:")
    print(yearly_summary)
    
    # Plot annual population trends
    plt.figure(figsize=(12, 6))
    plt.plot(yearly_summary['year'], yearly_summary['total_population'] / 1_000_000, marker='o')
    plt.title('Total Norwegian Population by Year')
    plt.xlabel('Year')
    plt.ylabel('Population (Millions)')
    plt.grid(True)
    plt.savefig('total_population_by_year.png')
    print("Saved population trend chart to 'total_population_by_year.png'")
    
    return yearly_summary

def analyze_data_completeness(df):
    """Analyze how many grunnkretser have data for all years"""
    print("\nAnalyzing data completeness...")
    
    # Count years per grunnkrets
    year_counts = df.groupby('grunnkretsnummer')['year'].nunique()
    
    # Get total number of years
    total_years = df['year'].nunique()
    print(f"Total number of years in the dataset: {total_years}")
    
    # Count how many grunnkretser have data for each number of years
    completeness_counts = year_counts.value_counts().sort_index()
    
    print("Number of grunnkretser by years of data:")
    for years, count in completeness_counts.items():
        print(f"  {years} years: {count} grunnkretser")
    
    # Count grunnkretser with complete data
    complete_grunnkretser = year_counts[year_counts == total_years].index.tolist()
    print(f"\nNumber of grunnkretser with data for all {total_years} years: {len(complete_grunnkretser)}")
    
    # Plot distribution
    plt.figure(figsize=(10, 6))
    plt.bar(completeness_counts.index, completeness_counts.values)
    plt.title('Number of Grunnkretser by Years of Available Data')
    plt.xlabel('Number of Years with Data')
    plt.ylabel('Number of Grunnkretser')
    plt.xticks(range(1, total_years + 1))
    plt.grid(axis='y')
    plt.savefig('grunnkrets_data_completeness.png')
    print("Saved data completeness chart to 'grunnkrets_data_completeness.png'")
    
    return complete_grunnkretser

def main():
    print("Starting full dataset processing...")
    
    # Process all GML files
    df = process_all_gml_files()
    
    if df is None:
        print("Failed to process data.")
        return
    
    # Create annual summary
    yearly_summary = create_annual_summary(df)
    
    # Analyze data completeness
    complete_grunnkretser = analyze_data_completeness(df)
    
    # Save the list of complete grunnkretser for use in modeling
    if complete_grunnkretser:
        pd.DataFrame({'grunnkretsnummer': complete_grunnkretser}).to_csv(
            'complete_grunnkretser.csv', index=False
        )
        print(f"Saved list of {len(complete_grunnkretser)} complete grunnkretser to 'complete_grunnkretser.csv'")
    
    print("Full dataset processing complete!")

if __name__ == "__main__":
    main()