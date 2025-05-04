"""
Main script for Population Growth Prediction model
Orchestrates the entire workflow from data loading to prediction
"""
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

from config import ModelConfig
from data_pipeline import DataPipeline
from model import PopulationGrowthModel
from evaluation import plot_training_history, evaluate_model, compare_models
from prediction import PopulationPredictor
from utils import timing_decorator, check_gpu_availability, save_model_metadata, save_scalers, convert_to_tfjs

@timing_decorator
def train_model_workflow(args):
    """Full workflow for training the model"""
    print("Starting model training workflow...")
    
    # Initialize configuration
    config = ModelConfig()
    config.ensure_directories()
    
    # Check GPU availability
    check_gpu_availability()
    
    # Initialize data pipeline
    data_pipeline = DataPipeline(config)
    
    # Load data
    df = data_pipeline.load_data()
    if df is None:
        return
    
    # Select subset of municipalities
    filtered_df, selected_kommuner = data_pipeline.select_municipalities(df, num_municipalities=args.municipalities)
    
    # Preprocess data
    filtered_df = data_pipeline.preprocess_data(filtered_df)
    
    # Find complete sequences
    complete_grunnkretser = data_pipeline.find_complete_sequences(
        filtered_df, 
        seq_length=config.SEQ_LENGTH,
        min_count=args.min_sequences
    )
    
    if not complete_grunnkretser:
        print("No complete sequences found. Cannot proceed with modeling.")
        return
    
    # Create time series datasets
    result = data_pipeline.create_time_series_datasets(
        filtered_df, 
        complete_grunnkretser,
        target_col='folketilvekst',
        seq_length=config.SEQ_LENGTH
    )
    
    if result[0] is None:
        return
    
    X, y, metadata_df, feature_cols = result
    
    # Save feature list for future use
    feature_df = pd.DataFrame({'feature': feature_cols})
    feature_df.to_csv(config.FEATURE_LIST_FILE, index=False)
    print(f"Saved feature list to {config.FEATURE_LIST_FILE}")
    
    # Split data
    (X_train, y_train, metadata_train), (X_val, y_val, metadata_val), (X_test, y_test, metadata_test) = data_pipeline.split_data(X, y, metadata_df)
    
    # Initialize model
    model_trainer = PopulationGrowthModel(config)
    
    # Scale data
    scaled_data = model_trainer.scale_data(X_train, X_val, X_test, y_train, y_val, y_test)
    X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled = scaled_data
    
    # Build and train model
    history = model_trainer.train(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled)
    
    # Plot training history
    plot_training_history(history, output_file=f'training_history_{args.output_prefix}.png')
    
    # Evaluate model
    results_df, mae, mse, rmse = evaluate_model(
        model_trainer.model, 
        X_test_scaled, 
        y_test_scaled, 
        model_trainer.target_scaler, 
        metadata_test,
        output_prefix=args.output_prefix
    )
    
    # Save model metadata and scalers
    save_model_metadata(
        feature_cols, 
        model_info={
            'input_sequence_length': config.SEQ_LENGTH,
            'mae': float(mae),
            'rmse': float(rmse),
            'created_on': pd.Timestamp.now().strftime('%Y-%m-%d')
        }
    )
    
    save_scalers(model_trainer.feature_scaler, model_trainer.target_scaler)
    
    # Compare with original model if it exists
    if os.path.exists('prediction_results.csv'):
        compare_models(
            'prediction_results.csv', 
            f'{args.output_prefix}prediction_results.csv'
        )
    
    # Convert model to TensorFlow.js
    if args.convert_to_js:
        convert_to_tfjs(
            config.MODEL_PATH, 
            config.TFJS_MODEL_DIR,
            feature_cols=feature_cols,
            scaler_params={
                'feature_scaler': {
                    'center_': model_trainer.feature_scaler.center_.tolist(),
                    'scale_': model_trainer.feature_scaler.scale_.tolist()
                },
                'target_scaler': {
                    'center_': model_trainer.target_scaler.center_.tolist(),
                    'scale_': model_trainer.target_scaler.scale_.tolist()
                }
            }
        )
    
    print("\nModel training workflow completed successfully!")

@timing_decorator
def predict_workflow(args):
    """Workflow for making predictions with trained model"""
    print("Starting prediction workflow...")
    
    # Initialize configuration
    config = ModelConfig()
    config.ensure_directories()
    
    # Initialize predictor
    predictor = PopulationPredictor(config)
    
    # Generate predictions
    summary_df = predictor.generate_predictions(args.start_year, args.end_year)
    
    if summary_df is not None:
        # Plot prediction summary
        plt.figure(figsize=(10, 6))
        plt.plot(summary_df['year'], summary_df['total_population'] / 1_000_000, marker='o')
        plt.title('Predicted Norwegian Population by Year')
        plt.xlabel('Year')
        plt.ylabel('Population (Millions)')
        plt.grid(True)
        plt.savefig(f"{config.PREDICTION_OUTPUT_DIR}/population_prediction.png")
        print(f"Saved population prediction chart to '{config.PREDICTION_OUTPUT_DIR}/population_prediction.png'")
    
    print("\nPrediction workflow completed successfully!")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Population Growth Prediction Model')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--municipalities', type=int, default=25,
                              help='Number of municipalities to include in training')
    train_parser.add_argument('--min-sequences', type=int, default=700,
                              help='Minimum number of sequences to train on')
    train_parser.add_argument('--output-prefix', type=str, default='',
                              help='Prefix for output files')
    train_parser.add_argument('--convert-to-js', action='store_true',
                              help='Convert model to TensorFlow.js format')
    
    # Prediction command
    predict_parser = subparsers.add_parser('predict', help='Make predictions with trained model')
    predict_parser.add_argument('--start-year', type=int, default=2025,
                               help='Start year for predictions')
    predict_parser.add_argument('--end-year', type=int, default=2030,
                               help='End year for predictions')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run selected command
    if args.command == 'train':
        train_model_workflow(args)
    elif args.command == 'predict':
        predict_workflow(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()