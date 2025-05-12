"""
Evaluation module for Population Growth Prediction model
Contains functions for evaluating and visualizing model performance
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_training_history(history, output_file='training_history.png'):
    """Plot training history"""
    plt.figure(figsize=(12, 4))
    
    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Update these lines to use 'mean_absolute_error' instead of 'mae'
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mean_absolute_error'], label='Training MAE')  # Changed from 'mae'
    plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')  # Changed from 'val_mae'
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Saved training history plot to '{output_file}'")

def evaluate_model(model, X_test, y_test, target_scaler, metadata_test, output_prefix=''):
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
    results_file = f'{output_prefix}prediction_results.csv'
    results_df.to_csv(results_file, index=False)
    print(f"Saved detailed prediction results to '{results_file}'")
    
    # Create visualizations
    create_prediction_visualizations(results_df, output_prefix)
    
    return results_df, mae, mse, rmse

def create_prediction_visualizations(results_df, output_prefix=''):
    """Create visualizations of prediction results"""
    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['actual'], results_df['predicted'], alpha=0.5)
    plt.plot([min(results_df['actual']), max(results_df['actual'])], 
             [min(results_df['actual']), max(results_df['actual'])], 'r--')
    plt.title('Predicted vs Actual Population Growth')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.savefig(f'{output_prefix}predictions_vs_actual.png')
    print(f"Saved predictions vs actual plot to '{output_prefix}predictions_vs_actual.png'")
    
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
    plt.savefig(f'{output_prefix}error_by_municipality.png')
    print(f"Saved error by municipality plot to '{output_prefix}error_by_municipality.png'")
    
    # Plot error distribution
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['error'], bins=30)
    plt.title('Prediction Error Distribution')
    plt.xlabel('Error (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.savefig(f'{output_prefix}error_distribution.png')
    print(f"Saved error distribution plot to '{output_prefix}error_distribution.png'")
    
    # Plot errors across municipalities
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='kommunenummer', y='error', data=results_df)
    plt.title('Error Distribution by Municipality')
    plt.xlabel('Municipality')
    plt.ylabel('Error')
    plt.xticks(rotation=90)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}error_boxplot.png')
    print(f"Saved error boxplot to '{output_prefix}error_boxplot.png'")

def compare_models(original_results_file, new_results_file):
    """Compare performance of two different models"""
    if not os.path.exists(original_results_file):
        print(f"Original results file {original_results_file} not found. Cannot compare models.")
        return
        
    if not os.path.exists(new_results_file):
        print(f"New results file {new_results_file} not found. Cannot compare models.")
        return
    
    print(f"Comparing model performance: {original_results_file} vs {new_results_file}")
    
    # Load results
    original_results = pd.read_csv(original_results_file)
    new_results = pd.read_csv(new_results_file)
    
    # Calculate metrics
    original_mae = original_results['abs_error'].mean()
    original_mse = np.mean(np.square(original_results['error']))
    original_rmse = np.sqrt(original_mse)
    
    new_mae = new_results['abs_error'].mean()
    new_mse = np.mean(np.square(new_results['error']))
    new_rmse = np.sqrt(new_mse)
    
    # Print comparison
    print("\nModel Comparison:")
    print(f"{'Metric':<10} {'Original':<10} {'New':<10} {'Difference':<10} {'Improvement':<10}")
    print("-" * 60)
    
    mae_diff = original_mae - new_mae
    mae_improvement = (mae_diff / original_mae) * 100 if original_mae > 0 else 0
    print(f"{'MAE':<10} {original_mae:<10.4f} {new_mae:<10.4f} {mae_diff:<10.4f} {mae_improvement:<10.2f}%")
    
    rmse_diff = original_rmse - new_rmse
    rmse_improvement = (rmse_diff / original_rmse) * 100 if original_rmse > 0 else 0
    print(f"{'RMSE':<10} {original_rmse:<10.4f} {new_rmse:<10.4f} {rmse_diff:<10.4f} {rmse_improvement:<10.2f}%")
    
    size_diff = len(new_results) - len(original_results)
    size_change = (size_diff / len(original_results)) * 100 if len(original_results) > 0 else 0
    print(f"{'Size':<10} {len(original_results):<10} {len(new_results):<10} {size_diff:<10} {size_change:<10.2f}%")
    
    # Create comparison plots
    plt.figure(figsize=(12, 6))
    
    # Plot MAE by municipality for both models
    kommuner_original = original_results.groupby('kommunenummer')['abs_error'].mean()
    kommuner_new = new_results.groupby('kommunenummer')['abs_error'].mean()
    
    # Get intersection of municipalities
    common_kommuner = list(set(kommuner_original.index) & set(kommuner_new.index))
    common_kommuner.sort()
    
    if common_kommuner:
        plt.figure(figsize=(14, 6))
        x = np.arange(len(common_kommuner))
        width = 0.35
        
        plt.bar(x - width/2, [kommuner_original.loc[k] for k in common_kommuner], width, label='Original Model')
        plt.bar(x + width/2, [kommuner_new.loc[k] for k in common_kommuner], width, label='New Model')
        
        plt.xticks(x, [str(k) for k in common_kommuner], rotation=90)
        plt.xlabel('Municipality')
        plt.ylabel('Mean Absolute Error')
        plt.title('MAE by Municipality - Model Comparison')
        plt.legend()
        plt.tight_layout()
        plt.savefig('model_comparison_by_kommune.png')
        print(f"Saved model comparison plot to 'model_comparison_by_kommune.png'")
    
    # Create summary text
    if mae_improvement > 0:
        print("\nThe new model shows improved accuracy!")
    else:
        print("\nThe new model has different performance characteristics.")