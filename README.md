# Norwegian Population Growth Prediction Model

This repository contains a machine learning model for predicting population growth in Norwegian geographic regions (grunnkretser) using TensorFlow. The model uses demographic time series data to forecast population changes.

## Project Structure

```
├── config.py              # Configuration parameters
├── data_pipeline.py       # Data loading and preprocessing
├── evaluation.py          # Model evaluation and visualization
├── features.py            # Feature engineering functions
├── main.py                # Main entry point script
├── model.py               # Model definition and training
├── prediction.py          # Functions for making predictions
├── utils.py               # Utility functions
├── README.md              # This file
```

## Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

Install requirements with:

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
```

For TensorFlow.js conversion:

```bash
pip install tensorflowjs
```

### Training the Model

To train the model with default settings:

```bash
python main.py train
```

Optional parameters:

- `--municipalities`: Number of municipalities to include (default: 25)
- `--min-sequences`: Minimum number of sequences to train on (default: 700)
- `--output-prefix`: Prefix for output files (e.g., "v2_")
- `--convert-to-js`: Convert the model to TensorFlow.js format

Example:

```bash
python main.py train --municipalities 30 --min-sequences 1000 --output-prefix "v2_" --convert-to-js
```

### Making Predictions

To make predictions with a trained model:

```bash
python main.py predict
```

Optional parameters:

- `--start-year`: Starting year for predictions (default: 2025)
- `--end-year`: End year for predictions (default: 2030)

Example:

```bash
python main.py predict --start-year 2025 --end-year 2035
```

## Model Details

The model uses an LSTM (Long Short-Term Memory) neural network architecture with the following features:

- Multiple LSTM layers with batch normalization and dropout for regularization
- Dense layers for final predictions
- Mean Squared Error (MSE) as the loss function
- Mean Absolute Error (MAE) as the evaluation metric

## Data Features

The model uses the following types of features:

- Age distribution ratios across different categories
- Gender ratios (male/female)
- Historical population totals
- Previous growth rates
- Population changes (lag features)

## Output

The training process produces the following outputs:

- Trained model file (`population_growth_model.h5`)
- Feature list (`model_features.csv`)
- Training history plot
- Prediction evaluation metrics and visualizations

The prediction process produces:

- Population predictions for each year in CSV format
- Summary of population predictions
- Visualization of predicted population trends

## Web Deployment

The model can be converted to TensorFlow.js format for web deployment. This creates:

- TensorFlow.js model files
- Feature list in JSON format
- Scaler parameters for proper input/output transformation

## License

This project is licensed under the MIT License - see the LICENSE file for details.