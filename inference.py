"""inference.py

Helpers to load saved training artefacts and perform inference on new
examples.
"""

import pickle
import pandas as pd
import numpy as np
import config
from pathlib import Path


def load_artefacts(model_path: str = None, scaler_path: str = None):
    """Load model and scaler from disk. Defaults to paths in config."""
    model_path = model_path or config.MODEL_PATH
    scaler_path = scaler_path or config.SCALER_PATH

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not Path(scaler_path).exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler


def _load_training_columns(path: str = None):
    path = path or config.TRAINING_COLUMNS_PATH
    cols_path = Path(path)
    if not cols_path.exists():
        raise FileNotFoundError(f"Training columns file not found: {path}")
    with open(cols_path, 'r') as f:
        cols = [line.strip() for line in f if line.strip()]
    return cols


def predict(model, scaler, new_data: dict):
    """Predict heart disease label and probability for a single example.

    Args:
        model: trained classifier with predict and predict_proba
        scaler: fitted scaler compatible with training features
        new_data: dict mapping feature -> value (may include one-hot keys)

    Returns:
        (prediction_label, probability_vector)
    """
    df = pd.DataFrame([new_data])
    df = pd.get_dummies(df)

    training_cols = _load_training_columns()
    # Ensure all training columns exist; fill missing with 0
    df = df.reindex(columns=training_cols, fill_value=0)

    # Scale numeric columns
    df[config.SCALING_COLS] = scaler.transform(df[config.SCALING_COLS])

    pred = model.predict(df)
    proba = model.predict_proba(df)
    return int(pred[0]), proba[0]


if __name__ == '__main__':
    # Example usage
    try:
        model, scaler = load_artefacts()
    except FileNotFoundError as e:
        print(e)
        print("Ensure you have trained and saved model artefacts before running inference.")
        raise

    sample_data = {
        'Sex': 1,
        'Age_Category': 60,
        'BMI': 25.3,
        'Smoking_History': 1,
        'Exercise': 1,
        'Alcohol_Consumption': 5,
        'Fruit_Consumption': 30,
        'Green_Vegetables_Consumption': 12,
        # Example one-hot encoded columns (set to 0/1)
    }

    pred, prob = predict(model, scaler, sample_data)
    print("Prediction:", pred)
    print("Probabilities:", prob)