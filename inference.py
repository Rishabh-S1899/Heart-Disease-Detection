# inference.py

import pickle
import pandas as pd
import numpy as np
import config

def load_artefacts(model_path, scaler_path):
    """Loads the saved model and scaler."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

def predict(model, scaler, new_data):
    """
    Preprocesses new data, scales it, and makes a prediction.
    'new_data' should be a dictionary.
    """
    # Convert new_data to a DataFrame
    df = pd.DataFrame([new_data])

    # Apply the same preprocessing steps as in training
    # Note: This is a simplified version. For a robust pipeline,
    # the exact same one-hot encoding columns must be present.
    df = pd.get_dummies(df)

    # Reindex the dataframe to match the training data columns
    # This handles missing columns from one-hot encoding
    with open('training_columns.txt', 'r') as f:
        training_cols = [line.strip() for line in f]
    
    df = df.reindex(columns=training_cols, fill_value=0)

    # Scale the numerical features
    df[config.SCALING_COLS] = scaler.transform(df[config.SCALING_COLS])

    # Make prediction
    prediction = model.predict(df)
    prediction_proba = model.predict_proba(df)
    
    return prediction[0], prediction_proba[0]


if __name__ == '__main__':
    # Load the model and scaler
    model, scaler = load_artefacts(config.MODEL_PATH, config.SCALER_PATH)

    # Example new data point (as a dictionary)
    sample_data = {
        'Sex': 1, # 1 for Male, 0 for Female
        'Age_Category': 60,
        'BMI': 25.3,
        'Smoking_History': 1, # 1 for Yes, 0 for No
        'Exercise': 1, # 1 for Yes, 0 for No
        'Alcohol_Consumption': 5,
        'Fruit_Consumption': 30,
        'Green_Vegetables_Consumption': 12,
        'General_Health_Excellent': 1, # Example one-hot encoded
        'Checkup_Within the past year': 1 # Example one-hot encoded
        # Add other one-hot encoded columns with 0 if not applicable
    }

    # Get prediction
    prediction, probability = predict(model, scaler, sample_data)

    print("--- Inference Result ---")
    if prediction == 1:
        print("Prediction: Positive for Heart Disease")
    else:
        print("Prediction: Negative for Heart Disease")
        
    print(f"Confidence (Probability): {probability[prediction]:.2%}")