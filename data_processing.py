# data_processing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
import config

def load_data(path):
    """Loads the dataset from a CSV file."""
    return pd.read_csv(path)

def preprocess_data(df):
    """Applies all preprocessing steps to the dataframe."""
    
    # Convert Age_Category to numerical
    temp = df.to_numpy()
    for i in range(len(temp)):
        if temp[i][10] != "80+":
            t = (int(temp[i][10][3:]) + int(temp[i][10][:2])) / 2
            temp[i][10] = t
        else:
            temp[i][10] = int(85)
    df = pd.DataFrame(temp, columns=df.columns)

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=config.CATEGORICAL_COLS)

    # Convert binary columns to 0/1
    for col in config.BINARY_COLS:
        df[col] = np.where(df[col] == "No", 0, 1)

    df["Sex"] = np.where(df["Sex"] == "Female", 0, 1)
    df["Smoking_History"] = np.where(df["Smoking_History"] == "No", 0, 1)

    # Drop unnecessary columns
    df.drop(config.DROP_COLS, axis=1, inplace=True)
    
    return df

def scale_features(df):
    """Scales the numerical features in the dataframe."""
    scaler = StandardScaler()
    df[config.SCALING_COLS] = scaler.fit_transform(df[config.SCALING_COLS])
    return df

def resample_data(X, y):
    """Resamples the data to handle class imbalance."""
    rus = RepeatedEditedNearestNeighbours(sampling_strategy='auto', max_iter=30)
    X_res, y_res = rus.fit_resample(X, y)
    return X_res, y_res