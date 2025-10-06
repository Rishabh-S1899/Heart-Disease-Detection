import pandas as pd
from sklearn.preprocessing import StandardScaler

import config
import data_processing
import model as model_utils
import plotting

def main():
    # Load and preprocess the data
    df = data_processing.load_data(config.DATA_PATH)
    df_processed = data_processing.preprocess_data(df)
    
    X = df_processed.drop(config.TARGET_COL, axis=1)
    y = df_processed[config.TARGET_COL]

    # Resample
    X_res, y_res = data_processing.resample_data(X, y)

    # Scale the features
    scaler = StandardScaler()
    X_res[config.SCALING_COLS] = scaler.fit_transform(X_res[config.SCALING_COLS])
    
    # Train the model
    model, X_test, y_test = model_utils.train_model(X_res, y_res)

    # Evaluate the model
    model_utils.evaluate_model(model, X_test, y_test)

    # Save model, scaler, and columns for inference
    model_utils.save_artefacts(model=model, scaler=scaler, columns=X_res.columns)
    print(f"Model, scaler, and column list saved.")

    # Generate and save plots
    plotting.plot_pca_2d(X_res, y_res)
    plotting.plot_pca_3d(X_res, y_res)

if __name__ == '__main__':
    main()