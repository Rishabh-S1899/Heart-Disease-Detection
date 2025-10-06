from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import config
# model.py

# ... (imports remain the same) ...
import config

def train_model(X, y):
    """
    Splits data, trains the XGBoost classifier, and returns the trained
    model and test set.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = XGBClassifier()
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    # ... (function remains the same) ...
    pass

def save_artefacts(model, scaler, columns):
    """Saves the trained model, scaler, and training columns."""
    with open(config.MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(config.SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    with open(config.TRAINING_COLUMNS_PATH, 'w') as f:
        for col in columns:
            f.write(f"{col}\n")