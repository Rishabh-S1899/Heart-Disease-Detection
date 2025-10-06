from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import config

"""model.py

Contains training, evaluation and artifact save/load helpers for the
Heart Disease project.
"""

def train_model(X, y, test_size: float = 0.3, random_state: int = 42):
    """Train an XGBoost classifier and return trained model + test split.

    Returns:
        model: trained XGBClassifier
        X_test, y_test: test split for evaluation
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = XGBClassifier()
    model.fit(X_train, y_train)

    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    """Evaluate model on provided test set and print common metrics."""
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds)

    print("Model evaluation")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)

    return {
        'accuracy': acc,
        'confusion_matrix': cm,
        'report': report,
    }


def save_artefacts(model, scaler, columns):
    """Saves model, scaler and training column list to locations from config."""
    with open(config.MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(config.SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    with open(config.TRAINING_COLUMNS_PATH, 'w') as f:
        for col in columns:
            f.write(f"{col}\n")