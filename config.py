# config.py

# Path to the input CSV file
DATA_PATH = "/content/drive/MyDrive/CVD_cleaned.csv"

# Columns to be one-hot encoded
CATEGORICAL_COLS = ['General_Health', 'Checkup']

# Columns with binary 'Yes'/'No' values to be converted to 0/1
BINARY_COLS = [
    "Exercise", "Heart_Disease", "Skin_Cancer", "Other_Cancer",
    "Depression", "Diabetes", "Arthritis"
]

# Columns with sex information to be converted to 0/1
SEX_COLS = ["Sex"]

# Columns with smoking history to be converted to 0/1
SMOKING_COLS = ["Smoking_History"]

# Columns to be dropped from the dataframe
DROP_COLS = ['Height_(cm)', 'Weight_(kg)', 'FriedPotato_Consumption']

# Columns to be scaled
SCALING_COLS = [
    'Age_Category', 'BMI', 'Alcohol_Consumption',
    'Fruit_Consumption', 'Green_Vegetables_Consumption'
]

# Target variable
TARGET_COL = 'Heart_Disease'


# Artefact save paths
MODEL_PATH = "xgboost_model.pkl"
SCALER_PATH = "scaler.pkl"
TRAINING_COLUMNS_PATH = "training_columns.txt"