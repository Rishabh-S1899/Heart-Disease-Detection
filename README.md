# Heart Disease Detection

An end-to-end project that preprocesses a health dataset, trains a classifier to predict heart disease, and provides utilities for inference and visualization.

## Overview

This repository implements a machine learning pipeline for detecting heart disease from survey / clinical features. The notebook `heart_condition2.ipynb` contains the exploratory analysis and experiments. The Python modules in the repo provide a reusable pipeline for preprocessing, training, saving/loading model artefacts, and plotting PCA visualizations.

Key features:
- Data loading and preprocessing that handles categorical encoding, binary conversions, and scaling.
- Resampling utilities to handle class imbalance (uses imbalanced-learn).
- Model training using XGBoost and helper functions to evaluate and save artefacts.
- Scriptable entrypoint `main.py` to run the end-to-end pipeline and save PCA plots.
- Inference helper in `inference.py` to load saved artefacts and predict new examples.

## Repository structure

- `heart_condition2.ipynb` - original exploratory notebook (includes PCA plots and confusion matrices).
- `config.py` - paths and configuration constants (update `DATA_PATH` to point to your CSV).
- `data_processing.py` - data loading and preprocessing functions.
- `model.py` - training, evaluation and artefact save helpers.
- `plotting.py` - PCA plotting utilities (2D and 3D). Plots are saved to files when run.
- `inference.py` - helpers to load artefacts and run single-example predictions.
- `main.py` - minimal end-to-end script: preprocess -> resample -> train -> evaluate -> save artefacts -> save PCA plots.

## Note about images from the notebook

The original notebook generated several PCA scatter plots and confusion matrix figures. The notebook file in this repository does not contain embedded image outputs. To reproduce those images locally and include them in documentation, run the `main.py` script which will create two image files:

- `pca_2d.png`
- `pca_3d.png`

You can then embed those images into this README (or publish them alongside the repo). If you already have PNGs exported from the notebook, place them in the repo root and reference them using standard Markdown image syntax:

![2D PCA](pca_2d.png)


## Quickstart (Windows PowerShell)

1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Make dataset available locally and update `config.py`:
- The default `DATA_PATH` is pointing to a Google Drive path used in the notebook. Replace `config.DATA_PATH` with the path to your `CVD_cleaned.csv`, for example `C:\data\CVD_cleaned.csv`.

3. Run the full training pipeline and generate plots:

```powershell
python main.py
```

This will:
- Preprocess the CSV.
- Resample and scale features.
- Train an XGBoost model and print evaluation metrics.
- Save artefacts: `xgboost_model.pkl`, `scaler.pkl`, and `training_columns.txt`.
- Save PCA plots: `pca_2d.png`, `pca_3d.png`.

4. Run an example single prediction (edit `inference.py` sample data or import functions):

```powershell
python -c "from inference import load_artefacts,predict; m,s=load_artefacts(); print(predict(m,s,{'Sex':1,'Age_Category':60,'BMI':25.3,'Smoking_History':1,'Exercise':1,'Alcohol_Consumption':5,'Fruit_Consumption':30,'Green_Vegetables_Consumption':12}))"
```

(Or run `python inference.py` after artefacts are generated — it includes a sample usage block.)

## Dependencies

See `requirements.txt` for a pinned list. Install with `pip install -r requirements.txt`.

## Tips and next steps

- If your dataset is large, consider running training on a machine with more RAM / CPU and enabling `n_jobs` where supported.
- Add unit tests to verify preprocessing invariants (column names, missing values handling).
- Add a small script to export notebook images into `docs/` and auto-embed them into the README using a simple script.
- Consider Dockerizing the environment for reproducible runs.

## License

Add your license here (e.g., MIT).

---

If you'd like, I can:
- Run the pipeline here to generate the two PCA images (I can't do that unless you provide the dataset in the repo or update `config.DATA_PATH` to a local path accessible in this workspace).
- Embed the generated images into the README after you add them, or provide a PR-ready README that already references images you upload.

