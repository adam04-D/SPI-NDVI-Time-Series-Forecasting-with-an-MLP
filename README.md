README — NDVI Forecasting Streamlit App
======================================

Description
-----------
This repository contains code and data for monthly NDVI forecasting using a neural network (MLP/LSTM in the notebook). It includes a Streamlit app (`app.py`) that loads a trained model and scalers and shows a one-month NDVI forecast.

Quick start (recommended: use the provided Conda environment)
------------------------------------------------------------
1. Activate the Conda environment used for development (or use the base Conda environment where TensorFlow and Streamlit are installed):

```powershell
conda activate base
```

2. Install Python requirements if needed:

```powershell
pip install -r requirements.txt
```

3. Run the Streamlit app from the repository root:

```powershell
streamlit run app.py
```

The app will be available at http://localhost:8502 by default.

Files of interest
-----------------
- `app.py` — Streamlit app; loads model & scalers and displays forecast + plot.
- `main.ipynb` — Notebook with data preparation, training and evaluation (contains the model training code).
- `master_dataset_bmk.csv` — Master dataset used by the app.

Model & scalers
----------------
This repo currently contains trained artefacts (`ndvi_model.keras`, `scaler_x.pkl`, `scaler_y.pkl`) which the app expects to find in the repository root. Note:
- Committing large binaries (models, pickles) to git is usually not recommended. If you'd prefer, I can remove them from the repository history and add them to `.gitignore`, and provide instructions or a script to download them from external storage instead.

If the app shows errors about missing model or scaler files, place the model folder (`ndvi_model.keras`) and two pickles (`scaler_x.pkl`, `scaler_y.pkl`) in the project root, or update `app.py` to point to their location.

Debugging
---------
If the Streamlit UI shows errors, check the running terminal for detailed logs. Common fixes:
- Ensure the right Python interpreter / Conda env is active (TensorFlow requires a supported Python version).
- Verify `master_dataset_bmk.csv` is present and has NDVI columns (`ndvi_final` or `ndvi_max_monthly`).

Want me to:
- Remove large model files from git history and add `.gitignore`? (This rewrites history; say "yes" if you want me to proceed.)
- Add a small UI page to let you upload model/scaler files from your machine? (I can implement that.)

License / Notes
----------------
Add licensing and authorship information here if needed.
