# SPI-NDVI Time Series Forecasting — Usage

This repository contains a Streamlit dashboard that visualizes NDVI time series and provides a one-month forecast using a saved Keras model (MLP) and scikit-learn scalers.

Required files (place them in the repository):

- `models/ndvi_model.keras` — saved Keras model for NDVI forecasting
- `models/scaler_x.pkl` — input scaler (pickle)
- `models/scaler_y.pkl` — output scaler (pickle)
- `data/processed/master_dataset_bmk.csv` — master dataset with `date` as index

Quick start (Windows PowerShell):

```powershell
cd "C:\Users\CE PC\Drought_monitoring"
# Activate the Conda environment that contains TensorFlow, plotly and pydeck
conda activate <your-conda-env>
streamlit run app.py
```

Notes:
- If TensorFlow is not available in the active environment, the prediction features will be disabled, but the visualizations will still work.
- For interactive polygon maps, install `geopandas` and `pydeck` in the environment.
- If you plan to deploy to Streamlit Cloud, add required packages to `requirements.txt`. TensorFlow may not be supported on some deployment platforms; consider exporting the model to ONNX and using `onnxruntime` if needed.

If you want, I can run the app locally now and report the startup logs and any runtime errors.
