import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
try:
    import matplotlib.pyplot as plt
except Exception as _e:
    plt = None
    # we'll fallback to plotly for plotting in this environment
import plotly.express as px
from typing import Optional, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- 1. Configuration de la Page ---
st.set_page_config(page_title="Prévision NDVI - Adam D.", layout="wide")
st.title("🚀 Tableau de Bord de Prévision du NDVI")
st.write("Projet de stage : Prévision de la santé de la végétation (NDVI) à l'aide d'un réseau de neurones MLP, basé sur la sécheresse (SPI).")
st.write("---")

# --- 2. Fonctions de Chargement (Mises en cache) ---
# @st.cache_resource garantit que nous ne chargeons le modèle qu'une seule fois
@st.cache_resource
def load_assets():
    print("Chargement des actifs (modèle et scalers)...")
    # S'assurer que les fichiers existent avant de les charger
    # Assets are optional for the dashboard; return Nones if missing
    model = None
    scaler_X = None
    scaler_Y = None

    model_path = 'ndvi_model.keras'
    scaler_x_path = 'scaler_x.pkl'
    scaler_y_path = 'scaler_y.pkl'

    if os.path.exists(model_path) and os.path.exists(scaler_x_path) and os.path.exists(scaler_y_path):
        try:
            # import tensorflow lazily to avoid ModuleNotFoundError at import time
            from tensorflow.keras.models import load_model as _load_model
        except Exception as e:  # ModuleNotFoundError or others
            st.warning(f"TensorFlow not available in this environment: {e}\nPrediction functionality will be disabled until TensorFlow is installed.")
            return None, None, None

        try:
            model = _load_model(model_path)
            with open(scaler_x_path, 'rb') as f:
                scaler_X = pickle.load(f)
            with open(scaler_y_path, 'rb') as f:
                scaler_Y = pickle.load(f)
            print("Actifs chargés.")
        except Exception as e:
            st.error(f"Erreur lors du chargement des actifs: {e}")
            return None, None, None

    else:
        # not fatal: show info to user
        missing = []
        if not os.path.exists(model_path):
            missing.append(model_path)
        if not os.path.exists(scaler_x_path):
            missing.append(scaler_x_path)
        if not os.path.exists(scaler_y_path):
            missing.append(scaler_y_path)
        if missing:
            st.info(f"Les fichiers suivants sont absents et la fonction de prédiction sera désactivée: {missing}")

    return model, scaler_X, scaler_Y

@st.cache_data
def load_data():
    print("Chargement des données (master_dataset)...")
    if not os.path.exists("master_dataset_bmk.csv"):
        st.error("ERREUR : Fichier 'master_dataset_bmk.csv' introuvable. Veuillez ré-exécuter la Leçon 9 (Fusion).")
        return None

    df_full = pd.read_csv("master_dataset_bmk.csv", index_col='date', parse_dates=True)
    print("Données chargées.")
    return df_full

# --- 3. Logique de Prévision (Notre Leçon 14) ---
def get_prediction(df_full, model, scaler_X, scaler_Y):
    print("Génération de la prévision...")
    # Recréer les "features"
    df_features = pd.DataFrame(index=df_full.index)
    # Support multiple possible column names (fallbacks)
    def find_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    ndvi_col = find_col(df_full, ['ndvi_max_monthly', 'ndvi_final', 'ndvi'])
    spi3_col = find_col(df_full, ['spi_3', 'spi3'])
    spi6_col = find_col(df_full, ['spi_6', 'spi6'])

    if ndvi_col is None:
        raise KeyError(f"Aucune colonne NDVI trouvée dans les données. Colonnes disponibles: {list(df_full.columns)}")

    df_features['X_ndvi_lag1'] = df_full[ndvi_col].shift(1)
    if spi3_col is not None:
        df_features['X_spi3_lag1'] = df_full[spi3_col].shift(1)
        df_features['X_spi3_lag2'] = df_full[spi3_col].shift(2)
    else:
        # fill with NaN columns to keep feature set shape
        df_features['X_spi3_lag1'] = np.nan
        df_features['X_spi3_lag2'] = np.nan

    if spi6_col is not None:
        df_features['X_spi6_lag3'] = df_full[spi6_col].shift(3)
    else:
        df_features['X_spi6_lag3'] = np.nan
    
    # Isoler la dernière ligne disponible (avec protection si dropna() est vide)
    df_nonnull = df_features.dropna()
    if not df_nonnull.empty:
        last_known_data = df_nonnull.iloc[[-1]]
    else:
        # fallback: forward-fill then take last row
        last_known_data = df_features.ffill().iloc[[-1]]
        if last_known_data.isna().any(axis=None):
            raise ValueError("Impossible de construire une ligne de features complète pour la prédiction (trop de NaN). Vérifiez les données d'entrée.")

    # Align columns with scaler if scaler exposes expected feature names
    try:
        if hasattr(scaler_X, 'feature_names_in_'):
            expected = list(scaler_X.feature_names_in_)
            # ensure we're working on a copy to avoid SettingWithCopyWarning
            last_known_data = last_known_data.copy()
            # add any missing expected columns with NaN and reorder
            for col in expected:
                if col not in last_known_data.columns:
                    last_known_data.loc[:, col] = np.nan
            last_known_data = last_known_data[expected]
    except Exception:
        # if alignment fails, continue and let transform raise a clear error
        pass

    # Normaliser
    last_known_data_scaled = scaler_X.transform(last_known_data)
    
    # Prédire
    prediction_scaled = model.predict(last_known_data_scaled, verbose=0)
    
    # Dé-normaliser
    prediction_unscaled = scaler_Y.inverse_transform(prediction_scaled)
    predicted_ndvi = prediction_unscaled[0][0]
    
    return last_known_data, predicted_ndvi, ndvi_col


def _plot_line_with_fallback(index, series, title=None, y_label=None, color=None):
    """Plot a single time series using matplotlib if available, else plotly."""
    if plt is not None:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(index, series, label=series.name if hasattr(series, 'name') else None, color=color)
        if title:
            ax.set_title(title)
        if y_label:
            ax.set_ylabel(y_label)
        ax.set_xlabel('Date')
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)
        # small caption area for context under the plot
        if title:
            st.caption(f"{title} — visualisation temporelle. Interprétez les tendances saisonnières et anomalies éventuelles.")
    else:
        dfp = pd.DataFrame({series.name if hasattr(series, 'name') else 'y': series.values}, index=index)
        fig = px.line(dfp, x=dfp.index, y=dfp.columns[0], title=title, labels={dfp.columns[0]: y_label or ''})
        st.plotly_chart(fig, use_container_width=True)
        if title:
            st.caption(f"{title} — visualisation temporelle. Interprétez les tendances saisonnières et anomalies éventuelles.")


def compute_model_diagnostics(df_full: pd.DataFrame, model, scaler_X, scaler_Y, ndvi_col: str):
    """Compute in-sample predictions and basic diagnostics.
    Returns: dict(metrics), pd.Series(predicted), pd.Series(actual)
    """
    # Build features the same way as get_prediction but for all rows
    df_features = pd.DataFrame(index=df_full.index)
    df_features['X_ndvi_lag1'] = df_full[ndvi_col].shift(1)
    # prefer spi_3 names
    spi3 = 'spi_3' if 'spi_3' in df_full.columns else ('spi3' if 'spi3' in df_full.columns else None)
    spi6 = 'spi_6' if 'spi_6' in df_full.columns else ('spi6' if 'spi6' in df_full.columns else None)
    if spi3 is not None:
        df_features['X_spi3_lag1'] = df_full[spi3].shift(1)
        df_features['X_spi3_lag2'] = df_full[spi3].shift(2)
    else:
        df_features['X_spi3_lag1'] = np.nan
        df_features['X_spi3_lag2'] = np.nan
    if spi6 is not None:
        df_features['X_spi6_lag3'] = df_full[spi6].shift(3)
    else:
        df_features['X_spi6_lag3'] = np.nan

    # align to scaler expected features if available
    try:
        if hasattr(scaler_X, 'feature_names_in_'):
            expected = list(scaler_X.feature_names_in_)
            for col in expected:
                if col not in df_features.columns:
                    df_features.loc[:, col] = np.nan
            df_features = df_features[expected]
    except Exception:
        pass

    # drop rows with NaNs in features or target
    df_comb = pd.concat([df_features, df_full[[ndvi_col]]], axis=1)
    df_comb = df_comb.dropna()
    if df_comb.empty:
        return None, None, None

    X = df_comb.iloc[:, :-1]
    y = df_comb[ndvi_col]

    X_scaled = scaler_X.transform(X)
    y_pred_scaled = model.predict(X_scaled, verbose=0)
    y_pred = scaler_Y.inverse_transform(y_pred_scaled).reshape(-1)
    y_actual = y.values.reshape(-1)

    mse = mean_squared_error(y_actual, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_actual, y_pred))
    r2 = float(r2_score(y_actual, y_pred))
    # MAPE, guard divide by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = float(np.nanmean(np.abs((y_actual - y_pred) / np.where(np.abs(y_actual) < 1e-9, np.nan, y_actual))) * 100)

    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape_percent': mape,
        'n_samples': len(y_actual)
    }

    predicted_series = pd.Series(y_pred, index=df_comb.index, name='predicted_ndvi')
    actual_series = pd.Series(y_actual, index=df_comb.index, name='actual_ndvi')

    return metrics, predicted_series, actual_series


# --- 4. Exécution de l'Application (Dashboard) ---
try:
    # Charger les données et le modèle
    model, scaler_X, scaler_Y = load_assets()
    df_full = load_data()

    if df_full is None:
        st.stop()

    # détecter colonne NDVI
    ndvi_col = None
    for cand in ['ndvi_max_monthly', 'ndvi_final', 'ndvi']:
        if cand in df_full.columns:
            ndvi_col = cand
            break

    if ndvi_col is None:
        st.error(f"Aucune colonne NDVI trouvée. Colonnes disponibles: {list(df_full.columns)}")
        st.stop()

    # --- Sidebar: filtres et options ---
    st.sidebar.header("Filtres")
    # Option: prefer using local files present in the workspace (default)
    use_local_files = st.sidebar.checkbox("Utiliser les fichiers locaux (aucun upload)", value=True, help="Si coché, l'application utilisera les fichiers présents dans le répertoire du projet (ndvi_model.keras, scaler_x.pkl, scaler_y.pkl, history.csv/json). Décochez pour téléverser vos propres fichiers.")
    st.sidebar.markdown("---")
    # Upload widgets are shown only when the user opts out of local files
    uploaded_model = uploaded_scaler_x = uploaded_scaler_y = uploaded_history = None
    if not use_local_files:
        st.sidebar.subheader("Charger un modèle (optionnel)")
        uploaded_model = st.sidebar.file_uploader("Modèle Keras (.keras/.h5)", type=['keras','h5','hdf5'], help='Téléversez un fichier .keras ou .h5 entraîné si vous ne souhaitez pas placer les fichiers sur le disque.')
        uploaded_scaler_x = st.sidebar.file_uploader("Scaler X (pickle)", type=['pkl'], help='Scaler des features (scaler_x.pkl)')
        uploaded_scaler_y = st.sidebar.file_uploader("Scaler Y (pickle)", type=['pkl'], help='Scaler de la cible (scaler_y.pkl)')

        # Training history uploader (CSV with columns 'loss' and optionally 'val_loss' OR Keras history JSON)
        uploaded_history = st.sidebar.file_uploader("Historique d'entraînement (CSV/JSON)", type=['csv','json'], help='Fichier history.history() exporté ou CSV avec colonnes loss, val_loss')
    # GeoJSON visualization toggle
    st.sidebar.markdown('---')
    show_geo = st.sidebar.checkbox('Afficher la zone GeoJSON (Béni Mellal-Khénifra)', value=True)

    def _plot_training_history(uploaded_file_or_path):
        """Accept either a Streamlit UploadedFile or a filesystem path to CSV/JSON history."""
        try:
            import json
            # If a path string is provided, read from disk
            if isinstance(uploaded_file_or_path, str):
                path = uploaded_file_or_path
                if path.lower().endswith('.csv'):
                    dfh = pd.read_csv(path)
                else:
                    raw = open(path, 'r', encoding='utf-8').read()
                    j = json.loads(raw)
                    if 'history' in j and isinstance(j['history'], dict):
                        h = j['history']
                    else:
                        h = j
                    dfh = pd.DataFrame(h)
            else:
                # assume Streamlit uploaded file
                if uploaded_file_or_path.name.lower().endswith('.csv'):
                    dfh = pd.read_csv(uploaded_file_or_path)
                else:
                    raw = uploaded_file_or_path.getvalue().decode('utf-8')
                    j = json.loads(raw)
                    if 'history' in j and isinstance(j['history'], dict):
                        h = j['history']
                    else:
                        h = j
                    dfh = pd.DataFrame(h)

            # prefer columns named loss and val_loss
            cols = [c for c in ['loss','val_loss'] if c in dfh.columns]
            if not cols:
                st.info('Le fichier d\'historique ne contient pas loss/val_loss attendus.')
                return

            st.subheader("Historique d'entraînement")
            # use plotly for training history for interactivity
            fig = px.line(dfh[cols].reset_index().rename(columns={'index':'epoch'}), x='epoch', y=cols, labels={'value':'loss','variable':'courbe'})
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Impossible de lire l'historique d'entraînement : {e}")

    def _load_uploaded_assets(uploaded_model, uploaded_scaler_x, uploaded_scaler_y):
        """If user uploaded assets, save them to disk temporarily and attempt to load them.
        Returns (model, scaler_X, scaler_Y) or (None, None, None) on failure/absence.
        """
        if uploaded_model is None and uploaded_scaler_x is None and uploaded_scaler_y is None:
            return None, None, None

        tmp_model_path = None
        tmp_x = None
        tmp_y = None
        try:
            if uploaded_model is not None:
                tmp_model_path = os.path.join(os.getcwd(), 'uploaded_ndvi_model.keras')
                with open(tmp_model_path, 'wb') as f:
                    f.write(uploaded_model.getbuffer())
            if uploaded_scaler_x is not None:
                tmp_x = os.path.join(os.getcwd(), 'uploaded_scaler_x.pkl')
                with open(tmp_x, 'wb') as f:
                    f.write(uploaded_scaler_x.getbuffer())
            if uploaded_scaler_y is not None:
                tmp_y = os.path.join(os.getcwd(), 'uploaded_scaler_y.pkl')
                with open(tmp_y, 'wb') as f:
                    f.write(uploaded_scaler_y.getbuffer())

            # Try to load if tensorflow is available
            try:
                from tensorflow.keras.models import load_model as _load_model
            except Exception as e:
                st.warning(f"TensorFlow non disponible; impossible de charger le modèle téléversé: {e}")
                return None, None, None

            model_l = None
            scaler_x_l = None
            scaler_y_l = None
            if tmp_model_path is not None:
                model_l = _load_model(tmp_model_path)
            if tmp_x is not None:
                with open(tmp_x, 'rb') as f:
                    scaler_x_l = pickle.load(f)
            if tmp_y is not None:
                with open(tmp_y, 'rb') as f:
                    scaler_y_l = pickle.load(f)

            return model_l, scaler_x_l, scaler_y_l
        except Exception as e:
            st.error(f"Échec du chargement des fichiers téléversés: {e}")
            return None, None, None
    min_date = df_full.index.min().date()
    max_date = df_full.index.max().date()
    start_date, end_date = st.sidebar.date_input("Période", value=[min_date, max_date], min_value=min_date, max_value=max_date)
    if isinstance(start_date, list) or isinstance(start_date, tuple):
        # streamlit may return a list if user picks range differently
        start_date, end_date = start_date[0], start_date[-1]

    # Ensure proper ordering
    if start_date > end_date:
        start_date, end_date = end_date, start_date

    # subset data for display
    df_view = df_full.loc[str(start_date):str(end_date)].copy()

    # If user uploaded model/scalers, try to load and override the disk assets
    uploaded_override = None
    if uploaded_model is not None or uploaded_scaler_x is not None or uploaded_scaler_y is not None:
        uploaded_override = _load_uploaded_assets(uploaded_model, uploaded_scaler_x, uploaded_scaler_y)
        if uploaded_override is not None:
            m_u, sx_u, sy_u = uploaded_override
            if m_u is not None:
                model = m_u
                st.success('Modèle téléversé chargé et utilisé.')
            if sx_u is not None:
                scaler_X = sx_u
                st.success('Scaler X téléversé chargé et utilisé.')
            if sy_u is not None:
                scaler_Y = sy_u
                st.success('Scaler Y téléversé chargé et utilisé.')

        # (previously the training-history helper was here; it's defined above so we can use local files too)
    # After uploaded assets handling: decide how to show training history
    if not use_local_files:
        if uploaded_history is not None:
            _plot_training_history(uploaded_history)
    else:
        # User chose to use local files. Try to find a training history on disk and plot it.
        possible_history_files = ['history.csv','training_history.csv','history.json','training_history.json']
        found = None
        for p in possible_history_files:
            if os.path.exists(p):
                found = p
                break
        if found is not None:
            _plot_training_history(found)

    # Top summary metrics
    st.header("📊 Résumé")
    c1, c2, c3 = st.columns(3)
    last_idx = df_view.index.max()
    last_ndvi = df_view[ndvi_col].loc[last_idx]
    c1.metric("Dernière valeur NDVI", f"{last_ndvi:.4f}", delta=None)

    # moyenne sur 12 mois si possible
    try:
        mean_12 = df_view[ndvi_col].iloc[-12:].mean()
        c2.metric("Moyenne 12 mois (NDVI)", f"{mean_12:.4f}")
    except Exception:
        c2.metric("Moyenne 12 mois (NDVI)", "N/A")

    if 'precip_median' in df_view.columns:
        c3.metric("Précipitation (médiane) dernière", f"{df_view['precip_median'].loc[last_idx]:.2f}")
    else:
        c3.metric("Précipitation (médiane) dernière", "N/A")

    # --- Time series NDVI ---
    st.subheader("Série Temporelle NDVI")
    _plot_line_with_fallback(df_view.index, df_view[ndvi_col], title='Série Temporelle NDVI', y_label='NDVI', color='tab:green')

    # --- SPI plots if present ---
    spi_cols = [c for c in ['spi_3', 'spi3', 'spi_6', 'spi6', 'spi_12'] if c in df_view.columns]
    if spi_cols:
        st.subheader("Indices SPI")
        # If matplotlib exists, plot stacked subplots; else plot each series separately with Plotly
        if plt is not None:
            fig2, axes = plt.subplots(len(spi_cols), 1, figsize=(14, 3 * len(spi_cols)), sharex=True)
            if len(spi_cols) == 1:
                axes = [axes]
            for ax, col in zip(axes, spi_cols):
                ax.plot(df_view.index, df_view[col], label=col, color='tab:blue')
                ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
                ax.set_ylabel(col)
                ax.legend()
            st.pyplot(fig2)
            st.caption('Indices SPI — Indicateurs standards de la sécheresse. Les valeurs négatives indiquent conditions plus sèches que la normale, positives indiquent conditions plus humides.')
        else:
            for col in spi_cols:
                _plot_line_with_fallback(df_view.index, df_view[col], title=f'SPI: {col}', y_label=col, color='tab:blue')

    # --- Precipitation ---
    if 'precip_median' in df_view.columns:
        st.subheader('Précipitation (médiane)')
    _plot_line_with_fallback(df_view.index, df_view['precip_median'], title='Précipitation (médiane)', y_label='Précipitation (mm)', color='tab:purple')
    st.caption('Précipitation médiane — valeurs mensuelles agrégées. Utilisez ceci pour relier les anomalies pluviométriques aux variations de NDVI.')

    # --- Prediction panel ---
    if model is not None and scaler_X is not None and scaler_Y is not None:
        st.header('🔮 Prédiction (modèle chargé)')
        try:
            last_known_data, predicted_ndvi, _ = get_prediction(df_full, model, scaler_X, scaler_Y)
            last_date = last_known_data.index[0]
            future_date = last_date + pd.DateOffset(months=1)

            col_a, col_b = st.columns(2)
            value_last = float(df_full[ndvi_col].loc[last_date])
            col_a.metric(label=f"Dernière Donnée — {last_date.strftime('%Y-%m')}", value=f"{value_last:.4f}")
            delta = float(predicted_ndvi - value_last)
            col_b.metric(label=f"Prévision — {future_date.strftime('%Y-%m')}", value=f"{predicted_ndvi:.4f}", delta=delta)

            # Add forecast point to NDVI plot (recent history)
            st.subheader('Historique + Prévision (36 mois)')
            recent_history = df_full[ndvi_col].iloc[-36:].copy()
            prediction_point = pd.Series([predicted_ndvi], index=[future_date])
            if plt is not None:
                fig4, ax4 = plt.subplots(figsize=(14, 5))
                ax4.plot(recent_history.index, recent_history, label='Historique', color='tab:green')
                ax4.plot(prediction_point.index, prediction_point, 'r*', markersize=14, label='Prévision')
                ax4.set_title('NDVI : Historique et Prévision')
                ax4.set_ylabel('NDVI')
                ax4.grid(True)
                ax4.legend()
                st.pyplot(fig4)
                st.caption('Historique et prévision — point rouge représente la prévision du mois suivant basée sur le modèle chargé. Vérifiez l\'incertitude si le modèle a peu d\'échantillons.')
            else:
                # Use plotly to combine history and point
                df_hist = pd.DataFrame({ 'NDVI': recent_history.values }, index=recent_history.index)
                fig_pl = px.line(df_hist, x=df_hist.index, y='NDVI', title='NDVI : Historique et Prévision')
                fig_pl.add_scatter(x=prediction_point.index, y=prediction_point.values, mode='markers', marker=dict(size=12, color='red', symbol='star'), name='Prévision')
                st.plotly_chart(fig_pl, use_container_width=True)
                st.caption('Historique et prévision — point rouge représente la prévision du mois suivant basée sur le modèle chargé. Vérifiez l\'incertitude si le modèle a peu d\'échantillons.')

            # --- Diagnostics: in-sample predictions and metrics ---
            try:
                diag = compute_model_diagnostics(df_full, model, scaler_X, scaler_Y, ndvi_col)
                if diag is not None and diag[0] is not None:
                    metrics, pred_series, actual_series = diag
                    with st.expander('🔍 Diagnostics du Modèle & Verdict (in-sample)'):
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric('RMSE', f"{metrics['rmse']:.4f}")
                        m2.metric('MAE', f"{metrics['mae']:.4f}")
                        m3.metric('R²', f"{metrics['r2']:.3f}")
                        m4.metric('MAPE (%)', f"{metrics['mape_percent']:.2f}")

                        # brief interpretation
                        st.markdown('**Interprétation rapide :**')
                        interp = []
                        # RMSE benchmarks for NDVI (range ~0..1): thresholds are heuristic
                        if metrics['rmse'] < 0.02:
                            interp.append('- RMSE faible : bonne précision du modèle sur les données historiques.')
                        elif metrics['rmse'] < 0.05:
                            interp.append('- RMSE modéré : performance acceptable, surveiller erreurs saisonnières.')
                        else:
                            interp.append('- RMSE élevé : le modèle a des erreurs significatives; examiner les features et la pré-traitement.')
                        if metrics['r2'] > 0.7:
                            interp.append('- R² élevé : le modèle explique bien la variance observée.')
                        elif metrics['r2'] > 0.4:
                            interp.append('- R² moyen : le modèle capture partiellement la tendance.')
                        else:
                            interp.append('- R² faible : le modèle n\'explique pas bien la variance; reconsidérer la structure ou les données.')
                        st.write('\n'.join(interp))

                        # Plots: actual vs predicted time series
                        st.subheader('Série : Observé vs Prédit (In-sample)')
                        df_plot = pd.concat([actual_series, pred_series], axis=1)
                        df_plot.columns = ['observed', 'predicted']
                        if plt is not None:
                            fig_ts, ax_ts = plt.subplots(figsize=(14,5))
                            ax_ts.plot(df_plot.index, df_plot['observed'], label='Observé', color='tab:green')
                            ax_ts.plot(df_plot.index, df_plot['predicted'], label='Prévu', color='tab:red', alpha=0.8)
                            ax_ts.set_title('Observé vs Prévu (In-sample)')
                            ax_ts.legend()
                            ax_ts.grid(True)
                            st.pyplot(fig_ts)
                            st.caption('Observé vs Prévu — série temporelle in-sample. Un fort décalage saisonnier peut indiquer que le modèle ne capture pas la saisonnalité.')
                        else:
                            fig_px = px.line(df_plot.reset_index(), x='index', y=['observed','predicted'], labels={'index':'date'})
                            st.plotly_chart(fig_px, use_container_width=True)
                            st.caption('Observé vs Prévu — série temporelle in-sample. Un fort décalage saisonnier peut indiquer que le modèle ne capture pas la saisonnalité.')

                        # Scatter plot and residuals
                        st.subheader('Scatter : Observé vs Prévu')
                        df_sc = df_plot.dropna().reset_index()
                        fig_sc = px.scatter(df_sc, x='observed', y='predicted', trendline='ols', height=400)
                        fig_sc.add_shape(type='line', x0=df_sc['observed'].min(), x1=df_sc['observed'].max(), y0=df_sc['observed'].min(), y1=df_sc['observed'].max(), line=dict(dash='dash'))
                        st.plotly_chart(fig_sc, use_container_width=True)
                        st.caption('Scatter Observé vs Prévu — la ligne en pointillé est la diagonale idéale; l\'écart montre les biais et la dispersion des prédictions.')

                        st.subheader('Distribution des résidus (observé - prédit)')
                        residuals = df_plot['observed'] - df_plot['predicted']
                        if plt is not None:
                            fig_r, ax_r = plt.subplots(figsize=(10,4))
                            ax_r.hist(residuals.dropna(), bins=40, color='grey', alpha=0.7)
                            ax_r.set_title('Histogramme des résidus')
                            st.pyplot(fig_r)
                            st.caption('Distribution des résidus — recherchez une distribution centrée sur zéro; longue queue indique erreurs extrêmes.')
                        else:
                            fig_hr = px.histogram(residuals.reset_index(), x=0, nbins=40, title='Histogramme des résidus')
                            st.plotly_chart(fig_hr, use_container_width=True)
                            st.caption('Distribution des résidus — recherchez une distribution centrée sur zéro; longue queue indique erreurs extrêmes.')

                        # allow download of diagnostics
                        df_diag_out = df_plot.reset_index()
                        csv_diag = df_diag_out.to_csv(index=False)
                        st.download_button('Télécharger diagnostics (CSV)', csv_diag, file_name='model_diagnostics.csv')
                else:
                    st.info('Pas assez de données propres pour calculer les diagnostics du modèle.')
            except Exception as e:
                st.warning(f"Impossible de calculer les diagnostics détaillés: {e}")

        except Exception as e:
            st.error(f"Erreur lors de la génération de la prédiction : {e}")
    else:
        st.info('Modèle non chargé : la fonctionnalité de prédiction est désactivée. Placez les fichiers ndvi_model.keras, scaler_x.pkl et scaler_y.pkl dans le répertoire du projet et relancez l\'application.')

    # (Data preview and small diagnostics panel removed per user request)

    # --- GeoJSON visualization ---
    if show_geo:
        st.subheader('Carte : Zone de Béni Mellal-Khénifra')
        geo_path = 'beni_mellal_khenifra.geojson'
        if os.path.exists(geo_path):
            try:
                import json
                try:
                    import geopandas as gpd
                    import pydeck as pdk
                    gdf = gpd.read_file(geo_path)
                    # convert to GeoJSON dict for pydeck
                    gj = json.loads(gdf.to_json())
                    # compute centroid for initial view
                    centroid = gdf.geometry.centroid.unary_union.centroid
                    initial_view = pdk.ViewState(latitude=centroid.y, longitude=centroid.x, zoom=8)
                    geo_layer = pdk.Layer(
                        "GeoJsonLayer",
                        data=gj,
                        stroked=True,
                        filled=True,
                        get_fill_color=[180, 0, 200, 80],
                        get_line_color=[0, 0, 0],
                        pickable=True
                    )
                    deck = pdk.Deck(layers=[geo_layer], initial_view_state=initial_view, tooltip={"text": "Béni Mellal-Khénifra region"})
                    st.pydeck_chart(deck)
                    st.caption('Limite de la région Béni Mellal-Khénifra affichée (source GeoJSON). Utilisez le zoom pour inspecter les sous-zones.')
                except Exception:
                    # fallback: show raw geojson or simple centroid in st.map
                    with open(geo_path, 'r', encoding='utf-8') as f:
                        gj = json.load(f)
                    # try to extract centroids for display
                    centroids = []
                    try:
                        for feat in gj.get('features', []):
                            geom = feat.get('geometry', {})
                            if geom.get('type') == 'Polygon' or geom.get('type') == 'MultiPolygon':
                                # compute bbox centroid approximately
                                coords = []
                                if geom.get('type') == 'Polygon':
                                    coords = geom.get('coordinates', [[]])[0]
                                else:
                                    # MultiPolygon: take first polygon
                                    coords = geom.get('coordinates', [[[]]])[0][0]
                                xs = [c[0] for c in coords if isinstance(c, (list, tuple))]
                                ys = [c[1] for c in coords if isinstance(c, (list, tuple))]
                                if xs and ys:
                                    centroids.append({'lat': sum(ys)/len(ys), 'lon': sum(xs)/len(xs)})
                    except Exception:
                        centroids = []
                    if centroids:
                        import pandas as _pd
                        st.map(_pd.DataFrame(centroids))
                        st.caption('Affichage approximatif : centroids des polygones (fallback). Pour une carte polygonale interactive, installez geopandas et pydeck.')
                    else:
                        st.write('GeoJSON (raw):')
                        st.json(gj)
                        st.info('Pour une carte interactive, installez geopandas et pydeck sur l\'environnement de déploiement.')
            except Exception as e:
                st.warning(f"Impossible d'afficher le GeoJSON : {e}")
        else:
            st.info(f"Fichier GeoJSON introuvable: {geo_path}")

except Exception as e:
    st.error(f"Une erreur inattendue est survenue : {e}")
    st.info("Veuillez vérifier les logs du terminal pour plus de détails.")