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
    else:
        dfp = pd.DataFrame({series.name if hasattr(series, 'name') else 'y': series.values}, index=index)
        fig = px.line(dfp, x=dfp.index, y=dfp.columns[0], title=title, labels={dfp.columns[0]: y_label or ''})
        st.plotly_chart(fig, use_container_width=True)


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
    # Optional: allow user to upload model and scalers from the UI (useful for deployment without file access)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Charger un modèle (optionnel)")
    uploaded_model = st.sidebar.file_uploader("Modèle Keras (.keras/.h5)", type=['keras','h5','hdf5'], help='Téléversez un fichier .keras ou .h5 entraîné si vous ne souhaitez pas placer les fichiers sur le disque.')
    uploaded_scaler_x = st.sidebar.file_uploader("Scaler X (pickle)", type=['pkl'], help='Scaler des features (scaler_x.pkl)')
    uploaded_scaler_y = st.sidebar.file_uploader("Scaler Y (pickle)", type=['pkl'], help='Scaler de la cible (scaler_y.pkl)')

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
        else:
            for col in spi_cols:
                _plot_line_with_fallback(df_view.index, df_view[col], title=f'SPI: {col}', y_label=col, color='tab:blue')

    # --- Precipitation ---
    if 'precip_median' in df_view.columns:
        st.subheader('Précipitation (médiane)')
    _plot_line_with_fallback(df_view.index, df_view['precip_median'], title='Précipitation (médiane)', y_label='Précipitation (mm)', color='tab:purple')

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
            else:
                # Use plotly to combine history and point
                df_hist = pd.DataFrame({ 'NDVI': recent_history.values }, index=recent_history.index)
                fig_pl = px.line(df_hist, x=df_hist.index, y='NDVI', title='NDVI : Historique et Prévision')
                fig_pl.add_scatter(x=prediction_point.index, y=prediction_point.values, mode='markers', marker=dict(size=12, color='red', symbol='star'), name='Prévision')
                st.plotly_chart(fig_pl, use_container_width=True)

        except Exception as e:
            st.error(f"Erreur lors de la génération de la prédiction : {e}")
    else:
        st.info('Modèle non chargé : la fonctionnalité de prédiction est désactivée. Placez les fichiers ndvi_model.keras, scaler_x.pkl et scaler_y.pkl dans le répertoire du projet et relancez l\'application.')

    # --- Data table and download ---
    st.subheader('Aperçu des Données')
    st.dataframe(df_view.head(200))
    csv = df_view.to_csv(index=True)
    st.download_button('Télécharger les données filtrées (CSV)', csv, file_name='master_dataset_filtered.csv')

    # Diagnostic panel
    with st.expander('Diagnostics'):
        st.write('Colonnes détectées :', list(df_full.columns))
        st.write('Colonne NDVI utilisée :', ndvi_col)
        st.write('Taille du jeu de données :', df_full.shape)

except Exception as e:
    st.error(f"Une erreur inattendue est survenue : {e}")
    st.info("Veuillez vérifier les logs du terminal pour plus de détails.")