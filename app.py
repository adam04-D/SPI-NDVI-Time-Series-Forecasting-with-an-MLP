import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os

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
    if not os.path.exists('ndvi_model.keras'):
        st.error("ERREUR : Fichier 'ndvi_model.keras' introuvable. Veuillez ré-exécuter le script de sauvegarde dans votre notebook.")
        return None, None, None
    if not os.path.exists('scaler_x.pkl'):
        st.error("ERREUR : Fichier 'scaler_x.pkl' introuvable. Veuillez ré-exécuter le script de sauvegarde.")
        return None, None, None
    if not os.path.exists('scaler_y.pkl'):
        st.error("ERREUR : Fichier 'scaler_y.pkl' introuvable. Veuillez ré-exécuter le script de sauvegarde.")
        return None, None, None
        
    model = load_model('ndvi_model.keras')
    with open('scaler_x.pkl', 'rb') as f:
        scaler_X = pickle.load(f)
    with open('scaler_y.pkl', 'rb') as f:
        scaler_Y = pickle.load(f)
    print("Actifs chargés.")
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

# --- 4. Exécution de l'Application ---
try:
    # Charger les données et le modèle
    model, scaler_X, scaler_Y = load_assets()
    df_full = load_data()

    # Vérifier que tout est chargé avant de continuer
    if model is not None and df_full is not None:
        st.header("📈 Prévision pour le Mois Prochain")

        # Faire la prévision
        last_known_data, predicted_ndvi, ndvi_col = get_prediction(df_full, model, scaler_X, scaler_Y)

        last_date = last_known_data.index[0]
        future_date = last_date + pd.DateOffset(months=1)

    # Afficher la prévision (en grosses boîtes)
    col1, col2 = st.columns(2)
    # show date in the label, use numeric value for the metric and numeric delta
    value_last = float(df_full[ndvi_col].loc[last_date])
    col1.metric(label=f"Dernière Donnée (Réelle) — {last_date.strftime('%Y-%m')}",
            value=f"{value_last:.4f}")

    delta = float(predicted_ndvi - value_last)
    col2.metric(label=f"Prévision (Mois Prochain) — {future_date.strftime('%Y-%m')}",
            value=f"{predicted_ndvi:.4f}",
            delta=delta)

    # --- 5. Visualisation (Notre Leçon 15) ---
    st.subheader("Contexte de la Prévision (Historique sur 3 ans)")

    recent_history = df_full[ndvi_col].iloc[-36:]
    prediction_point = pd.Series([predicted_ndvi], index=[future_date])

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(recent_history.index, recent_history,
        label='Données Réelles (Historique Récent)',
        color='blue', linewidth=2, marker='o', markersize=5)
    ax.plot(prediction_point.index, prediction_point,
        'r*', markersize=15,
        label=f'Prévision Future ({future_date.strftime("%Y-%m")}) = {predicted_ndvi:.4f}')
    ax.set_title('Prévision du NDVI pour le Mois Prochain')
    ax.set_ylabel('NDVI (Max Mensuel)')
    ax.set_xlabel('Date')
    ax.legend()
    ax.grid(True)

    # Afficher le graphique dans Streamlit
    st.pyplot(fig)

except Exception as e:
    st.error(f"Une erreur inattendue est survenue : {e}")
    st.info("Veuillez vérifier les logs du terminal pour plus de détails.")