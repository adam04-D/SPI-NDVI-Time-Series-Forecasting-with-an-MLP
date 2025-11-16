# Fichier: pages/1_accueil.py (MODIFIÉ)

import dash
from dash import html, dash_table
import pandas as pd
import os
import dash_bootstrap_components as dbc

# --- Enregistrement de la page ---
dash.register_page(
    __name__, 
    path='/', 
    name='Accueil & Données',
    icon='fa-solid fa-house me-2' # Icône de FontAwesome
)

# --- Fonction de chargement des données (copiée de vos autres pages) ---
def load_master_dataset():
    """Try several common locations for the master dataset and return a DataFrame."""
    candidates = [
        "master_dataset_bmk.csv",
        os.path.join("data", "processed", "master_dataset_bmk.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                # On lit les données, mais on garde 'date' comme colonne
                df = pd.read_csv(p, parse_dates=["date"]) 
                return df
            except Exception as e:
                print(f"Error reading {p}: {e}")
                return None
    return None

# --- Chargement des données ---
df = load_master_dataset()

# --- Définition du contenu (Layout) de la page ---
if df is None:
    # Message d'erreur si les données ne sont pas trouvées
    layout = html.Div([
        html.H1("Bienvenue sur le Dashboard de Prévision NDVI"),
        html.P("Ceci est la page d'accueil. Utilisez la navigation pour explorer :"),
        html.Hr(),
        html.H3("Vos Données (master_dataset_bmk.csv)"),
        dbc.Alert(
            "Dataset introuvable. Veuillez placer `master_dataset_bmk.csv` à la racine.",
            color="danger",
            className="m-4"
        )
    ])
else:
    # Si les données sont chargées, on crée le layout AVEC la table
    
    # Formatte la date pour un meilleur affichage (optionnel mais propre)
    if 'date' in df.columns:
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')

    layout = html.Div([
        html.H1("Bienvenue sur le Dashboard de Prévision NDVI"),
        html.P("Ceci est la page d'accueil. Utilisez la navigation pour explorer :"),
        html.Ul([
            html.Li("Analyse Exploratoire : Voir les données historiques NDVI et SPI."),
            html.Li("Modélisation & Prévision : Lancer une nouvelle prévision."),
        ]),
        html.Hr(),
        html.H3("Vos Données (master_dataset_bmk.csv)"),
        
        # --- AJOUT DU TABLEAU DE DONNÉES ---
        dash_table.DataTable(
            id='data-table',
            
            # Prépare les données pour le tableau
            columns=[{"name": i, "id": i} for i in df.columns],
            data=df.to_dict('records'),
            
            # --- Style pour le thème sombre (CYBORG / custom) ---
            style_as_list_view=True,
            style_header={
                'backgroundColor': '#2a3d4a', # Fond de l'en-tête
                'color': 'white',
                'fontWeight': 'bold',
                'border': '1px solid #1f3040'
            },
            style_cell={
                'backgroundColor': '#1f3040', # Fond des cellules
                'color': 'white',
                'border': '1px solid #2a3d4a',
                'padding': '8px',
                'textAlign': 'left',
                'fontFamily': 'sans-serif'
            },
            
            # Pagination et barres de défilement
            page_size=15, # Affiche 15 lignes par page
            style_table={
                'overflowX': 'auto',  # Défilement horizontal
                'height': '450px', 
                'overflowY': 'auto'   # Défilement vertical
            }
        )
    ])