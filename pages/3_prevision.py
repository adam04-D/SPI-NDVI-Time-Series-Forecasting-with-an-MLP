# Fichier: pages/3_prevision.py (MIS À JOUR)

import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import datetime as dt
import dash_bootstrap_components as dbc

# --- Enregistrement de la page ---
dash.register_page(
    __name__, 
    name='Modélisation & Prévision',
    icon='fa-solid fa-magic-wand-sparkles me-2' # Icône
)

# --- Fonction de chargement des données ---
def load_master_dataset():
    """Try several common locations for the master dataset and return a DataFrame."""
    candidates = [
        "master_dataset_bmk.csv",
        os.path.join("data", "processed", "master_dataset_bmk.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p, parse_dates=["date"], index_col="date")
                df.index.freq = pd.infer_freq(df.index) # Tente d'inférer la fréquence
                return df
            except Exception as e:
                print(f"Error reading {p}: {e}")
                return None
    return None

# --- Fonction de Prévision (votre modèle vient ici) ---
def simple_seasonal_forecast(series, n_periods):
    """
    Une prévision "naïve saisonnière" simple.
    REMPLACEZ CECI par votre vrai modèle (SARIMA, Prophet, etc.)
    """
    if series.empty:
        return pd.Series(dtype=float), pd.Index([], dtype='datetime64[ns]')

    last_date = series.index.max()
    freq = series.index.freq or 'MS' 
    
    if isinstance(freq, str):
        offset = pd.DateOffset(months=1) if 'M' in freq else pd.DateOffset(days=1)
    else:
        offset = freq

    future_dates = pd.date_range(
        start=last_date + offset, 
        periods=n_periods, 
        freq=freq
    )

    last_12_values = series.iloc[-12:].values
    forecast_values = np.tile(last_12_values, (n_periods // 12) + 1)[:n_periods]
    
    return future_dates, forecast_values

# --- Chargement des données ---
df = load_master_dataset()

# --- Définition du contenu (Layout) de la page ---
if df is None:
    layout = dbc.Alert(
        "Dataset introuvable. Veuillez placer `master_dataset_bmk.csv` à la racine.",
        color="danger",
        className="m-4"
    )
else:
    ndvi_col = next((c for c in ["ndvi_final", "ndvi_max_monthly", "ndvi"] if c in df.columns), None)
    
    layout = dbc.Container(
        [
            html.H1("Modélisation et Prévision NDVI", className="mb-4"),
            
            # Ligne des contrôles
            dbc.Row([
                dbc.Col([
                    html.Label("Mois à prévoir"),
                    dcc.Input(
                        id="forecast-months-prevision", # ID unique
                        type="number",
                        value=12,
                        min=1,
                        max=60,
                        step=1,
                        className="form-control"
                    ),
                ], md=4) 
            ], className="mb-4"),
            
            # --- NOUVEAU BLOC : Métriques de Performance ---
            dbc.Row([
                # Carte 1: RMSE
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H4("RMSE du Modèle", className="card-title"),
                            html.P("3.97 %", className="card-text fs-3 fw-bold")
                        ]),
                        color="info", # Couleur bleue
                        inverse=True
                    ),
                    md=4, # Utilise 4 colonnes sur 12
                    className="mb-3"
                ),
                
                # Carte 2: Exemple de Prédiction
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H4("Exemple de Prédiction", className="card-title"),
                            html.Div([
                                "Prévision: ", 
                                html.Strong("0.3345", className="text-warning"), # En orange
                                html.Br(),
                                "Réalité: ", 
                                html.Strong("0.3881", className="text-success") # En vert
                            ], 
                            className="card-text fs-5" # Taille de police 5
                           )
                        ]),
                        color="secondary", # Couleur grise
                        inverse=True
                    ),
                    md=4, # Utilise 4 colonnes sur 12
                    className="mb-3"
                )
            ], className="mb-4"), # Marge en bas de la ligne
            # --- FIN DU NOUVEAU BLOC ---

            # Ligne du Graphique de Prévision
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="forecast-graph-prevision") # ID unique
                ], width=12)
            ]),
        ],
        fluid=True
    )

# --- Callback de Prévision (INCHANGÉ) ---
@dash.callback(
    Output("forecast-graph-prevision", "figure"),
    Input("forecast-months-prevision", "value"),
)
def update_forecast_chart(forecast_months):
    if df is None or not forecast_months:
        return go.Figure(layout=go.Layout(title="Erreur de données ou Période de prévision non valide", template="plotly_dark"))

    try:
        forecast_months = int(forecast_months)
    except (ValueError, TypeError):
        return go.Figure(layout=go.Layout(title="Veuillez entrer un nombre valide.", template="plotly_dark"))

    ndvi_col = next((c for c in ["ndvi_final", "ndvi_max_monthly", "ndvi"] if c in df.columns), None)
    
    fig_ndvi = go.Figure()
    
    if ndvi_col and ndvi_col in df.columns:
        fig_ndvi.add_trace(go.Scatter(
            x=df.index, 
            y=df[ndvi_col], 
            mode='lines',
            name='NDVI Historique',
            line=dict(color="#2ca02c")
        ))
        
        forecast_dates, forecast_values = simple_seasonal_forecast(df[ndvi_col].dropna(), forecast_months)
        
        fig_ndvi.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_values,
            mode='lines',
            name='NDVI Prévu',
            line=dict(color="#FF851B", dash='dash')
        ))
        
        fig_ndvi.update_layout(
            title="NDVI Historique et Prévision", 
            legend=dict(x=0.01, y=0.99),
            template="plotly_dark"
        )
    else:
        fig_ndvi.update_layout(title="NDVI: colonne introuvable", template="plotly_dark")

    return fig_ndvi