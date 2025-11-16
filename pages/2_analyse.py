# Fichier: pages/2_analyse.py

import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
import os
import datetime as dt
import dash_bootstrap_components as dbc # Pour le style

# --- Enregistrement de la page ---
dash.register_page(
    __name__, 
    name='Analyse Exploratoire',
    icon='fa-solid fa-chart-line me-2' # Icône
)

# --- Fonction de chargement des données (de votre script original) ---
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
    layout = dbc.Alert(
        "Dataset introuvable. Veuillez placer `master_dataset_bmk.csv` à la racine.",
        color="danger",
        className="m-4"
    )
else:
    # Variables (de votre script original)
    ndvi_col = next((c for c in ["ndvi_final", "ndvi_max_monthly", "ndvi"] if c in df.columns), None)
    spi3_col = next((c for c in ["spi_3", "spi3"] if c in df.columns), None)
    min_date = df.index.min().date()
    max_date = df.index.max().date()

    # Le contenu (Layout)
    layout = dbc.Container(
        [
            html.H1("Analyse Exploratoire (NDVI & SPI)", className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    html.Label("Période"),
                    dcc.DatePickerRange(
                        id="date-range-analyse", # ID unique pour cette page
                        min_date_allowed=min_date,
                        max_date_allowed=max_date,
                        start_date=min_date,
                        end_date=max_date,
                        display_format="YYYY-MM",
                        className="d-block"
                    ),
                ], width=12)
            ], className="mb-4"),

            # Ligne pour les métriques (de votre script original)
            dbc.Row(id="metrics-row-analyse", className="mb-4"),

            # Ligne pour les graphiques
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="ndvi-graph-analyse")
                ], width=12)
            ]),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="spi-graph-analyse")
                ], width=12)
            ]),
        ],
        fluid=True # Utilise tout l'espace
    )


# --- Callback (de votre script original) ---
# IMPORTANT: On utilise @dash.callback au lieu de @app.callback
@dash.callback(
    Output("ndvi-graph-analyse", "figure"),
    Output("spi-graph-analyse", "figure"),
    Output("metrics-row-analyse", "children"),
    Input("date-range-analyse", "start_date"),
    Input("date-range-analyse", "end_date"),
)
def update_charts(start_date, end_date):
    # On vérifie si les données ont été chargées
    if df is None:
        return {}, {}, "Erreur de données"

    # On re-vérifie les colonnes au cas où
    ndvi_col = next((c for c in ["ndvi_final", "ndvi_max_monthly", "ndvi"] if c in df.columns), None)
    spi3_col = next((c for c in ["spi_3", "spi3"] if c in df.columns), None)

    # Filtrage (de votre script original)
    start = pd.to_datetime(start_date) if start_date else df.index.min()
    end = pd.to_datetime(end_date) if end_date else df.index.max()
    dff = df.loc[(df.index >= start) & (df.index <= end)].copy()

    # --- Métriques ---
    metrics = []
    if ndvi_col in dff.columns:
        last_ndvi = dff[ndvi_col].dropna().iloc[-1] if not dff[ndvi_col].dropna().empty else np.nan
        card_ndvi = dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.H4("Dernier NDVI", className="card-title"),
                    html.P(f"{last_ndvi:.4f}", className="card-text fs-3")
                ])
            ], color="success", inverse=True), # 'inverse' pour texte blanc sur fond sombre
            md=4
        )
        metrics.append(card_ndvi)

    if spi3_col in dff.columns:
        last_spi3 = dff[spi3_col].dropna().iloc[-1] if not dff[spi3_col].dropna().empty else np.nan
        card_spi = dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.H4("Dernier SPI-3", className="card-title"),
                    html.P(f"{last_spi3:.3f}", className="card-text fs-3")
                ])
            ], color="primary", inverse=True),
            md=4
        )
        metrics.append(card_spi)

    # --- Figure NDVI ---
    if ndvi_col and ndvi_col in dff.columns:
        fig_ndvi = px.line(dff, x=dff.index, y=ndvi_col, title="NDVI Time Series", labels={ndvi_col: "NDVI", "x": "Date"})
        fig_ndvi.update_traces(line=dict(color="#2ca02c"))
        fig_ndvi.update_layout(template="plotly_dark") # Adapte au thème sombre
    else:
        fig_ndvi = px.line(title="NDVI: colonne introuvable").update_layout(template="plotly_dark")

    # --- Figure SPI ---
    if spi3_col and spi3_col in dff.columns:
        fig_spi = px.line(dff, x=dff.index, y=spi3_col, title="SPI-3 Time Series", labels={spi3_col: "SPI-3", "x": "Date"})
        fig_spi.update_traces(line=dict(color="#0d6efd"))
        fig_spi.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_spi.add_hline(y=-1.5, line_dash="dot", line_color="red")
        fig_spi.update_layout(template="plotly_dark") # Adapte au thème sombre
    else:
        fig_spi = px.line(title="SPI: colonne introuvable").update_layout(template="plotly_dark")

    return fig_ndvi, fig_spi, metrics