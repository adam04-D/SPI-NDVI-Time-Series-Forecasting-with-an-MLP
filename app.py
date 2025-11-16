# Fichier: app.py (MODIFIÉ ET AJUSTÉ)

import dash
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, page_container

# --- Initialisation de l'App ---
app = Dash(__name__, 
           use_pages=True,
           external_stylesheets=[
               dbc.themes.CYBORG, 
               dbc.icons.FONT_AWESOME,
               '/assets/custom.css'  # Assure-toi que ton CSS est ici
           ],
           suppress_callback_exceptions=True)

server = app.server

# --- Importe le CONTENU de la navigation (de assets/nav.py) ---
from assets.nav import sidebar_nav_content

# --- Définition des styles ---
# Nous utilisons une largeur fixe pour plus de fiabilité
SIDEBAR_WIDTH_REM = "18rem"

# Style de la barre latérale (Sidebar)
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": SIDEBAR_WIDTH_REM, # Largeur fixe
    "padding": "1rem",         # Un peu d'espace intérieur
    "overflow-y": "auto",    # Permet de scroller si les liens sont nombreux
}

# Style du contenu principal (Content)
CONTENT_STYLE = {
    "margin-left": SIDEBAR_WIDTH_REM, # Doit être égal à la largeur de la sidebar
    "padding": "2rem",              # Espace intérieur pour le contenu
}

# --- Structure de la Page (Layout) ---

# 1. La barre latérale
sidebar = html.Div(
    sidebar_nav_content,    # Le contenu vient de assets/nav.py
    className="bg-dark",      # Utilise le style de fond de custom.css
    style=SIDEBAR_STYLE
)

# 2. Le contenu
content = html.Div(
    page_container,         # C'est ici que tes pages s'affichent
    style=CONTENT_STYLE
)

# Layout final (Sidebar + Content)
app.layout = html.Div([sidebar, content])


# --- Lancement de l'App ---
if __name__ == '__main__':
    app.run(debug=True)