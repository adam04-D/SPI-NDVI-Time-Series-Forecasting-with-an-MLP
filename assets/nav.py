# Fichier: assets/nav.py (MODIFIÉ)
import dash
import dash_bootstrap_components as dbc
from dash import html

# Récupère les informations des pages du dossier 'pages/'
pages = dash.page_registry.values()

# Crée les liens de navigation
nav_links = [
    dbc.NavLink(
        [
            html.I(className=page["icon"]), # Ajoute une icône
            page["name"]
        ],
        href=page["relative_path"], # Lien vers la page
        active="exact",
    ) for page in pages
]

# MODIFICATION :
# 'sidebar_nav_content' n'est plus un 'html.Div' complet.
# C'est juste le *contenu* que app.py placera dans la sidebar.
sidebar_nav_content = [
    html.H4("Tableau de Bord NDVI", className="my-3 ms-3 text-white"),
    html.Hr(className="text-white"),
    
    dbc.Nav(
        nav_links,
        vertical=True, # Navigation verticale
        pills=True, # Style "pilule"
        className="flex-column",
    ),
]