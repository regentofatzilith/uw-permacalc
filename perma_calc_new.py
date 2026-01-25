"""
UW PermaCalc (New) - Standalone deployment version for Render.com

This wrapper creates a standalone Dash application for deployment on Render.com with gunicorn.
For multi-page app integration, see pages/perma_calc_new.py
"""

import sys
import os
import dash
import dash_bootstrap_components as dbc

# Add project root to path to import core module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import all core functionality from the shared module
from perma_calc_core import layout

# NOTE: For Render deployment, we do NOT use dash.register_page()
# This is a standalone app, not part of a multi-page application

if __name__ == "__main__":
    # Standalone mode: run as independent Dash app
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        suppress_callback_exceptions=True
    )
    
    app.layout = layout
    
    print("Starting UW PermaCalc in standalone mode...")
    print("Open your browser and navigate to: http://127.0.0.1:8050")
    print("Press Ctrl+C to stop the server.")
    
    app.run(debug=True, host='127.0.0.1', port=8050)
else:
    # Production mode for deployment (e.g., Render.com with gunicorn)
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        suppress_callback_exceptions=True
    )
    
    app.layout = layout
    server = app.server  # Expose server for gunicorn
