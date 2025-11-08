from dash import html, dcc
from models.registry import MODEL_REGISTRY

def serve_layout():
    return html.Div([
        html.H1("Epidemiological Models"),

        # Model selection
        dcc.Dropdown(
            id="model-select",
            options=[{"label": k.upper(), "value": k} for k in MODEL_REGISTRY.keys()],
            placeholder="Select a model"
        ),

        # Container for dynamic parameter inputs
        html.Div(id="param-inputs", style={"margin": "10px 0"}),
        html.Button("Run Model", id="run", n_clicks=0),

        # Output graph & metadata
        dcc.Graph(id="model-graph"),
        html.Div(id="model-meta")
    ], style={"padding": "20px"})
