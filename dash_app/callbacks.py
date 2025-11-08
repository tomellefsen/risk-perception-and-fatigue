from dash import Input, Output, State, ALL, html, dcc, no_update
import plotly.graph_objects as go
from models.registry import MODEL_REGISTRY
import numpy as np
import inspect

# -----------------------------------
# HELPERS

def get_default_params(func):
    """
    Returns a dict of parameter name -> default value for a function.
    Only includes parameters that have a default value.
    """
    sig = inspect.signature(func)
    return {
        k: v.default
        for k, v in sig.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def standardize_result(t, compartments, meta, default_names=None):
    """
    Standardize the model output into a dictionary:
    {
        "t": time array,
        "compartments": {name: series, ...},
        "meta": {dict with model-specific metadata}
    }
    """
    if isinstance(compartments, dict):
        comp_dict = compartments
    else:
        # Use default or generic names
        if default_names is None:
            default_names = [f"Compartment {i+1}" for i in range(len(compartments))]
        comp_dict = {name: arr for name, arr in zip(default_names, compartments)}

    return {"t": t, "compartments": comp_dict, "meta": meta}


def model_unpacking(model_name):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found in registry.")
    
    model_info = MODEL_REGISTRY[model_name]
    func = model_info["function"]
    param_names = model_info["params"]
    default_names = model_info.get("default_names", None)
    plot_type = model_info.get("plot", "default")

    return func, param_names, default_names, plot_type

    
# END OF HELPERS
# -----------------------------------


def register_callbacks(app):
    
    ###### GENERATE PARAMETER INPUTS ######
    @app.callback(
        Output("param-inputs", "children"),
        Input("model-select", "value")
    )
    def update_param_inputs(model_name):
        if not model_name or model_name not in MODEL_REGISTRY:
            return []

        # Unpack model info
        func, param_names, default_names, plot_type = model_unpacking(model_name)

        # Get defaults from function signature
        defaults = get_default_params(func)

        input_fields = [
            html.Div([
                html.Label(name, style={"display": "block"}),
                dcc.Input(
                    id={"type": "param-input", "param": name},
                    type="number",
                    value=defaults.get(name, None),
                    debounce=True,
                    style={"width": "80px"}
                )
            ], style={"margin-right": "15px"})
            for name in param_names
        ]

        # Wrap all inputs in a flex container
        return html.Div(input_fields, style={"display": "flex", "align-items": "center"})

    ###### RUN MODEL & UPDATE GRAPH ######
    @app.callback(
        [Output("model-graph", "figure"),
         Output("model-meta", "children")],
        Input("run", "n_clicks"),
        State("model-select", "value"),
        State({"type": "param-input", "param": ALL}, "value"),
        State({"type": "param-input", "param": ALL}, "id"),
    )
    def update_graph(n_clicks, model_name, values, ids):
        if not model_name or model_name not in MODEL_REGISTRY:
            return no_update, "Invalid model"

        # Unpack model info
        func, param_names, default_names, plot_type = model_unpacking(model_name)

        # Map values to param dict
        params = {}
        for val, id_obj in zip(values, ids):
            param_name = id_obj["param"]
            if val is not None:
                try:
                    params[param_name] = float(val)
                except (ValueError, TypeError):
                    return no_update, f"Invalid value for {param_name}"

        # Run model
        t, compartments, meta = func(**params)
        
        # Standardize
        result = standardize_result(t, compartments, meta, default_names)

        # Plot with Plotly
        if plot_type == "default":
            fig = go.Figure()
            for name, data in result["compartments"].items():
                fig.add_trace(go.Scatter(x=result["t"], y=data, mode="lines", name=name))
            fig.update_layout(title=f"{model_name.upper()} Model", height=600)
        else:
            if plot_type == "sircp_dashboard":
                from models.sircp_dash import plot_sircp_dashboard
                fig = plot_sircp_dashboard(t, compartments, meta)
            elif plot_type == "sicr_pf_dashboard":
                from models.sicr_pf import plot_sicr_pf_dashboard
                fig = plot_sicr_pf_dashboard(t, compartments, meta)
            else:
                raise ValueError(f"Unknown plot type: {plot_type}")

        # Metadata
        meta_text = " | ".join(f"{k}: {v}" for k, v in result["meta"].items() 
                               if not isinstance(v, (list, np.ndarray)))

        return fig, meta_text