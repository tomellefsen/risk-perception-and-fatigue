from dash import Dash
from layouts.main_layout import serve_layout
from callbacks import register_callbacks

app = Dash(__name__, suppress_callback_exceptions=True)
app.layout = serve_layout

register_callbacks(app)

server = app.server

if __name__ == "__main__":
    app.run(debug=True)
