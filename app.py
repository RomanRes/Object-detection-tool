import os
import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from dash import dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate

from utils.plot import img_to_plotly_fig
from utils.loadimage import load_image_pixels

# ---------------------------
# TensorFlow Setup
# ---------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU detected: {len(gpus)} device(s)")
    except RuntimeError as e:
        print(e)
else:
    print("⚠️ No GPU found, using CPU")


# ---------------------------
# Default Figure
# ---------------------------
def create_default_figure():
    image = load_image_pixels(None)
    image_w, image_h = image.size

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[0, image_w],
        y=[0, image_h],
        mode="markers",
        marker_opacity=0,
        showlegend=False,
    ))

    fig.add_layout_image(
        dict(
            source=image,
            x=0,
            y=0,
            sizex=image_w,
            sizey=image_h,
            xref="x",
            yref="y",
            layer="below",
        )
    )

    fig.update_xaxes(visible=False, range=[0, image_w], constrain="domain")
    fig.update_yaxes(visible=False, range=[image_h, 0], scaleanchor="x")

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        autosize=True,
    )

    return fig


# ---------------------------
# App Setup
# ---------------------------
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.LITERA],
    meta_tags=[{"name": "viewport", "content": "width=device-width"}],
)

app.title = "YOLOv3 Detection Tool"
app.update_title = "Processing..."


# ---------------------------
# Layout
# ---------------------------
app.layout = html.Div(
    [
        # Header
        dbc.Row(
            dbc.Col(
                html.H3("YOLOv3 Detection Tool", className="text-center"),
                width=12,
            ),
            style={
                "flex": "0 0 auto",
                "marginBottom": "20px"
            },
        ),

        # Graph (nimmt Resthöhe)
        dbc.Row(
            [
                dbc.Col(
                    dbc.Spinner(
                        dcc.Graph(
                            id="graph_figure",
                            figure=create_default_figure(),
                            style={
                                "width": "100%",
                                "height": "100%",
                                "flex": "1",
                            },
                            responsive=True,
                        )
                    ),
                    width=12,
                    style={
                        "height": "100%",
                        "display": "flex",
                        "flexDirection": "column",
                    },
                )
            ],
            style={
                "flex": "1",
                "minHeight": 0,
                "display": "flex",
            },
        ),

        # Controls
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H6("IoU Threshold (NMS)", className="text-center"),
                        dcc.Slider(
                            id="nms_thresh",
                            min=0,
                            max=1,
                            step=0.1,
                            marks={i / 10: str(i / 10) for i in range(11)},
                            value=0.5,
                        ),
                    ],
                    width=12,
                    md=4,
                ),

                dbc.Col(
                    [
                        html.H6("Confidence Threshold", className="text-center"),
                        dcc.Slider(
                            id="class_threshold",
                            min=0,
                            max=1,
                            step=0.1,
                            marks={i / 10: str(i / 10) for i in range(11)},
                            value=0.5,
                        ),
                    ],
                    width=12,
                    md=4,
                ),

                dbc.Col(
                    dbc.Button(
                        "🔍 Detect",
                        id="submit_button",
                        n_clicks=0,
                        color="primary",
                        className="w-100",
                    ),
                    width=12,
                    md=4,
                ),
            ],
            style={"flex": "0 0 auto", "padding": "10px"},
        ),

        # Upload
        dbc.Row(
            [
                dbc.Col(
                    dcc.Upload(
                        id="upload-data",
                        children=html.Div(
                            "Drag and drop a file or click to upload"
                        ),
                        style={
                            "width": "100%",
                            "height": "60px",
                            "lineHeight": "60px",
                            "borderWidth": "1px",
                            "borderStyle": "dashed",
                            "borderRadius": "5px",
                            "textAlign": "center",
                        },
                        multiple=False,
                    ),
                    width=12,
                )
            ],
            style={"flex": "0 0 auto", "padding": "10px"},
        ),
    ],
    style={
        "height": "100vh",
        "display": "flex",
        "flexDirection": "column",
        "overflow": "hidden",
    },
)


# ---------------------------
# Callbacks
# ---------------------------

# Bild anzeigen (ohne Boxes)
@app.callback(
    Output("graph_figure", "figure", allow_duplicate=True),
    Input("upload-data", "contents"),
    prevent_initial_call=True,
)
def show_uploaded_image(contents):

    image = load_image_pixels(contents)
    image_w, image_h = image.size

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[0, image_w],
        y=[0, image_h],
        mode="markers",
        marker_opacity=0,
        showlegend=False,
    ))

    fig.add_layout_image(
        dict(
            source=image,
            x=0,
            y=0,
            sizex=image_w,
            sizey=image_h,
            xref="x",
            yref="y",
            layer="below",
        )
    )

    fig.update_xaxes(visible=False, range=[0, image_w], constrain="domain")
    fig.update_yaxes(visible=False, range=[image_h, 0], scaleanchor="x")

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        autosize=True,
    )

    return fig


# Detection (mit Button)
@app.callback(
    Output("graph_figure", "figure"),
    Input("submit_button", "n_clicks"),
    State("upload-data", "contents"),
    State("class_threshold", "value"),
    State("nms_thresh", "value"),
)
def run_detection(n_clicks, contents, class_threshold, nms_thresh):

    if n_clicks == 0:
        raise PreventUpdate

    image = load_image_pixels(contents)

    return img_to_plotly_fig(
        image,
        class_threshold=class_threshold,
        nms_thresh=nms_thresh,
    )


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    app.run(debug=False)