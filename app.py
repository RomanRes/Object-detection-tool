import os
import base64
from typing import Optional, Union, Any


import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import tensorflow as tf
from dash import dcc, html, Input, Output, State, callback
from dash.exceptions import PreventUpdate


from utils.plot import img_to_plotly_fig
from utils.loadimage import load_image_pixels


# TENSORFLOW CONFIGURATION
# Reduce TF logging (0 = all, 1 = no info, 2 = no warnings, 3 = no errors)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def setup_tensorflow() -> None:
    """Checks for GPU availability and configures memory growth."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                # Prevent TF from hogging all VRAM at once
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU detected: {len(gpus)} device(s)")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPU found, falling back to CPU mode.")


setup_tensorflow()


# PLOTLY FIGURE DEFAULTS

def create_default_figure() -> go.Figure:
    """
    Creates an empty placeholder figure to show before any image is uploaded.
    Returns:
        go.Figure: A Plotly figure object with a placeholder layout.
    """
    # Loading an empty state (assuming load_image_pixels handles None)
    image: Any = load_image_pixels(None)
    image_w, image_h = image.size

    fig = go.Figure()

    # Invisible scatter to define coordinate space
    fig.add_trace(go.Scatter(
        x=[0, image_w],
        y=[0, image_h],
        mode="markers",
        marker_opacity=0,
        showlegend=False,
    ))

    # Add the base image as a layout background
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

    # Clean UI: hide axes and fix aspect ratio
    fig.update_xaxes(visible=False, range=[0, image_w], constrain="domain")
    fig.update_yaxes(visible=False, range=[image_h, 0], scaleanchor="x")

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        autosize=True,
        template="none"
    )

    return fig



# DASH APP INITIALIZATION

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.LITERA],
    # Important for mobile responsiveness
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

app.title = "YOLOv3 Detection Tool"


# APP LAYOUT

app.layout = html.Div(
    [
        # HEADER SECTION
        dbc.Row(
            dbc.Col(
                html.H3("YOLOv3 Object Detection Tool", className="text-center py-3"),
                width=12,
            ),
            style={"flex": "0 0 auto"}
        ),

        # MAIN GRAPH SECTION (Flexible height)
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
                                "minHeight": "0px",
                            },
                            responsive=True,
                            config={'displayModeBar': True}
                        )
                    ),
                    width=12,
                    style={
                        "flex": "1",
                        "minHeight": 0,
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

        # CONTROLS SECTION
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("IoU Threshold (NMS)", className="fw-bold"),
                        dcc.Slider(
                            id="nms_thresh",
                            min=0, max=1, step=0.05,
                            marks={i / 10: str(i / 10) for i in range(11)},
                            value=0.5,
                        ),
                    ],
                    width=12, md=4,
                ),

                dbc.Col(
                    [
                        html.Label("Confidence Threshold", className="fw-bold"),
                        dcc.Slider(
                            id="class_threshold",
                            min=0, max=1, step=0.05,
                            marks={i / 10: str(i / 10) for i in range(11)},
                            value=0.5,
                        ),
                    ],
                    width=12, md=4,
                ),

                dbc.Col(
                    dbc.Button(
                        "Start Detection",
                        id="submit_button",
                        n_clicks=0,
                        color="primary",
                        className="w-100 h-100",
                    ),
                    width=12, md=4, className="d-flex align-items-center"
                ),
            ],
            className="p-3 shadow-sm border-top bg-light",
            style={"flex": "0 0 auto", "zIndex": 10},
        ),

        # UPLOAD SECTION
        dbc.Row(
            [
                dbc.Col(
                    dcc.Upload(
                        id="upload-data",
                        children=html.Div([
                            "Drag and Drop or Select an Image"
                        ]),
                        style={
                            "width": "100%", "height": "50px", "lineHeight": "50px",
                            "borderWidth": "1px", "borderStyle": "dashed",
                            "borderRadius": "5px", "textAlign": "center",
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
        "height": "100vh",  # Fixed viewport height
        "display": "flex",
        "flexDirection": "column",
        "overflow": "hidden",  # Prevent double scrolling
    }
)



# CALLBACKS

@app.callback(
    Output("graph_figure", "figure", allow_duplicate=True),
    Input("upload-data", "contents"),
    prevent_initial_call=True,
)
def show_uploaded_image(contents: Optional[str]) -> go.Figure:
    """Displays the uploaded image without running detection, keeping axis consistency."""

    # Load image from base64 string
    image: Any = load_image_pixels(contents)
    image_w, image_h = image.size

    # Re-building the Plotly figure exactly as specified
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


@app.callback(
    Output("graph_figure", "figure"),
    Input("submit_button", "n_clicks"),
    State("upload-data", "contents"),
    State("class_threshold", "value"),
    State("nms_thresh", "value"),
)
def run_detection(
    n_clicks: int,
    contents: Optional[str],
    class_threshold: float,
    nms_thresh: float
) -> go.Figure:
    """Runs detection only when the button is clicked."""
    if n_clicks == 0:
        raise PreventUpdate

    image = load_image_pixels(contents)


    return img_to_plotly_fig(
        image,
        class_threshold=class_threshold,
        nms_thresh=nms_thresh,
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)