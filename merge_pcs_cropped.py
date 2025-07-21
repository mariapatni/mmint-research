import open3d as o3d
import numpy as np
import os
from open3d.visualization import draw_plotly_server
from dash import Dash, html, dcc, Input, Output, State
import plotly.graph_objs as go

def aabb_to_wireframe(box, color="red"):
    lines = [
        [0, 1], [1, 3], [3, 2], [2, 0],  # bottom face
        [4, 5], [5, 7], [7, 6], [6, 4],  # top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # vertical lines
    ]
    points = np.asarray(box.get_box_points())

    # Flatten line segments into x, y, z for Plotly
    x_lines = []
    y_lines = []
    z_lines = []
    for edge in lines:
        for idx in edge:
            x_lines.append(points[idx][0])
            y_lines.append(points[idx][1])
            z_lines.append(points[idx][2])
        x_lines.append(None)  # to break line between segments
        y_lines.append(None)
        z_lines.append(None)

    return go.Scatter3d(
        x=x_lines, y=y_lines, z=z_lines,
        mode='lines',
        line=dict(color=color, width=4),
        name='Crop Box',
        hoverinfo='skip',
        showlegend=False
    )


# === Config ===
INPUT_FILE = "./lego_lidar/0000000.ply"
OUTPUT_DIR = "./lego_lidar_cropped"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Load point cloud
pcd = o3d.io.read_point_cloud(INPUT_FILE)
points = np.asarray(pcd.points)


# Get Dash app and figure, but don’t auto-run
app, fig = draw_plotly_server([pcd], window_name="Point Cloud Viewer", port=8050)

# === Layout ===
app.layout = html.Div([
    html.H2("Point Cloud Cropper", style={"textAlign": "center"}),

    dcc.Graph(id='pointcloud', figure=fig),

    html.Div([
        html.H4("Bounding Box Crop Controls"),
        html.Div([
            html.Label("Min [x, y, z]:"),
            dcc.Input(id='min-x', type='number', placeholder="x", step=0.01),
            dcc.Input(id='min-y', type='number', placeholder="y", step=0.01),
            dcc.Input(id='min-z', type='number', placeholder="z", step=0.01),
        ], style={"marginBottom": "10px"}),

        html.Div([
            html.Label("Max [x, y, z]:"),
            dcc.Input(id='max-x', type='number', placeholder="x", step=0.01),
            dcc.Input(id='max-y', type='number', placeholder="y", step=0.01),
            dcc.Input(id='max-z', type='number', placeholder="z", step=0.01),
        ], style={"marginBottom": "10px"}),

        html.Button("Crop and Save", id='crop-btn', n_clicks=0),
        html.Button("Preview Crop Box", id='preview-btn', n_clicks=0),
        html.Div(id='status', style={"marginTop": "15px", "color": "green"})
    ], style={"margin": "30px"})
])

# === Crop + Save Callback ===
@app.callback(
    Output('status', 'children'),
    Input('crop-btn', 'n_clicks'),
    State('min-x', 'value'), State('min-y', 'value'), State('min-z', 'value'),
    State('max-x', 'value'), State('max-y', 'value'), State('max-z', 'value'),
)
def crop_and_save(n_clicks, min_x, min_y, min_z, max_x, max_y, max_z):
    if n_clicks == 0:
        return ""

    if None in [min_x, min_y, min_z, max_x, max_y, max_z]:
        return "⚠️ Please enter all six bounding box values."

    min_bound = np.array([min_x, min_y, min_z])
    max_bound = np.array([max_x, max_y, max_z])
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

    cropped = pcd.crop(bbox)
    output_path = os.path.join(OUTPUT_DIR, "cropped_0000000.ply")
    o3d.io.write_point_cloud(output_path, cropped)

    return f"✅ Cropped point cloud saved to: {output_path}"

@app.callback(
    Output('pointcloud', 'figure'),
    Input('preview-btn', 'n_clicks'),
    State('min-x', 'value'), State('min-y', 'value'), State('min-z', 'value'),
    State('max-x', 'value'), State('max-y', 'value'), State('max-z', 'value'),
)
def preview_crop(n_clicks, min_x, min_y, min_z, max_x, max_y, max_z):
    fig = draw_plotly_server([pcd], return_fig=True)  # regenerate base view

    if n_clicks == 0 or None in [min_x, min_y, min_z, max_x, max_y, max_z]:
        return fig

    min_bound = np.array([min_x, min_y, min_z])
    max_bound = np.array([max_x, max_y, max_z])
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

    red_box = aabb_to_wireframe(bbox, color="red")
    fig.add_trace(red_box)

    return fig


# === Run Dash app ===
if __name__ == "__main__":
    app.run(port=8050)

