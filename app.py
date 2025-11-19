import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pydicom
import io
import imageio.v3 as iio
import plotly.express as px
from extractor import ViewExtractor, ViewParams
from extractor import utils_3d, math_utils
import tempfile
import os
import time
import plotly.graph_objects as go
import pandas as pd
import json


views = ["A2C", "A3C", "A4C", "A5C", "PLAX",  "PSAX_(level_of_apex)","PSAX_(level_of_papillary_muscles)", "PSAX_(level_of_MV)"]
view_params={}
for view in views:
    view_df = pd.read_csv(f"sample_output/{view}.csv")
    view_params[view] = ViewParams(
        name=view,
        d=view_df['d'].values[0],
        phi=view_df['phi'].values[0],
        theta=view_df['theta'].values[0],
        flipu=view_df['flipu'].values[0],
        flipv=view_df['flipv'].values[0],
        rotation_angle=view_df['rotation_angle'].values[0],
        cm_pix=view_df['cm_pix'].values[0],
        h_cm=view_df['h_cm'].values[0],
        w_cm=view_df['w_cm'].values[0],
    )

# Page configuration
st.set_page_config(
    page_title="EchoSlicer 3D Visualization",
    page_icon="🔬",
    layout="wide"
)
st.markdown("""
    <style>
        /* Let the main container stretch edge-to-edge */
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        /* Stretch Plotly charts to the full column width */
        .stPlotlyChart {
            width: 100% !important;
        }
        /* Make the video responsive */
        .stVideo, .stVideo > div {
            width: 100% !important;
        }
    
        .stVideo video {
            width: 100% !important;
            height: 600px !important;
            width: 600px !important;
            object-fit: contain;
            background-color: black;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the 3D data"""
    dicom_path = 'sample_3d.dcm'
    view_extractor = ViewExtractor(original_weights=True)
    dcm = pydicom.dcmread(dicom_path)
    fps = int(dcm[('0008','2144')].value)
    data, bounds = utils_3d.read_3d(dcm) 
    rho_min, rho_max = bounds[0]
    phi_min, phi_max = bounds[1]
    theta_min, theta_max = bounds[2] 
    
    # Create linspace for rho, phi, and theta
    rho_coords = np.linspace(rho_min, rho_max, data.shape[0])
    phi_coords = np.linspace(phi_min, phi_max, data.shape[1])
    theta_coords = np.linspace(theta_min, theta_max, data.shape[2])
    positions = (rho_coords, phi_coords, theta_coords)
    
    data_at_first_time_point = data[:,:,:,0]
    position_cartesian, values_cartesian = math_utils.convert_to_cartesian(
        data_at_first_time_point, rho_coords, phi_coords, theta_coords
    )

    random_mask = np.random.random(len(position_cartesian)) < 0.05
    js_coords = list(position_cartesian[random_mask].reshape(-1).squeeze())
    js_vals = values_cartesian[random_mask].astype(float).squeeze()
    js_vals = np.repeat(js_vals, 3)
    js_vals = js_vals.reshape(-1).squeeze()
    #normalize js_vals to 0-1
    js_vals = (js_vals - js_vals.min()) / (js_vals.max() - js_vals.min())
    js_vals = list(js_vals)

    
    return view_extractor, data, positions, fps, position_cartesian, values_cartesian,js_coords,js_vals

def make_plane_surface(d,phi,theta):
    # to point normal form
    n = math_utils.normal_from_phitheta(math_utils.degree_to_radian(phi), math_utils.degree_to_radian(theta))
    P0 = math_utils.point_from_scalar(d,n)

    # to parametric form
    a = np.array([1,0,0], dtype=float)
    if np.allclose(n, a) or np.allclose(n, -a):
        a = np.array([0,1,0], dtype=float)

    # Create orthonormal basis
    u = np.cross(n, a)
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    s = np.linspace(-100, 100, 10)
    t = np.linspace(-100, 100, 10)
    S, T = np.meshgrid(s, t)
    X = P0[0] + S*u[0] + T*v[0]
    Y = P0[1] + S*u[1] + T*v[1]
    Z = P0[2] + S*u[2] + T*v[2]
    return X, Y, Z

def create_threejs_visualization(js_coords, js_vals, normal_vector, point_on_plane):
    """Create a Three.js visualization with scatter points and plane"""
    
    with open('js_viz.html', 'r') as f:
        html_code = f.read()
    html_code = html_code.replace('$JS_COORDS$', json.dumps(js_coords))
    html_code = html_code.replace('$JS_VALS$', json.dumps(js_vals))
    html_code = html_code.replace('$NORMAL_VECTOR$', ', '.join(map(str, normal_vector)))
    html_code = html_code.replace('$POINT_ON_PLANE$', ', '.join(map(str, point_on_plane)))

    return html_code


def numpy_video_to_mp4_bytes(video_array: np.ndarray, fps):
    """Convert a NumPy array to MP4 bytes in memory."""
    # Convert to uint8 if needed
    if video_array.dtype != np.uint8:
        video_array = ((video_array - video_array.min()) / 
                      (video_array.max() - video_array.min()) * 255).astype(np.uint8)
    
    buf = io.BytesIO()
    iio.imwrite(
        buf,
        video_array,
        extension=".mp4",
        fps=fps,
        codec='h264',
    )
    buf.seek(0)
    return buf

def render_video(view_extractor, data, positions, fps, position_cartesian, d, phi, theta, flipu, flipv, rotation_angle):
    """Render video with given parameters"""
    view_params = ViewParams(
        name="A4C",
        d=d,
        phi=phi,  
        theta=theta, 
        flipu=flipu,
        flipv=flipv,
        rotation_angle=rotation_angle,
        cm_pix=0,
        h_cm=0,
        w_cm=0
    )
    
    spherical_grid, (width, height), (width_px, height_px) = view_extractor.slice_volume(
        view_params, position_cartesian
    )
    video = math_utils.interpolate(positions, data, spherical_grid, height_px, width_px)
    return video

def main():

    view_extractor, data, positions, fps, position_cartesian, values_cartesian, js_coords, js_vals = load_data()

        
    visualization_container = st.container()
    # split visualization container into 3 columns
    visualization_container_1, visualization_container_2, visualization_container_3 = st.columns(3)
    controls_container = st.container()

    with visualization_container_3:
        for view in views:
            if st.button(view):
                st.session_state['distance_slider'] = int(view_params[view].d)
                st.session_state['azimuthal_angle_slider'] = int(view_params[view].phi)
                st.session_state['elevation_angle_slider'] = int(view_params[view].theta)
                st.session_state['flipu_checkbox'] = bool(view_params[view].flipu)
                st.session_state['flipv_checkbox'] = bool(view_params[view].flipv)
                st.session_state['rotation_angle_slider'] = int(view_params[view].rotation_angle)
        if 'distance_slider' not in st.session_state:
            st.session_state['distance_slider'] = 0
        if 'azimuthal_angle_slider' not in st.session_state:
            st.session_state['azimuthal_angle_slider'] = 0
        if 'elevation_angle_slider' not in st.session_state:
            st.session_state['elevation_angle_slider'] = 0
        if 'rotation_angle_slider' not in st.session_state:
            st.session_state['rotation_angle_slider'] = 0
        if 'flipu_checkbox' not in st.session_state:
            st.session_state['flipu_checkbox'] = False
        if 'flipv_checkbox' not in st.session_state:
            st.session_state['flipv_checkbox'] = False
    
    with controls_container:
        col1, col2 = st.columns(2)
        with col1:
            d = st.slider(
                "Distance (d)", 
                min_value=-200, 
                max_value=200, 
                key="distance_slider",
                help="Distance Parameter for the view"
            )
            phi = st.slider(
                "Azimuthal Angle (phi)", 
                min_value=-90, 
                max_value=90, 
                step=1,
                key="azimuthal_angle_slider",
                help="Azimuthal Angle Parameter for the view"
            )
            theta = st.slider(
                "Elevation Angle (theta)", 
                min_value=-180, 
                max_value=180, 
                step=1,
                key="elevation_angle_slider",
                help="Elevation Angle Parameter for the view"
            )
        with col2:
            rotation_angle = st.slider(
                "Rotation Angle",
                min_value=0,
                max_value=360,
                step=1,
                key="rotation_angle_slider",
                help="Rotation Angle Parameter for the view"
            )
            flipu = st.checkbox(
                "Flip U",
                key="flipu_checkbox",
                help="Flip U Parameter for the view"
            )
            flipv = st.checkbox(
                "Flip V",
                key="flipv_checkbox",
                help="Flip V Parameter for the view"
            )



    with visualization_container_1:
        normal_vector = math_utils.normal_from_phitheta(math_utils.degree_to_radian(phi), math_utils.degree_to_radian(theta)).reshape(-1).squeeze()
        point_on_plane = list(math_utils.point_from_scalar(d, normal_vector).reshape(-1).squeeze())
        normal_vector = list(normal_vector)
    
        html_viz = create_threejs_visualization(js_coords, js_vals, normal_vector, point_on_plane)
        components.html(html_viz, height=550)

                
    with visualization_container_2:
        video = render_video(
                    view_extractor, data, positions, fps, position_cartesian,
                    d, phi, theta, flipu, flipv, rotation_angle
                )
        mp4_bytes = numpy_video_to_mp4_bytes(video, fps)
        st.video(mp4_bytes, autoplay=True, loop=True)
    

if __name__ == "__main__":
    main()
