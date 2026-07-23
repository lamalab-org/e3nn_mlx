


import ase
from ase import visualize, io

from e3nn_mlx import o3

import math
import ipywidgets
import numpy as np

import plotly.graph_objects as go
import plotly.express as px
from scipy.spatial.transform import Rotation

import mlx.core as mx



axis = dict(
    showbackground=False,
    showticklabels=False,
    showgrid=False,
    zeroline=False,
    title='',
    nticks=3,
)

layout = dict(
    width=690,
    height=160,
    scene=dict(
        xaxis=dict(
            **axis,
            range=[-8, 8]
        ),
        yaxis=dict(
            **axis,
            range=[-2, 2]
        ),
        zaxis=dict(
            **axis,
            range=[-2, 2]
        ),
        aspectmode='manual',
        aspectratio=dict(x=8, y=2, z=2),
        camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=0, y=-5, z=5),
            projection=dict(type='orthographic'),
        ),
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=0, b=0)
)


#torch.set_default_dtype(torch.float32)
np.set_printoptions(precision=2)

def view(atoms: ase.Atom, centre=True):
    viewer = visualize.view(atoms, viewer='x3d')
    # if not centre:
    # nglview.view.center(selection='0')
    return viewer

def show_array(positions, calc=None):
    positions = np.asarray(positions)

    if calc is None:
        calc = lambda x: x

    initial_data = np.asarray(calc(positions))

    fig = px.imshow(
        initial_data,
        color_continuous_scale="RdBu",
        zmin=-5,
        zmax=5,
    )
    widget = go.FigureWidget(fig)

    @ipywidgets.interact(
        xrot=(0, 360, 1.0),
        xtrans=(-10, 10, 1.0),
        swap=(0, len(positions) - 1, 1),
    )
    def update(xrot=0, xtrans=0, swap=0):
        with widget.batch_update():
            pos = positions.copy()

            if swap:
                pos[[swap, 0]] = pos[[0, swap]]

            if xtrans:
                pos[:, 0] += xtrans

            if xrot:
                rot = Rotation.from_euler("x", xrot, degrees=True)
                pos = rot.apply(pos)

            # Plotly requires NumPy arrays, lists, tuples, or pandas objects.
            data = np.asarray(calc(pos)).T
            widget.data[0].z = data

    return widget


def s2_grid():
    betas = mx.linspace(0, math.pi, 40)
    alphas = mx.linspace(0, 2 * math.pi, 80)
    beta, alpha = mx.meshgrid(betas, alphas)
    return o3.angles_to_xyz(alpha, beta)

import numpy as np


def trace(r, f, c, radial_abs=True):
    # Ensure inputs are NumPy arrays so mathematical operations work consistently
    r = np.asarray(r)
    f = np.asarray(f)
    c = np.asarray(c)

    # Use np.abs(f) instead of f.abs()
    if radial_abs:
        a = np.abs(f)
    else:
        a = 1.0

    return dict(
        x=a * r[..., 0] + c[0],
        y=a * r[..., 1] + c[1],
        z=a * r[..., 2] + c[2],
        surfacecolor=f,
    )

def plot(data, radial_abs=True):
    data=np.asarray(data)
    r = s2_grid()
    r=np.asarray(r)
    n = data.shape[-1]
    traces = [
        trace(r, data[..., i], mx.array([2.0 * i - (n - 1.0), 0.0, 0.0]), radial_abs=radial_abs)
        for i in range(n)
    ]
    cmax = max(d['surfacecolor'].abs().max().item() for d in traces)
    traces = [go.Surface(**d, colorscale='RdBu', cmin=-cmax, cmax=cmax) for d in traces]
    fig = go.Figure(data=traces, layout=layout)
    fig.show()

def plot_sphere(r):
    r=np.asarray(r)
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=r[..., 0].flatten(),
                y=r[..., 1].flatten(),
                z=r[..., 2].flatten(),
                mode='markers',
                marker=dict(
                    size=1,
                ),
            )
        ],
        layout=dict(
            width=500,
            height=300,
            scene=dict(
                xaxis=dict(
                    **axis,
                    range=[-1, 1]
                ),
                yaxis=dict(
                    **axis,
                    range=[-1, 1]
                ),
                zaxis=dict(
                    **axis,
                    range=[-1, 1]
                ),
                aspectmode='manual',
                aspectratio=dict(x=3, y=3, z=3),
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=0, y=-5, z=5),
                    projection=dict(type='orthographic'),
                ),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0)
        )
    )
    fig.show()


def random_point_signal(lmax, N=5, r_min=1e-1, std=1.):
    n = N
    points = mx.norm.random(n, 3) * std / mx.sqrt(3.)
    points /= points.norm(2, -1, keepdim=True)
#     select = (points.norm(2, -1) > r_min).nonzero()
    return io.SphericalTensor(lmax, p_val=1, p_arg=-1).from_geometry_adjusted(points)



def plot(data, radial_abs=True, layout=None):
    # Ensure data and grid are numpy arrays
    data = np.asarray(data)
    r = np.asarray(s2_grid())
    n = data.shape[-1]

    # 1. Generate raw trace dicts (using np.array for type consistency)
    raw_traces = [
        trace(
            r,
            data[..., i],
            np.array([2.0 * i - (n - 1.0), 0.0, 0.0]),
            radial_abs=radial_abs,
        )
        for i in range(n)
    ]

    # 2. Fix .abs() call by using np.abs()
    cmax = max(np.abs(d["surfacecolor"]).max().item() for d in raw_traces)

    # 3. Create Plotly Surface objects
    traces = [
        go.Surface(**d, colorscale="RdBu", cmin=-cmax, cmax=cmax)
        for d in raw_traces
    ]

    # 4. Provide a default 3D layout if none was passed
    if layout is None:
        layout = go.Layout(
            scene=dict(aspectmode="data"), margin=dict(l=0, r=0, b=0, t=0)
        )

    fig = go.Figure(data=traces, layout=layout)
    fig.show()