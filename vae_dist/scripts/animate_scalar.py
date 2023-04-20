#!export CUDA_VISIBLE_DEVICES=0

import torch, json
import numpy as np

# import matplotlib.pyplot as plt
from pylab import *
from mpl_toolkits.mplot3d import Axes3D

from copy import deepcopy

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from vae_dist.dataset.dataset import FieldDataset
from vae_dist.core.training_utils import construct_model
from vae_dist.dataset.fields import split_and_filter


def plot_sfield(
    scalar_field,
    z_ind=11,
    z_level=0.0,
    cmax=None,
    cmin=None,
    colorbar=None,
    x_y_dims={"x": [-3.0, 3.3, 0.3], "y": [-3.0, 3.3, 0.3]},
):
    # plot iso surface of a given scalar field
    # z_level: iso surface level

    zvals = scalar_field[:, :, z_ind]
    # print(z_ind, z_level)
    # create flat surface with color values = zvals at z_level
    surface = go.Surface(
        x=np.arange(x_y_dims["x"][0], x_y_dims["x"][1], x_y_dims["x"][2]),
        y=np.arange(x_y_dims["y"][0], x_y_dims["y"][1], x_y_dims["y"][2]),
        z=z_level * np.ones_like(zvals),
        showscale=True,
        surfacecolor=zvals,
        text=zvals,
        cmax=cmax,
        cmin=cmin,
        colorbar=colorbar,
        colorscale="PuBu",
        name="{}".format(z_ind),
    )
    return surface
    # return xvals, yvals, zvals


def show_in_out_plots(
    in_field,
    model,
    device,
    shape_dict={"x": 21, "y": 21, "z": 21},
    bounds_dict={"x": [-3.3, 3.0], "y": [-3.3, 3.0], "z": [-3.3, 3.0]},
):
    z_step = (bounds_dict["z"][1] - bounds_dict["z"][0]) / (shape_dict["z"])
    x = in_field.reshape(1, 1, shape_dict["x"], shape_dict["y"], shape_dict["z"]).to(
        device
    )
    x_out = model.forward(x)
    x_out = (
        x_out.to("cpu")
        .detach()
        .numpy()
        .reshape(1, 1, shape_dict["x"], shape_dict["y"], shape_dict["z"])
    )
    max_val = max(
        np.max(
            x.cpu().numpy().reshape(shape_dict["x"], shape_dict["y"], shape_dict["z"])
        ),
        np.max(x_out.reshape(shape_dict["x"], shape_dict["y"], shape_dict["z"])),
    )
    # max_val = np.max(x_out.reshape(21, 21, 21))
    min_val = min(
        np.min(
            x.cpu().numpy().reshape(shape_dict["x"], shape_dict["y"], shape_dict["z"])
        ),
        np.min(x_out.reshape(shape_dict["x"], shape_dict["y"], shape_dict["z"])),
    )
    print("max_val: ", max_val)
    print("min_val: ", min_val)

    frames_in = []
    frames_out = []
    scalar_in = (
        x.cpu().numpy().reshape(shape_dict["x"], shape_dict["y"], shape_dict["z"])
    )
    scalar_out = x_out.reshape(shape_dict["x"], shape_dict["y"], shape_dict["z"])
    for i in range(shape_dict["z"]):
        if i == 0:
            frame_init = plot_sfield(
                scalar_in,
                z_ind=0,
                z_level=float(bounds_dict["z"][0]),
                cmin=min_val,
                cmax=max_val,
            )
            frame_init_out = plot_sfield(
                scalar_out,
                z_ind=0,
                z_level=float(bounds_dict["z"][0]),
                cmin=min_val,
                cmax=max_val,
                colorbar=dict(thickness=20, ticklen=4),
            )

        frames_in.append(
            plot_sfield(
                scalar_in,
                z_ind=i,
                z_level=float(bounds_dict["z"][0] + z_step * i),
                cmin=min_val,
                cmax=max_val,
            )
        )

        frames_out.append(
            plot_sfield(
                scalar_out,
                z_ind=i,
                z_level=float(bounds_dict["z"][0] + z_step * i),
                cmin=min_val,
                cmax=max_val,
            )
        )

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "surface"}, {"type": "surface"}]],
        subplot_titles=("Scalar In", "Scalar Out"),
    )
    fig.frames = [
        go.Frame(data=[frames_in[i], frames_out[i]], name=str(i), traces=[0, 1])
        for i in range(shape_dict["z"])
    ]

    fig.add_trace(frame_init, row=1, col=1)
    fig.add_trace(frame_init_out, row=1, col=2)

    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[f.name], frame_args(0)],
                    "label": str(k),
                    "method": "animate",
                }
                for k, f in enumerate(fig.frames)
            ],
        }
    ]

    # Layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                nticks=10,
                range=bounds_dict["x"],
            ),
            yaxis=dict(
                nticks=10,
                range=bounds_dict["y"],
            ),
            zaxis=dict(
                nticks=10,
                range=bounds_dict["z"],
            ),
        ),
        scene2=dict(
            xaxis=dict(
                nticks=10,
                range=bounds_dict["x"],
            ),
            yaxis=dict(
                nticks=10,
                range=bounds_dict["y"],
            ),
            zaxis=dict(
                nticks=10,
                range=bounds_dict["z"],
            ),
        ),
        title="Slices in volumetric data",
        width=2000,
        height=1000,
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, frame_args(50)],
                        "label": "&#9654;",  # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;",  # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
        ],
        sliders=sliders,
    )

    fig.show()


def frame_args(duration):
    return {
        "frame": {"duration": duration},
        "mode": "immediate",
        "fromcurrent": True,
        "transition": {"duration": duration, "easing": "linear"},
    }


# plot isosurface of a given scalar field
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_list = []
    model_names = "model_single_datapoint.ckpt"

    for model_type in ["cnn"]:
        options = json.load(
            open("./options/options_{}_default.json".format(model_type))
        )
        model_temp = construct_model(model_type, options)
        model_temp.load_model("./log_version_{}_1/{}".format(model_type, model_names))
        model_temp.to(device)
        model_list.append(deepcopy(model_temp))

    root = "../../data/augment_test/"

    dataset_single = FieldDataset(
        root,
        transform=False,
        augmentation=False,
        standardize=False,
        lower_filter=True,
        log_scale=True,
        min_max_scale=False,
        wrangle_outliers=False,
        scalar=False,
        device=device,
    )

    show_in_out_plots(dataset_single[0], model_list[0], device)


main()
