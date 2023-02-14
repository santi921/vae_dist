#!export CUDA_VISIBLE_DEVICES=0

import torch, os
import numpy as np 

import matplotlib.pyplot as plt

from copy import deepcopy
from plotly.offline import init_notebook_mode

from vae_dist.dataset.dataset import FieldDataset
from vae_dist.data.visualize import get_latent_space
from vae_dist.core.training_utils import construct_model 
from vae_dist.dataset.fields import split_and_filter
import json 

import plotly.graph_objects as go
from plotly.subplots import make_subplots
# plot isosurface of a given scalar field 

init_notebook_mode(connected=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model_list = []
model_names = "model_single_datapoint.ckpt"

for model_type in ['auto']:
    options = json.load(open('./options/options_{}_default.json'.format(model_type)))
    model_temp = construct_model(model_type, options)
    model_temp.load_model("./log_version_{}_1/{}".format(model_type, model_names))
    model_temp.to(device)
    model_list.append(deepcopy(model_temp))

root = '../../data/single_field/'

dataset_single = FieldDataset(
    root, 
    transform=False, 
    augmentation=False,
    standardize=True,
    lower_filter=True,
    log_scale=True, 
    scalar=True,
    device=device
    )

dataset_loader_full = torch.utils.data.DataLoader(
                dataset_single, 
                batch_size=1,
                shuffle=True,
                num_workers=0
            )

import plotly.graph_objects as go
import numpy as np

def plot_sfield(scalar_field, z_level=0.0):
    # plot iso surface of a given scalar field
    # z_level: iso surface level
    # sweep x, y and get z as value of scalar field at z level
    x = np.linspace(0, 1, 21)
    y = np.linspace(0, 1, 21)
    z = scalar_field[:, : , 11]

    iso =go.Surface(x=x, y=y, z=z, showscale=False)
    return iso


def show_in_out_plots(in_field, model):
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'surface'}]])

    model = model_list[0]
    x = in_field.reshape(1, 1, 21, 21, 21).to(device)
    x_out = model.forward(x)
    x_out = x_out.to('cpu').detach().numpy().reshape(1, 21, 21, 21)

    field_in =  plot_sfield(x.cpu().numpy())
    field_out = plot_sfield(x_out)
    #plot surface in subplots
    # add traces
    fig.add_trace(field_in, 1, 1)
    fig.add_trace(field_out, 1, 2)
    
    fig.update_layout(
    title_text='Differnt 3D subplots with different color scale',
    height=900,
    width=900)
    fig.show()
show_in_out_plots(dataset_single[0], model_list[0])

