import numpy as np
import networkx as nx
from plotly.offline import iplot
import plotly.graph_objects as go
from typing import List, Tuple
import pytorch_lightning as pl

from vae_dist.data.rdkit import xyz2AC_vdW, pdb_to_xyz, get_AC
from vae_dist.dataset.fields import split_and_filter, pull_fields
from vae_dist.data.dictionaries import *
from vae_dist.dataset.dataset import FieldDataset

def filter_xyz_by_distance(
        xyz: np.ndarray, 
        center: List = [0, 0, 0], 
        distance: float = 5.0):
    
    xyz = np.array(xyz, dtype=float)
    center = np.array(center, dtype=float)
    return xyz[np.linalg.norm(xyz - center, axis=1) < distance]


def filter_other_by_distance(
        xyz: np.ndarray, 
        other: List, 
        center: List = [0, 0, 0], 
        distance: float = 5):
    xyz = np.array(xyz, dtype=float)
    center = np.array(center, dtype=float)
    mask = np.linalg.norm(xyz - center, axis=1) < distance
    mask = [i for i in range(len(mask)) if mask[i]]
    return [other[i] for i in mask]


def connectivity_to_list_of_bonds(
        connectivity_mat: np.ndarray):
    bonds = []
    for i in range(len(connectivity_mat)):
        for j in range(i + 1, len(connectivity_mat)):
            if connectivity_mat[i][j] > 0:
                bonds.append([i, j])
    return bonds


def get_nodes_and_edges_from_pdb(
    file: str = "../../data/pdbs_processed/1a4e.pdb", 
    distance_filter: float = 5.0
):
    xyz, charge, atom = pdb_to_xyz(file)
    filtered_xyz = filter_xyz_by_distance(
        xyz, center=[130.581, 41.541, 38.350], distance=distance_filter
    )
    # filtered_charge = filter_other_by_distance(xyz, charge, center = [130.581,  41.541,  38.350], distance = distance_filter)
    filtered_atom = filter_other_by_distance(
        xyz, atom, center=[130.581, 41.541, 38.350], distance=distance_filter
    )
    connectivity_mat, rdkit_mol = xyz2AC_vdW(filtered_atom, filtered_xyz)
    connectivity_mat = get_AC(filtered_atom, filtered_xyz, covalent_factor=1.3)

    bonds = connectivity_to_list_of_bonds(connectivity_mat)
    return filtered_atom, bonds, filtered_xyz


def shift_and_rotate(
    xyz_list: list, 
    center: list = [0, 0, 0], 
    x_axis: list = [1, 0, 0], 
    y_axis: list = [0, 1, 0], 
    z_axis: list = [0, 0, 1]
):
    for i in range(len(xyz_list)):
        xyz_list[i] = xyz_list[i] - center
        xyz_list[i] = np.array(
            [
                np.dot(xyz_list[i], x_axis),
                np.dot(xyz_list[i], y_axis),
                np.dot(xyz_list[i], z_axis),
            ]
        )
    return xyz_list


def get_latent_space(
    model: pl.LightningModule, 
    dataset: FieldDataset, 
    comp:list=[0, 2], 
    latent_dim:int=10, 
    field_dims:Tuple=(3, 21, 21, 21)
):
    latent_space = []
    # convert load to numpy
    print("Total number of fields: ", len(dataset))
    for ind in range(len(dataset)):
        field = dataset[ind].reshape(
            1, field_dims[0], field_dims[1], field_dims[2], field_dims[3]
        )
        latent = model.latent(field)
        # print(latent.shape)
        latent_space.append(latent.detach().cpu().numpy())
    if comp == []:
        return np.array(latent_space).reshape(-1, latent_dim)
    else:
        return np.array(latent_space).reshape(-1, latent_dim)[:, comp]


def plot_vfield(
    mat,
    cutoff_low: float=95,
    cutoff_high: float=99.999,
    min_max: bool=True,
    scale: int=10,
    bounds_dict: dict={"x": [-3, 3.3], "y": [-3, 3.3], "z": [-3, 3.3]},
    steps_dict: dict={"x": 0.3, "y": 0.3, "z": 0.3},
):
    # mat has shape (1, 3, 21, 21, 21)
    x = mat
    print(x.min(), x.max())
    u_1, v_1, w_1 = split_and_filter(
        x, cutoff_low_percentile=cutoff_low, cutoff_high_percentile=cutoff_high
    )
    x, y, z = np.meshgrid(
        np.arange(bounds_dict["x"][0], bounds_dict["x"][1], steps_dict["x"]),
        np.arange(bounds_dict["y"][0], bounds_dict["y"][1], steps_dict["y"]),
        np.arange(bounds_dict["z"][0], bounds_dict["z"][1], steps_dict["z"]),
    )

    # max value of each dimension
    max_u = np.max(u_1)
    max_v = np.max(v_1)
    max_w = np.max(w_1)

    # print(max_u, max_v, max_w)
    # print(a.shape, b.shape, c.shape, u_1.shape, v_1.shape, w_1.shape)
    cones = go.Cone(
        x=x.flatten(), y=y.flatten(), z=z.flatten(), u=u_1, v=v_1, w=w_1, sizeref=scale
    )

    components = {"u": u_1, "v": v_1, "w": w_1}
    return cones, components
