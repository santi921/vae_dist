import numpy as np 
import networkx as nx
from plotly.offline import iplot
import plotly.graph_objects as go

from vae_dist.data.rdkit import xyz2AC_vdW, pdb_to_xyz, get_AC
from vae_dist.dataset.fields import split_and_filter, pull_fields 
from vae_dist.data.dictionaries import *

def filter_xyz_by_distance(xyz, center = [0,0,0], distance = 5):
    xyz = np.array(xyz, dtype = float)
    center = np.array(center, dtype = float)
    return xyz[np.linalg.norm(xyz - center, axis = 1) < distance]


def filter_other_by_distance(xyz, other, center = [0,0,0], distance = 5):
    xyz = np.array(xyz, dtype = float)
    center = np.array(center, dtype = float)
    mask = np.linalg.norm(xyz - center, axis = 1) < distance
    mask = [i for i in range(len(mask)) if mask[i]]
    return [other[i] for i in mask]


def connectivity_to_list_of_bonds(connectivity_mat):
    bonds = []
    for i in range(len(connectivity_mat)):
        for j in range(i+1, len(connectivity_mat)):
            if connectivity_mat[i][j] > 0:
                bonds.append([i,j])
    return bonds


def get_nodes_and_edges_from_pdb(file = '../../data/pdbs_processed/1a4e.pdb', distance_filter = 5.0):
    
    xyz, charge, atom = pdb_to_xyz(file)
    filtered_xyz = filter_xyz_by_distance(xyz, center = [130.581,  41.541,  38.350], distance = distance_filter)
    #filtered_charge = filter_other_by_distance(xyz, charge, center = [130.581,  41.541,  38.350], distance = distance_filter)
    filtered_atom = filter_other_by_distance(xyz, atom, center = [130.581,  41.541,  38.350], distance = distance_filter)
    connectivity_mat, rdkit_mol = xyz2AC_vdW(filtered_atom, filtered_xyz)
    connectivity_mat = get_AC(filtered_atom, filtered_xyz, covalent_factor=1.3)
     
    bonds = connectivity_to_list_of_bonds(connectivity_mat)
    return filtered_atom, bonds, filtered_xyz


def shift_and_rotate(xyz_list, center = [0,0,0], x_axis = [1,0,0], y_axis = [0,1,0], z_axis = [0,0,1]): 
    for i in range(len(xyz_list)):
        xyz_list[i] = xyz_list[i] - center
        xyz_list[i] = np.array([np.dot(xyz_list[i], x_axis), np.dot(xyz_list[i], y_axis), np.dot(xyz_list[i], z_axis)])
    return xyz_list


def plot_nodes_edge(file = "../../data/pdbs_processed/1a4e.pdb"): 
    
    G = nx.Graph()
    atom_list, bond_list, xyz_list = get_nodes_and_edges_from_pdb("../../data/pdbs_processed/1a4e.pdb", distance_filter= 8.0)
    
    NA_pos = [129.775,  39.761,  38.051]
    NB_pos = [130.581,  41.865,  36.409]
    NC_pos = [131.320,  43.348,  38.639]
    ND_pos = [130.469,  41.267,  40.273]
    Fe_pos = [130.581,  41.541,  38.350]
    center = np.mean([NA_pos, NB_pos, NC_pos, ND_pos], axis = 0)
    x_axis = np.array(NA_pos) - np.array(Fe_pos)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.array(NB_pos) - np.array(Fe_pos)
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = np.cross(y_axis, x_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)

    xyz_list = shift_and_rotate(
        xyz_list, 
        center = center, 
        x_axis = x_axis,
        y_axis = y_axis,
        z_axis = z_axis
    )


    for i in range(len(atom_list)):
        G.add_node(i, 
        xyz=xyz_list[i], 
        atom=atom_list[i]
        )
        
    for i in range(len(bond_list)):
        G.add_edge(
            bond_list[i][0], 
            bond_list[i][1]
            )


    edge_x, edge_y, edge_z = [], [], []
    node_x, node_y, node_z = [], [], []

    for edge in G.edges():
        x0, y0, z0  = G.nodes[edge[0]]['xyz']
        x1, y1, z1 = G.nodes[edge[1]]['xyz']
        edge_x+=[x0, x1, None]
        edge_y+=[y0, y1, None]
        edge_z+=[z0, z1, None]

    for node in G.nodes():
        x, y, z = G.nodes[node]['xyz']
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)

    scalar = 10
    color = [atom_colors[int_atom_dict[G.nodes[i]["atom"]]] for i in G.nodes]
    size = [scalar * atomic_size[int_atom_dict[G.nodes[i]["atom"]]] for i in G.nodes]

    trace_nodes = go.Scatter3d(x=node_x, 
                            y=node_y, 
                            z=node_z, 
                            mode="markers",
                            #hoverinfo='text',
                            #hover_name='title',
                            text = [int_atom_dict[i] for i in atom_list],
                            marker = dict(
                                    symbol='circle', 
                                    size=size,
                                    color=color,
                                    colorscale='Viridis',
                                    opacity= 0.8
                            ))
        
    trace_edges = go.Scatter3d(
        x=edge_x, 
        y=edge_y, 
        z=edge_z, 
        line=dict(width=1, color="#000000"), 
        hoverinfo='none', 
        mode='lines')


    return trace_edges, trace_nodes
    #fig.show()


def get_cones_viz_from_pca(
        vector_scale = 3, 
        components = 10, 
        dir_fields = "../../data/cpet/",
        bounds_dict = {'x': [-3, 3.3], 'y': [-3, 3.3], 'z': [-3, 3.3]},
        steps_dict = {"x": 0.3, "y": 0.3, "z": 0.3}
    ): 

    cones = []

    x= pull_fields(root_dir = dir_fields, verbose = False, write = False)
    arr_min, arr_max,  = np.min(x), np.max(x)
    #x = (x - arr_min) / np.abs(arr_max - arr_min + 0.1)
    # getting sign of every element
    x_sign = np.sign(x)
    # getting absolute value of every element
    x_abs = np.abs(x)
    # applying log1p
    x_log1p = np.log1p(x_abs)
    # getting sign back
    x = np.multiply(x_log1p, x_sign)
    
    x_untransformed = x
    x_pca, pca_obj = pca(x, verbose = True, pca_comps = components, write = False) 
    shape_mat = x.shape


    for ind,pca_comp in enumerate(pca_obj.components_):
        comp_vect_field = pca_comp.reshape(shape_mat[1], shape_mat[2], shape_mat[3], shape_mat[4])

        x, y, z = np.meshgrid(
                np.arange(bounds_dict['x'][0], bounds_dict['x'][1], steps_dict['x']),
                np.arange(bounds_dict['y'][0], bounds_dict['y'][1], steps_dict['y']),
                np.arange(bounds_dict['z'][0], bounds_dict['z'][1], steps_dict['z'])
                )

        u_1, v_1, w_1 = split_and_filter(
            comp_vect_field, 
            cutoff=95, 
            std_mean=True, 
            min_max=False
            )
        
        cones.append(go.Cone(
            x=x.flatten(), 
            y=y.flatten(), 
            z=z.flatten(), 
            u=u_1,
            v=v_1, 
            w=w_1,
            sizeref=vector_scale,
            opacity=0.99))
        
    return cones 
        

def get_latent_space(model, dataset, comp=[0, 2], latent_dim=10, field_dims=(3, 21, 21, 21)):
    latent_space = []
    # convert load to numpy 
    dataset_loader_np = []
    print("Total number of fields: ", len(dataset))
    for ind in range(len(dataset)):
        field=dataset[ind].reshape(1, field_dims[0], field_dims[1], field_dims[2], field_dims[3])
        latent = model.latent(field)
        #print(latent.shape)
        latent_space.append(latent.detach().cpu().numpy())
    if comp == []:
        return np.array(latent_space).reshape(-1, latent_dim)
    else: 
        return np.array(latent_space).reshape(-1, latent_dim)[:, comp]


def plot_vfield(
        mat, 
        cutoff_low = 95, 
        cutoff_high = 99.999, 
        min_max = True, 
        scale = 10, 
        bounds_dict = {'x': [-3, 3.3], 'y': [-3, 3.3], 'z': [-3, 3.3]},
        steps_dict = {"x": 0.3, "y": 0.3, "z": 0.3}
    ):
        # mat has shape (1, 3, 21, 21, 21)
        x = mat 
        print(x.min(), x.max())
        u_1, v_1, w_1 = split_and_filter(x, 
                                         cutoff_low_percentile = cutoff_low, 
                                         cutoff_high_percentile = cutoff_high)  
        x, y, z = np.meshgrid(
                np.arange(bounds_dict['x'][0], bounds_dict['x'][1], steps_dict['x']),
                np.arange(bounds_dict['y'][0], bounds_dict['y'][1], steps_dict['y']),
                np.arange(bounds_dict['z'][0], bounds_dict['z'][1], steps_dict['z'])
                )
        
        #max value of each dimension
        max_u = np.max(u_1)
        max_v = np.max(v_1)
        max_w = np.max(w_1)

        #print(max_u, max_v, max_w)
        #print(a.shape, b.shape, c.shape, u_1.shape, v_1.shape, w_1.shape)
        cones = go.Cone(
                x=x.flatten(), 
                y=y.flatten(), 
                z=z.flatten(), 
                u=u_1 ,
                v=v_1 , 
                w=w_1 ,
                sizeref=scale)
                
        components = {"u": u_1, "v": v_1, "w": w_1}
        return cones, components