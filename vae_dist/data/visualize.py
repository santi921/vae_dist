import numpy as np 
import plotly.graph_objects as go
import networkx as nx
from vae_dist.data.rdkit import xyz2AC_vdW, pdb_to_xyz, get_AC
from vae_dist.dataset.fields import split_and_filter, pull_fields, pca 


atom_int_dict = {
    'H': 1,
    'C': 6,
    'N': 7,
    'O': 8,
    'F': 9,
    'P': 15,
    'S': 16,
    'Cl': 17,
    'Br': 35,
    'Fe': 26, 
    'FE': 26, 
    'I': 53
}


int_atom_dict = {
    1: 'H',
    6: 'C',
    7: 'N',
    8: 'O',
    9: 'F',
    15: 'P',
    16: 'S',
    17: 'Cl',
    35: 'Br',
    26: 'Fe',
    53: 'I'
}


atomic_size = {
    'H': 0.5,
    'C': 1.7,
    'N': 1.55,
    'O': 1.52,
    'F': 1.47,
    'P': 1.80,
    'S': 1.80,
    'Cl': 1.75,
    'Br': 1.85,
    'Fe': 1.80,
    'I': 1.98
}


atom_colors = {
    'H': 'white',
    'C': 'black',
    'N': 'blue',
    'O': 'red',
    'F': 'orange',
    'P': 'green',
    'S': 'yellow',
    'Cl': 'green',
    'Br': 'brown',
    'Fe': 'orange',
    'I': 'purple'
}


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


def get_cones_viz_from_pca(vector_scale = 3, components = 10, dir_fields = "../../data/cpet/"): 

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
                        np.arange(-3, 3.3, 0.3),
                        np.arange(-3, 3.3, 0.3),
                        np.arange(-3, 3.3, 0.3)
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
        

#trace_edges, trace_nodes = plot_nodes_edge()