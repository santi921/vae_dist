import os 
import numpy as np 


def split_and_filter(mat, cutoff = 95, min_max = True, std_mean = False):
    """
    Filters some noise in the data by setting all values below a certain percentile to 0.
    Also normalizes the data by either min-max or standard deviation and mean.
    Takes 
        mat: 4D numpy array of shape (x, y, z, 3) where x, y, z are the number of steps in each direction and 3 is the number of components
        cutoff: percentile to filter out
        min_max: boolean to normalize by min-max
        std_mean: boolean to normalize by standard deviation and mean
    Returns
        u, v, w: 1D numpy arrays of shape (x*y*z) where x, y, z are the number of steps in each direction
    """
    arr_mean, arr_std, arr_min, arr_max  = np.mean(mat), np.std(mat), np.min(mat), np.max(mat)
    if(min_max):
        mat = (mat - arr_min) / (arr_max - arr_min + 10e-10)

    if(std_mean):
        mat = (mat - arr_mean) / (arr_std)
   
    try:
        u = mat[0][:,:,:,0].flatten()
        v = mat[0][:,:,:,1].flatten()
        w = mat[0][:,:,:,2].flatten()
    except:
        u = mat[:,:,:,0].flatten()
        v = mat[:,:,:,1].flatten()
        w = mat[:,:,:,2].flatten()

    component_distro = [np.sqrt(u[ind]**2 + v[ind]**2 + w[ind]**2) for ind in range(len(u))]
    cutoff = np.percentile(component_distro, cutoff)

    for ind, i in enumerate(component_distro): 
        if (i < cutoff): 
            u[ind], v[ind], w[ind] = 0,0,0  

    u = np.around(u, decimals=2)
    v = np.around(v, decimals=2)
    w = np.around(w, decimals=2)

    return u, v, w


def pull_fields(root): 
    # mat pull on every file in root ending in .dat
    # returns a list of 4D numpy arrays of shape (x, y, z, 3) where x, y, z are the number of steps in each direction and 3 is the number of components
    mats = []
    for file in os.listdir(root):
        if file.endswith(".dat"):
            mats.append(mat_pull(root + file))
    mats = np.array(mats)
    return mats


def mat_pull(file):

    with open(file) as f: 
        lines = f.readlines()

    steps_x = 2 * int(lines[0].split()[2]) + 1
    steps_y = 2 * int(lines[0].split()[3]) + 1
    steps_z = 2 * int(lines[0].split()[4][:-1]) + 1
    mat = np.zeros((steps_x, steps_y, steps_z, 3))

    for ind, i in enumerate(lines[7:]):
        line_split = i.split()
        #print(i)
        mat[int(ind/(steps_z*steps_y)), int(ind/steps_z % steps_y), ind%steps_z, 0] = float(line_split[-3])
        mat[int(ind/(steps_z*steps_y)), int(ind/steps_z % steps_y), ind%steps_z, 1] = float(line_split[-2])
        mat[int(ind/(steps_z*steps_y)), int(ind/steps_z % steps_y), ind%steps_z, 2] = float(line_split[-1])
    return mat  

