import os 
import numpy as np 

def filter(mat, cutoff_low_percentile = 50, cutoff_high_percentile=False, dim = 3):
    # mat is 5D numpy array of shape (1, 3, x, y, z) where x, y, z are the number of steps in each direction and 3 is the number of components
    # throw assertion error if dim is not 1 or 3
    assert dim == 1 or dim == 3, "Dimension must be 1 or 3"

    if dim == 3:
        u = mat[0,0,:,:,:].reshape(-1)
        v = mat[0,1,:,:,:].reshape(-1)
        w = mat[0,2,:,:,:].reshape(-1)

        component_distro = np.sqrt((mat**2).sum(axis=1))
        cutoff_low = np.percentile(component_distro, cutoff_low_percentile)
        cutoff_high = np.percentile(component_distro, cutoff_high_percentile)


        for ind, i in enumerate(component_distro.flatten()): 
            if (i < cutoff_low):
                u[ind], v[ind], w[ind] = 0,0,0  
            
            if(cutoff_high_percentile):
                if (i > cutoff_high):
                    u[ind], v[ind], w[ind] = 0,0,0
        
        u = np.around(u, decimals=2)
        v = np.around(v, decimals=2)
        w = np.around(w, decimals=2)
        new_mat = np.zeros(mat.shape)
        new_mat[0,0,:,:,:] = u.reshape(mat.shape[2], mat.shape[3], mat.shape[4])
        new_mat[0,1,:,:,:] = v.reshape(mat.shape[2], mat.shape[3], mat.shape[4])
        new_mat[0,2,:,:,:] = w.reshape(mat.shape[2], mat.shape[3], mat.shape[4])
    
    if dim == 1: 
        mag = mat[0,0,:,:,:].reshape(-1)
        component_distro = np.sqrt((mat**2).sum(axis=1))
        cutoff_low = np.percentile(component_distro, cutoff_low_percentile)
        cutoff_high = np.percentile(component_distro, cutoff_high_percentile)
        # filter out 
        for ind, i in enumerate(component_distro.flatten()):
            if (i < cutoff_low):
                mag[ind] = 0
            if(cutoff_high_percentile):
                if (i > cutoff_high):
                    mag[ind] = 0
        # put back into matrix form 
        new_mat = np.zeros(mat.shape)
        new_mat[0,0,:,:,:] = mag.reshape(mat.shape[2], mat.shape[3], mat.shape[4])

    return new_mat


def split_and_filter(mat, cutoff_low_percentile = 95, cutoff_high_percentile= 99.99999):
    """
    Filters some noise in the data by setting all values below a certain percentile to 0.
    Also normalizes the data by either min-max or standard deviation and mean.
    Takes 
        mat: 5D numpy array of shape (1, 3, x, y, z) where x, y, z are the number of steps in each direction and 3 is the number of components
        cutoff: percentile to filter out
        min_max: boolean to normalize by min-max
        std_mean: boolean to normalize by standard deviation and mean
    Returns
        u, v, w: 1D numpy arrays of shape (x*y*z) where x, y, z are the number of steps in each direction
    """
    u = mat[0,0,:,:,:].reshape(-1)
    v = mat[0,1,:,:,:].reshape(-1)
    w = mat[0,2,:,:,:].reshape(-1)

    component_distro = np.sqrt((mat**2).sum(axis=1))
    cutoff_low = np.percentile(component_distro, cutoff_low_percentile)
    cutoff_high = np.percentile(component_distro, cutoff_high_percentile)


    for ind, i in enumerate(component_distro.flatten()): 
        if (i < cutoff_low):
            u[ind], v[ind], w[ind] = 0,0,0  
        
        if(cutoff_high_percentile):
            if (i > cutoff_high):
                u[ind], v[ind], w[ind] = 0,0,0
    
    u = np.around(u, decimals=2)
    v = np.around(v, decimals=2)
    w = np.around(w, decimals=2)

    return u, v, w


def pull_fields(root, ret_names = False): 
    # mat pull on every file in root ending in .dat
    # returns a list of 4D numpy arrays of shape (x, y, z, 3) where x, y, z are the number of steps in each direction and 3 is the number of components
    mats = []
    names = []
    for file in os.listdir(root):
        if file.endswith(".dat"):
            mat = mat_pull(root + file, meta_data=False)
            #shape = [meta["steps_x"], meta["steps_y"], meta["steps_z"]]
            mats.append(mat)
            names.append(file)
    meta = mat_pull(root + file, meta_data=True)
    shape = [meta["steps_x"], meta["steps_y"], meta["steps_z"]]
    mats = np.array(mats)
    if ret_names: 
        return mats, shape, names
    return mats, shape


def mat_pull(file, meta_data=False):

    with open(file) as f: 
        lines = f.readlines()

    if meta_data:
        steps_x = 2 * int(lines[0].split()[2]) + 1
        steps_y = 2 * int(lines[0].split()[3]) + 1
        steps_z = 2 * int(lines[0].split()[4][:-1]) + 1
        x_size = float(lines[0].split()[-3])
        y_size = float(lines[0].split()[-2])
        z_size = float(lines[0].split()[-1])


        meta_dict = {
            "steps_x": steps_x,
            "steps_y": steps_y,
            "steps_z": steps_z,
            "step_size_x": np.round(x_size / float(lines[0].split()[2]), 4),
            "step_size_y": np.round(y_size / float(lines[0].split()[3]), 4),
            "step_size_z": np.round(z_size / float(lines[0].split()[4][:-1]), 4),
            "first_line": lines[0]
        }

        return meta_dict
    
    else: 
        steps_x = 2 * int(lines[0].split()[2]) + 1
        steps_y = 2 * int(lines[0].split()[3]) + 1
        steps_z = 2 * int(lines[0].split()[4][:-1]) + 1
        mat = np.zeros((steps_x, steps_y, steps_z, 3))

        # gap_x = round(np.abs(float(lines[steps_x*steps_y + 7].split()[0]) - float(lines[7].split()[0])), 4)
        # gap_y = round(np.abs(float(lines[steps_x+8].split()[1]) - float(lines[7].split()[1])), 4)
        # gap_z = round(np.abs(float(lines[8].split()[2]) - float(lines[7].split()[2])), 4)

        for ind, i in enumerate(lines[7:]):
            line_split = i.split()
            # print(i)
            mat[
                int(ind / (steps_z * steps_y)),
                int(ind / steps_z % steps_y),
                ind % steps_z,
                0,
            ] = float(line_split[-3])
            mat[
                int(ind / (steps_z * steps_y)),
                int(ind / steps_z % steps_y),
                ind % steps_z,
                1,
            ] = float(line_split[-2])
            mat[
                int(ind / (steps_z * steps_y)),
                int(ind / steps_z % steps_y),
                ind % steps_z,
                2,
            ] = float(line_split[-1])
        
        return mat



def helmholtz_hodge_decomp_approx(mat):
    Vf = mat
    shape = Vf.shape
    
    # reshape to (x, y, z, 3)
    Vf = Vf.reshape(shape[1], shape[2], shape[3], 3)
    NX, NY, NZ = Vf[:,:,:,1].shape
    Vfx = Vf[:,:,:,0]
    Vfy = Vf[:,:,:,1]
    Vfz = Vf[:,:,:,2]

    vx_f = np.fft.fftn(Vfx)
    vy_f = np.fft.fftn(Vfy)
    vz_f = np.fft.fftn(Vfz)
    
    kx = np.fft.fftfreq(NX).reshape(NX,1,1)
    ky = np.fft.fftfreq(NY).reshape(NY,1)
    kz = np.fft.fftfreq(NZ)
    k2 = kx**2 + ky**2 + kz**2
    k2[0,0,0] = 1. # to avoid inf. we do not care about the k=0 component

    div_Vf_f = (vx_f * kx +  vy_f * ky + vz_f * kz) #* 1j
    V_compressive_overk = div_Vf_f / k2
    V_compressive_x = np.fft.ifftn(V_compressive_overk * kx) #[:,np.newaxis,np.newaxis])
    V_compressive_y = np.fft.ifftn(V_compressive_overk * ky)
    V_compressive_z = np.fft.ifftn(V_compressive_overk * kz)

    V_solenoidal_x = Vfx - V_compressive_x
    V_solenoidal_y = Vfy - V_compressive_y
    V_solenoidal_z = Vfz - V_compressive_z

    # check if the solenoidal part really divergence-free
    divVs = np.fft.ifftn((np.fft.fftn(V_solenoidal_x) * kx + np.fft.fftn(V_solenoidal_y) * ky + np.fft.fftn(V_solenoidal_z) * kz) * 1j * 2. * np.pi)
    
    # reshape to (x, y, z, 3)
    sol_x = V_solenoidal_x.real.reshape(shape[1], shape[2], shape[3])
    sol_y = V_solenoidal_y.real.reshape(shape[1], shape[2], shape[3])
    sol_z = V_solenoidal_z.real.reshape(shape[1], shape[2], shape[3])
    V_compressive_x = V_compressive_x.real.reshape(shape[1], shape[2], shape[3])
    V_compressive_y = V_compressive_y.real.reshape(shape[1], shape[2], shape[3])
    V_compressive_z = V_compressive_z.real.reshape(shape[1], shape[2], shape[3])
    
    solenoidal = {"x": sol_x, "y":sol_y, "z": sol_z}
    compressize = {"x": V_compressive_x, "y":V_compressive_y, "z": V_compressive_z}
    
    # reshape as matrix
    out_mat = np.zeros((3, shape[1], shape[2], shape[3]))
    out_mat_compressize = np.zeros((3, shape[1], shape[2], shape[3]))

    # put solenoidal and compressize in the same matrix
    out_mat[0,:,:,:] = solenoidal["x"]
    out_mat[1,:,:,:] = solenoidal["y"]
    out_mat[2,:,:,:] = solenoidal["z"]
    out_mat_compressize[0,:,:,:] = compressize["x"]
    out_mat_compressize[1,:,:,:] = compressize["y"]
    out_mat_compressize[2,:,:,:] = compressize["z"]
    return np.array(out_mat)