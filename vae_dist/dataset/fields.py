import os 
import numpy as np 

def filter(mat, cutoff_low_percentile = 50, cutoff_high_percentile=False):
    try:
        u = mat[0][:,:,:,0].flatten()
        v = mat[0][:,:,:,1].flatten()
        w = mat[0][:,:,:,2].flatten()
    except:
        u = mat[:,:,:,0].flatten()
        v = mat[:,:,:,1].flatten()
        w = mat[:,:,:,2].flatten()

    component_distro = [np.sqrt(u[ind]**2 + v[ind]**2 + w[ind]**2) for ind in range(len(u))]
    cutoff_low = np.percentile(component_distro, cutoff_low_percentile)
    cutoff_high = np.percentile(component_distro, cutoff_high_percentile)


    for ind, i in enumerate(component_distro): 
        if (i < cutoff_low):
            u[ind], v[ind], w[ind] = 0,0,0  
        
        if(cutoff_high_percentile):
            if (i > cutoff_high):
                u[ind], v[ind], w[ind] = 0,0,0
    
    u = np.around(u, decimals=2)
    v = np.around(v, decimals=2)
    w = np.around(w, decimals=2)
    # put back into matrix form 
    new_mat = np.zeros(mat.shape)
    new_mat[:,:,:,0] = u.reshape(mat.shape[0], mat.shape[1], mat.shape[2])
    new_mat[:,:,:,1] = v.reshape(mat.shape[0], mat.shape[1], mat.shape[2])
    new_mat[:,:,:,2] = w.reshape(mat.shape[0], mat.shape[1], mat.shape[2])

    return new_mat


def split_and_filter(mat, cutoff_low = 95, cutoff_high= 99.99999, min_max = True, std_mean = False):
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
    cutoff_low = np.percentile(component_distro, cutoff_low)
    cutoff_high = np.percentile(component_distro, cutoff_high)

    for ind, i in enumerate(component_distro): 
        #if (i < cutoff): 
        if (i < cutoff_low or i > cutoff_high):
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
            mat, shape = mat_pull(root + file)
            mats.append(mat)
            names.append(file)
    mats = np.array(mats)
    if ret_names: 
        return mats, shape, names
    return mats, shape


def mat_pull(file):

    with open(file) as f: 
        lines = f.readlines()

    steps_x = 2 * int(lines[0].split()[2]) + 1
    steps_y = 2 * int(lines[0].split()[3]) + 1
    steps_z = 2 * int(lines[0].split()[4][:-1]) + 1
    mat = np.zeros((steps_x, steps_y, steps_z, 3))

    for ind, i in enumerate(lines[7:]):
        line_split = i.split()
        # warn if number is larger than 10 or smaller than -10
        #if (float(line_split[-3]) > 100 or float(line_split[-3]) < -100):
        #    print("Warning: value of {} is larger than 100 or smaller than -100".format(float(line_split[-3])))
        #    print(file) 
        #if (float(line_split[-2]) > 100 or float(line_split[-2]) < -100):
        #    print("Warning: value of {} is larger than 100 or smaller than -100".format(float(line_split[-2])))
        #    print(file) 
        #if (float(line_split[-1]) > 100 or float(line_split[-1]) < -100):
        #    print("Warning: value of {} is larger than 100 or smaller than -100".format(float(line_split[-1])))
        #    print(file)    
        mat[int(ind/(steps_z*steps_y)), int(ind/steps_z % steps_y), ind%steps_z, 0] = float(line_split[-3])
        mat[int(ind/(steps_z*steps_y)), int(ind/steps_z % steps_y), ind%steps_z, 1] = float(line_split[-2])
        mat[int(ind/(steps_z*steps_y)), int(ind/steps_z % steps_y), ind%steps_z, 2] = float(line_split[-1])
    return mat, [steps_x, steps_y, steps_z]


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