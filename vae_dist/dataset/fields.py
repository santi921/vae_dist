import os
import numpy as np
import pandas as pd


def filter(mat, cutoff_low_percentile=50, cutoff_high_percentile=False, dim=3):
    # mat is 5D numpy array of shape (1, 3, x, y, z) where x, y, z are the number of steps in each direction and 3 is the number of components
    # throw assertion error if dim is not 1 or 3
    assert dim == 1 or dim == 3, "Dimension must be 1 or 3"

    if dim == 3:
        u = mat[0, 0, :, :, :].reshape(-1)
        v = mat[0, 1, :, :, :].reshape(-1)
        w = mat[0, 2, :, :, :].reshape(-1)

        component_distro = np.sqrt((mat**2).sum(axis=1))
        cutoff_low = np.percentile(component_distro, cutoff_low_percentile)
        cutoff_high = np.percentile(component_distro, cutoff_high_percentile)

        for ind, i in enumerate(component_distro.flatten()):
            if i < cutoff_low:
                u[ind], v[ind], w[ind] = 0, 0, 0

            if cutoff_high_percentile:
                if i > cutoff_high:
                    u[ind], v[ind], w[ind] = 0, 0, 0

        u = np.around(u, decimals=2)
        v = np.around(v, decimals=2)
        w = np.around(w, decimals=2)
        new_mat = np.zeros(mat.shape)
        new_mat[0, 0, :, :, :] = u.reshape(mat.shape[2], mat.shape[3], mat.shape[4])
        new_mat[0, 1, :, :, :] = v.reshape(mat.shape[2], mat.shape[3], mat.shape[4])
        new_mat[0, 2, :, :, :] = w.reshape(mat.shape[2], mat.shape[3], mat.shape[4])

    if dim == 1:
        mag = mat[0, 0, :, :, :].reshape(-1)
        component_distro = np.sqrt((mat**2).sum(axis=1))
        cutoff_low = np.percentile(component_distro, cutoff_low_percentile)
        cutoff_high = np.percentile(component_distro, cutoff_high_percentile)
        # filter out
        for ind, i in enumerate(component_distro.flatten()):
            if i < cutoff_low:
                mag[ind] = 0
            if cutoff_high_percentile:
                if i > cutoff_high:
                    mag[ind] = 0
        # put back into matrix form
        new_mat = np.zeros(mat.shape)
        new_mat[0, 0, :, :, :] = mag.reshape(mat.shape[2], mat.shape[3], mat.shape[4])

    return new_mat


def split_and_filter(
    mat,
    cutoff=95,
    min_max=True,
    std_mean=False,
    log1=False,
    unlog1=False,
    cos_center_scaling=False,
):
    mag = np.sqrt(np.sum(mat**2, axis=3))

    arr_mean = np.mean(mag)
    arr_std = np.std(mag)
    arr_min = np.min(mag)
    arr_max = np.max(mag)

    if log1:
        x_sign = np.sign(mat)
        # getting absolute value of every element
        x_abs = np.abs(mat)
        # applying log1p
        x_log1p = np.log1p(x_abs)
        # getting sign back
        mat = np.multiply(x_log1p, x_sign)

    if unlog1:
        print("invert log operation")
        x_sign = np.sign(mat)
        # getting absolute value of every element
        x_abs = np.abs(mat)
        # applying log1p
        x_unlog1p = np.expm1(x_abs)
        # getting sign back
        mat = np.multiply(x_unlog1p, x_sign)

    if min_max:
        mat = (mat - arr_min) / (arr_max - arr_min + 10e-10)

    if std_mean:
        mat = (mat - arr_mean) / (arr_std)

    if cos_center_scaling:
        shape = mat.shape
        center_ind = np.array(
            [np.ceil(shape[0] // 2), np.ceil(shape[1] // 2), np.ceil(shape[2] // 2)]
        )
        scale_mat = np.zeros_like(mat)
        max_dist = np.sqrt(np.sum(center_ind) ** 2)

        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    scale_mat[i, j, k] = 1 + 5 * np.cos(
                        np.sqrt(np.sum((center_ind - np.array([i, j, k])) ** 2))
                        / max_dist
                        * np.pi
                        / 2
                    )
        multiply = np.multiply(mat, scale_mat)
        mat = multiply

    try:
        u = mat[0][:, :, :, 0].flatten()
        v = mat[0][:, :, :, 1].flatten()
        w = mat[0][:, :, :, 2].flatten()
    except:
        u = mat[:, :, :, 0].flatten()
        v = mat[:, :, :, 1].flatten()
        w = mat[:, :, :, 2].flatten()

    component_distro = [
        np.sqrt(u[ind] ** 2 + v[ind] ** 2 + w[ind] ** 2) for ind in range(len(u))
    ]
    cutoff = np.percentile(component_distro, cutoff)

    for ind, i in enumerate(component_distro):
        if i < cutoff:
            u[ind], v[ind], w[ind] = 0, 0, 0

    u = np.around(u, decimals=2)
    v = np.around(v, decimals=2)
    w = np.around(w, decimals=2)

    return u, v, w


def pull_fields(root, ret_names=False, offset=0):
    # mat pull on every file in root ending in .dat
    # returns a list of 4D numpy arrays of shape (x, y, z, 3) where x, y, z are the number of steps in each direction and 3 is the number of components
    mats = []
    names = []
    for file in os.listdir(root):
        if file.endswith(".dat"):
            mat = mat_pull(root + file, meta_data=False, offset=offset)
            # shape = [meta["steps_x"], meta["steps_y"], meta["steps_z"]]
            mats.append(mat)
            names.append(file)
    meta = mat_pull(root + file, meta_data=True, offset=offset)
    shape = [meta["steps_x"], meta["steps_y"], meta["steps_z"]]
    mats = np.array(mats)
    if ret_names:
        return mats, shape, names
    return mats, shape


def pull_fields_w_label(root, supervised_file, ret_names=False, offset=0, label_ind=2):
    # mat pull on every file in root ending in .dat
    # returns a list of 4D numpy arrays of shape (x, y, z, 3) where x, y, z are the number of steps in each direction and 3 is the number of components
    mats = []
    names = []
    labels = []
    df = pd.read_csv(supervised_file)

    for file in os.listdir(root):
        if file.endswith(".dat"):
            protein = file.split(".")[0].split("_")[label_ind]
            label = df.loc[df["name"] == protein]["label"].values[0]
            if label == "Y":
                label = 0
            elif label == "H":
                label = 1
            else:
                label = 2

            mat = mat_pull(root + file, meta_data=False, offset=offset)
            mats.append(mat)
            labels.append(label)
            names.append(file)

    meta = mat_pull(root + file, meta_data=True, offset=offset)
    shape = [meta["steps_x"], meta["steps_y"], meta["steps_z"]]
    mats = np.array(mats)
    labels = np.array(labels)
    if ret_names:
        return mats, shape, names, labels

    return mats, shape, labels


def mat_pull(file, meta_data=False, offset=0):
    with open(file) as f:
        lines = f.readlines()

    if meta_data:
        steps_x = 2 * int(lines[0].split()[2]) + offset
        steps_y = 2 * int(lines[0].split()[3]) + offset
        steps_z = 2 * int(lines[0].split()[4][:-1]) + offset
        x_size = float(lines[0].split()[-3])
        y_size = float(lines[0].split()[-2])
        z_size = float(lines[0].split()[-1])

        meta_dict = {
            "steps_x": steps_x,
            "steps_y": steps_y,
            "steps_z": steps_z,
            "step_size_x": np.round(x_size / float(lines[0].split()[2]), 5),
            "step_size_y": np.round(y_size / float(lines[0].split()[3]), 5),
            "step_size_z": np.round(z_size / float(lines[0].split()[4][:-1]), 5),
            "first_line": lines[0],
        }

        return meta_dict

    else:
        steps_x = 2 * int(lines[0].split()[2]) + offset
        steps_y = 2 * int(lines[0].split()[3]) + offset
        steps_z = 2 * int(lines[0].split()[4][:-1]) + offset
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
    NX, NY, NZ = Vf[:, :, :, 1].shape
    Vfx = Vf[:, :, :, 0]
    Vfy = Vf[:, :, :, 1]
    Vfz = Vf[:, :, :, 2]

    vx_f = np.fft.fftn(Vfx)
    vy_f = np.fft.fftn(Vfy)
    vz_f = np.fft.fftn(Vfz)

    kx = np.fft.fftfreq(NX).reshape(NX, 1, 1)
    ky = np.fft.fftfreq(NY).reshape(NY, 1)
    kz = np.fft.fftfreq(NZ)
    k2 = kx**2 + ky**2 + kz**2
    k2[0, 0, 0] = 1.0  # to avoid inf. we do not care about the k=0 component

    div_Vf_f = vx_f * kx + vy_f * ky + vz_f * kz  # * 1j
    V_compressive_overk = div_Vf_f / k2
    V_compressive_x = np.fft.ifftn(
        V_compressive_overk * kx
    )  # [:,np.newaxis,np.newaxis])
    V_compressive_y = np.fft.ifftn(V_compressive_overk * ky)
    V_compressive_z = np.fft.ifftn(V_compressive_overk * kz)

    V_solenoidal_x = Vfx - V_compressive_x
    V_solenoidal_y = Vfy - V_compressive_y
    V_solenoidal_z = Vfz - V_compressive_z

    # check if the solenoidal part really divergence-free
    divVs = np.fft.ifftn(
        (
            np.fft.fftn(V_solenoidal_x) * kx
            + np.fft.fftn(V_solenoidal_y) * ky
            + np.fft.fftn(V_solenoidal_z) * kz
        )
        * 1j
        * 2.0
        * np.pi
    )

    # reshape to (x, y, z, 3)
    sol_x = V_solenoidal_x.real.reshape(shape[1], shape[2], shape[3])
    sol_y = V_solenoidal_y.real.reshape(shape[1], shape[2], shape[3])
    sol_z = V_solenoidal_z.real.reshape(shape[1], shape[2], shape[3])
    V_compressive_x = V_compressive_x.real.reshape(shape[1], shape[2], shape[3])
    V_compressive_y = V_compressive_y.real.reshape(shape[1], shape[2], shape[3])
    V_compressive_z = V_compressive_z.real.reshape(shape[1], shape[2], shape[3])

    solenoidal = {"x": sol_x, "y": sol_y, "z": sol_z}
    compressize = {"x": V_compressive_x, "y": V_compressive_y, "z": V_compressive_z}

    # reshape as matrix
    out_mat = np.zeros((3, shape[1], shape[2], shape[3]))
    out_mat_compressize = np.zeros((3, shape[1], shape[2], shape[3]))

    # put solenoidal and compressize in the same matrix
    out_mat[0, :, :, :] = solenoidal["x"]
    out_mat[1, :, :, :] = solenoidal["y"]
    out_mat[2, :, :, :] = solenoidal["z"]
    out_mat_compressize[0, :, :, :] = compressize["x"]
    out_mat_compressize[1, :, :, :] = compressize["y"]
    out_mat_compressize[2, :, :, :] = compressize["z"]
    return np.array(out_mat)
