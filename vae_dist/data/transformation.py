import numpy as np
from sklearn.decomposition import PCA
from vae_dist.dataset.fields import mat_pull, split_and_filter


class Transform(object):
    def __init__(self, pca=None, helm=None):
        self.pca = pca
        self.helm = helm
        self.shape = None

        if pca == True:
            self.pca = PCA(n_components=10)
        elif pca != None and pca != False:
            self.pca = pca
        else:
            self.pca = None

        if helm != None:
            self.helm = helm

    def __call__(self, mat):
        if self.pca != None:
            mat = self.pca_transform(mat)

        elif self.helm != None:
            mat = self.helmholtz_full(mat)
        else:
            mat = mat
        return mat

    def pca_transform(self, mat):
        if self.shape == None:
            self.shape = mat.shape

        mat_transform = mat.reshape(
            mat.shape[0], mat.shape[1] * mat.shape[2] * mat.shape[3] * mat.shape[4]
        )

        try:
            mat_transform = self.pca.transform(mat_transform)
        except:
            mat_transform = self.pca.fit_transform(mat_transform)

        return mat_transform

    def helmholtz_full(self, mat):
        matret = helmholtz_hodge_decomp_approx(mat[0])
        matret = matret.reshape(
            mat.shape[0], mat.shape[1], mat.shape[2], mat.shape[3], mat.shape[4]
        )
        return matret

    def unwrap_pca(self, mat):
        mat = self.pca.inverse_transform(mat)
        mat = mat.reshape(
            len(mat), self.shape[1], self.shape[2], self.shape[3], self.shape[4]
        )
        return mat


def helmholtz_hodge_decomp_approx(mat):
    # rotate indices to be in the order (x,y,z,component), with einsum
    mat = np.einsum("lijk->ijkl", mat)

    NX, NY, NZ = mat[:, :, :, 1].shape

    Vfx = mat[:, :, :, 0]
    Vfy = mat[:, :, :, 1]
    Vfz = mat[:, :, :, 2]

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

    solenoidal = {
        "x": V_solenoidal_x.real,
        "y": V_solenoidal_y.real,
        "z": V_solenoidal_z.real,
    }
    # compressize = {"x": V_compressive_x.real, "y":V_compressive_y.real, "z": V_compressive_z.real}

    # remap the solenoidal part to x, y, z, u, w, v
    solenoidal_mat = np.stack(
        [solenoidal["x"], solenoidal["y"], solenoidal["z"]], axis=-1
    )
    solenoidal_mat = np.einsum("ijkl->lijk", solenoidal_mat)

    return solenoidal_mat
