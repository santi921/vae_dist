import numpy as np 


class Augment(object):
    def __init__(self, xy = True, z = False):
        self.xy = xy
        self.z = z

    def __call__(self, mat):
        return self.aug_all(mat)


    def aug_all(self, mat):
        full_aug  = []
        for _, i in enumerate(mat):
            x_aug = augment_mat_field(i, self.xy, self.z)
            [full_aug.append(j) for j in x_aug]
        return np.array(full_aug)    


def augment_mat_field(mat, xy, z):
        aug_mat = []
        
        if(xy):
            x_flip = np.array(np.flip(mat, axis = 0), dtype=float)
            y_flip = np.array(np.flip(mat, axis = 1), dtype=float)
            xy_flip = np.array(np.flip(np.flip(mat, axis = 1), axis = 0), dtype=float)

            x_flip[:,:,:,0] = -1*x_flip[:,:,:,0]
            y_flip[:,:,:,1] = -1*y_flip[:,:,:,1]
            xy_flip[:,:,:,0] = -1*xy_flip[:,:,:,0]
            xy_flip[:,:,:,1] = -1*xy_flip[:,:,:,1]
            
            aug_mat.append(mat)
            aug_mat.append(x_flip)
            aug_mat.append(y_flip)
            aug_mat.append(xy_flip)

        if(z):
            z_flip = np.array(np.flip(mat, axis = 2), dtype=float)
            xz_flip = np.array(np.flip(np.flip(mat, axis = 0), axis = 2), dtype=float)
            yz_flip = np.array(np.flip(np.flip(mat, axis = 1), axis = 2), dtype=float)
            xyz_flip = np.array(np.flip(np.flip(np.flip(mat, axis = 2), axis = 1), axis = 0), dtype=float)

            z_flip[:,:,:,0] = -1*z_flip[:,:,:,0]
            xz_flip[:,:,:,0] = -1*xz_flip[:,:,:,0]
            xz_flip[:,:,:,2] = -1*xz_flip[:,:,:,2]
            yz_flip[:,:,:,1] = -1*yz_flip[:,:,:,1]
            yz_flip[:,:,:,2] = -1*yz_flip[:,:,:,2]
            xyz_flip[:,:,:,0] = -1*xyz_flip[:,:,:,0]
            xyz_flip[:,:,:,1] = -1*xyz_flip[:,:,:,1]
            xyz_flip[:,:,:,2] = -1*xyz_flip[:,:,:,2]
            
            aug_mat.append(z_flip)
            aug_mat.append(xz_flip)
            aug_mat.append(yz_flip)
            aug_mat.append(xyz_flip)

            
        return aug_mat