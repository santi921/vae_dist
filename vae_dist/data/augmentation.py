import numpy as np 
import torch 
from torch.functional import F

class Augment(object):
    def __init__(self, xy = True, z = False, rot = 0):
        self.xy = xy
        self.z = z
        self.rot = rot 
        

    def __call__(self, mat):
        return self.aug_all(mat)


    def aug_all(self, mat):
        full_aug  = []
        for _, i in enumerate(mat):
            x_aug = augment_mat_field(i, self.xy, self.z, self.rot)
            [full_aug.append(j) for j in x_aug]
        print(len(full_aug))
        return np.array(full_aug)    


def augment_mat_field(mat, xy, z, rot = 0):
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

        if(rot>0):
            if rot == 4:
                # add 90, 180, 270 rotations
                aug_mat.append(np.rot90(mat, axes = (0,1)))
                aug_mat.append(np.rot90(mat, axes = (0,1), k = 2))
                aug_mat.append(np.rot90(mat, axes = (0,1), k = 3))
            elif rot == 2:
                # add 180
                aug_mat.append(np.rot90(mat, axes = (0,1), k = 2))
        return aug_mat 



class RandomFlips(torch.nn.Module):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p_x=0.5, p_y=0.5, p_z=0.5, x=True, y=True, z=True):
        super().__init__()
        self.p_x = p_x
        self.p_y = p_y
        self.p_z = p_z
        self.x = x
        self.y = y
        self.z = z


    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if self.x:
            if torch.rand(1) < self.p_x:
                img = F.flip(img, dims=[0])
                img[0,:,:,:] = -1*img[0,:,:,:]
        if self.y:        
            if torch.rand(1) < self.p_y:
                img = F.flip(img, dims=[1])
                #return F.flip(img, dims=[1])
                img[1,:,:,:] = -1*img[1,:,:,:]
        if self.z:    
            if torch.rand(1) < self.p_z:
                img = F.flip(img, dims=[2])
                # flip z axis of vector field 
                img[2,:,:,:] = -1*img[2,:,:,:]
        
        return img
        #return mat

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"