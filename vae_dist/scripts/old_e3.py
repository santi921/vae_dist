import torch
import torch.nn as nn
import numpy as np
import pyvista as pv
from scipy import ndimage
from e3nn import rs
from e3nn.image.convolution import Convolution

class Simple(nn.Module):
    def __init__(self, fuzzy_pixels):
        #super(Simple, self).__init__()
        super().__init__()

        size = 3
        self.f = torch.nn.Sequential( Convolution(Rs_in=[0], Rs_out=[0, 1], size=size,  steps=(1., 1., 1.), fuzzy_pixels=fuzzy_pixels),
        )
    def forward(self, x):
        out = self.f(x)
        return out

def rotate(inp, rotation_angle):    
    inp = ndimage.interpolation.rotate(inp,
                                       angle = rotation_angle,
                                       axes=(2,3),
                                       reshape=False,
                                       order=1,
                                       mode= 'nearest',#'constant',
                                       cval=0.0)
    return inp


def VoxPositions(dim, res):
    pos = np.empty(shape=(dim*dim*dim,3))
    l = 0
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                pos[l, 0] = i*res + res/2
                pos[l, 1] = j*res + res/2
                pos[l, 2] = k*res + res/2
                l = l + 1 
    return pos

if __name__=='__main__':

    dim = 16;  chans = 1
    torch.manual_seed(1000)

    # get input
    inp = torch.zeros(1, chans, dim, dim, dim)
    inp[:, :, dim//2, dim//2, dim//2] = 1.0
    inp[:, :, dim//2, 1, dim//2] = 1.0
    model = Simple(fuzzy_pixels=True)
    
    # plotting setup
    pl = pv.Plotter(shape=(1, 3))
    pl.open_movie('Simple.mp4')
    fs = 12 #text font size
    cents = VoxPositions(dim=dim-2, res = 1.0)

    for i in range(180):
        inpR = rotate(inp.numpy(), rotation_angle= i*2.0)
        inpR = torch.from_numpy(inpR).float()#.to('cuda')
        inpR = torch.einsum('tixyz->txyzi', inpR) #permute
                
        model.eval()
        outR = model(inpR)
        outR = torch.einsum('txyzi->tixyz', outR) #unpermute
        inpR = torch.einsum('txyzi->tixyz', inpR) #unpermute

        OutSca = outR[0, 0, :, :, :].detach().numpy()
        OutVec = outR[0, 1:4, :, :, :]
        OutVec = OutVec.flatten(1).detach().numpy()

        vec = np.array([OutVec[2], OutVec[0], OutVec[1]]).T
             
        text = "angle = " + str(2*i)
        pl.subplot(0, 0);  
        pl.add_text("Input", position = 'lower_left', font_size = fs)
        pl.add_text(text, position = 'upper_left', font_size = fs)
        pl.add_volume(inpR[0][0].detach().numpy(), cmap = "viridis_r",
                      opacity = "linear", show_scalar_bar=False)

        pl.subplot(0, 1);  
        pl.add_text("Out Vector", position = 'lower_left', font_size = fs)
        pl.add_arrows(cents, vec, mag=20, show_scalar_bar=False)
        
        pl.subplot(0, 2);  
        pl.add_text("Output", position = 'lower_left', font_size = fs)
        OutSca[OutSca < 0.01] = 0.0
        pl.add_volume(OutSca, cmap = "viridis_r", show_scalar_bar=False)
        #pl.add_axes()

        if i == 0 :
            pl.show(auto_close=False)
          
        pl.write_frame()
        pl.clear()

    pl.close()