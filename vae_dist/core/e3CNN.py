import torch
import torch.nn as nn
from torch.nn import functional as F

import pytorch_lightning as pl

from vae_dist.core.e3layers import (
    Down, 
    Up
)

from e3nn.o3 import Irreps, Linear
from e3nn.nn import Activation, FullyConnectedNet


class e3CNN(pl.LightningModule):
    def __init__(
        self,
        irreps_in,
        irreps_out,
        n, 
        n_blocks_down, 
        n_blocks_up,
        hidden_dim,
        stride,
        device,
        learning_rate=1e-4,
        latent_dim = 4,
        dropout_prob=0.1,
        cutoff=False,
        kernel_size=5,
        down_op='maxpool3d',
        batch_norm='instance',
        lmax=2,
        n_downsample=2,
        equivariance='SO3',
        bias = True,
        scalar_upsampling=False,
        scale=2,
        **kwargs
    ):
        super().__init__()

        modules_enc, modules_dec = [], []
        assert batch_norm in ['None','batch','instance'], "batch_norm needs to be 'batch', 'instance', or 'None'"
        assert down_op in ['maxpool3d','average','lowpass'], "down_op needs to be 'maxpool3d', 'average', or 'lowpass'"
    
        self.n_downsample = n_downsample
        latent_irreps = Irreps(f"{hidden_dim}x0e")
        if down_op == 'lowpass':
            up_op = 'lowpass'
            self.odd_resize = True

        else:
            up_op = 'upsample'
            self.odd_resize = False

        if equivariance == 'SO3':
            activation = [torch.relu]
            irreps_sh = Irreps.spherical_harmonics(lmax, 1)
            ne = n
            no = 0
        elif equivariance == 'O3':
            activation = [torch.relu,torch.tanh]
            irreps_sh = Irreps.spherical_harmonics(lmax, -1)
            ne = n
            no = n

        self.learning_rate = learning_rate
        params = {
            'irreps_in': irreps_in,
            'irreps_out': irreps_out,
            'irreps_latent': latent_irreps,
            'down_op': down_op,
            'up_op': up_op,
            'kernel_size': kernel_size,
            'stride': stride,
            'bias': bias,
            'latent_dim': latent_dim,
            'hidden_dim': hidden_dim,
            'activation': activation,
            'batch_norm': batch_norm,
            'device': device,
            'kwargs': kwargs,
            'learning_rate': learning_rate,
            'dropout_prob': dropout_prob,
            'cutoff': cutoff,
            'scalar_upsampling': scalar_upsampling,
            'irreps_sh': irreps_sh,
            'ne': ne,
            'no': no,
            'n_downsample': n_downsample,
            'n_blocks_down': n_blocks_down,
            'n_blocks_up': n_blocks_up,
            'lmax': lmax,
            'irreps_sh': irreps_sh,
            'n': n,
            'equivariance': equivariance,
            'scale': scale

        }

        self.hparams.update(params)
        self.save_hyperparameters()

        self.down = Down(
            n_blocks_down=n_downsample,
            activation=activation,
            irreps_sh=irreps_sh,
            ne=ne,
            no=no,
            BN=batch_norm,
            input=irreps_in,
            kernel_size=kernel_size,
            down_op=down_op,
            scale=scale,
            stride=stride,
            drop_prob=dropout_prob,
            cutoff=cutoff)

        self.latent = Linear(
            self.up.up_blocks[-1].irreps_out, 
            latent_irreps,
            activation = torch.ReLU)
    
        ne *= 2**(n_downsample-1)
        no *= 2**(n_downsample-1)

        self.up = Up(
            n_blocks_up=n_blocks_up,
            activation=activation,
            irreps_sh=irreps_sh,
            ne=ne,
            no=no,
            BN=batch_norm,
            downblock_irreps=latent_irreps,
            kernel_size=kernel_size,
            up_op=up_op,
            scale=scale,
            stride=stride,
            dropout_prob=dropout_prob,
            scalar_upsampling=scalar_upsampling,
            cutoff=cutoff
        )
        out_layer_dim = [self.up.up_blocks[-1].irreps_out, irreps_out]
        self.out = FullyConnectedNet(
            out_layer_dim,
            out_act=True,
            activation=torch.ReLU 
            )
        

        self.encoder = nn.Sequential(*modules_enc)
        self.decoder = nn.Sequential(*modules_dec)


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


    def encode(self, x):
        def resize(s,n_downsample,odd):
            f = 2**n_downsample
            if odd:
                t = (s - 1) % f
            else:
                t = s % f

            if t != 0:
                s = s + f - t
            return s

        pad = [resize(s,self.n_downsample,self.odd_resize) - s for s in x.shape[-3:]]
        x = torch.nn.functional.pad(x, (pad[-1], 0, pad[-2], 0, pad[-3], 0))

        #down_ftrs = self.down(x)
        #x = self.up(down_ftrs[-1], down_ftrs)
        #x = self.out(x.transpose(1, 4)).transpose(1, 4)
        x = self.encoder(x)
        x = x[..., pad[0]:, pad[1]:, pad[2]:]
        return self.latent(x)


    def decode(self, x):
        x = self.decoder(x)
        return self.out(x)


    def loss_function(self, x, x_hat): 
        return nn.MSELoss()(x_hat, x).to(self.device)
        

    def training_step(self, batch, batch_idx):
        predict = self.forward(batch)
        loss = self.loss_function(batch, predict)
        self.log("train_loss", loss)     
        return loss


    def test_step(self, batch, batch_idx):
        predict = self.forward(batch)
        loss = self.loss_function(batch, predict)
        self.log("test_loss", loss)
        return loss
    

    def validation_step(self, batch, batch_idx):
        predict = self.forward(batch)
        loss = self.loss_function(batch, predict)
        self.log("val_loss", loss)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.1, 
            patience=10, 
            verbose=True, 
            threshold=0.0001, 
            threshold_mode='rel', 
            cooldown=0, 
            min_lr=0, 
            eps=1e-08
            )
        lr_scheduler = {
            "scheduler": scheduler, 
            "monitor": "val_loss"
            }
        return [optimizer], [lr_scheduler]
    
