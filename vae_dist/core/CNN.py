import torch.nn as nn
import torch
#from torchConvNd import ConvNd, ConvTransposeNd
from torchsummary import summary
from vae_dist.core.layers import UpConvBatch, ConvBatch, ResNetBatch
import pytorch_lightning as pl

class CNNAutoencoderLightning(pl.LightningModule):
    def __init__(
        self,
        irreps, 
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
        padding_mode,
        latent_dim,
        num_layers,
        hidden_dim,
        activation,
        dropout,
        batch_norm,
        beta,
        device,
        **kwargs
    ):
        super().__init__()

        """self.irreps = irreps
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.beta = beta
        self.kwargs = kwargs
        self.device = device"""

        params = {
            'irreps': irreps,
            'in_channels': in_channels,
            'out_channels': out_channels,
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'dilation': dilation,
            'groups': groups,
            'bias': bias,
            'padding_mode': padding_mode,
            'latent_dim': latent_dim,
            'num_layers': num_layers,
            'hidden_dim': hidden_dim,
            'activation': activation,
            'dropout': dropout,
            'batch_norm': batch_norm,
            'beta': beta,
            'device': device,
            'kwargs': kwargs,
        }
        self.hparams.update(params)
        self.save_hyperparameters()

        modules_enc, modules_dec = [], []
        modules_enc.append(
            ConvBatch(
                    in_channels = self.hparams.in_channels,
                    out_channels = self.hparams.out_channels,
                    kernel_size = self.hparams.kernel_size,
                    stride = self.hparams.stride,
                    padding = self.hparams.padding,
                    dilation = self.hparams.dilation,
                    groups = self.hparams.groups,
                    bias = self.hparams.bias,
                    padding_mode = self.hparams.padding_mode,
            )
        )
        modules_enc.append(nn.Flatten())
        modules_enc.append(
            nn.Linear(
                in_features = self.hparams.out_channels * 17 * 17 * 17,
                out_features = self.hparams.hidden_dim
            ))
        modules_enc.append(nn.ReLU())
        self.encoder = nn.Sequential(*modules_enc)

        #modules_dec.append(
        #    nn.Linear(
        #        in_features = self.hidden_dim,
        #        out_features = self.out_channels
        #    ))
        #modules_dec.append(nn.ReLU())
        modules_dec.append(
            nn.Linear(
                in_features = self.hparams.hidden_dim,
                out_features = self.hparams.out_channels * 17 * 17 * 17
            ))
        modules_dec.append(nn.ReLU()),
        modules_dec.append(nn.Unflatten(1, (self.hparams.out_channels, 17, 17, 17)))
        modules_dec.append(
            UpConvBatch(
                    in_channels = self.hparams.out_channels,
                    out_channels = self.hparams.in_channels,
                    kernel_size = self.hparams.kernel_size,
                    stride = self.hparams.stride,
                    padding = self.hparams.padding,
                    dilation = self.hparams.dilation,
                    groups = self.hparams.groups,
                    bias = self.hparams.bias,
                    padding_mode = self.hparams.padding_mode,
            )
        )
        modules_dec.append(nn.Tanh())
        self.decoder = nn.Sequential(*modules_dec)

        #self.decoder.to(self.device)
        #self.encoder.to(self.device)
        #summary(self.encoder, (3, 21, 21, 21), device="cuda")
        #summary(self.decoder, (32), device="cuda")


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


    def encode(self, x):
        return self.encoder(x)


    def decode(self, x):
        return self.decoder(x)


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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
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
    
"""nn.ConvTranspose3d(
    in_channels = self.out_channels,
    out_channels = self.in_channels,
    kernel_size = self.kernel_size,
    stride = self.stride,
    padding = self.padding,
    dilation = self.dilation,
    groups = self.groups,
    bias = self.bias,
    padding_mode = self.padding_mode
),"""

"""nn.Conv3d(
    in_channels = self.in_channels,
    out_channels = self.out_channels,
    kernel_size = self.kernel_size,
    stride = self.stride,
    padding = self.padding,
    dilation = self.dilation,
    groups = self.groups,
    bias = self.bias,
    padding_mode = self.padding_mode
),
nn.BatchNorm3d(self.out_channels),
nn.ReLU(),"""      