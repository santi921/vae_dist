# TODO: make this lightening
import torch.nn as nn
import torch
from torchConvNd import ConvNd, ConvTransposeNd
from torchsummary import summary


class Unflatten(nn.Module):
    def __init__(self, *args):
        super(Unflatten, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class baselineCNNAutoencoder(nn.Module):
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
        self.irreps = irreps
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
        self.device = device

        self.encoder = nn.Sequential(
            nn.Conv3d(
                in_channels = self.in_channels,
                out_channels = self.out_channels,
                kernel_size = 3,
                stride = self.stride,
                padding = self.padding,
                dilation = self.dilation,
                groups = self.groups,
                bias = self.bias,
                padding_mode = self.padding_mode
            ),
            nn.BatchNorm3d(self.out_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(, self.latent_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 16*16*16),
            nn.ReLU(),
            Unflatten(-1, 16, 16, 16),
            nn.BatchNorm3d(self.out_channels),
            nn.ReLU(),
            nn.ConvTranspose3d(
                in_channels = self.out_channels,
                out_channels = self.in_channels,
                kernel_size = self.kernel_size,
                stride = self.stride,
                padding = self.padding,
                dilation = self.dilation,
                groups = self.groups,
                bias = self.bias,
                padding_mode = self.padding_mode
            ),
            
            nn.Sigmoid()
        )

        self.decoder.to(self.device)
        self.encoder.to(self.device)
        #summary(self.encoder)
        #summary(self.decoder)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


    def encode(self, x):
        return self.encoder(x)


    def decode(self, x):
        return self.decoder(x)


    def loss(self, x): # todo
        x_hat = self.forward(x)
        return nn.MSELoss()(x_hat, x).to(self.device)
        
    
    