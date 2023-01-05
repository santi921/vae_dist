import torch.nn as nn
import torch

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

        self.encoder = nn.Sequential(
            nn.Conv3d(
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
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                in_features = self.out_channels * 32 * 32 * 32,
                out_features = self.hidden_dim
            ),
            nn.ReLU(),
            nn.Linear(
                in_features = self.hidden_dim,
                out_features = self.latent_dim
            )
        )

        self.decoder = nn.Sequential(
            nn.Linear(
                in_features = self.latent_dim,
                out_features = self.hidden_dim
            ),
            nn.ReLU(),
            nn.Linear(
                in_features = self.hidden_dim,
                out_features = self.out_channels * 32 * 32 * 32
            ),
            nn.ReLU(),
            nn.Unflatten(1, (self.out_channels, 32, 32, 32)),
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

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def loss(self, x): # this is autoencoder loss, not VAE loss
        # KL divergence
        z_mu, z_log_var = self.encode(x)
        x_hat = self.forward(x)
        return nn.MSELoss()(x_hat, x) + self.beta * torch.sum(torch.exp(z_log_var) + z_mu**2 - 1. - z_log_var)
        
    
    