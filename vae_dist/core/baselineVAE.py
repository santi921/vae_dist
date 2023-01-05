import torch.nn as nn
import torch
import pytorch_lightning as pl

from pl_bolts.models.autoencoders.components import (
    resnet18_decoder,
    resnet18_encoder,
)

class baselineVAEAutoencoder(pl.LightningModule):
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
        self.kl = 0


        #self.N = torch.distributions.Normal(0, 1)
        #self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        #self.N.scale = self.N.scale.cuda()
        #self.fc_mu = nn.Linear(hidden_dim[-1]*4, latent_dim) # 4 dimensions
        #self.fc_var = nn.Linear(hidden_dim[-1]*4, latent_dim)
        self.save_hyperparameters()

        self.fc_mu = nn.Linear(hidden_dim[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dim[-1], latent_dim)
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

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
            ),

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

    def forward(self, x  : torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        # alternatively, use the following: result = torch.flatten(result, start_dim=1)
        return self.fc_mu(x), self.fc_var(x)

    def decode(self, x: torch.Tensor)  -> torch.Tensor:
        return self.decoder(x)

    #def loss(self, x): # this is autoencoder loss, not VAE loss
    #    # KL divergence
    #    z_mu, z_log_var = self.encode(x)
    #    x_hat = self.forward(x)
    #    return nn.MSELoss()(x_hat, x) + self.beta * torch.sum(torch.exp(z_log_var) + z_mu**2 - 1. - z_log_var)
    
    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1, 2, 3))

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        
        # this prior isn't necessarily normal
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std)) 
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl
    
    def training_step(self, batch, batch_idx):
        x, _ = batch

        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded
        x_hat = self.decoder(z)

        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)

        # kl
        kl = self.kl_divergence(z, mu, std, self.beta)

        # elbo
        elbo = (kl - recon_loss)
        elbo = elbo.mean()

        self.log_dict({
            'elbo': elbo,
            'kl': kl.mean(),
            'recon_loss': recon_loss.mean(),
            'reconstruction': recon_loss.mean(),
            'kl': kl.mean(),
        })

        return elbo