import torch, wandb
import torch.nn as nn
from torch.nn import functional as F

import pytorch_lightning as pl
from vae_dist.core.layers import UpConvBatch, ConvBatch, ResNetBatch

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
        learning_rate,
        log_wandb=False,
        **kwargs
    ):
        super().__init__()
        self.learning_rate = learning_rate
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
            'kwargs': kwargs,
            'learning_rate': learning_rate,
            'log_wandb': log_wandb
        }

        self.hparams.update(params)
        self.save_hyperparameters()

        self.fc_mu = nn.Linear(self.hparams.hidden_dim, self.hparams.latent_dim)
        self.fc_var = nn.Linear(self.hparams.hidden_dim, self.hparams.latent_dim)
        # initialize the log scale to 0
        
        nn.init.zeros_(self.fc_var.bias)
        #nn.init.ones_(self.fc_mu.bias)

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
                out_features = self.hparams.hidden_dim,
            ))
        modules_enc.append(nn.ReLU())

        modules_dec.append(
            nn.Linear(
                in_features = self.hparams.latent_dim,
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
        modules_dec.append(nn.LeakyReLU())
        modules_dec.append(nn.Sigmoid())
        self.encoder = nn.Sequential(*modules_enc)
        self.decoder = nn.Sequential(*modules_dec)
        self.model = nn.Sequential(self.encoder, self.decoder)

    def forward(self, x  : torch.Tensor) -> torch.Tensor:
        mu, var = self.encode(x)
        std = torch.exp(var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()       
        x = self.decoder(z)
        return x


    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)
        # alternatively, use the following: result = torch.flatten(result, start_dim=1)
        return self.fc_mu(x), self.fc_var(x)


    def latent(self, x: torch.Tensor) -> torch.Tensor:
        mu, log_var = self.encode(x)
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return z

        
    def decode(self, x: torch.Tensor)  -> torch.Tensor:
        return self.decoder(x)


    def training_step(self, batch, batch_idx):
        x = batch
        # encode x to get the mu and variance parameters
        mu, log_var = self.encode(x)
        #print(x_encoded.shape)
        #mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded
        x_hat = self.decoder(z)

        # reconstruction loss
        #recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)
        recon_loss = F.mse_loss(x_hat, x, reduction='mean')
        
        # kl
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        kl = torch.distributions.kl_divergence(q, p)
        kl = kl.mean()
        
        # elbo
        #print(kl.tolist(), recon_loss.tolist())
        elbo = (kl + self.hparams.beta * recon_loss)
        mape = torch.mean(torch.abs((x_hat - batch) / torch.abs(batch)))
        medpe = torch.median(torch.abs((x_hat - batch) / torch.abs(batch)))

        out_dict = {
            'elbo_train': elbo,
            'kl_train': kl,
            'recon_loss_train': recon_loss,
            'train_loss': elbo,
            'mape_train': mape,
            'medpe_train': medpe,
        }

        if self.hparams.log_wandb:wandb.log(out_dict)
        self.log_dict(out_dict)
        return elbo


    def test_step(self, batch, batch_idx):
        x = batch
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)
        
        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded
        x_hat = self.decoder(z)

        # reconstruction loss
        #recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)
        recon_loss = F.mse_loss(x_hat, x, reduction='mean')

        # kl
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        kl = torch.distributions.kl_divergence(q, p).mean()

        # elbo
        elbo = (kl + self.hparams.beta * recon_loss)
        mape = torch.mean(torch.abs((x_hat - batch) / torch.abs(batch)))
        medpe = torch.median(torch.abs((x_hat - batch) / torch.abs(batch)))

        out_dict = {
            'elbo_test': elbo,
            'kl_test': kl,
            'recon_loss_test': recon_loss,
            'test_loss': elbo,
            'test_mape': mape,
            'test_medpe': medpe
        }
        
        if self.hparams.log_wandb:wandb.log(out_dict)
        self.log_dict(out_dict)

        return elbo


    def validation_step(self, batch, batch_idx):
        x = batch
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)
        
        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded
        x_hat = self.decoder(z)

        # reconstruction loss
        #recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)
        recon_loss = F.mse_loss(x_hat, x, reduction='mean')

        # kl
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        kl = torch.distributions.kl_divergence(q, p).mean()

        # elbo
        elbo = (kl + self.hparams.beta * recon_loss)
        mape = torch.mean(torch.abs((x_hat - batch) / torch.abs(batch)))
        medpe = torch.median(torch.abs((x_hat - batch) / torch.abs(batch)))

        out_dict = {
            'elbo_val': elbo,
            'kl_val': kl,
            'recon_loss_val': recon_loss,
            'val_loss': elbo,
            'mape_val': mape,
            'medpe_val': medpe
        }
        if self.hparams.log_wandb:wandb.log(out_dict)
        self.log_dict(out_dict)

        return elbo


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
            "monitor": "elbo_val"
            }
        return [optimizer], [lr_scheduler]


    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location='cuda:0'), strict=False)
        print('Model Created!')