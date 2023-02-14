import torch, wandb
import torch.nn as nn
from torch.nn import functional as F

import pytorch_lightning as pl
from vae_dist.core.layers import UpConvBatch, ConvBatch
from vae_dist.core.losses import stepwise_inverse_huber_loss, inverse_huber
from torchsummary import summary

class baselineVAEAutoencoder(pl.LightningModule):
    def __init__(
        self,
        irreps, 
        channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
        padding_mode,
        latent_dim,
        fully_connected_layers,
        activation,
        dropout,
        batch_norm,
        learning_rate,
        beta = 1.0,
        max_pool=False, 
        max_pool_kernel_size=2,
        max_pool_loc=[3],
        log_wandb=False,
        im_dim=21,
        reconstruction_loss='mse',
        **kwargs
    ):

        super().__init__()
        self.learning_rate = learning_rate
        params = {
            'irreps': irreps,
            'channels': channels,
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'dilation': dilation,
            'groups': groups,
            'bias': bias,
            'padding_mode': padding_mode,
            'latent_dim': latent_dim,
            'activation': activation,
            'dropout': dropout,
            'batch_norm': batch_norm,
            'beta': beta,
            'kwargs': kwargs,
            'learning_rate': learning_rate,
            'log_wandb': log_wandb,
            'im_dim': im_dim,
            'max_pool': max_pool,
            'fully_connected_layers': fully_connected_layers,
            'max_pool_kernel_size': max_pool_kernel_size,
            'max_pool_loc': max_pool_loc,
            'reconstruction_loss': reconstruction_loss
        }
        assert len(channels) == len(stride) + 1, "channels and stride must be the same length"
        assert len(stride) == len(kernel_size), "stride and kernel_size must be the same length"

        self.hparams.update(params)
        self.save_hyperparameters()


        modules_enc, modules_dec = [], []
        self.list_enc_fully = []
        self.list_dec_fully = []
        self.list_enc_conv = []
        self.list_dec_conv = []
        trigger = 0
        inner_dim = im_dim
        for i in range(len(self.hparams.channels)-1):
            inner_dim = int((inner_dim - (self.hparams.kernel_size[i] - 1)) / self.hparams.stride[i])
            if self.hparams.max_pool:
                if i in self.hparams.max_pool_loc:    
                    inner_dim = int(1 + (inner_dim - self.hparams.max_pool_kernel_size + 1 ) / self.hparams.max_pool_kernel_size )
    
        print("inner_dim: ", inner_dim)

        for ind, h in enumerate(self.hparams.fully_connected_layers):
            # if it's the last item in the list, then we want to output the latent dim
                
            if ind == 0:
                self.list_dec_fully.append(torch.nn.Unflatten(1, (self.hparams.channels[-1], inner_dim, inner_dim, inner_dim)))        
                self.list_enc_fully.append(torch.nn.Flatten())
                h_in = self.hparams.channels[-1] * inner_dim * inner_dim * inner_dim
                h_out = h
            else:
                h_in = self.hparams.fully_connected_layers[ind-1]
                h_out = h

            self.list_enc_fully.append(torch.nn.Linear(h_in , h_out))
            self.list_enc_fully.append(torch.nn.ReLU())
            
            if self.hparams.dropout > 0:
                self.list_enc_fully.append(torch.nn.Dropout(self.hparams.dropout))
                self.list_dec_fully.append(torch.nn.Dropout(self.hparams.dropout))

            if ind == len(self.hparams.fully_connected_layers)-1:
                self.list_dec_fully.append(torch.nn.Sigmoid())
            else: 
                self.list_dec_fully.append(torch.nn.ReLU())
            
            self.list_dec_fully.append(torch.nn.Linear(h_out, h_in))


        self.fc_mu = nn.Linear(self.hparams.fully_connected_layers[-1], self.hparams.latent_dim)
        self.fc_var = nn.Linear(self.hparams.fully_connected_layers[-1], self.hparams.latent_dim)
        h_out = self.hparams.latent_dim
        nn.init.zeros_(self.fc_var.bias)
        self.list_dec_fully.append(torch.nn.Linear(self.hparams.latent_dim, self.hparams.fully_connected_layers[-1]))


        for ind in range(len(self.hparams.channels)-1):
            channel_in = self.hparams.channels[ind]
            channel_out = self.hparams.channels[ind+1]
            kernel = self.hparams.kernel_size[ind]
            stride = self.hparams.stride[ind]

            self.list_enc_conv.append(
                ConvBatch(
                        in_channels = channel_in,
                        out_channels = channel_out,
                        kernel_size = kernel,
                        stride = stride,
                        padding = self.hparams.padding,
                        dilation = self.hparams.dilation,
                        groups = self.hparams.groups,
                        bias = self.hparams.bias,
                        padding_mode = self.hparams.padding_mode,
                )
            )
            
            output_padding = 0
            if inner_dim%2 == 1 and ind == len(self.hparams.channels)-1:
                output_padding = 1

            if dropout > 0: 
                self.list_enc_conv.append(torch.nn.Dropout(dropout))


            if trigger:
                self.list_dec_conv.append(torch.nn.Upsample(
                    scale_factor = self.hparams.max_pool_kernel_size))
                trigger = 0

            if (self.hparams.max_pool and ind in self.hparams.max_pool_loc):
                self.list_enc_conv.append(torch.nn.MaxPool3d(
                    self.hparams.max_pool_kernel_size))
                trigger = 1    

            if dropout > 0:
                self.list_dec_conv.append(torch.nn.Dropout(dropout))
            
            if ind == 0: 
                output_layer = True
            else:
                output_layer = False
            self.list_dec_conv.append(
                UpConvBatch(
                        in_channels = channel_out,
                        out_channels = channel_in,
                        kernel_size = kernel,
                        stride = stride,
                        padding = self.hparams.padding,
                        dilation = self.hparams.dilation,
                        groups = self.hparams.groups,
                        bias = self.hparams.bias,
                        padding_mode = self.hparams.padding_mode,
                        output_padding=output_padding, 
                        output_layer = output_layer
                )
            )


        # reverse the list
        self.list_dec_fully.reverse()
        self.list_dec_conv.reverse()
        [modules_enc.append(i) for i in self.list_enc_conv]
        [modules_enc.append(i) for i in self.list_enc_fully]
        [modules_dec.append(i) for i in self.list_dec_fully]
        [modules_dec.append(i) for i in self.list_dec_conv]
        # for debugging have the conv layers be sequential

        self.encoder_conv = nn.Sequential(*self.list_enc_conv)
        self.decoder_conv = nn.Sequential(*self.list_dec_conv)
        self.decoder_dense = nn.Sequential(*self.list_dec_fully)
        self.encoder = nn.Sequential(*modules_enc)
        self.decoder = nn.Sequential(*modules_dec)
        self.model = nn.Sequential(self.encoder, self.decoder)

        #self.decoder.to(self.device)
        #self.encoder.to(self.device)
        summary(self.encoder_conv, (3, 21, 21, 21), device="cpu")
        summary(self.encoder, (3, 21, 21, 21), device="cpu")
        summary(self.decoder_dense, tuple([latent_dim]), device="cpu")
        summary(self.decoder_conv, (channels[-1], inner_dim, inner_dim, inner_dim), device="cpu")



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
        rmse_loss = torch.sqrt(recon_loss)

        # kl
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        kl = torch.distributions.kl_divergence(q, p)
        kl = kl.mean()
        
        # elbo
        #print(kl.tolist(), recon_loss.tolist())
        elbo = (kl + self.hparams.beta * recon_loss)
        mape = torch.mean(torch.abs((x_hat - batch) / (torch.abs(batch) + 1e-8)))
        medpe = torch.median(torch.abs((x_hat - batch) / (torch.abs(batch) + 1e-8)))
        
        out_dict = {
            'elbo_train': elbo,
            'kl_train': kl,
            'recon_loss_train': recon_loss,
            'rmse_train': rmse_loss,
            'train_loss': elbo,
            'mape_train': mape,
            'medpe_train': medpe,
        }

        if self.hparams.log_wandb:wandb.log(out_dict)
        self.log_dict(out_dict)
        return elbo


    def loss_function(self, x, x_hat, q, p): 
        if self.hparams.reconstruction_loss == "mse":
            recon_loss = F.mse_loss(x_hat, x, reduction='mean')
        
        elif self.hparams.reconstruction_loss == "l1":
            recon_loss = F.l1_loss(x_hat, x, reduction='mean')

        elif self.hparams.reconstruction_loss == "huber":
            recon_loss = F.huber_loss(x_hat, x, reduction='mean')
        elif self.hparams.reconstruction_loss == 'many_step_inverse_huber':
            recon_loss = stepwise_inverse_huber_loss(x_hat, x, reduction='mean')
        elif self.hparams.reconstruction_loss == 'inverse_huber': 
            recon_loss = inverse_huber(x_hat, x, reduction='mean')
        #recon_l_1_2 = torch.sqrt(recon_loss)
        kl = torch.distributions.kl_divergence(q, p).mean()
        elbo = (kl + self.hparams.beta * recon_loss)
        return elbo, kl, recon_loss



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
        # kl
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        elbo, kl, recon_loss = self.loss_function(batch, x_hat, q, p)
        elbo = (kl + self.hparams.beta * recon_loss)
        mape = torch.mean(torch.abs((x_hat - batch) / (torch.abs(batch) + 1e-8)))
        medpe = torch.median(torch.abs((x_hat - batch) / (torch.abs(batch) + 1e-8)))
        
        out_dict = {
            'elbo_test': elbo,
            'kl_test': kl,
            'recon_loss_test': recon_loss,
            'rmse_test': recon_loss,
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
        rmse_loss = torch.sqrt(recon_loss)

        # kl
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        kl = torch.distributions.kl_divergence(q, p).mean()

        # elbo
        elbo = (kl + self.hparams.beta * recon_loss)
        mape = torch.mean(torch.abs((x_hat - batch) / (torch.abs(batch) + 1e-8)))
        medpe = torch.median(torch.abs((x_hat - batch) / (torch.abs(batch) + 1e-8)))
        
        out_dict = {
            'elbo_val': elbo,
            'kl_val': kl,
            'recon_loss_val': recon_loss,
            'rmse_val': rmse_loss,
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