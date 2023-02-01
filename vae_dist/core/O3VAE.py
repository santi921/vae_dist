import torch, wandb                                                      
from torchsummary import summary
from torch.nn import functional as F

import pytorch_lightning as pl
from escnn import nn, group                                        

from vae_dist.core.escnnlayers import R3Upsampling


class R3VAE(pl.LightningModule):
    def __init__(
        self, 
        learning_rate, 
        gspace,  
        group,
        feat_type_in, 
        feat_type_out, 
        kernel_size=5, 
        latent_dim=1, 
        beta = 1.0,
        fully_connected_dims = [64],
        log_wandb=False):

        super().__init__()
        self.learning_rate = learning_rate
        
        params = {
            'in_type': feat_type_in,
            'out_type': feat_type_out,
            'kernel_size': kernel_size,
            'padding': 0,
            'bias': False,
            'learning_rate': learning_rate,
            'latent_dim': latent_dim,
            'group': group,
            'gspace': gspace,
            'fully_connected_dims': fully_connected_dims,
            'beta': beta,
            'log_wandb': log_wandb
        }
        self.hparams.update(params)
        #self.save_hyperparameters()

        ############
        # NOTE THAT RELU ISNT EQUIVARIANT
        # NOTE THAT RELU ISNT EQUIVARIANT
        # NOTE THAT RELU ISNT EQUIVARIANT
        ########
        self.list_dec_fully = []
        self.list_enc_fully  = []
        self.decoder_conv_list = [] 
        self.encoder_conv_list = []

        self.gspace = gspace     
        self.group = group
        self.feat_type_in  = feat_type_in
        self.feat_type_out = feat_type_out
        self.feat_type_hidden = nn.FieldType(self.gspace, latent_dim*[self.gspace.trivial_repr])
        self.channels_inner = 48
        self.channels_outer = 24
        
        # convolution 1
        self.fc_mu = torch.nn.Linear(self.hparams.latent_dim, self.hparams.latent_dim)
        self.fc_var = torch.nn.Linear(self.hparams.latent_dim, self.hparams.latent_dim)
        # initialize the log scale to 0
        
        torch.nn.init.zeros_(self.fc_var.bias)

        ######################### Encoder ########################################
        in_type_og  = feat_type_in        
        
        out_type = nn.FieldType(self.gspace, self.channels_outer*[self.gspace.trivial_repr])
        # we choose 24 feature fields, each transforming under the regular representation of C8        
        self.encoder_conv_list.append(
            nn.R3Conv(in_type_og, out_type, kernel_size=kernel_size, padding=0, bias=False))
        
        # works
        nn.ReLU(out_type, inplace=True)
        nn.PointwiseNonLinearity(out_type, function='p_relu')
        nn.QuotientFourierPointwise(
            gspace = self.gspace,
            channels = 8,
            irreps=self.group.bl_sphere_representation(L=1).irreps,
            grid = self.group.grid(type='thomson', N=8),
            subgroup_id=(True, 'so3'),
            function='p_relu',
            inplace=True
        )

        # Not working 
        #nn.FourierPointwise(self.gspace, 
        #                    channels = 8,
        #                    irreps = self.group.bl_sphere_representation(L=1).irreps,
        #                    grid = self.group.grid(type='thomson', N=9),
        #                    function='p_relu', 
        #                    inplace=True)
        #nn.FourierELU(self.gspace,
        #            channels = 8,
        #            irreps= self.group.bl_sphere_representation(L=1).irreps,
        #            grid = self.group.grid(type='thomson', N=16),
        #            subgroup_id=(True, 'so3'),
        #            inplace=True
        #              )
        # not working
        #nn.InducedNormNonLinearity(out_type, function='n_sigmoid', bias=True)        
        #nn.GatedNonLinearityUniform(out_type)
        


        
        self.encoder_conv_list.append(
            nn.QuotientFourierPointwise(
            gspace = self.gspace,
            channels = 8,
            irreps=self.group.bl_sphere_representation(L=1).irreps,
            grid = self.group.grid(type='thomson', N=8),
            subgroup_id=(True, 'so3'),
            function='p_relu',
            inplace=True
            )
        )
        #activations


        self.encoder_conv_list.append(
            nn.PointwiseDropout(out_type, p=0.1)
        )

        #self.encoder_conv_list.append(nn.ReLU(out_type, inplace=True))
        self.encoder_conv_list.append(nn.PointwiseAvgPoolAntialiased3D(out_type, sigma=0.66, stride=3))

        in_type = out_type
        out_type = nn.FieldType(self.gspace, self.channels_inner*[self.gspace.trivial_repr])
        self.encoder_conv_list.append(nn.R3Conv(in_type, out_type, kernel_size=kernel_size, padding=0, bias=False))
        self.encoder_conv_list.append(
            nn.QuotientFourierELU(
                gspace = self.gspace, 
                channels = 3, 
                irreps=self.group.bl_sphere_representation(L=1).irreps,
                grid = self.group.sphere_grid(type='thomson', N=16),
                subgroup_id=(True, 'so3'),
                inplace=True
                )
            )
        #self.encoder_conv_list.append(nn.ReLU(out_type, inplace=True))
        self.encoder_conv_list.append(nn.PointwiseAvgPoolAntialiased3D(out_type, sigma=0.66, stride=3))
        self.encoder_conv_list.append(nn.ReLU(out_type, inplace=True))
        
        #self.encoder_conv_list.append(nn.GroupPooling(out_type))

        # number of output channels
        
        self.im_dim = 1

        for ind, h in enumerate(fully_connected_dims):
            # if it's the last item in the list, then we want to output the latent dim
            if ind == len(fully_connected_dims)-1:
                h_out = latent_dim

            else: 
                h_out = fully_connected_dims[ind+1]

            if ind == 0:
                self.list_dec_fully.append(torch.nn.Unflatten(1, (self.channels_inner, self.im_dim, self.im_dim, self.im_dim)))        
                self.list_enc_fully.append(torch.nn.Flatten())
                h_in = self.channels_inner*self.im_dim*self.im_dim*self.im_dim
            else:
                h_in = h

           
            self.list_enc_fully.append(torch.nn.Linear(h_in , h_out))
            self.list_enc_fully.append(torch.nn.ReLU())
            
            
            self.list_dec_fully.append(torch.nn.Sigmoid())
            self.list_dec_fully.append(torch.nn.Linear(h_out, h_in))

            
        # reverse the list
        self.list_dec_fully.reverse()
            
        self.dense_out_type = nn.FieldType(self.gspace,  self.channels_inner * [self.gspace.trivial_repr])
        out_type = nn.FieldType(self.gspace, self.hparams.latent_dim * self.channels_outer*[self.gspace.trivial_repr])
        
        self.decoder_conv_list.append(nn.R3ConvTransposed(self.dense_out_type, out_type, kernel_size=kernel_size, padding=2, bias=False))
        self.decoder_conv_list.append(nn.ReLU(out_type, inplace=True))
        self.decoder_conv_list.append(R3Upsampling(out_type, scale_factor=3, mode='nearest', align_corners=False))
        self.decoder_conv_list.append(nn.R3ConvTransposed(
            out_type, 
            in_type_og, 
            kernel_size=kernel_size, 
            padding=0, 
            bias=False))
        self.decoder_conv_list.append(nn.ReLU(in_type_og, inplace=True))
        self.decoder_conv_list.append(R3Upsampling(in_type_og, scale_factor=3, mode='nearest', align_corners=False))
        
        self.encoder_fully_net = torch.nn.Sequential(*self.list_enc_fully)
        self.decoder_fully_net = torch.nn.Sequential(*self.list_dec_fully)
        self.decoder = nn.SequentialModule(*self.decoder_conv_list)
        self.encoder = nn.SequentialModule(*self.encoder_conv_list)
        # summary on encoder    
        #print(self.encoder)
        #print(self.encoder_fully_net)
        #print(self.decoder_fully_net)
        #print(self.decoder)
        #self.model = nn.SequentialModule(self.encoder, self.encoder_fully_net, self.decoder_fully_net, self.decoder)
        #summary(self.encoder, (1, 32, 32, 32))
        
    def encode(self, x):
        x = self.feat_type_in(x)
        x = self.encoder(x)
        x = x.tensor
        x = self.encoder_fully_net(x)
        mu, var = torch.clamp(self.fc_mu(x), min=0.000001), torch.clamp(self.fc_var(x), min=0.000001)
        return mu, var
        

    def decode(self, x):
        x = self.decoder_fully_net(x)
        #x = x.reshape(x.shape[0], self.channels_inner, self.im_dim, self.im_dim, self.im_dim)
        x = self.dense_out_type(x)
        x = self.decoder(x)
        x = x.tensor
        return x
   
    
    def latent(self, x): 
        mu, var = self.encode(x)
        std = torch.exp(var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()    
        return z 

        
    def forward(self, x: torch.Tensor):
        mu, var = self.encode(x)
        std = torch.exp(var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()    
        x = self.decode(z)
        return x


    def loss_function(self, x, x_hat, q, p): 
        recon_loss = F.mse_loss(x_hat, x, reduction='mean')
        kl = torch.distributions.kl_divergence(q, p).mean()
        elbo = (kl + self.hparams.beta * recon_loss)
        return elbo, kl, recon_loss


    def training_step(self, batch, batch_idx):
        mu, log_var = self.encode(batch)
        #mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)
        std = torch.exp(log_var / 2)        
        q = torch.distributions.Normal(mu, log_var)
        z = q.rsample()
        x_hat = self.decode(z)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))

        elbo, kl, recon_loss = self.loss_function(batch, x_hat, q, p)
        mape = torch.mean(torch.abs((x_hat - batch) / torch.abs(batch)))
        medpe = torch.median(torch.abs((x_hat - batch) / torch.abs(batch)))
        out_dict = {
            'elbo_train': elbo,
            'kl_train': kl,
            'recon_loss_train': recon_loss,
            'train_loss': elbo,
            'mape_train': mape, 
            "training_median_percent_error": medpe
            }     
        if self.hparams.log_wandb:wandb.log(out_dict)
        self.log_dict(out_dict)
        return elbo


    def test_step(self, batch, batch_idx):
        
        mu, log_var = self.encode(batch)
        #mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)
        std = torch.exp(log_var / 2)        
        q = torch.distributions.Normal(mu, log_var)
        z = q.rsample()
        x_hat = self.decode(z)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))

        elbo, kl, recon_loss = self.loss_function(batch, x_hat, q, p)
        mape = torch.mean(torch.abs((x_hat - batch) / torch.abs(batch)))
        medpe = torch.median(torch.abs((x_hat - batch) / torch.abs(batch)))
        out_dict = {
            'elbo_test': elbo,
            'kl_test': kl,
            'recon_loss_test': recon_loss,
            'test_loss': elbo,
            'test_mape': mape,
            "training_median_percent_error": medpe
            }     
        if self.hparams.log_wandb:wandb.log(out_dict)
        self.log_dict(out_dict)

        return elbo
    

    def validation_step(self, batch, batch_idx):
        
        mu, log_var = self.encode(batch)
        # = self.fc_mu(x_encoded), self.fc_var(x_encoded)
        std = torch.exp(log_var / 2)        
        q = torch.distributions.Normal(mu, log_var)
        z = q.rsample()
        x_hat = self.decode(z)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))

        elbo, kl, recon_loss = self.loss_function(batch, x_hat, q, p)
        mape = torch.mean(torch.abs((x_hat - batch) / torch.abs(batch)))
        medpe = torch.median(torch.abs((x_hat - batch) / torch.abs(batch)))
        out_dict = {
            'elbo_val': elbo,
            'kl_val': kl,
            'recon_loss_val': recon_loss,
            'val_loss': elbo,
            'mape_val': mape,
            "training_median_percent_error": medpe
            }     

        if self.hparams.log_wandb:wandb.log(out_dict)
        self.log_dict(out_dict)

        return elbo


    def load_model(self, path):
        self.load_state_dict(torch.load(path, map_location='cuda:0'), strict=False)
        print('Model Created!')


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

