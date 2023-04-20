import torch, wandb                                                      
from torchsummary import summary
from torch.nn import functional as F

import pytorch_lightning as pl
from torchmetrics import MeanSquaredError, MeanAbsoluteError

from escnn import nn                                        

from vae_dist.core.losses import stepwise_inverse_huber_loss, inverse_huber
from vae_dist.core.escnnlayers import R3Upsampling
#from vae_dist.core.training_utils import pull_default_escnn_params

def pull_default_escnn_params(dim = 3):
    from escnn import gspaces, nn, group 
    g = group.o3_group()
    gspace = gspaces.flipRot3dOnR3(maximum_frequency=10) 
    input_out_reps = dim*[gspace.trivial_repr]
    feat_type_in  = nn.FieldType(gspace,  input_out_reps) 
    feat_type_out = nn.FieldType(gspace,  input_out_reps) 
    return group, gspace, feat_type_in, feat_type_out, input_out_reps

class R3VAE(pl.LightningModule):
    def __init__(
        self, 
        learning_rate, 
        channels,
        #gspace,  
        #group,
        #feat_type_in, 
        #feat_type_out, 
        dropout,
        kernel_size_in=5,
        kernel_size_out=5, 
        latent_dim=1, 
        beta = 1.0,
        batch_norm=False,
        max_pool=False, 
        bias = True, 
        stride=[1],
        fully_connected_layers=[64, 64, 64],
        log_wandb=False,
        im_dim=21,
        max_pool_kernel_size_in=2,
        max_pool_loc_in=[3],
        stride_in=[1],
        max_pool_kernel_size_out=2,
        max_pool_loc_out=[3],
        stride_out=[1],
        reconstruction_loss='mse',
        **kwargs
        ):

        super().__init__()
        self.learning_rate = learning_rate
        
        params = {
            #'in_type': feat_type_in,
            #'out_type': feat_type_out,
            'channels': channels,
            'padding': 0,
            'bias': bias,
            'stride_in': stride_in, 
            'stride_out': stride_out,
            'learning_rate': learning_rate,
            'latent_dim': latent_dim,
            #'group': group,
            #'gspace': gspace,
            'fully_connected_layers': fully_connected_layers,
            'dropout': dropout,
            'batch_norm': batch_norm,
            'max_pool': max_pool,
            'kernel_size_in': kernel_size_in,
            'kernel_size_out': kernel_size_out,
            'max_pool_kernel_size_in': max_pool_kernel_size_in,
            'max_pool_kernel_size_out': max_pool_kernel_size_out,
            'max_pool_loc_in': max_pool_loc_in,
            'max_pool_loc_out': max_pool_loc_out,
            'beta': beta,
            'log_wandb': log_wandb,
            'im_dim': im_dim,
            'reconstruction_loss': reconstruction_loss,
        }

        assert len(channels) == len(stride_in) == len(stride_out), "channels and stride must be the same length"
        assert len(stride_in) == len(stride_out) == len(kernel_size_in) == len(kernel_size_out), "stride and kernel_size must be the same length"
        
        self.hparams.update(params)
        self.save_hyperparameters()

        #params["in_type"] = feat_type_in
        #params["out_type"] = feat_type_out
        #params["group"] = group
        #params["gspace"] = gspace

        self.list_dec_fully = []
        self.list_enc_fully  = []
        self.decoder_conv_list = [] 
        self.encoder_conv_list = []
        
        group, gspace, feat_type_in, feat_type_out, input_out_reps = pull_default_escnn_params()
        gspace = gspace     
        group = group
        self.feat_type_in  = feat_type_in
        feat_type_out = feat_type_out
        feat_type_hidden = nn.FieldType(gspace, latent_dim*[gspace.trivial_repr])
        self.dense_out_type = nn.FieldType(gspace,  self.hparams.channels[-1] * [gspace.trivial_repr])

        trigger = 0 
        inner_dim = self.hparams.im_dim
        # number of output channels
        for i in range(len(self.hparams.channels)):
            inner_dim = int((inner_dim - (self.hparams.kernel_size_in[i] - 1)) / self.hparams.stride_in[i])
            if self.hparams.max_pool:
                if i in self.hparams.max_pool_loc_in:    
                    inner_dim = int(1 + (inner_dim - self.hparams.max_pool_kernel_size_in + 1 ) / self.hparams.max_pool_kernel_size_in)
    
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
            self.list_enc_fully.append(torch.nn.LeakyReLU())


            if self.hparams.batch_norm:
                self.list_enc_fully.append(torch.nn.BatchNorm1d(h_out))

            if self.hparams.dropout > 0:
                self.list_enc_fully.append(torch.nn.Dropout(self.hparams.dropout))
                self.list_dec_fully.append(torch.nn.Dropout(self.hparams.dropout))

            if self.hparams.batch_norm and ind != 0:
                self.list_dec_fully.append(torch.nn.BatchNorm1d(h_in))

            if ind == len(self.hparams.fully_connected_layers)-1:
                self.list_dec_fully.append(torch.nn.LeakyReLU())
            else: 
                self.list_dec_fully.append(torch.nn.LeakyReLU())
            
            self.list_dec_fully.append(torch.nn.Linear(h_out, h_in))


        for ind in range(len(self.hparams.channels)):
            if ind == 0:
                in_type  = feat_type_in        
                channel_out = self.hparams.channels[0]

            else: 
                channel_in = self.hparams.channels[ind-1]
                channel_out = self.hparams.channels[ind]
                in_type = nn.FieldType(gspace, channel_in * [gspace.trivial_repr])
            
            out_type = nn.FieldType(gspace, channel_out * [gspace.trivial_repr])
    
            print('in_type: {} out_type: {}'.format(in_type, out_type))
            
            self.encoder_conv_list.append(
                nn.R3Conv(
                    in_type = in_type, 
                    out_type = out_type, 
                    kernel_size = kernel_size_in[ind], 
                    stride = stride_in[ind],
                    bias=self.hparams.bias,
                )
            )
            if self.hparams.batch_norm:
                self.encoder_conv_list.append(nn.IIDBatchNorm3d(out_type, affine=True))

            self.encoder_conv_list.append(nn.ReLU(out_type, inplace=False))  
    

            output_padding = 0
            #if inner_dim%2 == 1 and ind == len(self.hparams.channels)-1:
            #    output_padding = 1

            if dropout > 0: 
                self.encoder_conv_list.append(nn.PointwiseDropout(out_type, p=dropout))

            if trigger:
                self.decoder_conv_list.append(R3Upsampling(
                    in_type, 
                    scale_factor=self.hparams.max_pool_kernel_size_out, 
                    mode='nearest', 
                    align_corners=False))
                trigger = 0

            if (self.hparams.max_pool and ind in self.hparams.max_pool_kernel_size_in):
                self.encoder_conv_list.append(
                    nn.PointwiseAvgPoolAntialiased3D(
                        out_type, 
                        stride=self.hparams.max_pool_kernel_size_in,
                        sigma = 0.66))
                trigger = 1    

            if dropout > 0:
                self.decoder_conv_list.append(nn.PointwiseDropout(in_type, p=dropout))
            
            # decoder
            if ind == 0:
                self.decoder_conv_list.append(nn.IdentityModule(in_type))
            else: 
                self.decoder_conv_list.append(nn.ReLU(in_type, inplace=False))
            
            if self.hparams.batch_norm:
                self.decoder_conv_list.append(nn.IIDBatchNorm3d(in_type, affine=True))
            
            self.decoder_conv_list.append(
                nn.R3ConvTransposed(
                    in_type=out_type, 
                    out_type=in_type, 
                    stride=stride_out[ind],
                    kernel_size=kernel_size_out[ind], 
                    bias=self.hparams.bias,
                    output_padding=output_padding
                )
            )



        # sampling layers
        self.fc_mu = torch.nn.Linear(self.hparams.fully_connected_layers[-1], self.hparams.latent_dim)
        self.fc_var = torch.nn.Linear(self.hparams.fully_connected_layers[-1], self.hparams.latent_dim)
        h_out = self.hparams.latent_dim
        torch.nn.init.zeros_(self.fc_var.bias)
        torch.nn.init.zeros_(self.fc_mu.bias)
        self.list_dec_fully.append(torch.nn.Linear(self.hparams.latent_dim, self.hparams.fully_connected_layers[-1]))

        # reverse the list
        self.list_dec_fully.reverse()
        self.decoder_conv_list.reverse()
        self.encoder_fully_net = torch.nn.Sequential(*self.list_enc_fully)
        self.decoder_fully_net = torch.nn.Sequential(*self.list_dec_fully)        
        self.encoder = nn.SequentialModule(*self.encoder_conv_list)
        self.decoder = nn.SequentialModule(*self.decoder_conv_list)
        self.model = nn.SequentialModule(self.encoder, self.decoder)

        summary(self.encoder_fully_net, (self.hparams.channels[-1], inner_dim, inner_dim, inner_dim), device="cpu")
        summary(self.decoder_fully_net, tuple([latent_dim]), device="cpu")

        self.train_rmse = MeanSquaredError(squared=False)
        self.val_rmse = MeanSquaredError(squared=False)
        self.test_rmse = MeanSquaredError(squared=False)
        self.test_mae = MeanAbsoluteError()
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()



    def encode(self, x):
        x = self.feat_type_in(x)
        x = self.encoder(x)
        x = x.tensor
        #print("x.shape: ", x.shape)
        x = self.encoder_fully_net(x)
        mu, var = torch.clamp(self.fc_mu(x), min=0.000001), torch.clamp(self.fc_var(x), min=0.000001)
        return mu, var
        

    def decode(self, x):
        x = self.decoder_fully_net(x)
        #print("x.shape: ", x.shape)
        #x = x.reshape(x.shape[0], self.channels_inner, self.im_dim, self.im_dim, self.im_dim)
        x = self.dense_out_type(x)
        x = self.decoder(x)
        x = x.tensor
        #print("x.shape: ", x.shape)
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

    def shared_step(self, batch, mode):
        x=batch
        mu, log_var = self.encode(batch)
        std = torch.exp(log_var / 2)        
        q = torch.distributions.Normal(mu, log_var)
        z = q.rsample()
        x_hat = self.decode(z)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))


        if mode == 'train':
            self.train_rmse.update(x_hat, x)
            #self.train_kl.update(p, q)
            self.train_mae.update(x_hat, x)

        elif mode == 'val':
            self.val_rmse.update(x_hat, x)
            #self.val_kl.update(p, q)
            self.val_mae.update(x_hat, x)

        elif mode == 'test':
            self.test_rmse.update(x_hat, x)
            #self.test_kl.update(p, q)
            self.test_mae.update(x_hat, x)
            #self.test_mae.update(predict, batch_group)
            
        loss = self.loss_function(x, x_hat, q, p)
        #if self.hparams.log_wandb:wandb.log({f"{mode}_loss": loss})
        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch))
        return loss 

    def compute_metrics(self, mode):
        if mode == 'train':
            rmse = self.train_rmse.compute()
            mae = self.train_mae.compute()
            #kl = self.train_kl.compute()
            self.train_rmse.reset()      
            self.train_mae.reset()
            #self.train_kl.reset()

        elif mode == 'val':
            rmse = self.val_rmse.compute()
            mae = self.val_mae.compute()
            #kl = self.val_kl.compute()
            self.val_mae.reset()
            self.val_rmse.reset()
            #self.val_kl.reset()

        elif mode == 'test':
            rmse = self.test_rmse.compute()
            mae = self.test_mae.compute()
            #kl = self.test_kl.compute()
            self.test_rmse.reset()
            self.test_mae.reset()
            #self.test_kl.reset()
            
        return rmse, mae#, kl

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

        kl = torch.distributions.kl_divergence(p, q).mean()
        loss = (kl + self.hparams.beta * recon_loss)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, mode='train')

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, mode='test')
    
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, mode='val')
    

    def training_epoch_end(self, outputs):
        rmse, mae= self.compute_metrics(mode='train')
        out_dict = {
            'train_rmse': rmse,
            #'train_kl': kl,
            'train_mae': mae,
            #'train_elbo': kl + self.hparams.beta * rmse,
        }   
        if self.hparams.log_wandb:wandb.log(out_dict)
        self.log_dict(out_dict, prog_bar=True)

    def validation_epoch_end(self, outputs):
        rmse, mae= self.compute_metrics(mode='val')
        out_dict = {
            'val_rmse': rmse,
            #'val_kl': kl,
            'val_mae': mae,
            #'val_elbo': kl + self.hparams.beta * rmse,
        }   
        if self.hparams.log_wandb:wandb.log(out_dict)
        self.log_dict(out_dict, prog_bar=True)
    
    def test_epoch_end(self, outputs):
        rmse, mae = self.compute_metrics(mode='test')
        out_dict = {
            'test_rmse': rmse,
            #'test_kl': kl,
            'test_mae': mae,
            #'test_elbo': kl + self.hparams.beta * rmse,
        }   
        if self.hparams.log_wandb:wandb.log(out_dict)
        self.log_dict(out_dict, prog_bar=True)



    def load_model(self, path):
        self.load_state_dict(torch.load(path, map_location='cuda:0'), strict=False)
        print('Model Created!')


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        
        #optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.1, 
            patience=20, 
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

