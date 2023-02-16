from escnn import nn                                         
import torch, wandb                                                      
import pytorch_lightning as pl

from torchsummary import summary
from torch.nn import MSELoss

from vae_dist.core.escnnlayers import R3Upsampling

class R3CNN(pl.LightningModule):
    def __init__(
        self, 
        learning_rate, 
        channels,
        gspace,  
        group,
        feat_type_in, 
        feat_type_out, 
        dropout,
        kernel_size=5, 
        latent_dim=1, 
        max_pool=False, 
        max_pool_kernel_size=2,
        max_pool_loc=[3],
        stride=[1],
        fully_connected_layers=[64, 64, 64],
        log_wandb=False,
        im_dim=21,
        reconstruction_loss='mse',
        **kwargs
        ):

        #super(self).__init__()
        super().__init__()
        self.learning_rate = learning_rate
        params = {
            'in_type': feat_type_in,
            'out_type': feat_type_out,
            'kernel_size': kernel_size,
            'channels': channels,
            'padding': 0,
            'bias': False,
            'stride': stride, 
            'learning_rate': learning_rate,
            'latent_dim': latent_dim,
            'group': group,
            'gspace': gspace,
            'fully_connected_layers': fully_connected_layers,
            'dropout': dropout,
            'max_pool': max_pool,
            'max_pool_kernel_size': max_pool_kernel_size,
            'max_pool_loc': max_pool_loc,
            'log_wandb': log_wandb,
            'im_dim': im_dim,
            'reconstruction_loss': reconstruction_loss,
        }

        self.hparams.update(params)
        #self.save_hyperparameters()

        self.list_dec_fully = []
        self.list_enc_fully  = []
        self.decoder_conv_list = [] 
        self.encoder_conv_list = []

        self.gspace = gspace     
        self.group = group
        self.feat_type_in  = feat_type_in
        self.feat_type_out = feat_type_out
        self.feat_type_hidden = nn.FieldType(self.gspace, latent_dim*[self.gspace.trivial_repr])
        self.dense_out_type = nn.FieldType(self.gspace,  self.hparams.channels[-1] * [self.gspace.trivial_repr])

        trigger = 0 
        inner_dim = self.hparams.im_dim
        # number of output channels
        for i in range(len(self.hparams.channels)):
            inner_dim = int((inner_dim - (self.hparams.kernel_size[i] - 1)) / self.hparams.stride[i])
            if self.hparams.max_pool:
                if i in self.hparams.max_pool_loc:    
                    inner_dim = int(1 + (inner_dim - self.hparams.max_pool_kernel_size + 1 ) / self.hparams.max_pool_kernel_size )
    
        print("inner_dim: ", inner_dim)
        print(self.hparams.channels)
        ######################### Encoder ########################################
        for ind in range(len(self.hparams.channels)):
            if ind == 0:
                in_type  = feat_type_in        
                channel_out = self.hparams.channels[0]

            else: 
                channel_in = self.hparams.channels[ind-1]
                channel_out = self.hparams.channels[ind]
                in_type = nn.FieldType(self.gspace, channel_in * [self.gspace.trivial_repr])
            
            out_type = nn.FieldType(self.gspace, channel_out * [self.gspace.trivial_repr])
    
            print('in_type: {} out_type: {}'.format(in_type, out_type))
            
            self.encoder_conv_list.append(
                nn.R3Conv(
                    in_type, 
                    out_type, 
                    kernel_size=kernel_size[ind], 
                    padding=0, 
                    bias=True,
                )
            )
            #self.encoder_conv_list.append(nn.IIDBatchNorm3d(out_type))
            self.encoder_conv_list.append(nn.ReLU(out_type, inplace=False))  

            output_padding = 0
            if inner_dim%2 == 1 and ind == len(self.hparams.channels)-1:
                output_padding = 1

            if dropout > 0: 
                self.encoder_conv_list.append(nn.PointwiseDropout(out_type, p=dropout))

            if trigger:
                self.decoder_conv_list.append(R3Upsampling(
                    in_type, 
                    scale_factor=self.hparams.max_pool_kernel_size, 
                    mode='nearest', 
                    align_corners=False))
                trigger = 0

            if (self.hparams.max_pool and ind in self.hparams.max_pool_loc):
                self.encoder_conv_list.append(
                    nn.PointwiseAvgPoolAntialiased3D(
                        out_type, 
                        stride=self.hparams.max_pool_kernel_size,
                        sigma = 0.66))
                trigger = 1    

            if dropout > 0:
                self.decoder_conv_list.append(nn.PointwiseDropout(in_type, p=dropout))

            # decoder
            if ind == 0:
                self.decoder_conv_list.append(nn.IdentityModule(in_type))
            else: 
                self.decoder_conv_list.append(nn.ReLU(in_type, inplace=False))
            #self.decoder_conv_list.append(nn.IIDBatchNorm3d(in_type))
            self.decoder_conv_list.append(nn.R3ConvTransposed(
                out_type, 
                in_type, 
                kernel_size=kernel_size[ind], 
                output_padding=0, 
                bias=True))
            

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
                self.list_dec_fully.append(torch.nn.LeakyReLU())
            else: 
                self.list_dec_fully.append(torch.nn.ReLU())
            
            self.list_dec_fully.append(torch.nn.Linear(h_out, h_in))

        # sampling layers
        #self.fc_mu = torch.nn.Linear(self.hparams.fully_connected_layers[-1], self.hparams.latent_dim)
        #self.fc_var = torch.nn.Linear(self.hparams.fully_connected_layers[-1], self.hparams.latent_dim)
        h_out = self.hparams.latent_dim
        #torch.nn.init.zeros_(self.fc_var.bias)
        
        self.list_dec_fully.append(
            torch.nn.Linear(self.hparams.latent_dim, self.hparams.fully_connected_layers[-1]))
        self.list_enc_fully.append(
            torch.nn.Linear(self.hparams.fully_connected_layers[-1], self.hparams.latent_dim))
        self.list_enc_fully.append(torch.nn.ReLU())

        # reverse the list
        self.list_dec_fully.reverse()
        self.decoder_conv_list.reverse()
        self.encoder_fully_net = torch.nn.Sequential(*self.list_enc_fully)
        self.decoder_fully_net = torch.nn.Sequential(*self.list_dec_fully)
        summary(self.encoder_fully_net, (self.hparams.channels[-1], inner_dim, inner_dim, inner_dim), device="cpu")
        summary(self.decoder_fully_net, tuple([latent_dim]), device="cpu")        
        self.encoder = nn.SequentialModule(*self.encoder_conv_list)
        self.decoder = nn.SequentialModule(*self.decoder_conv_list)

    def encode(self, x):
        x = self.feat_type_in(x)
        x = self.encoder(x)
        x = x.tensor
        x = x.reshape(x.shape[0], -1)
        x = self.encoder_fully_net(x)
        return x 


    def decode(self, x):
        x = self.decoder_fully_net(x)
        #x = x.reshape(x.shape[0], self.channels_inner, self.inner_dim, self.inner_dim, self.inner_dim)
        x = self.dense_out_type(x)
        x = self.decoder(x)
        x = x.tensor
        return x


    def latent(self, x): 
        x = self.encode(x)
        return x 


    def forward(self, x: torch.Tensor):
        x = self.encode(x)
        x = self.decode(x)
        return x


    def loss_function(self, x, x_hat): 
        return MSELoss()(x_hat, x).to(self.device)


    def training_step(self, batch, batch_idx):
        predict = self.forward(batch)
        loss = self.loss_function(batch, predict)
        rmse_loss = torch.sqrt(loss)
        mape = torch.mean(torch.abs((predict - batch) / (torch.abs(batch) + 1e-8)))
        medpe = torch.median(torch.abs((predict - batch) / (torch.abs(batch) + 1e-8)))
        out_dict = {
            "train_loss": loss,
            "rmse_train": rmse_loss,
            "mape_train": mape,
            "medpe_train": medpe
            }     
        if self.hparams.log_wandb:wandb.log(out_dict)
        self.log_dict(out_dict)
          
        return loss


    def test_step(self, batch, batch_idx):
        predict = self.forward(batch)
        loss = self.loss_function(batch, predict)
        rmse_loss = torch.sqrt(loss)
        mape = torch.mean(torch.abs((predict - batch) / (torch.abs(batch) + 1e-8)))
        medpe = torch.median(torch.abs((predict - batch) / (torch.abs(batch) + 1e-8)))
        out_dict = {
            "test_loss": loss,
            "test_mape": mape,
            "mape_test": medpe,
            "rmse_test": rmse_loss
        }
        if self.hparams.log_wandb:wandb.log(out_dict)
        self.log_dict(out_dict)
        return loss
    

    def validation_step(self, batch, batch_idx):
        predict = self.forward(batch)
        loss = self.loss_function(batch, predict)
        rmse_loss = torch.sqrt(loss)
        mape = torch.mean(torch.abs((predict - batch) / (torch.abs(batch) + 1e-8)))
        medpe = torch.median(torch.abs((predict - batch) / (torch.abs(batch) + 1e-8)))
        out_dict = {
            "val_loss": loss,
            "rmse_val": rmse_loss,
            "mape_val": mape, 
            'medpe_val': medpe,

        }
        if self.hparams.log_wandb:wandb.log(out_dict)
        self.log_dict(out_dict)
        return loss


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

