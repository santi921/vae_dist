import torch.nn as nn
import torch, wandb
#from torchConvNd import ConvNd, ConvTransposeNd
from torchsummary import summary
from vae_dist.core.layers import UpConvBatch, ConvBatch, ResNetBatch
import pytorch_lightning as pl

class CNNAutoencoderLightning(pl.LightningModule):
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
        max_pool=False, 
        max_pool_kernel_size=2,
        max_pool_loc=[3],
        log_wandb=False,
        im_dim=21,
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
            'fully_connected_layers': fully_connected_layers,
            'activation': activation,
            'dropout': dropout,
            'batch_norm': batch_norm,
            'learning_rate': learning_rate,
            'log_wandb': log_wandb,
            'im_dim': im_dim,
            'max_pool': max_pool,
            'max_pool_kernel_size': max_pool_kernel_size,
            'max_pool_loc': max_pool_loc
        }
        # asset len(channels) == len(kernel_size)
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

        self.list_enc_fully.append(torch.nn.Linear(self.hparams.fully_connected_layers[-1] , self.hparams.latent_dim))
        self.list_enc_fully.append(torch.nn.ReLU())
        self.list_dec_fully.append(torch.nn.Linear(self.hparams.latent_dim, self.hparams.fully_connected_layers[-1]))

        
        for ind in range(len(self.hparams.channels)-1):
            channel_in = self.hparams.channels[ind]
            channel_out = self.hparams.channels[ind+1]
            kernel = self.hparams.kernel_size[ind]
            stride = self.hparams.stride[ind]
            print( "channel_in: ", channel_in)
            print( "channel_out: ", channel_out)
            print( "kernel: ", kernel)
            print( "stride: ", stride)
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
                        output_padding=output_padding
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
        

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


    def encode(self, x):
        return self.encoder(x)


    def latent(self, x):
        return self.encode(x)


    def decode(self, x):
        return self.decoder(x)


    def loss_function(self, x, x_hat): 
        return nn.MSELoss()(x_hat, x)
        

    def training_step(self, batch, batch_idx):
        predict = self.forward(batch)
        loss = self.loss_function(batch, predict)
        rmse_loss = torch.sqrt(loss)
        mape = torch.mean(torch.abs((predict - batch) / torch.abs(batch)))
        medpe = torch.median(torch.abs((predict - batch) / torch.abs(batch)))
        
        out_dict = {
            'train_loss': loss, 
            'rmse_train': rmse_loss,
            'mape_train': mape,
            'medpe_train': medpe

        }
        wandb.log(out_dict)
        self.log_dict(out_dict)
        return loss


    def test_step(self, batch, batch_idx):
        predict = self.forward(batch)
        loss = self.loss_function(batch, predict)
        rmse_loss = torch.sqrt(loss)
        mape = torch.mean(torch.abs((predict - batch) / torch.abs(batch)))
        medpe = torch.median(torch.abs((predict - batch) / torch.abs(batch)))
        out_dict = {
            'test_loss': loss, 
            'rmse_test': rmse_loss,
            'mape_test': mape,
            'medpe_test': medpe
        }
        wandb.log(out_dict)
        self.log_dict(out_dict)
        return loss
    

    def validation_step(self, batch, batch_idx):
        predict = self.forward(batch)
        loss = self.loss_function(batch, predict)
        rmse_loss = torch.sqrt(loss)
        mape = torch.mean(torch.abs((batch - predict) / torch.abs(batch)))
        medpe = torch.median(torch.abs((batch - predict) / torch.abs(batch)))
        out_dict = {
            'val_loss': loss,
            'rmse_val': rmse_loss,
            'mape_val': mape,
            'medpe_val': medpe
        }
        wandb.log(out_dict)
        self.log_dict(out_dict)
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
    

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location='cuda:0'), strict=False)
        print('Model Created!')