from escnn import nn                                         
import torch                                                      
import pytorch_lightning as pl

from torchsummary import summary
from torch.nn import MSELoss

from vae_dist.core.escnnlayers import R3Upsampling

class R3CNN(pl.LightningModule):
    def __init__(
        self, 
        learning_rate, 
        group, 
        feat_type_in, 
        feat_type_out, 
        kernel_size=5, 
        latent_dim=1, 
        fully_connected_dims = [64]):

        #super(self).__init__()
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
            'fully_connected_dims': fully_connected_dims
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

        self.group = group     
        self.feat_type_in  = feat_type_in
        self.feat_type_out = feat_type_out
        self.feat_type_hidden = nn.FieldType(self.group, latent_dim*[self.group.trivial_repr])
        self.channels_inner = 48
        self.channels_outer = 24
        
        # convolution 1
    

        ######################### Encoder ########################################
        in_type_og  = feat_type_in        
        
        out_type = nn.FieldType(self.group, self.channels_outer*[self.group.trivial_repr])
        # we choose 24 feature fields, each transforming under the regular representation of C8        
        self.encoder_conv_list.append(nn.R3Conv(in_type_og, out_type, kernel_size=kernel_size, padding=0, bias=False))
        self.encoder_conv_list.append(nn.ReLU(out_type, inplace=True))
        self.encoder_conv_list.append(nn.PointwiseAvgPoolAntialiased3D(out_type, sigma=0.66, stride=3))

        in_type = out_type
        out_type = nn.FieldType(self.group, self.channels_inner*[self.group.trivial_repr])
        self.encoder_conv_list.append(nn.R3Conv(in_type, out_type, kernel_size=kernel_size, padding=0, bias=False))
        self.encoder_conv_list.append(nn.ReLU(out_type, inplace=True))
        self.encoder_conv_list.append(nn.PointwiseAvgPoolAntialiased3D(out_type, sigma=0.66, stride=3))
        self.encoder_conv_list.append(nn.GroupPooling(out_type))
        self.encoder = nn.SequentialModule(*self.encoder_conv_list)

        # number of output channels
        c = self.channels_inner
        self.im_dim = 1

        for ind, h in enumerate(fully_connected_dims):
            # if it's the last item in the list, then we want to output the latent dim
            if ind == len(fully_connected_dims)-1:
                h_out = latent_dim
            else: 
                h_out = fully_connected_dims[ind+1]

            if ind == 0:
                h_in = c*self.im_dim*self.im_dim*self.im_dim
            else:
                h_in = h
            
            self.list_enc_fully.append(torch.nn.Linear(h_in, h_out))
            self.list_enc_fully.append(torch.nn.Sigmoid())
            self.list_dec_fully.append(torch.nn.Sigmoid())
            self.list_dec_fully.append(torch.nn.Linear(h_out, h_in))
            print(h_in, h_out)

        # reverse the list
        self.list_dec_fully.reverse()
            
        self.encoder_fully_net = torch.nn.Sequential(*self.list_enc_fully)
        self.decoder_fully_net = torch.nn.Sequential(*self.list_dec_fully)
            

        self.dense_out_type = nn.FieldType(group,  self.channels_inner * [self.group.trivial_repr])
        out_type = nn.FieldType(self.group, self.channels_outer*[self.group.trivial_repr])
        
        
        #self.decoder_conv_list.append(R3Upsampling(out_type, scale_factor=2, mode='nearest', align_corners=False))
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

        self.decoder = nn.SequentialModule(*self.decoder_conv_list)
        #self.model = nn.SequentialModule(self.encoder, self.encoder_fully_net, self.decoder_fully_net, self.decoder)

    def encode(self, x):
        x = self.feat_type_in(x)
        x = self.encoder(x)
        x = x.tensor
        x = x.reshape(x.shape[0], -1)
        x = self.encoder_fully_net(x)
        return x 


    def decode(self, x):
        x = self.decoder_fully_net(x)
        x = x.reshape(x.shape[0], self.channels_inner, self.im_dim, self.im_dim, self.im_dim)
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
        mape = torch.mean(torch.abs((predict - batch) / torch.abs(batch)))
        #self.log("train_loss", loss)     
        self.log_dict(
            {
                "training_loss": loss,
                "training_mape": mape
            })     
        return loss


    def test_step(self, batch, batch_idx):
        predict = self.forward(batch)
        loss = self.loss_function(batch, predict)
        mape = torch.mean(torch.abs((predict - batch) / torch.abs(batch)))
        self.log_dict(
            {
                "test_loss": loss,
                "test_mape": mape
            })        
        
        return loss
    

    def validation_step(self, batch, batch_idx):
        predict = self.forward(batch)
        loss = self.loss_function(batch, predict)
        #self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        mape = torch.mean(torch.abs((predict - batch) / torch.abs(batch)))
        self.log_dict(
            {
                "val_loss": loss,
                "val_mape": mape
            })
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

