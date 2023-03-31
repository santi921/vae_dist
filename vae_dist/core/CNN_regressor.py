import torch, wandb           
from torch import nn                                           
import pytorch_lightning as pl
from torch.nn import functional as F
from torchsummary import summary
import torchmetrics

from vae_dist.core.metrics import *
from vae_dist.core.layers import UpConvBatch, ConvBatch, ResNetBatch


class CNNRegressor(pl.LightningModule):
    def __init__(
        self, 
        channels,
        dilation,
        bias,
        padding_mode,
        activation,
        dropout,
        learning_rate, 
        groups, 
        kernel_size_in=5,
        latent_dim=3, # three output dimension for classification
        max_pool=False, 
        batch_norm=False,
        max_pool_kernel_size_in=2,
        max_pool_loc_in=[3],
        stride_in=[1],
        padding=0, 
        fully_connected_layers=[64, 64, 64],
        log_wandb=False,
        im_dim=21,
        reconstruction_loss='mse',
        **kwargs
        ):

        super().__init__()
        self.learning_rate = learning_rate
        params = {
            'channels': channels,
            'padding': padding,
            'padding_mode': padding_mode,
            'dilation': dilation,
            'groups': groups,
            'bias': bias,
            'stride_in': stride_in, 
            'latent_dim': latent_dim,
            'activation': activation,
            'batch_norm': batch_norm,
            'fully_connected_layers': fully_connected_layers,
            'activation': 'relu', 
            'dropout': dropout,
            'learning_rate': learning_rate,
            'log_wandb': log_wandb,
            'im_dim': im_dim,
            'max_pool': max_pool,
            'kernel_size_in': kernel_size_in,
            'max_pool_kernel_size_in': max_pool_kernel_size_in,
            'max_pool_loc_in': max_pool_loc_in,
            'reconstruction_loss': reconstruction_loss
        }

        assert len(channels)-1 == len(stride_in), "channels and stride must be the same length"
        assert len(stride_in)  == len(kernel_size_in), "stride and kernel_size must be the same length"
        
        self.hparams.update(params)
        self.list_enc_fully = []
        self.list_enc_conv = []
        modules_enc = []
        
        
        inner_dim = im_dim
        for i in range(len(self.hparams.channels)-1):
            inner_dim = int((inner_dim - (self.hparams.kernel_size_in[i] - 1)) / self.hparams.stride_in[i])
            if self.hparams.max_pool:
                if i in self.hparams.max_pool_loc_in:    
                    inner_dim = int(1 + (inner_dim - self.hparams.max_pool_kernel_size_in + 1 ) / self.hparams.max_pool_kernel_size_in)
    
        print("inner_dim: ", inner_dim)

        
        for ind, h in enumerate(self.hparams.fully_connected_layers):
            # if it's the last item in the list, then we want to output the latent dim
                
            if ind == 0:
                self.list_enc_fully.append(torch.nn.Flatten())
                h_in = self.hparams.channels[-1] * inner_dim * inner_dim * inner_dim
                h_out = h

            else:
                h_in = self.hparams.fully_connected_layers[ind-1]
                h_out = h

            self.list_enc_fully.append(torch.nn.Linear(h_in , h_out))
            self.list_enc_fully.append(torch.nn.LeakyReLU())
            
            if self.hparams.batch_norm and ind != 0:
                self.list_enc_fully.append(torch.nn.BatchNorm1d(h_out))
            
            if self.hparams.dropout > 0:
                self.list_enc_fully.append(torch.nn.Dropout(self.hparams.dropout))
                
        
        self.list_enc_fully.append(
            torch.nn.Linear(self.hparams.fully_connected_layers[-1], self.hparams.latent_dim))
        self.list_enc_fully.append(torch.nn.Softmax(dim = 1))

        for ind in range(len(self.hparams.channels)-1):
            channel_in = self.hparams.channels[ind]
            channel_out = self.hparams.channels[ind+1]

            output_padding = 0
            if inner_dim%2 == 1 and ind == len(self.hparams.channels)-1:
                output_padding = 1

            self.list_enc_conv.append(
                ConvBatch(
                        in_channels = channel_in,
                        out_channels = channel_out,
                        kernel_size = kernel_size_in[ind], 
                        stride = stride_in[ind],
                        padding = self.hparams.padding,
                        dilation = self.hparams.dilation,
                        groups = self.hparams.groups,
                        bias = self.hparams.bias,
                        padding_mode = self.hparams.padding_mode,
                )
            )


            if dropout > 0: 
                self.list_enc_conv.append(torch.nn.Dropout(dropout))

            if (self.hparams.max_pool and ind in self.hparams.max_pool_loc_in):
                self.list_enc_conv.append(
                    torch.nn.MaxPool3d(
                    self.hparams.max_pool_kernel_size_in))
 

        # reverse the list
        [modules_enc.append(item) for item in self.list_enc_conv]
        [modules_enc.append(item) for item in self.list_enc_fully]

        self.encoder_fully_net = torch.nn.Sequential(*self.list_enc_fully)
        self.encoder_conv = nn.Sequential(*self.list_enc_conv)
        self.encoder = nn.Sequential(*modules_enc)
        
        # this will need to be generalized for scalar fields
        self.example_input_array = torch.rand(1, 3, self.hparams.im_dim, self.hparams.im_dim, self.hparams.im_dim)


        
        summary(self.encoder_conv, (self.hparams.channels[0], im_dim, im_dim, im_dim), device="cpu")
        summary(self.encoder,      (self.hparams.channels[0], im_dim, im_dim, im_dim), device="cpu")

        self.train_acc = torchmetrics.Accuracy(task = 'multiclass', num_classes=self.hparams.latent_dim)
        self.val_acc = torchmetrics.Accuracy(task = 'multiclass', num_classes=self.hparams.latent_dim)
        self.test_acc = torchmetrics.Accuracy(task = 'multiclass', num_classes=self.hparams.latent_dim)
        self.train_f1 = torchmetrics.F1Score(task = 'multiclass', num_classes = self.hparams.latent_dim)
        self.val_f1 = torchmetrics.F1Score(task = 'multiclass', num_classes = self.hparams.latent_dim)
        self.test_f1 = torchmetrics.F1Score(task = 'multiclass', num_classes = self.hparams.latent_dim)
        self.loss = torch.nn.CrossEntropyLoss()


    def encode(self, x):
        return self.encoder(x)

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        return x

    def compute_loss(self, predict, label):
        loss = self.loss(predict, label)
        return loss
    
    def shared_step(self, batch, mode):
        batch, label = batch
        predict = self.forward(batch)
        label = label.reshape(-1)
    
        if mode == 'train':
            self.train_acc.update(predict, label)
            self.train_f1.update(predict, label)
        elif mode == 'val':
            self.val_acc.update(predict, label)
            self.val_f1.update(predict, label)
        elif mode == 'test':
            self.test_acc.update(predict, label)
            self.test_f1.update(predict, label)
        
        loss = self.compute_loss(predict, label)
        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(label))
        return loss 

    def compute_metrics(self, mode):
        if mode == 'train':
            acc = self.train_acc.compute()
            f1 = self.train_f1.compute()
            #loss = self.train_loss.compute()
            self.train_acc.reset()
            self.train_f1.reset()
            #self.train_loss.reset()

        elif mode == 'val':
            acc = self.val_acc.compute()
            f1 = self.val_f1.compute()
            #loss = self.val_loss.compute()
            self.val_acc.reset()
            self.val_f1.reset()
            #self.val_loss.reset()

        elif mode == 'test':
            acc = self.test_acc.compute()
            f1 = self.test_f1.compute()
            #loss = self.test_loss.compute()
            self.test_acc.reset()
            self.test_f1.reset()
            #self.test_loss.reset()

        return acc, f1
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, mode='train')

    def training_epoch_end(self, outputs):
        acc, f1 = self.compute_metrics(mode='train')
        out_dict = {
            "train_f1": f1,
            "train_acc": acc
        }
        if self.hparams.log_wandb:wandb.log(out_dict)
        self.log_dict(out_dict, prog_bar=True)

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, mode='test')

    def test_epoch_end(self, outputs):
        acc, f1 = self.compute_metrics(mode='test')
        out_dict = {
            "test_f1": f1,
            "test_acc": acc
        }
        if self.hparams.log_wandb:wandb.log(out_dict)
        self.log_dict(out_dict)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, mode='val')

    def validation_epoch_end(self, outputs):
        acc, f1 = self.compute_metrics(mode='val')
        out_dict = {
            "val_f1": f1,
            "val_acc": acc
        }
        if self.hparams.log_wandb:wandb.log(out_dict)
        self.log_dict(out_dict, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        #optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=0.5, 
            patience=30, 
            verbose=True, 
            threshold=0.0001, 
            threshold_mode='rel', 
            cooldown=0, 
            min_lr=1e-06, 
            eps=1e-08
            )
        lr_scheduler = {
            "scheduler": scheduler, 
            "monitor": "val_f1"
            }
        return [optimizer], [lr_scheduler]

    def load_model(self, path):
        self.load_state_dict(torch.load(path, map_location='cuda:0'), strict=False)
        print('Model Created!')