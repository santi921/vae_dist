import torch 
from escnn import gspaces, nn, group                                        

from vae_dist.core.O3VAE import R3VAE
from vae_dist.core.R3CNN import R3CNN
from vae_dist.core.VAE import baselineVAEAutoencoder
from vae_dist.core.CNN import CNNAutoencoderLightning
from escnn import gspaces, nn, group                                         
import pytorch_lightning as pl
import numpy as np 

class LogParameters(pl.Callback):
    # weight and biases to tensorbard
    def __init__(self):
        super().__init__()

    def on_fit_start(self, trainer, pl_module):
        self.d_parameters = {}
        for n,p in pl_module.named_parameters():
            self.d_parameters[n] = []

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.sanity_checking: # WARN: sanity_check is turned on by default
            lp = []
            for n,p in pl_module.named_parameters():
                trainer.logger.experiment.add_histogram(n, p.data, trainer.current_epoch)
                self.d_parameters[n].append(p.ravel().cpu().numpy())
                lp.append(p.ravel().cpu().numpy())
            p = np.concatenate(lp)
            trainer.logger.experiment.add_histogram('Parameters', p, trainer.current_epoch)
            


def train(model, data_loader, epochs=20):
    opt = torch.optim.Adam(model.parameters(),
                             lr = 1e-1,
                             weight_decay = 1e-8)

    for epoch in range(epochs):
        running_loss = 0.0
        for x in data_loader:            
            predict = model(x)
            loss = model.loss_function(x, predict)
            loss.backward()
            opt.step()
            running_loss += loss.item()
        print("epoch: {} loss: {}".format(epoch, running_loss))
    

def construct_model(model, options):
    """
    Constructs a model based on the model name and options.
    Takes 
        model: string, name of the model
        options: dict, options for the model
    Returns
        model: pl.LightningModule, the model
    """
    if model == 'esvae':
        #g = group.so3_group()
        #g = group.DihedralGroup(4)
        g = group.so3_group()
        gspace = gspaces.flipRot3dOnR3(maximum_frequency=16) 
        input_out_reps = 3*[gspace.trivial_repr]
        feat_type_in  = nn.FieldType(gspace,  input_out_reps) 
        feat_type_out = nn.FieldType(gspace,  input_out_reps) 
        model = R3VAE(**options, gspace=gspace, group=g, feat_type_in=feat_type_in, feat_type_out=feat_type_out)
    
    elif model == 'escnn':
        #g = group.so3_group()
        #g = group.DihedralGroup(4)
        g = group.so3_group()
        gspace = gspaces.flipRot3dOnR3(maximum_frequency=16) 

        input_out_reps = 3*[gspace.trivial_repr]
        feat_type_in  = nn.FieldType(gspace,  input_out_reps) 
        feat_type_out = nn.FieldType(gspace,  input_out_reps)  
        model = R3CNN(**options, gspace=gspace, group=g, feat_type_in=feat_type_in, feat_type_out=feat_type_out)   

    elif model == 'cnn':
        model = CNNAutoencoderLightning(**options)

    elif model == 'vae':
        model = baselineVAEAutoencoder(**options)

    return model


def construct_model_hyper(model, options):
    """
    Constructs a model based on the model name and options.
    Takes 
        model: string, name of the model
        options: dict, options for the model
    Returns
        model: pl.LightningModule, the model
    """
    if model == 'esvae':
        options_non_wandb = {
            "log_wandb": True,
            "im_dim": 21,
            "activation": "relu",
            "groups": 1,
            "padding": 0,
            "dilation": 1,
            "irreps": None 
        }
        g = group.o3_group()
        gspace = gspaces.flipRot3dOnR3(maximum_frequency=10) 
        input_out_reps = 3*[gspace.trivial_repr]
        feat_type_in  = nn.FieldType(gspace,  input_out_reps) 
        feat_type_out = nn.FieldType(gspace,  input_out_reps) 
        options.update(options['architecture'])
        options.update(options_non_wandb)
        model = R3VAE(**options, gspace=gspace, group=g, feat_type_in=feat_type_in, feat_type_out=feat_type_out)
    
    elif model == 'escnn':
        options_non_wandb = {
            "log_wandb": True,
            "im_dim": 21,
            "activation": "relu",
            "groups": 1,
            "padding": 0,
            "dilation": 1,
            "irreps": None 
        }
        g = group.o3_group()
        gspace = gspaces.flipRot3dOnR3(maximum_frequency=10) 
        #gspace = gspace.no_base_space(g)
        input_out_reps = 3*[gspace.trivial_repr]
        #kernel_size = 5
        feat_type_in  = nn.FieldType(gspace,  input_out_reps) 
        feat_type_out = nn.FieldType(gspace,  input_out_reps)  
        options.update(options['architecture'])
        options.update(options_non_wandb)
        model = R3CNN(**options, gspace=gspace, group=g, feat_type_in=feat_type_in, feat_type_out=feat_type_out)   

    elif model == 'auto':
        options_non_wandb = {
            "log_wandb": True,
            "im_dim": 21,
            "activation": "relu",
            "groups": 1,
            "padding": 0,
            "dilation": 1,
            "irreps": None 
        }
        options.update(options['architecture'])
        options.update(options_non_wandb)
        model = CNNAutoencoderLightning(**options)

    elif model == 'vae':
        options_non_wandb = {
            "log_wandb": True,
            "im_dim": 21,
            "activation": "relu",
            "groups": 1,
            "padding": 0,
            "dilation": 1,
            "irreps": None 
        }
        options.update(options['architecture'])
        options.update(options_non_wandb)
        model = baselineVAEAutoencoder(**options)

    return model


def hyperparameter_dicts():
    dict_ret = {}
    
    dict_escnn = {
        "architecture": {"values":[
                    {
                        "channels": [32, 64, 128, 256, 512],
                        "kernel_size": [4, 3, 3, 3, 3],
                        "stride":      [1, 1, 1, 1, 1],
                        "max_pool": True, 
                        "max_pool_kernel_size": 2, 
                        "max_pool_loc": [2],
                        "padding_mode": "zeros"
                    },
                    {
                        "channels": [32, 64, 128],
                        "kernel_size": [4, 5, 4],
                        "stride":      [1, 1, 1],
                        "max_pool": True, 
                        "max_pool_kernel_size": 2, 
                        "max_pool_loc": [1, 3],
                        "padding_mode": "zeros"
                    },

                    ]
                },
        "bias": {"values": [True]},
        "epochs": {"values": [100, 500, 1000]},
        "latent_dim": {"values": [2, 5, 10, 25]},
        "fully_connected_layers": {"values": [[1300, 10], [100], [100, 50, 10], [50]]},
        "batch_norm": {"values": [True, False]},
        "dropout": {'values': [0.0, 0.1, 0.25, 0.4]},
        "learning_rate": {"min": 0.05, "max": 0.5, "distribution": "log_uniform_values"},   
        "reconstruction_loss": {"values": ["mse", "l1", "huber", "inverse_huber", "many_step_inverse_huber"]},


    }
    
    dict_esvae = {
        "beta": {"values": [0.001, 0.01,1, 10, 100]},
        "architecture": {"values":[
                    {
                        "channels": [32, 64, 128, 256, 512],
                        "kernel_size": [4, 3, 3, 3, 3],
                        "stride":      [1, 1, 1, 1, 1],
                        "max_pool": True, 
                        "max_pool_kernel_size": 2, 
                        "max_pool_loc": [2],
                        "padding_mode": "zeros"
                    },

                    {
                        "channels": [32, 64, 128],
                        "kernel_size": [4, 5, 4],
                        "stride":      [1, 1, 1],
                        "max_pool": True, 
                        "max_pool_kernel_size": 2, 
                        "max_pool_loc": [1, 3],
                        "padding_mode": "zeros"
                    },

                    ]
                },
        "bias": {"values": [True]},
        "epochs": {"values": [100, 500, 1000]},
        "latent_dim": {"values": [2, 5, 10, 25]},
        "fully_connected_layers": {"values": [[100, 10], [100], [100, 50, 10], [50]]},
        "batch_norm": {"values": [True, False]},
        "dropout": {'values': [0.0, 0.1, 0.25, 0.4]},
        "learning_rate": {"min": 0.05, "max": 0.5, "distribution": "log_uniform_values"},   
        "reconstruction_loss": {"values": ["mse", "l1", "huber", "inverse_huber", "many_step_inverse_huber"]},
    }

    dict_vae = {
        "beta": {"values": [0.001, 0.01,1, 10, 100]},
         "architecture": {"values":[
                    {
                        "channels": [3, 32, 64, 128, 256, 512],
                        "kernel_size": [5, 4, 3, 2, 2],
                        "stride":      [1, 1, 1, 1, 1],
                        "max_pool": True, 
                        "max_pool_kernel_size": 2, 
                        "max_pool_loc": [2],
                        "padding_mode": "zeros"
                    },
                    {
                        "channels": [3, 8, 16, 32, 64, 128],
                        "kernel_size": [5, 4, 3, 2, 2],
                        "stride":      [1, 1, 1, 1, 1],
                        "max_pool": True, 
                        "max_pool_kernel_size": 2, 
                        "max_pool_loc": [2],
                        "padding_mode": "zeros"
                    },
                    {
                        "channels": [3, 32, 64, 128],
                        "kernel_size": [4, 5, 4],
                        "stride":      [1, 1, 1],
                        "max_pool": True, 
                        "max_pool_kernel_size": 2, 
                        "max_pool_loc": [1, 3],
                        "padding_mode": "zeros"
                    },
                    {
                        "channels": [3, 32, 64, 128, 256, 512],
                        "kernel_size": [2, 2, 2, 2, 2],
                        "stride":      [1, 1, 1, 1, 1],
                        "max_pool": True, 
                        "max_pool_kernel_size": 2, 
                        "max_pool_loc": [0, 2],
                        "padding_mode": "zeros"
                    }
                    ]
                },
        "bias": {"values": [True]},
        "epochs": {"values": [100, 500, 1000]},
        "latent_dim": {"values": [2, 5, 10, 25]},
        "fully_connected_layers": {"values": [[100, 10], [100], [100, 50, 10], [50]]},
        "batch_norm": {"values": [True, False]},
        "dropout": {'values': [0.0, 0.1, 0.25, 0.4]},
        "learning_rate": {"min": 0.05, "max": 0.5, "distribution": "log_uniform_values"},   
    }

    dict_auto = {
        "architecture": {"values":[
                    {
                        "channels": [3, 32, 64, 128, 256, 512],
                        "kernel_size": [5, 4, 3, 2, 2],
                        "stride":      [1, 1, 1, 1, 1],
                        "max_pool": True, 
                        "max_pool_kernel_size": 2, 
                        "max_pool_loc": [2],
                        "padding_mode": "zeros"
                    },
                    {
                        "channels": [3, 8, 16, 32, 64, 128],
                        "kernel_size": [5, 4, 3, 2, 2],
                        "stride":      [1, 1, 1, 1, 1],
                        "max_pool": True, 
                        "max_pool_kernel_size": 2, 
                        "max_pool_loc": [2],
                        "padding_mode": "zeros"
                    },
                    {
                        "channels": [3, 32, 64, 128],
                        "kernel_size": [4, 5, 4],
                        "stride":      [1, 1, 1],
                        "max_pool": True, 
                        "max_pool_kernel_size": 2, 
                        "max_pool_loc": [1, 3],
                        "padding_mode": "zeros"
                    },
                    {
                        "channels": [3, 32, 64, 128, 256, 512],
                        "kernel_size": [2, 2, 2, 2, 2],
                        "stride":      [1, 1, 1, 1, 1],
                        "max_pool": True, 
                        "max_pool_kernel_size": 2, 
                        "max_pool_loc": [0, 2],
                        "padding_mode": "zeros"
                    }
                    ]
                },
        "bias": {"values": [True]},
        "epochs": {"values": [100, 500, 1000]},
        "latent_dim": {"values": [2, 5, 10, 25]},
        "fully_connected_layers": {"values": [[100, 10], [100], [100, 50, 10], [50]]},
        "batch_norm": {"values": [True, False]},
        "dropout": {'values': [0.0, 0.1, 0.25, 0.4]},
        "learning_rate": {"min": 0.05, "max": 0.5, "distribution": "log_uniform_values"},   
    }

    dict_ret['escnn'] = dict_escnn
    dict_ret['esvae'] = dict_esvae
    dict_ret['auto'] = dict_auto
    dict_ret['vae'] = dict_vae
    return dict_ret