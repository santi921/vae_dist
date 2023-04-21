import torch
from escnn import gspaces, nn, group

from vae_dist.core.intializers import *
from vae_dist.core.O3VAE import R3VAE
from vae_dist.core.R3CNN import R3CNN
from vae_dist.core.VAE import baselineVAEAutoencoder
from vae_dist.core.CNN import CNNAutoencoderLightning
from vae_dist.core.R3CNN_regressor import R3CNNRegressor
from vae_dist.core.CNN_regressor import CNNRegressor

import pytorch_lightning as pl
import numpy as np


class CheckBatchGradient(pl.Callback):
    def on_train_start(self, trainer, model):
        n = 0

        example_input = model.example_input_array.to(model.device)
        example_input.requires_grad = True

        model.zero_grad()
        output = model(example_input)
        output[n].abs().sum().backward()

        zero_grad_inds = list(range(example_input.size(0)))
        zero_grad_inds.pop(n)

        if example_input.grad[zero_grad_inds].abs().sum().item() > 0:
            raise RuntimeError("Your model mixes data across the batch dimension!")


class LogParameters(pl.Callback):
    # weight and biases to tensorbard
    def __init__(self):
        super().__init__()

    def on_fit_start(self, trainer, pl_module):
        self.d_parameters = {}
        for n, p in pl_module.named_parameters():
            self.d_parameters[n] = []

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.sanity_checking:  # WARN: sanity_check is turned on by default
            lp = []
            for n, p in pl_module.named_parameters():
                trainer.logger.experiment.add_histogram(
                    n, p.data, trainer.current_epoch
                )
                self.d_parameters[n].append(p.ravel().cpu().numpy())
                lp.append(p.ravel().cpu().numpy())
            p = np.concatenate(lp)
            trainer.logger.experiment.add_histogram(
                "Parameters", p, trainer.current_epoch
            )


class InputMonitor(pl.Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if (batch_idx + 1) % trainer.log_every_n_steps == 0:
            x, y = batch
            logger = trainer.logger
            logger.experiment.add_histogram("input", x, global_step=trainer.global_step)
            logger.experiment.add_histogram(
                "target", y, global_step=trainer.global_step
            )


def train(model, data_loader, epochs=20):
    opt = torch.optim.Adam(model.parameters(), lr=1e-1, weight_decay=1e-8)

    for epoch in range(epochs):
        running_loss = 0.0
        for x in data_loader:
            predict = model(x)
            loss = model.loss_function(x, predict)
            loss.backward()
            opt.step()
            running_loss += loss.item()
        print("epoch: {} loss: {}".format(epoch, running_loss))


def construct_model(model, options, im_dim=21, scalar_field=False):
    """
    Constructs a model based on the model name and options.
    Takes
        model: string, name of the model
        options: dict, options for the model
    Returns
        model: pl.LightningModule, the model
    """
    options_non_wandb = {
        "log_wandb": True,
        "im_dim": im_dim,
        "groups": 1,
        "dilation": 1,
    }
    assert model in [
        "esvae",
        "vae",
        "cnn",
        "escnn",
        "escnn_supervised",
        "cnn_supervised",
        "escnn_regressor",
        "escnn_regressor_supervised",
    ], "Model must be vae, cnn, escnn, escnn_supervised, cnn_supervised, escnn_regressor, escnn_regressor_supervised"
    

    dim = 3
    if options["scalar"]: dim = 1

    if model == "esvae":
        print("building esvae...")
        options.update(options["architecture"])
        options.update(options_non_wandb)
        model = R3VAE(**options)

    elif model == "escnn":
        print("building escnn...")
        options.update(options["architecture"])
        options.update(options_non_wandb)
        model = R3CNN(**options)

    elif model == "cnn":
        print("building autoencoder...")
        options.update(options["architecture"])
        options.update(options_non_wandb)
        model = CNNAutoencoderLightning(**options)

    elif model == "vae":
        print("building vae...")
        options.update(options["architecture"])
        options.update(options_non_wandb)
        model = baselineVAEAutoencoder(**options)

    elif model == "cnn_supervised":
        print("building cnn supervised...")
        options.update(options["architecture"])
        options.update(options_non_wandb)
        model = CNNRegressor(**options)

    elif model == "escnn_supervised":
        print("building escnn supervised...")
        options.update(options["architecture"])
        options.update(options_non_wandb)
        model = R3CNNRegressor(**options)

    initializer = options["initializer"]
    if initializer == "kaiming":
        print("using kaiming initialization")
        kaiming_init(model)
    elif initializer == "xavier":
        print("using xavier initialization")
        xavier_init(model)
    elif initializer == "equi_var":
        print("using equi-var initialization")
        equi_var_init(model)
    else:
        raise ValueError("Initializer must be kaiming, xavier or equi_var")

    return model


def hyperparameter_dicts(image_size=21):
    assert image_size == 21 or image_size == 51, "image size must be 21 or 51"

    dict_ret = {}

    dict_escnn = {
        "initializer": {"values": ["equi_var", "xavier", "kaiming"]},
        "bias": {"values": [True]},
        "max_epochs": {"values": [100, 500, 1000]},
        "latent_dim": {"values": [2, 5, 10, 25]},
        "fully_connected_layers": {"values": [[1300, 10], [100], [100, 50, 10], [50]]},
        "batch_norm": {"values": [True, False]},
        "dropout": {"values": [0.0, 0.1, 0.25, 0.4]},
        "learning_rate": {
            "min": 0.05,
            "max": 0.5,
            "distribution": "log_uniform_values",
        },
        "reconstruction_loss": {
            "values": ["mse", "l1", "huber", "inverse_huber", "many_step_inverse_huber"]
        },
        "padding": {"values": [0]},
    }

    dict_esvae = {
        "initializer": {"values": ["equi_var", "xavier", "kaiming"]},
        "beta": {"values": [0.001, 0.01, 1, 10, 100]},
        "bias": {"values": [True]},
        "max_epochs": {"values": [100, 500, 1000]},
        "latent_dim": {"values": [2, 5, 10, 25]},
        "fully_connected_layers": {"values": [[100, 10], [100], [100, 50, 10], [50]]},
        "batch_norm": {"values": [True, False]},
        "dropout": {"values": [0.0, 0.1, 0.25, 0.4]},
        "learning_rate": {
            "min": 0.05,
            "max": 0.5,
            "distribution": "log_uniform_values",
        },
        "reconstruction_loss": {
            "values": ["mse", "l1", "huber", "inverse_huber", "many_step_inverse_huber"]
        },
        "padding": {"values": [0]},
    }

    dict_vae = {
        "initializer": {"values": ["equi_var", "kaiming"]},
        "beta": {"values": [0.001, 0.01, 1, 10, 100]},
        "bias": {"values": [True]},
        "max_epochs": {"values": [100, 500, 1000]},
        "latent_dim": {"values": [2, 5, 10, 25]},
        "fully_connected_layers": {"values": [[100, 10], [100], [100, 50, 10], [50]]},
        "batch_norm": {"values": [True, False]},
        "dropout": {"values": [0.0, 0.1, 0.25, 0.4]},
        "learning_rate": {
            "min": 0.05,
            "max": 0.5,
            "distribution": "log_uniform_values",
        },
        "padding": {"values": [0]},
    }

    dict_auto = {
        "initializer": {"values": ["equi_var", "xavier", "kaiming"]},
        "bias": {"values": [True]},
        "max_epochs": {"values": [100, 500, 1000]},
        "latent_dim": {"values": [2, 5, 10, 25]},
        "fully_connected_layers": {"values": [[100, 10], [100], [100, 50, 10], [50]]},
        "batch_norm": {"values": [True, False]},
        "dropout": {"values": [0.0, 0.1, 0.25, 0.4]},
        "learning_rate": {
            "min": 0.05,
            "max": 0.5,
            "distribution": "log_uniform_values",
        },
        "padding": {"values": [0]},
    }

    dict_cnn_supervised = {
        "initializer": {"values": ["equi_var", "xavier", "kaiming"]},
        "irreps": {"values": [None]},
        "fully_connected_layers": {"values": [[100, 10], [100], [100, 50, 10], [50]]},
        "batch_norm": {"values": [True, False]},
        "dropout": {"values": [0.0, 0.1, 0.25, 0.4]},
        "learning_rate": {
            "min": 0.001,
            "max": 0.05,
            "distribution": "log_uniform_values",
        },
        "padding": {"values": [0]},
        "padding_mode": {"values": ["zeros"]},
        "bias": {"values": [True]},
        "max_epochs": {"values": [100, 500, 1000]},
        "latent_dim": {"values": [3]},
        "activation": {"values": ["relu"]},
        "log_wandb": {"values": [True]},
        "gradient_clip_val": {"values": [0.5, 1.0, 5.0, 10.0]},
        "accumulate_grad_batches": {"values": [1, 2, 4, 8]},
        "standardize": {"values": [False, True]},
        "lower_filter": {"values": [False, True]},
        "log_scale": {"values": [False, True]},
        "min_max_scale": {"values": [False, True]},
        "wrangle_outliers": {"values": [False, True]},
        "scalar": {"values": [False]},
    }

    dict_escnn_supervised = {
        "initializer": {"values": ["xavier", "kaiming"]},
        "irreps": {"values": [None]},
        "fully_connected_layers": {"values": [[100, 10], [100], [100, 50, 10], [50]]},
        "batch_norm": {"values": [True, False]},
        "dropout": {"values": [0.0, 0.1, 0.25, 0.4]},
        "learning_rate": {
            "min": 0.001,
            "max": 0.05,
            "distribution": "log_uniform_values",
        },
        "padding": {"values": [0]},
        "padding_mode": {"values": ["zeros"]},
        "bias": {"values": [True]},
        "max_epochs": {"values": [100, 500, 1000]},
        "latent_dim": {"values": [3]},
        "activation": {"values": ["relu"]},
        "log_wandb": {"values": [True]},
        "gradient_clip_val": {"values": [0.5, 1.0, 5.0, 10.0]},
        "accumulate_grad_batches": {"values": [1, 2, 4, 8]},
        "standardize": {"values": [False, True]},
        "lower_filter": {"values": [False, True]},
        "log_scale": {"values": [False, True]},
        "min_max_scale": {"values": [False, True]},
        "wrangle_outliers": {"values": [False, True]},
        "lr_patience": {"values": [10, 20, 50]},
        "lr_decay_factor": {"values": [0.1, 0.5, 0.8]},
        "optimizer": {"values": ["Adam", "SGD"]},
        "scalar": {"values": [False]},
    }

    if image_size == 21:
        dict_cnn_supervised["architecture"] = {
            "values": [
                {
                    "channels": [3, 256, 128, 128],
                    "kernel_size_in": [9, 9, 3],
                    "stride_in": [1, 1, 3],
                    "max_pool": False,
                    "max_pool_kernel_size_in": [0],
                    "max_pool_loc_in": [0],
                    "padding": [0],
                    "padding_mode": "zeros",
                },
                {
                    "channels": [3, 32, 64, 128],
                    "kernel_size_in": [4, 5, 4],
                    "stride_in": [1, 1, 1],
                    "max_pool": True,
                    "max_pool_kernel_size_in": 2,
                    "max_pool_loc_in": [1, 3],
                    "padding_mode": "zeros",
                },
                {
                    "channels": [3, 32, 32, 64, 64, 256],
                    "kernel_size_in": [5, 5, 5, 5, 5],
                    "stride_in": [1, 1, 1, 1, 1],
                    "max_pool": False,
                    "max_pool_kernel_size_in": 2,
                    "max_pool_loc_in": [2],
                    "padding_mode": "zeros",
                },
            ]
        }

        dict_escnn_supervised["architecture"] = {
            "values": [
                {
                    "channels": [256, 128, 128],
                    "kernel_size_in": [9, 9, 3],
                    "stride_in": [1, 1, 3],
                    "max_pool": False,
                    "max_pool_kernel_size_in": [0],
                    "max_pool_loc_in": [0],
                    "padding": [0],
                    "padding_mode": "zeros",
                },
                {
                    "channels": [32, 64, 128],
                    "kernel_size_in": [4, 5, 4],
                    "stride_in": [1, 1, 1],
                    "max_pool": True,
                    "max_pool_kernel_size_in": 2,
                    "max_pool_loc_in": [1, 3],
                    "padding_mode": "zeros",
                },
                {
                    "channels": [32, 64, 128, 256, 512],
                    "kernel_size_in": [5, 5, 5, 5, 5],
                    "stride_in": [1, 1, 1, 1, 1],
                    "max_pool": False,
                    "max_pool_kernel_size_in": 2,
                    "max_pool_loc_in": [2],
                    "padding_mode": "zeros",
                },
            ]
        }

        dict_escnn["architecture"] = (
            {
                "values": [
                    {
                        "channels": [32, 32, 64, 128, 256],
                        "kernel_size_in": [4, 3, 3, 3, 3],
                        "stride_in": [1, 1, 1, 1, 1],
                        "kernel_size_out": [3, 3, 5, 5, 5, 5],
                        "stride_out": [1, 1, 1, 1, 1, 1],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [2],
                        "padding_mode": "zeros",
                    },
                    {
                        "channels": [32, 64, 128],
                        "kernel_size_in": [5, 9, 7],
                        "stride_in": [1, 1, 3],
                        "kernel_size_out": [7, 5, 3],
                        "stride_out": [1, 2, 3],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [1, 3],
                        "padding_mode": "zeros",
                    },
                    {
                        "channels": [32, 64, 128, 256],
                        "kernel_size_in": [7, 7, 5, 5],
                        "stride_in": [1, 1, 1, 1],
                        "kernel_size_out": [7, 7, 5, 5],
                        "stride_out": [1, 1, 1, 1],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [1, 3],
                        "padding_mode": "zeros",
                    },
                    {
                        "channels": [32, 64, 256],
                        "kernel_size_in": [7, 7, 7],
                        "stride_in": [1, 1, 3],
                        "kernel_size_out": [5, 7, 7],
                        "stride_out": [1, 2, 1],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [1, 3],
                        "padding_mode": "zeros",
                    },
                ]
            },
        )

        dict_auto["architecture"] = (
            {
                "values": [
                    {
                        "channels": [3, 32, 32, 64, 128, 256],
                        "kernel_size_in": [4, 3, 3, 3, 3],
                        "stride_in": [1, 1, 1, 1, 1],
                        "kernel_size_out": [3, 3, 5, 5, 5, 5],
                        "stride_out": [1, 1, 1, 1, 1, 1],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [2],
                        "padding_mode": "zeros",
                    },
                    {
                        "channels": [3, 32, 64, 128],
                        "kernel_size_in": [5, 9, 7],
                        "stride_in": [1, 1, 3],
                        "kernel_size_out": [7, 5, 3],
                        "stride_out": [1, 2, 3],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [1, 3],
                        "padding_mode": "zeros",
                    },
                    {
                        "channels": [3, 32, 64, 128, 256],
                        "kernel_size_in": [7, 7, 5, 5],
                        "stride_in": [1, 1, 1, 1],
                        "kernel_size_out": [7, 7, 5, 5],
                        "stride_out": [1, 1, 1, 1],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [1, 3],
                        "padding_mode": "zeros",
                    },
                    {
                        "channels": [3, 32, 64, 256],
                        "kernel_size_in": [7, 7, 7],
                        "stride_in": [1, 1, 3],
                        "kernel_size_out": [5, 7, 7],
                        "stride_out": [1, 2, 1],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [1, 3],
                        "padding_mode": "zeros",
                    },
                ]
            },
        )

        dict_vae["architecture"] = (
            {
                "values": [
                    {
                        "channels": [3, 32, 32, 64, 128, 256],
                        "kernel_size_in": [4, 3, 3, 3, 3],
                        "stride_in": [1, 1, 1, 1, 1],
                        "kernel_size_out": [3, 3, 5, 5, 5, 5],
                        "stride_out": [1, 1, 1, 1, 1, 1],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [2],
                        "padding_mode": "zeros",
                    },
                    {
                        "channels": [3, 32, 64, 128],
                        "kernel_size_in": [5, 9, 7],
                        "stride_in": [1, 1, 3],
                        "kernel_size_out": [7, 5, 3],
                        "stride_out": [1, 2, 3],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [1, 3],
                        "padding_mode": "zeros",
                    },
                    {
                        "channels": [3, 32, 64, 128, 256],
                        "kernel_size_in": [7, 7, 5, 5],
                        "stride_in": [1, 1, 1, 1],
                        "kernel_size_out": [7, 7, 5, 5],
                        "stride_out": [1, 1, 1, 1],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [1, 3],
                        "padding_mode": "zeros",
                    },
                    {
                        "channels": [3, 32, 64, 256],
                        "kernel_size_in": [7, 7, 7],
                        "stride_in": [1, 1, 3],
                        "kernel_size_out": [5, 7, 7],
                        "stride_out": [1, 2, 1],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [1, 3],
                        "padding_mode": "zeros",
                    },
                ]
            },
        )

        dict_esvae["architecture"] = (
            {
                "values": [
                    {
                        "channels": [32, 32, 64, 128, 256],
                        "kernel_size_in": [4, 3, 3, 3, 3],
                        "stride_in": [1, 1, 1, 1, 1],
                        "kernel_size_out": [3, 3, 5, 5, 5, 5],
                        "stride_out": [1, 1, 1, 1, 1, 1],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [2],
                        "padding_mode": "zeros",
                    },
                    {
                        "channels": [32, 64, 128],
                        "kernel_size_in": [5, 9, 7],
                        "stride_in": [1, 1, 3],
                        "kernel_size_out": [7, 5, 3],
                        "stride_out": [1, 2, 3],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [1, 3],
                        "padding_mode": "zeros",
                    },
                    {
                        "channels": [32, 64, 128, 256],
                        "kernel_size_in": [7, 7, 5, 5],
                        "stride_in": [1, 1, 1, 1],
                        "kernel_size_out": [7, 7, 5, 5],
                        "stride_out": [1, 1, 1, 1],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [1, 3],
                        "padding_mode": "zeros",
                    },
                    {
                        "channels": [32, 64, 256],
                        "kernel_size_in": [7, 7, 7],
                        "stride_in": [1, 1, 3],
                        "kernel_size_out": [5, 7, 7],
                        "stride_out": [1, 2, 1],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [1, 3],
                        "padding_mode": "zeros",
                    },
                ]
            },
        )

    else:
        dict_escnn["architecture"] = (
            {
                "values": [
                    {
                        "channels": [256, 512, 1024],
                        "kernel_size_in": [7, 7, 3],
                        "stride_in": [3, 3, 1],
                        "kernel_size_out": [7, 5, 3],
                        "stride_out": [1, 2, 3],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [2],
                        "padding_mode": "zeros",
                    },
                    {
                        "channels": [64, 128, 128, 256],
                        "kernel_size_in": [7, 7, 5, 5],
                        "stride_in": [3, 1, 1, 1],
                        "kernel_size_out": [5, 7, 7, 9],
                        "stride_out": [1, 3, 2, 1],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [2],
                        "padding_mode": "zeros",
                    },
                    {
                        "channels": [64, 128, 128, 256],
                        "kernel_size_in": [12, 9, 5],
                        "stride_in": [2, 2, 2],
                        "kernel_size_out": [11, 11, 11],
                        "stride_out": [1, 3, 1],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [2],
                        "padding_mode": "zeros",
                    },
                ]
            },
        )

        dict_auto["architecture"] = (
            {
                "values": [
                    {
                        "channels": [3, 256, 512, 1024],
                        "kernel_size_in": [7, 7, 3],
                        "stride_in": [3, 3, 1],
                        "kernel_size_out": [7, 5, 3],
                        "stride_out": [1, 2, 3],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [2],
                        "padding_mode": "zeros",
                    },
                    {
                        "channels": [3, 64, 128, 128, 256],
                        "kernel_size_in": [7, 7, 5, 5],
                        "stride_in": [3, 1, 1, 1],
                        "kernel_size_out": [5, 7, 7, 9],
                        "stride_out": [1, 3, 2, 1],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [2],
                        "padding_mode": "zeros",
                    },
                    {
                        "channels": [3, 64, 128, 128, 256],
                        "kernel_size_in": [12, 9, 5],
                        "stride_in": [2, 2, 2],
                        "kernel_size_out": [11, 11, 11],
                        "stride_out": [1, 3, 1],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [2],
                        "padding_mode": "zeros",
                    },
                ]
            },
        )

        dict_vae["architecture"] = (
            {
                "values": [
                    {
                        "channels": [3, 256, 512, 1024],
                        "kernel_size_in": [7, 7, 3],
                        "stride_in": [3, 3, 1],
                        "kernel_size_out": [7, 5, 3],
                        "stride_out": [1, 2, 3],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [2],
                        "padding_mode": "zeros",
                    },
                    {
                        "channels": [3, 64, 128, 128, 256],
                        "kernel_size_in": [7, 7, 5, 5],
                        "stride_in": [3, 1, 1, 1],
                        "kernel_size_out": [5, 7, 7, 9],
                        "stride_out": [1, 3, 2, 1],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [2],
                        "padding_mode": "zeros",
                    },
                    {
                        "channels": [3, 64, 128, 128, 256],
                        "kernel_size_in": [12, 9, 5],
                        "stride_in": [2, 2, 2],
                        "kernel_size_out": [11, 11, 11],
                        "stride_out": [1, 3, 1],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [2],
                        "padding_mode": "zeros",
                    },
                ]
            },
        )

        dict_escnn["architecture"] = (
            {
                "values": [
                    {
                        "channels": [256, 512, 1024],
                        "kernel_size_in": [7, 7, 3],
                        "stride_in": [3, 3, 1],
                        "kernel_size_out": [7, 5, 3],
                        "stride_out": [1, 2, 3],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [2],
                        "padding_mode": "zeros",
                    },
                    {
                        "channels": [64, 128, 128, 256],
                        "kernel_size_in": [7, 7, 5, 5],
                        "stride_in": [3, 1, 1, 1],
                        "kernel_size_out": [5, 7, 7, 9],
                        "stride_out": [1, 3, 2, 1],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [2],
                        "padding_mode": "zeros",
                    },
                    {
                        "channels": [64, 128, 128, 256],
                        "kernel_size_in": [12, 9, 5],
                        "stride_in": [2, 2, 2],
                        "kernel_size_out": [11, 11, 11],
                        "stride_out": [1, 3, 1],
                        "max_pool": False,
                        "max_pool_kernel_size_in": 2,
                        "max_pool_loc_in": [2],
                        "padding_mode": "zeros",
                    },
                ]
            },
        )

        dict_escnn_supervised["architecture"] = {
            "values": [
                {
                    "channels": [256, 512, 1024],
                    "kernel_size_in": [7, 7, 3],
                    "stride_in": [3, 3, 1],
                    "max_pool": False,
                    "max_pool_kernel_size_in": 2,
                    "max_pool_loc_in": [2],
                    "padding_mode": "zeros",
                },
                {
                    "channels": [64, 128, 128, 256],
                    "kernel_size_in": [7, 7, 5, 5],
                    "stride_in": [3, 1, 1, 1],
                    "max_pool": False,
                    "max_pool_kernel_size_in": 2,
                    "max_pool_loc_in": [2],
                    "padding_mode": "zeros",
                },
                {
                    "channels": [64, 128, 128, 256],
                    "kernel_size_in": [12, 9, 5],
                    "stride_in": [2, 2, 2],
                    "max_pool": False,
                    "max_pool_kernel_size_in": 2,
                    "max_pool_loc_in": [2],
                    "padding_mode": "zeros",
                },
            ]
        }

        dict_cnn_supervised["architecture"] = {
            "values": [
                {
                    "channels": [3, 256, 512, 1024],
                    "kernel_size_in": [7, 7, 3],
                    "stride_in": [3, 3, 1],
                    "max_pool": False,
                    "max_pool_kernel_size_in": 2,
                    "max_pool_loc_in": [2],
                    "padding_mode": "zeros",
                },
                {
                    "channels": [3, 64, 128, 128, 256],
                    "kernel_size_in": [7, 7, 5, 5],
                    "stride_in": [3, 1, 1, 1],
                    "max_pool": False,
                    "max_pool_kernel_size_in": 2,
                    "max_pool_loc_in": [2],
                    "padding_mode": "zeros",
                },
                {
                    "channels": [3, 64, 128, 128, 256],
                    "kernel_size_in": [12, 9, 5],
                    "stride_in": [2, 2, 2],
                    "max_pool": False,
                    "max_pool_kernel_size_in": 2,
                    "max_pool_loc_in": [2],
                    "padding_mode": "zeros",
                },
            ]
        }

    dict_ret["escnn"] = dict_escnn
    dict_ret["esvae"] = dict_esvae
    dict_ret["auto"] = dict_auto
    dict_ret["vae"] = dict_vae
    dict_ret["cnn_supervised"] = dict_cnn_supervised
    dict_ret["escnn_supervised"] = dict_escnn_supervised

    return dict_ret
