import torch
from escnn import gspaces, nn, group

from vae_dist.core.intializers import xavier_init, kaiming_init, equi_var_init
from vae_dist.core.O3VAE import R3VAE
from vae_dist.core.R3CNN import R3CNN
from vae_dist.core.VAE import baselineVAEAutoencoder
from vae_dist.core.CNN import CNNAutoencoderLightning
from vae_dist.core.R3CNN_regressor import R3CNNRegressor
from vae_dist.core.CNN_regressor import CNNRegressor
from vae_dist.core.parameters import hyperparameter_dicts, pull_escnn_params
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


def construct_model(
    model: str, options: dict, im_dim: int = 21, scalar_field: bool = False
):
    """
    Constructs a model based on the model name and options.
    Takes
        model: string, name of the model
        options: dict, options for the model
    Returns
        model: pl.LightningModule, the model
    """
    keys = options.keys()
    if "log_wandb" not in keys:
        options.update(options_non_wandb)
    if "groups" not in keys:
        options["groups"] = 1
    if "dilation" not in keys:
        options["dilation"] = 1
    if "im_dim" not in keys:
        options["im_dim"] = im_dim

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
    if options["scalar"]:
        dim = 1

    options.update(options["architecture"])

    if model in ["escnn", "esvae", "escnn_supervised"]:
        options["escnn_params"] = {
            "scalar": options["scalar"],
            "escnn_group": options["escnn_group"],
            "max_freq": options["max_freq"],
            "flips_r3": options["flips_r3"],
            "l_max": options["l_max"],
        }
    print(options)

    if model == "esvae":
        print("building esvae...")
        model = R3VAE(**options)

    elif model == "escnn":
        print("building escnn...")
        model = R3CNN(**options)

    elif model == "cnn":
        print("building autoencoder...")
        model = CNNAutoencoderLightning(**options)

    elif model == "vae":
        print("building vae...")
        model = baselineVAEAutoencoder(**options)

    elif model == "cnn_supervised":
        print("building cnn supervised...")
        model = CNNRegressor(**options)

    elif model == "escnn_supervised":
        print("building escnn supervised...")
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
