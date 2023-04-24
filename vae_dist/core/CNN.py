import torch.nn as nn
import torch, wandb

# from torchConvNd import ConvNd, ConvTransposeNd
import pytorch_lightning as pl
from torchsummary import summary
from torch.nn import functional as F
from torchmetrics import MeanSquaredError, MeanAbsoluteError

from vae_dist.core.layers import UpConvBatch, ConvBatch, ResBlock
from vae_dist.core.losses import stepwise_inverse_huber_loss, inverse_huber


class CNNAutoencoderLightning(pl.LightningModule):
    def __init__(
        self,
        channels,
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
        kernel_size_in=5,
        kernel_size_out=5,
        max_pool_kernel_size_in=2,
        max_pool_loc_in=[3],
        stride_in=[1],
        max_pool_kernel_size_out=2,
        max_pool_loc_out=[3],
        stride_out=[1],
        log_wandb=True,
        im_dim=21,
        reconstruction_loss="mse",
        optimizer="adam",
        lr_decay_factor=0.5,
        lr_patience=30,
        **kwargs,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        params = {
            "kernel_size_in": kernel_size_in,
            "kernel_size_out": kernel_size_out,
            "channels": channels,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
            "bias": bias,
            "padding_mode": padding_mode,
            "latent_dim": latent_dim,
            "fully_connected_layers": fully_connected_layers,
            "activation": activation,
            "dropout": dropout,
            "batch_norm": batch_norm,
            "learning_rate": learning_rate,
            "log_wandb": log_wandb,
            "im_dim": im_dim,
            "max_pool": max_pool,
            "max_pool_kernel_size_in": max_pool_kernel_size_in,
            "max_pool_kernel_size_out": max_pool_kernel_size_out,
            "max_pool_loc_in": max_pool_loc_in,
            "max_pool_loc_out": max_pool_loc_out,
            "stride_in": stride_in,
            "stride_out": stride_out,
            "reconstruction_loss": reconstruction_loss,
            "optimizer": optimizer,
            "lr_decay_factor": lr_decay_factor,
            "lr_patience": lr_patience,
        }
        assert (
            len(channels) - 1 == len(stride_in) == len(stride_out)
        ), "channels and stride must be the same length"
        assert (
            len(stride_in)
            == len(stride_out)
            == len(kernel_size_in)
            == len(kernel_size_out)
        ), "stride and kernel_size must be the same length"

        self.hparams.update(params)
        self.save_hyperparameters()

        modules_enc, modules_dec = [], []
        self.list_enc_fully = []
        self.list_dec_fully = []
        self.list_enc_conv = []
        self.list_dec_conv = []

        trigger = 0
        inner_dim = im_dim
        for i in range(len(self.hparams.channels) - 1):
            inner_dim = int(
                (inner_dim - (self.hparams.kernel_size_in[i] - 1))
                / self.hparams.stride_in[i]
            )
            if self.hparams.max_pool:
                if i in self.hparams.max_pool_loc_in:
                    inner_dim = int(
                        1
                        + (inner_dim - self.hparams.max_pool_kernel_size_in + 1)
                        / self.hparams.max_pool_kernel_size_in
                    )

        print("inner_dim: ", inner_dim)

        for ind, h in enumerate(self.hparams.fully_connected_layers):
            # if it's the last item in the list, then we want to output the latent dim

            if ind == 0:
                self.list_dec_fully.append(
                    torch.nn.Unflatten(
                        1, (self.hparams.channels[-1], inner_dim, inner_dim, inner_dim)
                    )
                )
                self.list_enc_fully.append(torch.nn.Flatten())
                h_in = self.hparams.channels[-1] * inner_dim * inner_dim * inner_dim
                h_out = h

            else:
                h_in = self.hparams.fully_connected_layers[ind - 1]
                h_out = h

            self.list_enc_fully.append(torch.nn.Linear(h_in, h_out))
            self.list_enc_fully.append(torch.nn.LeakyReLU())

            if self.hparams.batch_norm and ind != 0:
                self.list_enc_fully.append(torch.nn.BatchNorm1d(h_out))

            if self.hparams.dropout > 0:
                self.list_enc_fully.append(torch.nn.Dropout(self.hparams.dropout))
                self.list_dec_fully.append(torch.nn.Dropout(self.hparams.dropout))

            if self.hparams.batch_norm and ind != 0:
                self.list_dec_fully.append(torch.nn.BatchNorm1d(h_in))

            if ind == len(self.hparams.fully_connected_layers) - 1:
                self.list_dec_fully.append(torch.nn.LeakyReLU())
            else:
                self.list_dec_fully.append(torch.nn.LeakyReLU())

            self.list_dec_fully.append(torch.nn.Linear(h_out, h_in))

        self.list_dec_fully.append(
            torch.nn.Linear(
                self.hparams.latent_dim, self.hparams.fully_connected_layers[-1]
            )
        )

        self.list_enc_fully.append(
            torch.nn.Linear(
                self.hparams.fully_connected_layers[-1], self.hparams.latent_dim
            )
        )

        self.list_enc_fully.append(torch.nn.LeakyReLU())

        for ind in range(len(self.hparams.channels) - 1):
            channel_in = self.hparams.channels[ind]
            channel_out = self.hparams.channels[ind + 1]

            output_padding = 0
            if inner_dim % 2 == 1 and ind == len(self.hparams.channels) - 1:
                output_padding = 1

            self.list_enc_conv.append(
                ConvBatch(
                    in_channels=channel_in,
                    out_channels=channel_out,
                    kernel_size=kernel_size_in[ind],
                    stride=stride_in[ind],
                    padding=self.hparams.padding,
                    dilation=self.hparams.dilation,
                    groups=self.hparams.groups,
                    bias=self.hparams.bias,
                    padding_mode=self.hparams.padding_mode,
                )
            )

            if dropout > 0:
                self.list_enc_conv.append(torch.nn.Dropout(dropout))

            if trigger:
                self.list_dec_conv.append(
                    torch.nn.Upsample(
                        scale_factor=self.hparams.max_pool_kernel_size_out
                    )
                )
                trigger = 0

            if self.hparams.max_pool and ind in self.hparams.max_pool_kernel_size_in:
                self.list_enc_conv.append(
                    torch.nn.MaxPool3d(self.hparams.max_pool_kernel_size_in)
                )
                trigger = 1

            if dropout > 0:
                self.list_dec_conv.append(torch.nn.Dropout(dropout))

            if ind == 0:
                output_layer = True
            else:
                output_layer = False

            self.list_dec_conv.append(
                UpConvBatch(
                    in_channels=channel_out,
                    out_channels=channel_in,
                    stride=stride_out[ind],
                    kernel_size=kernel_size_out[ind],
                    padding=self.hparams.padding,
                    dilation=self.hparams.dilation,
                    groups=self.hparams.groups,
                    bias=self.hparams.bias,
                    padding_mode=self.hparams.padding_mode,
                    output_padding=output_padding,
                    output_layer=output_layer,
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

        summary(
            self.encoder_conv,
            (self.hparams.channels[0], im_dim, im_dim, im_dim),
            device="cpu",
        )
        summary(
            self.encoder,
            (self.hparams.channels[0], im_dim, im_dim, im_dim),
            device="cpu",
        )
        summary(self.decoder_dense, tuple([latent_dim]), device="cpu")
        summary(
            self.decoder_conv,
            (channels[-1], inner_dim, inner_dim, inner_dim),
            device="cpu",
        )

        self.train_rmse = MeanSquaredError(squared=False)
        self.val_rmse = MeanSquaredError(squared=False)
        self.test_rmse = MeanSquaredError(squared=False)
        self.test_mae = MeanAbsoluteError()
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()

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
        x = x.float()
        x_hat = x_hat.float()
        if self.hparams.reconstruction_loss == "mse":
            recon_loss = F.mse_loss(x_hat, x, reduction="mean")
        elif self.hparams.reconstruction_loss == "l1":
            recon_loss = F.l1_loss(x_hat, x, reduction="mean")
        elif self.hparams.reconstruction_loss == "huber":
            recon_loss = F.huber_loss(x_hat, x, reduction="mean")
        elif self.hparams.reconstruction_loss == "many_step_inverse_huber":
            recon_loss = stepwise_inverse_huber_loss(x_hat, x)
        elif self.hparams.reconstruction_loss == "inverse_huber":
            recon_loss = inverse_huber(x_hat, x)

        return recon_loss

    def shared_step(self, batch, mode):
        predict = self.forward(batch)

        if mode == "train":
            self.train_rmse.update(predict, batch)
            self.train_mae.update(predict, batch)
        elif mode == "val":
            self.val_rmse.update(predict, batch)
            self.val_mae.update(predict, batch)
        elif mode == "test":
            self.test_rmse.update(predict, batch)
            self.test_mae.update(predict, batch)

        loss = self.loss_function(batch, predict)
        if self.hparams.log_wandb:
            wandb.log({f"{mode}_loss": loss})
        self.log(
            f"{mode}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch),
        )
        return loss

    def compute_metrics(self, mode):
        if mode == "train":
            rmse = self.train_rmse.compute()
            mae = self.train_mae.compute()
            self.train_rmse.reset()
            self.train_mae.reset()

        elif mode == "val":
            rmse = self.val_rmse.compute()
            mae = self.val_mae.compute()
            self.val_mae.reset()
            self.val_rmse.reset()

        elif mode == "test":
            rmse = self.test_rmse.compute()
            mae = self.test_mae.compute()
            self.test_rmse.reset()
            self.test_mae.reset()

        return rmse, mae

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, mode="train")

    def training_epoch_end(self, outputs):
        rmse, mae = self.compute_metrics(mode="train")
        out_dict = {
            "train_rmse": rmse,
            "train_mae": mae,
        }
        if self.hparams.log_wandb:
            wandb.log(out_dict)
        self.log_dict(out_dict, prog_bar=True)

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, mode="test")

    def test_epoch_end(self, outputs):
        rmse, mae = self.compute_metrics(mode="test")
        out_dict = {
            "test_rmse": rmse,
            "test_mae": mae,
        }
        if self.hparams.log_wandb:
            wandb.log(out_dict)
        self.log_dict(out_dict)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, mode="val")

    def validation_epoch_end(self, outputs):
        rmse, mae = self.compute_metrics(mode="val")
        out_dict = {
            "val_rmse": rmse,
            "val_mae": mae,
        }
        if self.hparams.log_wandb:
            wandb.log(out_dict)
        self.log_dict(out_dict, prog_bar=True)

    def configure_optimizers(self):
        if self.hparams.optimizer == "Adam":
            print("Using Adam Optimizer")
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.hparams.learning_rate
            )
        else:
            print("Using SGD Optimizer")
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.hparams.learning_rate
            )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=self.hparams.lr_decay_factor,
            patience=self.hparams.lr_patience,
            verbose=True,
            threshold=0.0001,
            threshold_mode="rel",
            cooldown=0,
            min_lr=1e-06,
            eps=1e-08,
        )
        lr_scheduler = {"scheduler": scheduler, "monitor": "val_loss"}
        return [optimizer], [lr_scheduler]

    def load_model(self, path):
        self.model.load_state_dict(
            torch.load(path, map_location="cuda:0"), strict=False
        )
        print("Model Created!")
