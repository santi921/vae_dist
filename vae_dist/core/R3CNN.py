from escnn import nn
import torch, wandb
import pytorch_lightning as pl

from torchmetrics import MeanSquaredError, MeanAbsoluteError
from torchsummary import summary
from torch.nn import functional as F

from vae_dist.core.escnnlayers import R3Upsampling
from vae_dist.core.losses import stepwise_inverse_huber_loss, inverse_huber
from vae_dist.core.parameters import pull_escnn_params


class R3CNN(pl.LightningModule):
    def __init__(
        self,
        learning_rate,
        channels,
        dropout,
        bias,
        kernel_size_in=5,
        kernel_size_out=5,
        latent_dim=1,
        max_pool=False,
        batch_norm=False,
        max_pool_kernel_size_in=2,
        max_pool_loc_in=[3],
        stride_in=[1],
        max_pool_kernel_size_out=2,
        max_pool_loc_out=[3],
        stride_out=[1],
        fully_connected_layers=[64, 64, 64],
        log_wandb=True,
        im_dim=21,
        reconstruction_loss="mse",
        optimizer="adam",
        lr_decay_factor=0.5,
        lr_patience=30,
        escnn_params={},
        **kwargs,
    ):
        # super(self).__init__()
        super().__init__()
        self.learning_rate = learning_rate
        params = {
            "channels": channels,
            "padding": 0,
            "bias": bias,
            "stride_in": stride_in,
            "stride_out": stride_out,
            "latent_dim": latent_dim,
            "batch_norm": batch_norm,
            "fully_connected_layers": fully_connected_layers,
            "activation": "relu",
            "dropout": dropout,
            "learning_rate": learning_rate,
            "log_wandb": log_wandb,
            "im_dim": im_dim,
            "max_pool": max_pool,
            "kernel_size_in": kernel_size_in,
            "kernel_size_out": kernel_size_out,
            "max_pool_kernel_size_in": max_pool_kernel_size_in,
            "max_pool_kernel_size_out": max_pool_kernel_size_out,
            "max_pool_loc_in": max_pool_loc_in,
            "max_pool_loc_out": max_pool_loc_out,
            "reconstruction_loss": reconstruction_loss,
            "optimizer": optimizer,
            "lr_decay_factor": lr_decay_factor,
            "lr_patience": lr_patience,
            "escnn_params": escnn_params,
        }

        assert (
            len(channels) == len(stride_in) == len(stride_out)
        ), "channels and stride must be the same length"
        assert (
            len(stride_in)
            == len(stride_out)
            == len(kernel_size_in)
            == len(kernel_size_out)
        ), "stride and kernel_size must be the same length"
        self.hparams.update(params)
        self.save_hyperparameters()

        self.list_dec_fully = []
        self.list_enc_fully = []
        self.decoder_conv_list = []
        self.encoder_conv_list = []

        (
            group,
            gspace,
            feat_type_in,
        ) = pull_escnn_params(escnn_params)

        gspace = gspace
        group = group
        self.feat_type_in = feat_type_in
        self.dense_out_type = nn.FieldType(
            gspace, self.hparams.channels[-1] * [gspace.trivial_repr]
        )

        trigger = 0
        inner_dim = self.hparams.im_dim
        # number of output channels
        for i in range(len(self.hparams.channels)):
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

        for ind in range(len(self.hparams.channels)):
            if ind == 0:
                in_type = feat_type_in
                channel_out = self.hparams.channels[0]

            else:
                channel_in = self.hparams.channels[ind - 1]
                channel_out = self.hparams.channels[ind]
                in_type = nn.FieldType(gspace, channel_in * [gspace.trivial_repr])

            out_type = nn.FieldType(gspace, channel_out * [gspace.trivial_repr])

            print("in_type: {} out_type: {}".format(in_type, out_type))

            if stride_in[ind] == 2:
                sigma = None
                frequencies_cutoff = None
                rings = None
            else:
                sigma = None
                frequencies_cutoff = None
                rings = None

            self.encoder_conv_list.append(
                nn.R3Conv(
                    in_type=in_type,
                    out_type=out_type,
                    kernel_size=kernel_size_in[ind],
                    stride=stride_in[ind],
                    # padding = self.hparams.padding,
                    bias=self.hparams.bias,
                    sigma=sigma,
                    frequencies_cutoff=frequencies_cutoff,
                    rings=rings,
                )
            )
            if self.hparams.batch_norm:
                self.encoder_conv_list.append(nn.IIDBatchNorm3d(out_type))

            # self.encoder_conv_list.append(nn.ReLU(out_type))
            self.encoder_conv_list.append(nn.NormNonLinearity(out_type))

            # self.encoder_conv_list.append(nn.NormNonLinearity(out_type, 'n_relu'))
            output_padding = 0
            # if inner_dim%2 == 1 and ind == len(self.hparams.channels)-1:
            #    output_padding = 1

            if dropout > 0:
                self.encoder_conv_list.append(nn.PointwiseDropout(out_type, p=dropout))

            if trigger:
                self.decoder_conv_list.append(
                    R3Upsampling(
                        in_type,
                        scale_factor=self.hparams.max_pool_kernel_size_out,
                        mode="nearest",
                        align_corners=False,
                    )
                )
                trigger = 0

            if self.hparams.max_pool and ind in self.hparams.max_pool_loc_in:
                self.encoder_conv_list.append(
                    nn.PointwiseAvgPoolAntialiased3D(
                        out_type,
                        stride=self.hparams.max_pool_kernel_size_in,
                        sigma=0.66,
                    )
                )
                trigger = 1

            if dropout > 0:
                self.decoder_conv_list.append(nn.PointwiseDropout(in_type, p=dropout))

            # decoder
            if ind == 0:
                self.decoder_conv_list.append(nn.IdentityModule(in_type))
            else:
                # self.decoder_conv_list.append(nn.ReLU(in_type))
                self.decoder_conv_list.append(nn.NormNonLinearity(in_type))

            if self.hparams.batch_norm:
                self.decoder_conv_list.append(nn.IIDBatchNorm3d(in_type, affine=True))

            if stride_out[ind] == 2:
                sigma = None
                frequencies_cutoff = None
                rings = None
            else:
                sigma = None
                frequencies_cutoff = None
                rings = None

            self.decoder_conv_list.append(
                nn.R3ConvTransposed(
                    in_type=out_type,
                    out_type=in_type,
                    stride=stride_out[ind],
                    kernel_size=kernel_size_out[ind],
                    bias=self.hparams.bias,
                    output_padding=output_padding,
                    sigma=sigma,
                    frequencies_cutoff=frequencies_cutoff,
                    rings=rings,
                )
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

            if self.hparams.batch_norm:
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

        # reverse the list
        self.list_dec_fully.reverse()
        self.decoder_conv_list.reverse()
        self.encoder_fully_net = torch.nn.Sequential(*self.list_enc_fully)
        self.decoder_fully_net = torch.nn.Sequential(*self.list_dec_fully)
        self.encoder = nn.SequentialModule(*self.encoder_conv_list)
        self.decoder = nn.SequentialModule(*self.decoder_conv_list)
        self.model = nn.SequentialModule(self.encoder, self.decoder)
        self.train_rmse = MeanSquaredError(squared=False)
        self.val_rmse = MeanSquaredError(squared=False)
        self.test_rmse = MeanSquaredError(squared=False)
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()
        summary(
            self.encoder_fully_net,
            (self.hparams.channels[-1], inner_dim, inner_dim, inner_dim),
            device="cpu",
        )
        summary(self.decoder_fully_net, tuple([latent_dim]), device="cpu")

    def encode(self, x):
        x = self.feat_type_in(x)
        x = self.encoder(x)
        x = x.tensor
        x = x.reshape(x.shape[0], -1)
        x = self.encoder_fully_net(x)
        return x

    def decode(self, x):
        x = self.decoder_fully_net(x)
        # x = x.reshape(x.shape[0], self.channels_inner, self.inner_dim, self.inner_dim, self.inner_dim)
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
        # convert to tensor of type float
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
        batch_group = self.feat_type_in(batch).tensor

        if mode == "train":
            self.train_rmse.update(predict, batch_group)
            self.train_mae.update(predict, batch_group)
        elif mode == "val":
            self.val_rmse.update(predict, batch_group)
            self.val_mae.update(predict, batch_group)
        elif mode == "test":
            self.test_rmse.update(predict, batch_group)
            self.test_mae.update(predict, batch_group)

        loss = self.loss_function(batch_group, predict)
        # if self.hparams.log_wandb:wandb.log({f"{mode}_loss": loss})
        self.log(
            f"{mode}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch_group),
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
        self.load_state_dict(torch.load(path, map_location="cuda:0"), strict=False)
        print("Model Created!")
