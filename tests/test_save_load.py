import argparse, json, wandb
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from vae_dist.dataset.dataset import (
    FieldDataset,
    FieldDatasetSupervised,
    dataset_split_loader,
)
from vae_dist.core.training_utils import construct_model, LogParameters
from vae_dist.core.CNN_regressor import CNNRegressor
from vae_dist.core.R3CNN_regressor import R3CNNRegressor
from vae_dist.core.CNN import CNNAutoencoderLightning
from vae_dist.core.VAE import baselineVAEAutoencoder
from vae_dist.core.R3CNN import R3CNN
from vae_dist.core.O3VAE import R3VAE

torch.set_float32_matmul_precision("high")


class TestConstruction:
    log_save_dir = "./test_model_saves/"
    root = "../data/"
    dataset = "cpet_augmented"
    root = root + dataset + "/"
    supervised_file = "../data/protein_data.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pre_process_options = {
        "transform": False,
        "augmentation": False,
        "standardize": True,
        "lower_filter": True,
        "log_scale": True,
        "min_max_scale": False,
        "wrangle_outliers": False,
        "scalar": False,
        "offset": 1,
    }
    dataset_unsuper = FieldDataset(
        root,
        transform=pre_process_options["transform"],
        augmentation=pre_process_options["augmentation"],
        standardize=pre_process_options["standardize"],
        lower_filter=pre_process_options["lower_filter"],
        log_scale=pre_process_options["log_scale"],
        min_max_scale=pre_process_options["min_max_scale"],
        wrangle_outliers=pre_process_options["wrangle_outliers"],
        scalar=pre_process_options["scalar"],
        device=device,
        offset=pre_process_options["offset"],
    )

    dataset_super = FieldDatasetSupervised(
        root,
        supervised_file,
        device=device,
        transform=False,
        augmentation=False,
        standardize=pre_process_options["standardize"],
        lower_filter=pre_process_options["lower_filter"],
        log_scale=pre_process_options["log_scale"],
        min_max_scale=pre_process_options["min_max_scale"],
        wrangle_outliers=pre_process_options["wrangle_outliers"],
        scalar=pre_process_options["scalar"],
        offset=pre_process_options["offset"],
    )
    _, dataset_train_unsuper, _ = dataset_split_loader(
        dataset_unsuper,
        train_split=0.8,
        batch_size=64,
        supervised=True,
        num_workers=0,
    )
    _, dataset_train_super, _ = dataset_split_loader(
        dataset_super,
        train_split=0.8,
        batch_size=64,
        supervised=True,
        num_workers=0,
    )

    def test_escnn(self):
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Testing escnn")
        options = json.load(
            open("../vae_dist/scripts/options/options_escnn_default.json")
        )
        options["log_wandb"] = False
        options["lr_monitor"] = False
        model = construct_model("escnn", options)
        model.to(self.device)
        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="gpu",
            devices=[0],
            gradient_clip_val=0.2,
            detect_anomaly=True,
            precision=32,
        )
        trainer.fit(model, self.dataset_train_unsuper)
        model.eval()

        print("Saving model to: ", self.log_save_dir + "/escnn.ckpt")
        trainer.save_checkpoint(self.log_save_dir + "/escnn.ckpt")
        escnn = R3CNN.load_from_checkpoint(self.log_save_dir + "/escnn.ckpt")

    def test_esvae(self):
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Testing esvae")
        options = json.load(
            open("../vae_dist/scripts/options/options_esvae_default.json")
        )
        options["log_wandb"] = False
        options["lr_monitor"] = False
        model = construct_model("esvae", options)
        model.to(self.device)
        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="gpu",
            devices=[0],
            gradient_clip_val=0.2,
            detect_anomaly=True,
            precision=32,
        )
        trainer.fit(model, self.dataset_train_unsuper)
        model.eval()
        print("Saving model to: ", self.log_save_dir + "/esvae.ckpt")
        trainer.save_checkpoint(self.log_save_dir + "/esvae.ckpt")
        esvae = R3VAE.load_from_checkpoint(self.log_save_dir + "/esvae.ckpt")

    def test_cnn(self):
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Testing cnn")
        options = json.load(
            open("../vae_dist/scripts/options/options_cnn_default.json")
        )
        options["log_wandb"] = False
        options["lr_monitor"] = False
        model = construct_model("cnn", options)
        model.to(self.device)
        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="gpu",
            devices=[0],
            gradient_clip_val=0.2,
            detect_anomaly=True,
            precision=32,
        )
        trainer.fit(model, self.dataset_train_unsuper)
        model.eval()
        print("Saving model to: ", self.log_save_dir + "/cnn.ckpt")
        trainer.save_checkpoint(self.log_save_dir + "/cnn.ckpt")
        cnn = CNNAutoencoderLightning.load_from_checkpoint(
            checkpoint_path=self.log_save_dir + "/cnn.ckpt"
        )

    def test_vae(self):
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Testing vae")
        options = json.load(
            open("../vae_dist/scripts/options/options_vae_default.json")
        )
        options["log_wandb"] = False
        options["lr_monitor"] = False
        model = construct_model("vae", options)
        model.to(self.device)
        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="gpu",
            devices=[0],
            gradient_clip_val=0.2,
            detect_anomaly=True,
            precision=32,
        )
        trainer.fit(model, self.dataset_train_unsuper)
        model.eval()
        print("Saving model to: ", self.log_save_dir + "/vae.ckpt")
        trainer.save_checkpoint(self.log_save_dir + "/vae.ckpt")
        vae = baselineVAEAutoencoder.load_from_checkpoint(
            checkpoint_path=self.log_save_dir + "/vae.ckpt"
        )

    def test_cnn_supervised(self):
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Testing cnn_supervised")
        options = json.load(
            open("../vae_dist/scripts/options/options_cnn_default_supervised.json")
        )
        options["log_wandb"] = False
        options["lr_monitor"] = False
        model = construct_model("cnn_supervised", options)
        model.to(self.device)
        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="gpu",
            devices=[0],
            gradient_clip_val=0.2,
            detect_anomaly=True,
            precision=32,
        )
        trainer.fit(model, self.dataset_train_super)
        model.eval()
        print("Saving model to: ", self.log_save_dir + "/cnn_super.ckpt")
        trainer.save_checkpoint(self.log_save_dir + "/cnn_super.ckpt")
        cnn_super = CNNRegressor.load_from_checkpoint(
            checkpoint_path=self.log_save_dir + "/cnn_super.ckpt"
        )

    def test_escnn_supervised(self):
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Testing escnn_supervised")
        options = json.load(
            open("../vae_dist/scripts/options/options_escnn_default_supervised.json")
        )
        options["log_wandb"] = False
        options["lr_monitor"] = False
        model = construct_model("escnn_supervised", options)
        model.to(self.device)
        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="gpu",
            devices=[0],
            gradient_clip_val=0.2,
            detect_anomaly=True,
            precision=32,
        )
        trainer.fit(model, self.dataset_train_super)
        model.eval()
        print("Saving model to: ", self.log_save_dir + "/escnn_super.ckpt")
        trainer.save_checkpoint(self.log_save_dir + "/escnn_super.ckpt")
        escnn_super = R3CNNRegressor.load_from_checkpoint(
            checkpoint_path=self.log_save_dir + "/escnn_super.ckpt"
        )
