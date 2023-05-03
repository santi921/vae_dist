import argparse, json, wandb
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from vae_dist.dataset.dataset import FieldDataset, dataset_split_loader
from vae_dist.core.training_utils import construct_model

torch.set_float32_matmul_precision("high")
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from vae_dist.core.training_utils import construct_model, LogParameters


def test_model_construction():

    # root = "../../data/cpet/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_list = [
        "cnn_supervised",
        "escnn",
        "esvae",
        "cnn",
        "vae",
        "escnn_supervised",
    ]
    # model_list = ["escnn_supervised"]

    for model_select in model_list:
        if model_select == "escnn":
            options = json.load(
                open("../vae_dist/scripts/options/options_escnn_default.json")
            )
            model = construct_model("escnn", options)
            model.to(device)
        elif model_select == "esvae":
            options = json.load(
                open("../vae_dist/scripts/options/options_esvae_default.json")
            )
            model = construct_model("esvae", options)
            model.to(device)
        elif model_select == "cnn":
            options = json.load(
                open("../vae_dist/scripts/options/options_cnn_default.json")
            )
            model = construct_model("cnn", options)
            model.to(device)
        elif model_select == "vae":
            options = json.load(
                open("../vae_dist/scripts/options/options_vae_default.json")
            )
            model = construct_model("vae", options)
            model.to(device)
        elif model_select == "cnn_supervised":
            options = json.load(
                open("../vae_dist/scripts/options/options_cnn_default_supervised.json")
            )
            model = construct_model("cnn_supervised", options)
            model.to(device)
        elif model_select == "escnn_supervised":
            options = json.load(
                open(
                    "../vae_dist/scripts/options/options_escnn_default_supervised.json"
                )
            )
            model = construct_model("escnn_supervised", options)
            model.to(device)


def main():
    print("testing base data loading")
    test_model_construction()


main()
