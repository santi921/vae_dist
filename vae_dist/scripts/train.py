import argparse, json, wandb
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    EarlyStopping,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from vae_dist.dataset.dataset import (
    FieldDatasetSupervised,
    FieldDataset,
)

from vae_dist.core.parameters import set_enviroment
from vae_dist.core.training_utils import (
    construct_model,
    LogParameters,
    InputMonitor,
)

from vae_dist.core.parameters import set_enviroment


def grab_supervised(
    options,
    device,
    dataset_dir="../../data/cpet_augmented/",
    supervised_file="../../data/protein_data.csv",
):
    dataset_vanilla = FieldDatasetSupervised(
        dataset_dir,
        supervised_file,
        device=device,
        transform=False,
        augmentation=False,
        standardize=options["standardize"],
        lower_filter=options["lower_filter"],
        log_scale=options["log_scale"],
        min_max_scale=options["min_max_scale"],
        wrangle_outliers=options["wrangle_outliers"],
        scalar=options["scalar"],
        offset=options["offset"],
    )
    print("got supervised dataset")
    return dataset_vanilla


if __name__ == "__main__":
    set_enviroment()
    torch.multiprocessing.freeze_support()
    # create argparser that just takes a string for model type
    # and a string for the path to the data
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="escnn")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--dataset_dir", type=str, default="../../data/cpet_augmented/")
    parser.add_argument("--test_activity", action="store_true")

    args = parser.parse_args()

    epochs = args.epochs
    model_select = args.model
    dataset_dir = args.dataset_dir
    dataset_name = dataset_dir.split("/")[-2]
    test_activity = bool(args.test_activity)

    print("dataset_name: ", dataset_name)
    root = dataset_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # assert that the model is one of the following escnn, cnn, vae, esvae
    assert model_select in [
        "escnn",
        "cnn",
        "vae",
        "esvae",
    ], "Model must be one of the following: escnn, cnn, vae, esvae"

    run = wandb.init(
        project="{}_dist_{}".format(model_select, dataset_name), reinit=True
    )

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

    dataset_vanilla = FieldDataset(
        dataset_dir,
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

    if test_activity:
        supervised_dataset_loader = grab_supervised(pre_process_options, device=device)
    else:
        supervised_dataset_loader = None
    # supervised_dataset_loader = None

    # print("about to make model")
    if model_select == "escnn":
        options = json.load(open("./options/options_escnn_default.json"))
        log_save_dir = "./logs/log_version_escnn_1/"
        model = construct_model(
            model="escnn", options=options, supervised_loader=supervised_dataset_loader
        )

    elif model_select == "esvae":
        options = json.load(open("./options/options_esvae_default.json"))
        log_save_dir = "./logs/log_version_esvae_1/"
        model = construct_model(
            model="esvae",
            options=options,
            supervised_loader=supervised_dataset_loader,
        )

    elif model_select == "cnn":
        options = json.load(open("./options/options_cnn_default.json"))
        log_save_dir = "./logs/log_version_auto_1/"
        model = construct_model(
            model="cnn",
            options=options,
            supervised_loader=supervised_dataset_loader,
        )

    elif model_select == "vae":
        options = json.load(open("./options/options_vae_default.json"))
        log_save_dir = "./logs/log_version_vae_1/"
        model = construct_model(
            model="vae",
            options=options,
            supervised_loader=supervised_dataset_loader,
        )

    else:
        # throw error
        print("Model not found")

    # print("made model")
    wandb.config.update({"model": model_select, "epochs": epochs, "data": root})
    wandb.config.update(options)
    wandb.config.update(pre_process_options)

    # load model to gpu
    model.to(device)

    print(">" * 40 + "config_settings" + "<" * 40)
    for k, v in options.items():
        print("{}\t\t\t{}".format(str(k).ljust(20), str(v).ljust(20)))
    print(">" * 40 + "config_settings" + "<" * 40)

    # check if there are any inf or nan values in the model
    is_nan = torch.stack([torch.isnan(p).any() for p in model.parameters()]).any()
    print("Model has inf or nan values: ", is_nan)
    # check if dataset has any inf or nan values
    print(
        "Dataset has inf or nan values: ",
        torch.isnan(dataset_vanilla.dataset_to_tensor()).any(),
    )

    dataset_loader_full = torch.utils.data.DataLoader(
        dataset_vanilla, batch_size=32, shuffle=False, num_workers=0
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    # create early stopping callback
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=200, verbose=False, mode="min"
    )

    log_parameters = LogParameters(wandb=True)
    logger_tb = TensorBoardLogger(log_save_dir, name="test_logs")
    logger_wb = WandbLogger(project="{}_dist".format(model_select), name="test_logs")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=log_save_dir,
        filename="{epoch:02d}-{val_loss:.2f}",
        mode="min",
    )
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=[0],
        accumulate_grad_batches=5,
        enable_progress_bar=True,
        gradient_clip_val=5.0,
        callbacks=[
            early_stop_callback,
            lr_monitor,
            log_parameters,
            InputMonitor(),
            checkpoint_callback,
        ],
        enable_checkpointing=True,
        default_root_dir=log_save_dir,
        logger=[logger_tb, logger_wb],
        detect_anomaly=True,
        precision=32,
    )
    print("got to trainer.fit")
    trainer.fit(model, dataset_loader_full, dataset_loader_full)

    model.eval()
    # save state dict
    print(
        "Saving model to: ",
        log_save_dir + "/model_train_epoch_{}_{}.ckpt".format(epochs, model_select),
    )
    # torch.save(model.state_dict(), log_save_dir + "/model_single_datapoint.ckpt")
    trainer.save_checkpoint(
        log_save_dir + "/model_train_epoch_{}_{}.ckpt".format(epochs, model_select)
    )
    run.finish()
