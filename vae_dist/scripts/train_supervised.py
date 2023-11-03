import argparse, json, wandb
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    EarlyStopping,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from vae_dist.dataset.dataset import FieldDatasetSupervised, dataset_split_loader
from vae_dist.core.parameters import set_enviroment
from vae_dist.core.training_utils import (
    construct_model,
    LogParameters,
    InputMonitor,
)


if __name__ == "__main__":
    set_enviroment()
    torch.multiprocessing.freeze_support()
    # create argparser that just takes a string for model type
    # and a string for the path to the data
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="escnn")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--dataset_dir", type=str, default="../../data/cpet_augmented/")
    parser.add_argument(
        "-config", type=str, default="./options/options_escnn_default_supervised.json"
    )

    args = parser.parse_args()
    epochs = args.epochs
    model_select = args.model
    dataset = args.dataset_dir
    checkpoint = bool(args.checkpoint)
    config = args.config
    # remove ., /  from dataset name
    dataset_trimmed = dataset.split("/")[-2]

    # root = "../../data/"
    # dataset = "cpet_augmented"
    # root = root + dataset + "/"
    supervised_file = "../../data/protein_data.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_select == "escnn" or model_select == "cnn":
        run = wandb.init(
            project="supervised_vae_{}".format(dataset_trimmed),
            reinit=True,
            entity="santi",
        )

    assert model_select in ["escnn", "cnn"], "Model must be escnn or cnn"

    if model_select == "escnn":
        options = json.load(open(config))
        log_save_dir = "./logs/log_version_escnn/"
        model = construct_model("escnn_supervised", options)

    elif model_select == "cnn":
        options = json.load(open(config))
        log_save_dir = "./logs/log_version_cnn/"
        model = construct_model("cnn_supervised", options)

    wandb.config.update({"model": model_select, "epochs": epochs, "data": dataset})
    wandb.config.update(options)

    pre_process_options = {
        "transform": False,
        "augmentation": False,
        "standardize": True,
        "lower_filter": True,
        "log_scale": True,
        "min_max_scale": False,
        "wrangle_outliers": False,
        "scalar": False,
        "sparsify": -1,
        "offset": 1,
    }
    wandb.config.update(pre_process_options)
    options.update(pre_process_options)

    dataset_vanilla = FieldDatasetSupervised(
        dataset,
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
        sparsify=options["sparsify"],
    )

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

    """dataset_loader_full = torch.utils.data.DataLoader(
        dataset_vanilla, 
        batch_size=64,
        shuffle=False,
        num_workers=0
    )"""

    dataset_full, dataset_train, dataset_val = dataset_split_loader(
        dataset_vanilla,
        train_split=0.8,
        batch_size=options["batch_size"],
        supervised=True,
        num_workers=0,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # create early stopping callback
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=300, verbose=False, mode="max"
    )

    log_parameters = LogParameters(wandb=True)
    logger_tb = TensorBoardLogger(log_save_dir, name="test_logs")
    logger_wb = WandbLogger(
        project="{}_supervised_vae_dist".format(model_select), name="test_logs"
    )
    callbacks = [
        early_stop_callback,
        lr_monitor,
        log_parameters,
        InputMonitor(),
    ]
    if checkpoint:
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=log_save_dir,
            filename="{epoch:02d}-{val_loss:.2f}",
            mode="min",
        )
        callbacks.append(checkpoint_callback)

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=[0],
        accumulate_grad_batches=options["accumulate_grad_batches"],
        enable_progress_bar=True,
        log_every_n_steps=10,
        gradient_clip_val=options["gradient_clip_val"],
        callbacks=callbacks,
        enable_checkpointing=True,
        default_root_dir=log_save_dir,
        logger=[logger_tb, logger_wb],
        detect_anomaly=True,
        precision=32,
    )

    trainer.fit(model, dataset_train, dataset_val)
    model.eval()

    # save state dict
    print(
        "Saving model to: ",
        log_save_dir + "/model_supervised_{}.ckpt".format(model_select),
    )
    # torch.save(model.state_dict(), log_save_dir + "/model_supervised_datapoint.ckpt")
    trainer.save_checkpoint(
        log_save_dir + "/model_supervised_{}.ckpt".format(model_select)
    )
    run.finish()
