import wandb, argparse, torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    EarlyStopping,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from vae_dist.dataset.dataset import FieldDataset, dataset_split_loader
from vae_dist.core.training_utils import (
    construct_model,
    LogParameters,
    InputMonitor,
)
from vae_dist.core.parameters import set_enviroment, hyperparameter_dicts


class training:
    def __init__(
        self,
        data_dir,
        model,
        device,
        transform=False,
        augmentation=False,
        standardize=True,
        lower_filter=True,
        log_scale=True,
        min_max_scale=False,
        wrangle_outliers=False,
        scalar=False,
        offset=1,
        project="vae_dist",
    ):
        self.model = model
        self.project = project
        self.device = device
        self.transform = transform
        self.log_scale = log_scale
        self.offset = offset
        self.scalar = scalar
        self.wrangle_outliers = wrangle_outliers
        self.min_max_scale = min_max_scale
        self.lower_filter = lower_filter
        self.standardize = standardize
        self.aug = augmentation

        self.data_dir = data_dir

        print("dataset parameters")
        print("augmentation: ", self.aug)
        print("standardize: ", self.standardize)
        print("log_scale: ", self.log_scale)
        print("model: ", self.model)
        print("device: ", self.device)
        print("data_dir: ", self.data_dir)
        print("project: ", self.project)

        dataset = FieldDataset(
            data_dir,
            transform=self.transform,
            augmentation=self.aug,
            standardize=self.standardize,
            lower_filter=self.lower_filter,
            log_scale=self.log_scale,
            min_max_scale=self.min_max_scale,
            wrangle_outliers=self.wrangle_outliers,
            scalar=self.scalar,
            device=self.device,
            offset=self.offset,
        )

        # check if dataset has any inf or nan values
        print(
            "Dataset has inf or nan values: ",
            torch.isnan(dataset.dataset_to_tensor()).any(),
        )
        (
            dataset_loader_full,
            dataset_loader_train,
            dataset_loader_test,
        ) = dataset_split_loader(dataset, train_split=0.8, num_workers=0)
        self.data_loader_full = dataset_loader_full
        self.data_loader_train = dataset_loader_train
        self.data_loader_test = dataset_loader_test
        print("loaders  created " * 5)

    def make_model(self, config):
        model_obj = construct_model(model=self.model, options=config)
        model_obj.to(self.device)

        """initializer = config['initializer']
        if initializer == 'kaiming':
            kaiming_init(model)
        elif initializer == 'xavier':
            xavier_init(model)
        elif initializer == 'equi_var': # works
            equi_var_init(model)
        else:
            raise ValueError("Initializer must be kaiming, xavier or equi_var")
        """

        if self.model == "auto":
            log_save_dir = "./logs/log_version_autoenc_sweep/"
        elif self.model == "vae":
            log_save_dir = "./logs/log_version_vae_sweep/"
        elif self.model == "esvae":
            log_save_dir = "./logs/log_version_esvae_sweep/"
        elif self.model == "escnn":
            log_save_dir = "./logs/log_version_escnn_sweep/"
        else:
            raise ValueError("model not found")

        log_parameters = LogParameters()
        logger_tb = TensorBoardLogger(log_save_dir, name="test_logs")
        logger_wb = WandbLogger(
            project="{}_dist_sweep".format(self.model), name="test_logs"
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")
        early_stop_callback = EarlyStopping(
            monitor="val_loss", min_delta=0.00, patience=200, verbose=False, mode="min"
        )
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=log_save_dir,
            filename="{epoch:02d}-{val_loss:.2f}",
            mode="min",
        )

        lr_monitor = LearningRateMonitor(logging_interval="step")
        trainer = pl.Trainer(
            max_epochs=config["epochs"],
            accelerator="gpu",
            devices=[0],
            accumulate_grad_batches=3,
            enable_progress_bar=True,
            gradient_clip_val=0.5,
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

        return model_obj, trainer, log_save_dir

    def train(self):
        with wandb.init(project=self.project) as run:
            config = wandb.config
            model_obj, trainer, log_save_dir = self.make_model(config)
            print("-" * 20 + "Training" + "-" * 20)
            trainer.fit(model_obj, self.data_loader_train, self.data_loader_test)
            # update config to include dataset parameters
            config.update(
                {
                    "aug": self.aug,
                    "standardize": self.standardize,
                    "log_scale": self.log_scale,
                    "model": self.model,
                    "device": self.device,
                    "data_dir": self.data_dir,
                }
            )
            model_obj.eval()
            # save state dict
            torch.save(model_obj.state_dict(), log_save_dir + "/model_1.ckpt")
            # save model
            # torch.save(model_obj, log_save_dir + "/model_1.pt")
            run.finish()

        run.finish()


def set_enviroment():
    from torch.multiprocessing import set_start_method

    torch.set_float32_matmul_precision("high")
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass


if __name__ == "__main__":
    set_enviroment()

    parser = argparse.ArgumentParser(description="options for hyperparam tune")
    parser.add_argument(
        "-dataset",
        action="store",
        dest="dataset",
        default="../../data/cpet_augmented/",
        help="dataset to use",
    )

    parser.add_argument(
        "-count",
        action="store",
        dest="count",
        default=100,
        help="number of hyperparams",
    )

    parser.add_argument(
        "-model",
        action="store",
        dest="model",
        default="auto",
        help="model",
    )

    pca_tf = True
    method = "bayes"
    project_name = "vae_dist_sweep"

    sweep_config = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = parser.parse_args()
    model = str(results.model)
    data_dir = str(results.dataset)
    # dataset_int = int(results.dataset)
    count = int(results.count)

    dataset_name = str(results.dataset).split("/")[-1]

    dict_hyper = hyperparameter_dicts(image_size=21)

    sweep_config["parameters"] = dict_hyper[model]
    sweep_config["name"] = method + "_" + model + "_" + dataset_name
    sweep_config["method"] = method
    if method == "bayes":
        sweep_config["metric"] = {"name": "val_rmse", "goal": "minimize"}

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

    sweep_id = wandb.sweep(sweep_config, project=project_name)
    training_obj = training(
        data_dir,
        model=model,
        device=device,
        transform=pre_process_options["transform"],
        augmentation=pre_process_options["augmentation"],
        standardize=pre_process_options["standardize"],
        lower_filter=pre_process_options["lower_filter"],
        log_scale=pre_process_options["log_scale"],
        min_max_scale=pre_process_options["min_max_scale"],
        wrangle_outliers=pre_process_options["wrangle_outliers"],
        scalar=pre_process_options["scalar"],
        offset=pre_process_options["offset"],
        project=project_name,
    )

    wandb.agent(sweep_id, function=training_obj.train, count=count)
