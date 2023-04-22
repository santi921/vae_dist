import wandb, argparse, torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    EarlyStopping,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from vae_dist.dataset.dataset import FieldDatasetSupervised, dataset_split_loader
from vae_dist.core.training_utils import (
    construct_model,
    hyperparameter_dicts,
    LogParameters,
    InputMonitor,
)
from vae_dist.core.parameters import set_enviroment
from vae_dist.core.intializers import equi_var_init, xavier_init, kaiming_init


class training:
    def __init__(
        self,
        data_dir,
        supervised_file,
        model,
        device,
        transform=False,
        augmentation=False,
        scalar=False,
        offset=1,
        project="vae_dist",
    ):
        self.model = model
        self.project = project
        self.device = device
        self.transform = transform
        self.offset = offset
        self.scalar = scalar
        self.aug = augmentation
        self.data_dir = data_dir
        self.supervised_file = supervised_file

    def make_model(self, config):
        dataset = FieldDatasetSupervised(
            self.data_dir,
            self.supervised_file,
            device=self.device,
            transform=self.transform,
            augmentation=self.aug,
            standardize=config["standardize"],
            lower_filter=config["lower_filter"],
            log_scale=config["log_scale"],
            min_max_scale=config["min_max_scale"],
            wrangle_outliers=config["wrangle_outliers"],
            scalar=self.scalar,
            offset=self.offset,
        )
        print("-" * 30)
        print("Some info...")
        print("model: ", self.model)
        print("device: ", self.device)
        print("data_dir: ", self.data_dir)
        print("supervised_file: ", self.supervised_file)
        print("project: ", self.project)
        print("-" * 30)

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

        model_obj = construct_model(model=self.model, options=config)
        model_obj.to(self.device)

        initializer = config["initializer"]
        if initializer == "kaiming":
            kaiming_init(model_obj)
        elif initializer == "xavier":
            xavier_init(model_obj)
        elif initializer == "equi_var":
            equi_var_init(model_obj)
        else:
            raise ValueError("Initializer must be kaiming, xavier or equi_var")

        if self.model == "cnn_supervised":
            log_save_dir = "./logs/log_supervised_autoenc_sweep/"
        elif self.model == "escnn_supervised":
            log_save_dir = "./logs/log_supervised_escnn_sweep/"
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

        lr_monitor = LearningRateMonitor(logging_interval="step")
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=log_save_dir,
            filename="{epoch:02d}-{val_loss:.2f}",
            mode="min",
        )
        callbacks = [
            early_stop_callback,
            lr_monitor,
            log_parameters,
            InputMonitor(),
            checkpoint_callback,
        ]
        trainer = pl.Trainer(
            max_epochs=config["max_epochs"],
            accelerator="gpu",
            devices=[0],
            accumulate_grad_batches=config["accumulate_grad_batches"],
            enable_progress_bar=True,
            log_every_n_steps=10,
            gradient_clip_val=config["gradient_clip_val"],
            callbacks=callbacks,
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
            print("-" * 20 + "Training in tuner" + "-" * 20)

            wandb.log(
                {
                    "aug": self.aug,
                    "transform": self.transform,
                    "scalar": self.scalar,
                    "offset": self.offset,
                    "model": self.model,
                    # "device": self.device,
                    "data_dir": self.data_dir,
                }
            )
            # print("-" * 20 + "Training in tuner" + "-" * 20)
            trainer.fit(model_obj, self.data_loader_train, self.data_loader_test)

            model_obj.eval()
            # save state dict
            torch.save(model_obj.state_dict(), log_save_dir + "/model_1.ckpt")
            run.finish()


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
        "-supervised_file",
        action="store",
        dest="super_file",
        default="../../data/protein_data.csv",
        help="dataset to use, labels",
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
        default="cnn_supervised",
        help="model",
    )

    parser.add_argument(
        "-im_size", action="store", dest="im_size", default=21, help="image mesh size"
    )

    method = "bayes"
    project_name = "vae_dist_supervised_sweep"
    sweep_config = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = parser.parse_args()
    model = str(results.model)
    data_dir = str(results.dataset)
    super_file = str(results.super_file)
    count = int(results.count)
    image_size = int(results.im_size)
    dataset_name = str(results.dataset).split("/")[-2]

    assert model in [
        "cnn_supervised",
        "escnn_supervised",
    ], "supervised sweep requires cnn_supervised or escnn_supervised"
    dict_hyper = hyperparameter_dicts(image_size=image_size)

    sweep_config["parameters"] = dict_hyper[model]
    sweep_config["name"] = method + "_" + model + "_" + dataset_name
    sweep_config["method"] = method
    if method == "bayes":
        sweep_config["metric"] = {"name": "val_f1", "goal": "maximize"}

    pre_process_options = {
        "transform": False,
        "augmentation": False,
        "scalar": False,
        "offset": 1,
    }

    sweep_id = wandb.sweep(sweep_config, project=project_name)

    training_obj = training(
        data_dir,
        super_file,
        model=model,
        device=device,
        transform=pre_process_options["transform"],
        augmentation=pre_process_options["augmentation"],
        scalar=pre_process_options["scalar"],
        offset=pre_process_options["offset"],
        project=project_name,
    )

    wandb.agent(sweep_id, function=training_obj.train, count=count)
