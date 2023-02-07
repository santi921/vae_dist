import wandb, argparse
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor


from vae_dist.dataset.dataset import FieldDataset, dataset_split_loader
from vae_dist.core.training_utils import construct_model_hyper, hyperparameter_dicts

class training: 
    def __init__(self, data_dir, model, device, log_scale=True, standardize=True, aug=False, project="vae_dist"):
        self.aug = aug
        self.model = model  
        self.project = project
        self.device = device
        self.log_scale = log_scale
        self.standardize = standardize
        self.data_dir = data_dir   
        
        print("dataset parameters")
        print("augmentation: ", self.aug)
        print("standardize: ", self.standardize)
        print("log_scale: ", self.log_scale)
        print("model: ", self.model)
        print("device: ", self.device)
        print("data_dir: ", self.data_dir)
        print("project: ", self.project)

        dataset_vanilla = FieldDataset(
            data_dir, 
            transform=False, 
            augmentation=False,
            standardize=False,
            device=device, 
            log_scale=log_scale
            )

        dataset_loader_full, dataset_loader_train, dataset_loader_test= dataset_split_loader(dataset_vanilla, train_split=0.8, num_workers=0)
        self.data_loader_full = dataset_loader_full
        self.data_loader_train = dataset_loader_train
        self.data_loader_test = dataset_loader_test
        print("loaders  created " * 5)


    def make_model(self, config):
        
        model_obj = construct_model_hyper(model = self.model, options = config)
        model_obj.to(self.device)
        
        if self.model ==  'auto': 
            log_save_dir = "./log_version_auto_1/"
        elif self.model == 'vae':
            log_save_dir = "./log_version_vae_1/"
        elif self.model == 'esvae':
            log_save_dir = "./log_version_esvae_1/"
        elif self.model == 'escnn':
            log_save_dir = "./log_version_escnn_1/"
        else: 
            raise ValueError("model not found")
        
        lr_monitor = LearningRateMonitor(logging_interval='step')
        trainer = pl.Trainer(
                max_epochs=config['epochs'], 
                accelerator='gpu', 
                devices = [0],
                accumulate_grad_batches=5, 
                enable_progress_bar=True,
                callbacks=[
                    pl.callbacks.EarlyStopping(monitor='val_loss', patience=100, verbose = True),
                    lr_monitor],
                enable_checkpointing=True,
                default_root_dir=log_save_dir
        )

        return model_obj, trainer, log_save_dir


    def train(self):

        with wandb.init(project=self.project) as run:
            config = wandb.config
            model_obj, trainer, log_save_dir = self.make_model(config)
            print("-" * 20 + "Training" + "-" * 20)
            trainer.fit(
                model_obj, 
                self.data_loader_train, 
                self.data_loader_test
                )
            model_obj.eval()
            # save state dict
            torch.save(model_obj.state_dict(), log_save_dir + "/model_1.ckpt")
            run.finish() 

            
        run.finish()


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='options for hyperparam tune')
    parser.add_argument(
        "-dataset",
        action="store",
        dest="dataset",
        default=1,
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
        default=" auto",
        help="model",
    )

    parser.add_argument(
        "--aug", 
        action = "store_true",
        dest="aug",
        default = False,
        help="augment data"
    )
    
    pca_tf = True
    method = "bayes" 
    dataset_name = "base"
    project_name = "vae_dist"
    data_dir = '../../data/cpet/'
    sweep_config = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = parser.parse_args()
    model = str(results.model)
    dataset_int = int(results.dataset)
    count = int(results.count)
    aug = bool(results.aug)
    
    
    dict_hyper = hyperparameter_dicts()
    #print(dict_hyper)
    sweep_config["parameters"] = dict_hyper[model]
    sweep_config["name"] = method + "_" + model + "_" + dataset_name
    sweep_config["method"] = method 
    if(method == "bayes"):
        sweep_config["metric"] = {"name": "mape_val", "goal": "minimize"}
    
    sweep_id = wandb.sweep(sweep_config, project = project_name)
    training_obj = training(
        data_dir,
        model = model, 
        device = device,
        log_scale=True, 
        aug=aug,
        standardize=True,
        project = project_name)
    
    wandb.agent(sweep_id, function=training_obj.train, count=count)
    

    



