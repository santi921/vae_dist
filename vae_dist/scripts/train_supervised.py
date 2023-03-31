import argparse, json, wandb                                         
import torch                                                      
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from vae_dist.dataset.dataset import FieldDatasetSupervised
from vae_dist.core.training_utils import (
    construct_model, 
    LogParameters, 
    InputMonitor, 
    CheckBatchGradient
)
from vae_dist.core.intializers import *

def set_enviroment():
    from torch.multiprocessing import set_start_method
    torch.set_float32_matmul_precision("high")
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

if __name__ == '__main__':
    set_enviroment()
    torch.multiprocessing.freeze_support()
    # create argparser that just takes a string for model type 
    # and a string for the path to the data
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='escnn')
    parser.add_argument('--epochs', type=int, default=1000)

    args = parser.parse_args()
    epochs = args.epochs
    model_select = args.model
    
    root = "../../data/"
    dataset = "cpet_augmented"
    root = root + dataset + "/"
    supervised_file = "../../data/protein_data.csv"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pre_process_options = {
        "transform": False,
        "augmentation": False,
        "standardize": True,
        "lower_filter": True,
        "log_scale": True,
        "min_max_scale": False,
        "wrangle_outliers": False,
        "scalar": False,
        "offset": 1
    }

    dataset_vanilla = FieldDatasetSupervised(
        root, 
        supervised_file,
        transform=pre_process_options['transform'], 
        augmentation=pre_process_options['augmentation'],
        standardize=pre_process_options['standardize'],
        lower_filter=pre_process_options['lower_filter'],
        log_scale=pre_process_options['log_scale'], 
        min_max_scale=pre_process_options['min_max_scale'],
        wrangle_outliers=pre_process_options['wrangle_outliers'],
        scalar=pre_process_options['scalar'],
        device=device, 
        offset=pre_process_options['offset']
    )


    if model_select == 'escnn' or model_select == 'cnn':        
        run = wandb.init(project="supervised_vae_{}".format(dataset), reinit=True)
    
    assert model_select in ['escnn', 'cnn'], "Model must be escnn or cnn"
    
    if model_select == 'escnn':
        options = json.load(open('./options/options_escnn_default_supervised.json'))
        log_save_dir = "./logs/log_version_escnn/"
        model = construct_model("escnn_supervised", options)

    elif model_select == 'cnn':
        options = json.load(open('./options/options_cnn_default_supervised.json'))
        log_save_dir = "./logs/log_version_cnn/"
        model = construct_model("cnn_supervised", options)

    wandb.config.update({
        "model": model_select,
        "epochs": epochs,
        "data": root
    })
    wandb.config.update(options)
    
    # load model to gpu
    model.to(device)

    initializer = options['initializer']
    if initializer == 'kaiming':
        kaiming_init(model)
    elif initializer == 'xavier':
        xavier_init(model)
    elif initializer == 'equi_var': # works
        equi_var_init(model)
    else:
        raise ValueError("Initializer must be kaiming, xavier or equi_var")

    
    # check if there are any inf or nan values in the model
    is_nan = torch.stack([torch.isnan(p).any() for p in model.parameters()]).any()
    print("Model has inf or nan values: ", is_nan)
    # check if dataset has any inf or nan values
    print("Dataset has inf or nan values: ", torch.isnan(dataset_vanilla.dataset_to_tensor()).any())
    
    dataset_loader_full = torch.utils.data.DataLoader(
        dataset_vanilla, 
        batch_size=64,
        shuffle=False,
        num_workers=0
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    # create early stopping callback
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=300,
        verbose=False,
        mode='max'
    )

    log_parameters = LogParameters()
    logger_tb = TensorBoardLogger(log_save_dir, name="test_logs")
    logger_wb = WandbLogger(project="{}_supervised_vae_dist".format(model_select), name="test_logs")

    trainer = pl.Trainer(
        max_epochs=epochs, 
        accelerator='gpu', 
        devices = [0],
        accumulate_grad_batches=2, 
        enable_progress_bar=True,
        log_every_n_steps=10,
        gradient_clip_val=2.0,
        callbacks=[early_stop_callback,  
            lr_monitor, 
            log_parameters,
            InputMonitor()],
        enable_checkpointing=True,
        default_root_dir=log_save_dir,
        logger=[logger_tb, logger_wb],
        detect_anomaly=True,
        precision=16
    )
    
    trainer.fit(
        model, 
        dataset_loader_full, 
        dataset_loader_full
        )


    model.eval()
    # save state dict
    print("Saving model to: ", log_save_dir + "/model_supervised_datapoint.ckpt")
    torch.save(model.state_dict(), log_save_dir + "/model_supervised_datapoint.ckpt")
    run.finish()

