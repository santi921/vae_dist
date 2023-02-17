import argparse, json, wandb                                         
import torch                                                      
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from vae_dist.dataset.dataset import FieldDataset
from vae_dist.core.training_utils import construct_model, LogParameters
from vae_dist.core.intializers import *
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

def main():              

    # create argparser that just takes a string for model type 
    # and a string for the path to the data
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='escnn')
    parser.add_argument('--epochs', type=int, default=1000)

    args = parser.parse_args()

    root = "../../data/single_field/"
    epochs = args.epochs
    model_select = args.model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_select == 'escnn' or model_select == 'auto':        
        run = wandb.init(project="cnn_dist_single", reinit=True)
    else:
        run = wandb.init(project="vae_dist_single", reinit=True)
    

    dataset_vanilla = FieldDataset(
        root, 
        transform=False, 
        augmentation=False,
        standardize=False,
        lower_filter=True,
        log_scale=True, 
        min_max_scale=False,
        wrangle_outliers=False,
        scalar=False,
        device=device
    )

    if model_select == 'escnn':
        options = json.load(open('./options/options_escnn_default.json'))
        log_save_dir = "./log_version_escnn_1/"
        model = construct_model("escnn", options)


    elif model_select == 'esvae':
        options = json.load(open('./options/options_esvae_default.json'))
        log_save_dir = "./log_version_esvae_1/"
        model = construct_model("esvae", options)

    elif model_select == 'auto':
        options = json.load(open('./options/options_cnn_default.json'))
        log_save_dir = "./log_version_auto_1/"
        model = construct_model("auto", options)

    elif model_select == 'vae':
        options = json.load(open('./options/options_vae_default.json'))
        log_save_dir = "./log_version_vae_1/"
        model = construct_model("vae", options)

    else: 
        # throw error
        print("Model not found")
        return
    
    wandb.config.update({
        "model": model_select,
        "epochs": epochs,
        "data": root
    })
    wandb.config.update(options)
    
    # load model to gpu
    #kaiming_init(model)
    model.to(device)
    
    xavier_init(model)
    #visualize_weight_distribution(model)
    #visualize_activations(model, print_variance=True)
    # check if there are any inf or nan values in the model
    is_nan = torch.stack([torch.isnan(p).any() for p in model.parameters()]).any()
    print("Model has inf or nan values: ", is_nan)
    # check if dataset has any inf or nan values
    print("Dataset has inf or nan values: ", torch.isnan(dataset_vanilla.dataset_to_tensor()).any())
    
    dataset_loader_full = torch.utils.data.DataLoader(
        dataset_vanilla, 
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    # create early stopping callback
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=200,
        verbose=False,
        mode='min'
    )
    log_parameters = LogParameters()

    logger_tb = TensorBoardLogger(log_save_dir, name="test_logs")
    logger_wb = WandbLogger(project="cnn_dist_single", name="test_logs")

    trainer = pl.Trainer(
        max_epochs=epochs, 
        accelerator='gpu', 
        devices = [0],
        accumulate_grad_batches=1, 
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        callbacks=[early_stop_callback,  
            lr_monitor, 
            log_parameters],
        enable_checkpointing=True,
        default_root_dir=log_save_dir,
        logger=[logger_tb, logger_wb],
    )
    
    trainer.fit(
        model, 
        dataset_loader_full, 
        dataset_loader_full
        )


    model.eval()
    # save state dict
    torch.save(model.state_dict(), log_save_dir + "/model_single_datapoint.ckpt")
    run.finish()

main()
