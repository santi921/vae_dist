import argparse, json, wandb                                         
import torch                                                      
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from vae_dist.dataset.dataset import FieldDataset, dataset_split_loader
from vae_dist.core.training_utils import construct_model
torch.set_float32_matmul_precision("high")
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from vae_dist.core.training_utils import construct_model, LogParameters


def main():              

    # create argparser that just takes a string for model type 
    # and a string for the path to the data

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='escnn')
    parser.add_argument('--data', type=str, default='../../data/cpet/')
    parser.add_argument('--epochs', type=int, default=1000)
    #parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    root = args.data
    epochs = args.epochs
    model_select = args.model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_select == 'escnn' or model_select == 'cnn':        
        run = wandb.init(project="cnn_dist", reinit=True)
    else:
        run = wandb.init(project="vae_dist", reinit=True)
    

    dataset_vanilla = FieldDataset(
        root, 
        transform=False, 
        augmentation=False,
        standardize=False,
        lower_filter=True,
        log_scale=True, 
        min_max_scale=True,
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

    elif model_select == 'cnn':
        options = json.load(open('./options/options_cnn_default.json'))
        log_save_dir = "./log_version_cnn_1/"
        model = construct_model("cnn", options)

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
    model.to(device)
    # check if there are any inf or nan values in the model
    is_nan = torch.stack([torch.isnan(p).any() for p in model.parameters()]).any()
    print("Model has inf or nan values: ", is_nan)
    # check if dataset has any inf or nan values
    print("Dataset has inf or nan values: ", torch.isnan(dataset_vanilla.dataset_to_tensor()).any())
    dataset_loader_full, dataset_loader_train, dataset_loader_test= dataset_split_loader(dataset_vanilla, train_split=0.8, num_workers=0)


    log_parameters = LogParameters()
    logger_tb = TensorBoardLogger(log_save_dir, name="test_logs")
    logger_wb = WandbLogger(project="{}_dist_single".format(model_select), name="test_logs")
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=200,
        verbose=False,
        mode='min'
    )
    
    trainer = pl.Trainer(
        max_epochs=epochs, 
        accelerator='gpu', 
        devices = [0],
        accumulate_grad_batches=3, 
        enable_progress_bar=True,
        gradient_clip_val=0.5,
        callbacks=[early_stop_callback,  
            lr_monitor, 
            log_parameters],
        enable_checkpointing=True,
        default_root_dir=log_save_dir,
        logger=[logger_tb, logger_wb],
        detect_anomaly=True,
        #pin_memory=True,
        #precision=16
    )

    trainer.fit(
        model, 
        dataset_loader_train, 
        dataset_loader_test
        )

    model.eval()
    # save state dict
    torch.save(model.state_dict(), log_save_dir + "/model_1.ckpt")
    #torch.save(model, log_save_dir + "/model_1.pt")
    run.finish()

main()
