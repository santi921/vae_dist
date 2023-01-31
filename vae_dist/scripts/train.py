import argparse, json, wandb
from escnn import gspaces, nn                                         
import torch                                                      
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from vae_dist.dataset.dataset import FieldDataset, dataset_split_loader
from vae_dist.core.training import construct_model

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

    if model_select == 'escnn' or model_select == 'auto':        
        run = wandb.init(project="cnn_dist", reinit=True)
    else:
        run = wandb.init(project="vae_dist", reinit=True)
    
    dataset_vanilla = FieldDataset(
        root, 
        transform=False, 
        augmentation=False,
        standardize=False,
        device=device, 
        log_scale=True
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

    wandb.config.update(options)
    
    # load model to gpu
    model.to(device)
    
    dataset_loader_full, dataset_loader_train, dataset_loader_test= dataset_split_loader(dataset_vanilla, train_split=0.8)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(
        limit_train_batches=100, 
        max_epochs=epochs, 
        accelerator='gpu', 
        devices = [0],
        accumulate_grad_batches=5, 
        callbacks=[
            pl.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose = False),
            lr_monitor],
        enable_checkpointing=True,
        default_root_dir=log_save_dir

    )

    trainer.fit(
        model, 
        dataset_loader_train, 
        dataset_loader_test
        )

    model.eval()
    # save state dict
    torch.save(model.state_dict(), log_save_dir + "/model_1.ckpt")
    run.finish()

main()
