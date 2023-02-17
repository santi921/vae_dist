import argparse, json, wandb                                         
import torch                                                      
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from vae_dist.dataset.dataset import FieldDataset, dataset_split_loader
from vae_dist.core.training_utils import construct_model

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
    
    for transform in [False, True]:   
        for standardize in [False, True]:
            for lower_filter in [False, True]:
                for log_scale in [False, True]:
                    save_name = f"transform_{transform}_standardize_{standardize}_lower_filter_{lower_filter}_log_scale_{log_scale}"

                    if model_select == 'escnn' or model_select == 'auto':        
                        run = wandb.init(project="cnn_dist_dataset_sweep", reinit=True)
                    else:
                        run = wandb.init(project="vae_dist_dataset_sweep", reinit=True)
                    
                    dataset_vanilla = FieldDataset(
                        root, 
                        transform=transform, 
                        augmentation=False,
                        standardize=standardize,
                        lower_filter=lower_filter,
                        log_scale=log_scale, 
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
                    
                    # load model to gpu
                    model.to(device)
                    # check if there are any inf or nan values in the model
                    is_nan = torch.stack([torch.isnan(p).any() for p in model.parameters()]).any()
                    print("Model has inf or nan values: ", is_nan)
                    # check if dataset has any inf or nan values
                    print("Dataset has inf or nan values: ", torch.isnan(dataset_vanilla.dataset_to_tensor()).any())
                    dataset_loader_full, dataset_loader_train, dataset_loader_test= dataset_split_loader(dataset_vanilla, train_split=0.8, num_workers=0)

                    lr_monitor = LearningRateMonitor(logging_interval='step')
                    trainer = pl.Trainer(
                        max_epochs=epochs, 
                        accelerator='gpu', 
                        devices = [0],
                        accumulate_grad_batches=5, 
                        enable_progress_bar=True,
                        gradient_clip_val=0.5,
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
                    torch.save(model.state_dict(), log_save_dir + "/{}_{}.ckpt".format(model_select, save_name))
                    #torch.save(model, log_save_dir + "/{}_{}.pt".format(model_select, save_name))
                    run.finish()

main()