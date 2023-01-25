import argparse
from escnn import gspaces, nn                                         
import torch                                                      
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor

from vae_dist.core.O3VAE import R3VAE
from vae_dist.dataset.dataset import FieldDataset, dataset_split_loader


def main():              
    # create argparser that just takes a string for model type 
    # and a string for the path to the data

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='escnn')
    parser.add_argument('--data', type=str, default='../../data/cpet/')
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    root = args.data
    model_select = args.model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    
    dataset_vanilla = FieldDataset(
        root, 
        transform=False, 
        augmentation=False,
        standardize=True,
        device=device
        )

    if model_select == 'escnn':
        log_save_dir = "./log_version_escnn_1/"


    elif model_select == 'esvae':
        log_save_dir = "./log_version_esvae_1/"
        group = gspaces.flipRot3dOnR3(maximum_frequency=10) 
        input_out_reps = 3*[group.trivial_repr]
        kernel_size = 5
        feat_type_in  = nn.FieldType(group,  input_out_reps) 
        feat_type_out = nn.FieldType(group,  input_out_reps)    
        model = R3VAE(
            group = group,
            feat_type_in = feat_type_in, 
            feat_type_out = feat_type_out, 
            kernel_size=kernel_size,
            latent_dim=10, 
            learning_rate=0.0005,
            fully_connected_dims=[100, 100, 100]
        )

    elif model_select == 'auto':
        log_save_dir = "./log_version_auto_1/"
    elif model_select == 'vae':
        log_save_dir = "./log_version_vae_1/"
    else: 
        # throw error
        print("Model not found")
        return

    

    # load model to gpu
    model.to(device)
    
    dataset_loader_full, dataset_loader_train, dataset_loader_test= dataset_split_loader(dataset_vanilla, train_split=0.8)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(
        limit_train_batches=100, 
        max_epochs=100, 
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
        dataset_loader_test, 
        )

    model.eval()
    # save state dict
    torch.save(model.state_dict(), "./log_version_esvae_1/autoenc_1.ckpt")

main()
