from vae_dist.core.VAE import baselineVAEAutoencoder

import pytorch_lightning as pl
from vae_dist.dataset.dataset import FieldDataset
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging
import torch 


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = "../../data/cpet/"
    # load model to gpu
    dataset_vanilla = FieldDataset(
        root, 
        transform=None, 
        augmentation=None, 
        device=device
        )

    #torch.multiprocessing.set_start_method('spawn')# good solution !!!!

    # train test split - randomly split dataset into train and test
    train_size = int(0.8 * len(dataset_vanilla))
    test_size = len(dataset_vanilla) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset_vanilla, [train_size, test_size])


    dataset_loader_full = torch.utils.data.DataLoader(
        dataset_vanilla, 
        batch_size=40,
        shuffle=True,
        num_workers=0
    )

    dataset_loader_train= torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=10,
        shuffle=True,
        num_workers=0
    )

    dataset_loader_test= torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=10,
        shuffle=True,
        num_workers=0
    )

    defaults = {
        "irreps" : None,
        "in_channels" : 3,
        "out_channels" : 16,
        "kernel_size" : 5,
        "stride" : 1,
        "padding" : 0,
        "dilation" : 1,
        "groups" : 1,
        "bias" : True,
        "padding_mode" : 'zeros',
        "latent_dim" : 4, # final vae hidden layer
        "num_layers" : 2, # not used rn
        "hidden_dim" : 32,
        "activation" : 'relu', # not used rn
        "dropout" : 0.1, # not used rn
        "batch_norm" : False, # not used rn
        "beta" : 1.0,
        "device" : device,
        "learning_rate" : 0.001
    }

    vae = baselineVAEAutoencoder(**defaults)

    trainer = pl.Trainer(
            gpus=1, 
            max_epochs=30, 
            callbacks=[
                EarlyStopping(monitor='elbo_val', patience=10),
                #StochasticWeightAveraging(swa_lrs=1e-2)
                ],
            accumulate_grad_batches=5,
            gradient_clip_val=1.0, 
            gradient_clip_algorithm='value',
            precision=16,
            enable_checkpointing=True,
            default_root_dir="./log_version_vae_1/"
        )
    trainer.fit(vae, dataset_loader_train, dataset_loader_test)

main()