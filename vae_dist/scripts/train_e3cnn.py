import pytorch_lightning as pl
from vae_dist.core.O3CNN import e3CNN
from vae_dist.dataset.dataset import FieldDataset
from pytorch_lightning.callbacks import LearningRateMonitor
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

    model = e3CNN(
        irreps_in='3x0e',
        irreps_out='3x0e',
        n=3, 
        n_blocks_down=2, 
        n_blocks_up=2,
        hidden_dim=1,
        stride=1,
        device = device,
        learning_rate=1e-4,
        latent_dim = 4,
        dropout_prob=0.1,
        cutoff=False,        
        kernel_size = 5,
        down_op='maxpool3d',
        batch_norm='instance',
        lmax=2,
        n_downsample=2,
        equivariance='SO3',
        bias = True,
        scalar_upsampling=False,
        scale=2,
        num_radial_basis_down=4,
        num_radial_basis_up=4,
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(
        limit_train_batches=100, 
        max_epochs=100, 
        accelerator='gpu', 
        devices = [0],
        accumulate_grad_batches=5, 
        callbacks=[
            pl.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose = False),
            lr_monitor]
    )

    trainer.fit(model, dataset_loader_train, dataset_loader_test)

    # find lr 
    trainer_lr = pl.Trainer(auto_lr_find=True, max_epochs=100, gpus=1)
    lr_finder = trainer_lr.tuner.lr_find(model, dataset_loader_train, dataset_loader_test)
    fig = lr_finder.plot(suggest=True)
    fig.show()
    lr_finder.suggestion()

main()