from vae_dist.core.CNN import baselineCNNAutoencoder
from vae_dist.core.training import train
from vae_dist.dataset.dataset import FieldDataset
import torch 

def main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = "../../data/cpet/"
    dataset_loader_full = torch.utils.data.DataLoader(dataset_vanilla, batch_size=4, shuffle=True)
    # load model to gpu
    dataset_vanilla = FieldDataset(
        root, 
        transform=None, 
        augmentation=None, 
        device=device
        )

    model = baselineCNNAutoencoder(
        irreps = None, # not used rn 
        in_channels = 3,
        out_channels = 16,
        kernel_size = 5,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups = 1,
        bias = True,
        padding_mode = 'zeros',
        #padding_mode = 'constant',
        latent_dim = 4, # final vae hidden layer 
        num_layers = 2, # not used rn 
        hidden_dim = 32,
        activation = 'relu', # not used rn 
        dropout = 0.1, # not used rn 
        batch_norm = False, # not used rn 
        beta = 1.0,
        device = device
    )

        
    train(model, dataset_loader_full, device = device)

    


main()