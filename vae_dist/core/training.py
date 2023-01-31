import torch 
from escnn import gspaces, nn, group                                        

from vae_dist.core.O3VAE import R3VAE
from vae_dist.core.R3CNN import R3CNN
from vae_dist.core.VAE import baselineVAEAutoencoder
from vae_dist.core.CNN import CNNAutoencoderLightning
from escnn import gspaces, nn, group                                         


def train(model, data_loader, epochs=20):
    opt = torch.optim.Adam(model.parameters(),
                             lr = 1e-1,
                             weight_decay = 1e-8)

    for epoch in range(epochs):
        running_loss = 0.0
        for x in data_loader:            
            predict = model(x)
            loss = model.loss_function(x, predict)
            loss.backward()
            opt.step()
            running_loss += loss.item()
        print("epoch: {} loss: {}".format(epoch, running_loss))
    

def construct_model(model, options):
    """
    Constructs a model based on the model name and options.
    Takes 
        model: string, name of the model
        options: dict, options for the model
    Returns
        model: pl.LightningModule, the model
    """
    if model == 'esvae':
        g = group.o3_group()
        gspace = gspaces.flipRot3dOnR3(maximum_frequency=10) 
        input_out_reps = 3*[gspace.trivial_repr]
        kernel_size = 5
        feat_type_in  = nn.FieldType(gspace,  input_out_reps) 
        feat_type_out = nn.FieldType(gspace,  input_out_reps) 
        model = R3VAE(**options, gspace=gspace, group=g, feat_type_in=feat_type_in, feat_type_out=feat_type_out)
    
    elif model == 'escnn':
        g = group.o3_group()
        gspace = gspaces.flipRot3dOnR3(maximum_frequency=10) 
        #gspace = gspace.no_base_space(g)
        input_out_reps = 3*[gspace.trivial_repr]
        #kernel_size = 5
        feat_type_in  = nn.FieldType(gspace,  input_out_reps) 
        feat_type_out = nn.FieldType(gspace,  input_out_reps)  
        model = R3CNN(**options, gspace=gspace, group=g, feat_type_in=feat_type_in, feat_type_out=feat_type_out)   

    elif model == 'auto':
        model = CNNAutoencoderLightning(**options)

    elif model == 'vae':
        model = baselineVAEAutoencoder(**options)

    return model