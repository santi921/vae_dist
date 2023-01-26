import torch 
from escnn import gspaces, nn                                         

from vae_dist.core.O3VAE import R3VAE
from vae_dist.core.R3CNN import R3CNN
from vae_dist.core.VAE import baselineVAEAutoencoder
from vae_dist.core.CNN import CNNAutoencoderLightning
from escnn import gspaces, nn                                         


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

    if model == 'esvae':

        group = gspaces.flipRot3dOnR3(maximum_frequency=10) 
        input_out_reps = 3*[group.trivial_repr]
        kernel_size = 5
        feat_type_in  = nn.FieldType(group,  input_out_reps) 
        feat_type_out = nn.FieldType(group,  input_out_reps)    
        model = R3VAE(**options, group=group, feat_type_in=feat_type_in, feat_type_out=feat_type_out)
    
    elif model == 'escnn':

        group = gspaces.flipRot3dOnR3(maximum_frequency=10) 
        input_out_reps = 3*[group.trivial_repr]
        kernel_size = 5
        feat_type_in  = nn.FieldType(group,  input_out_reps) 
        feat_type_out = nn.FieldType(group,  input_out_reps)  
        model = R3CNN(**options, group=group, feat_type_in=feat_type_in, feat_type_out=feat_type_out)   

    elif model == 'auto':
        model = CNNAutoencoderLightning(**options)

    elif model == 'vae':
        model = baselineVAEAutoencoder(**options)

    return model

#def test(model, dataset_test):
#    tensor = torch.tensor(dataset_test)
#    loader = torch.utils.data.DataLoader(tensor, batch_size=4, shuffle=True)


#def train_lightening():
#    pl.seed_everything(1234)
#    vae = VAE()
#    trainer = pl.Trainer(gpus=1, max_epochs=30, progress_bar_refresh_rate=10)
#    trainer.fit(vae, cifar_10)