import torch 

def train(autoencoder, data, epochs=20): # ttodo
    opt = torch.optim.Adam(autoencoder.parameters())
    for epoch in range(epochs):
        for x, y in data:
            x = x.to(device) # GPU
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat)**2).sum() + autoencoder.encoder.kl
            loss.backward()
            opt.step()
    return autoencoder


    

#def train_lightening():
#    pl.seed_everything(1234)
#    vae = VAE()
#    trainer = pl.Trainer(gpus=1, max_epochs=30, progress_bar_refresh_rate=10)
#    trainer.fit(vae, cifar_10)