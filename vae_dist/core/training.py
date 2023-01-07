import torch 



def train(model, data_loader, epochs=20, device=None):
    
    opt = torch.optim.Adam(model.parameters(),
                             lr = 1e-1,
                             weight_decay = 1e-8)



    for epoch in range(epochs):
        running_loss = 0.0
        for x in data_loader:
            print(x.shape)
            #print(x.is_cuda)
            #print(next(model.parameters()).is_cuda)
            # show data type of x 
            print(x.dtype)
            predict = model(x)
            #print(x.get_device())
            loss = model.loss(predict, x)
            model.loss.backward()
            opt.step()
            running_loss += loss.item()
        print("running loss: {}".format(running_loss))
    return model


def test(model, dataset_test):
    tensor = torch.tensor(dataset_test)
    loader = torch.utils.data.DataLoader(tensor, batch_size=4, shuffle=True)





#def train_lightening():
#    pl.seed_everything(1234)
#    vae = VAE()
#    trainer = pl.Trainer(gpus=1, max_epochs=30, progress_bar_refresh_rate=10)
#    trainer.fit(vae, cifar_10)