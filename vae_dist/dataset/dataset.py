import torch 
from vae_dist.dataset.fields import pull_fields
import numpy as np 
import matplotlib.pyplot as plt

# create class for dataset
class FieldDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, augmentation=None, standardize=True, log_scale=False, device='cpu'):
        fields, shape, names = pull_fields(root, ret_names=True)
        data = fields.reshape([len(fields), 3, shape[0], shape[1], shape[2]])

        
        # compute magnitude of vectors in data and store
        #print(data.shape)
        self.mags = np.sqrt((data**2).sum(axis=1))
        #print(self.mags.shape)
        # find index of max magnitude
        max_mag_ind = np.unravel_index(self.mags.argmax(), self.mags.shape)
        # compute minimum vector magnitude
        self.min_mag = self.mags.min()
        self.max_mag = self.mags.max()
        self.st_mag = self.mags.std()
        self.mean_mag = self.mags.mean()

        if standardize:
            # standardize every field 
            data = (data - self.mean_mag) / (self.st_mag + 0.0001)
            #data = (data - self.min) / (self.max - self.min + 0.0001)
        
        if log_scale:
            x_sign = np.sign(data)
            # getting absolute value of every element
            x_abs = np.abs(data)
            # applying log1p
            x_log1p = np.log1p(x_abs)
            # getting sign back
            data = np.multiply(x_log1p, x_sign)
        
        if np.isnan(data).any():
            print("Nan values in dataset")
            
        self.max = data.max()
        self.min = data.min()

        # print largest and smallest values
        print("Largest value in dataset: ", self.max)
        print("Smallest value in dataset: ", self.min)
        self.shape = shape
        self.data = data
        self.dataraw = self.data
        self.transform = transform
        self.augmentation = augmentation
        self.transform = transform        
        self.device = device
        self.standardize = standardize
        self.names = names
        
        
    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):

        if isinstance(index, int):
            index = [index]

        if self.augmentation:
            if self.transform:
                data = self.transform(self.data[index])
                data = self.augmentation(data)
            else:
                data = self.augmentation(self.data[index])
            
        elif self.transform:
            data = self.transform(self.data[index])

        else:
            data = self.data[index]

        #return tensor 
        data = torch.tensor(data)

        if not self.augmentation and len(index) == 1:
            data = data.reshape([3, self.shape[0], self.shape[1], self.shape[2]])
            
        #if len(index) == 1 and self.augmentation:
        #    data = data.reshape([3, self.shape[0], self.shape[1], self.shape[2]])
        
        return data.to(self.device, dtype=torch.float)


    def dataset_to_tensor(self):
        data_tensor = torch.tensor(self.data).to(self.device)
        return data_tensor

    def dataset_to_numpy(self): 
        data_np = self.data.numpy()
        return data_np


def dataset_split_loader(dataset, train_split, batch_size=1, num_workers=0, shuffle=True):

    # train test split - randomly split dataset into train and test
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])


    dataset_loader_full = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    dataset_loader_train= torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    dataset_loader_test= torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return dataset_loader_full, dataset_loader_train, dataset_loader_test
