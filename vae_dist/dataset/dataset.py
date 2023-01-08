import torch 
from vae_dist.dataset.fields import pull_fields
import numpy as np 
# create class for dataset
class FieldDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, augmentation=None, standardize=True, device='cpu'):
        fields, shape = pull_fields(root)
        data = fields.reshape([len(fields), 3, shape[0], shape[1], shape[2]])
        
        self.data_std = data.std(axis = (0,1,2,3), keepdims=True)
        self.data_mean = data.mean(axis = (0,1,2,3), keepdims=True)
        self.data_max = data.max(axis = (0,1,2,3), keepdims=True)
        self.data_min = data.min(axis = (0,1,2,3), keepdims=True)

        if standardize:
            data = (data - self.data_min) / (self.data_max - self.data_min + 0.0001)
        # print if any values are nan
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

        if len(index) == 1:
            data = data.reshape([3, self.shape[0], self.shape[1], self.shape[2]])
        
        return data.to(self.device, dtype=torch.float)

    def dataset_to_tensor(self):
        self.data = torch.tensor(self.data).to(self.device)
        
    def dataset_to_numpy(self): 
        self.data = self.data.numpy()
        