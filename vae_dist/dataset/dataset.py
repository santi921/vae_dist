import torch 
from vae_dist.dataset.fields import pull_fields
import numpy as np 

# create class for dataset
class FieldDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, augmentation=None, standardize=True, device='cpu'):
        fields, shape = pull_fields(root)
        data = fields.reshape([len(fields), 3, shape[0], shape[1], shape[2]])
        


        #self.data_std = data.std(axis = (0,1,2,3), keepdims=True)
        #self.data_mean = data.mean(axis = (0,1,2,3), keepdims=True)
        #self.data_max = data.max(axis = (0,1,2,3), keepdims=True)
        #self.data_min = data.min(axis = (0,1,2,3), keepdims=True)
        
        # compute maximum vector magnitude
        self.max_mag = np.sqrt((data**2).sum(axis=1)).max()
        # compute minimum vector magnitude
        self.min_mag = np.sqrt((data**2).sum(axis=1)).min()

        if standardize:
            data = (data - self.min_mag) / (self.max_mag - self.min_mag + 0.0001)
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

        if not self.augmentation and len(index) == 1:
            data = data.reshape([3, self.shape[0], self.shape[1], self.shape[2]])
            
        #if len(index) == 1 and self.augmentation:
        #    data = data.reshape([3, self.shape[0], self.shape[1], self.shape[2]])
        
        return data.to(self.device, dtype=torch.float)


    def dataset_to_tensor(self):
        self.data = torch.tensor(self.data).to(self.device)
        

    def dataset_to_numpy(self): 
        self.data = self.data.numpy()
        


def dataset_split_loader(dataset, train_split, batch_size=10, num_workers=0, shuffle=True):

    # train test split - randomly split dataset into train and test
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])


    dataset_loader_full = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
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
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    return dataset_loader_full, dataset_loader_train, dataset_loader_test
