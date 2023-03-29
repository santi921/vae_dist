import torch 
from vae_dist.dataset.fields import pull_fields, filter, helmholtz_hodge_decomp_approx
import numpy as np 
import matplotlib.pyplot as plt
from copy import deepcopy
# create class for dataset
class FieldDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            root, 
            transform=None, 
            augmentation=None, 
            standardize=True, 
            log_scale=False,
            lower_filter=False, 
            scalar=False,
            device='cpu', 
            mean_mag=None,
            wrangle_outliers=False,
            min_max_scale=False,
            show=False,
            st_mag=None):
        
        fields, shape, names = pull_fields(root, ret_names=True)
        data = fields.reshape([len(fields), 3, shape[0], shape[1], shape[2]])
        self.shape = data.shape
        self.dataraw = deepcopy(data)
        self.mags = np.sqrt((data**2).sum(axis=1))
        dim = 3

        if scalar:
            data = self.mags.reshape([len(fields), 1, shape[0], shape[1], shape[2]])
            dim = 1
            

        if log_scale:

            if scalar:                      
                x_log1p = np.log1p(data)
                data = x_log1p

            else: 
                # get magnitude of vector field
                mags = np.sqrt((data**2).sum(axis=1))
                mags = mags.reshape([len(fields), 1, shape[0], shape[1], shape[2]])
                mags_log1p = np.log1p(mags)
                # get ratio magnitude to mags_log1p
                ratio = mags_log1p / mags
                multiply = np.multiply(data, ratio) 
                data = multiply


        if wrangle_outliers: 
            if log_scale: multiplier = 5
            else: multiplier = 3
            
            # get mean and std of data
            mags = np.sqrt((data**2).sum(axis=1))
            mean = mags.mean()
            std = mags.std()
            min_mag = mags.min()
            max_mag = mags.max()
            print("wrangle data mean {}, std {}".format(mean, std))
            print("wrangle min {}, max {}".format(min_mag, max_mag))
            # get indices of outliers
            # create matrix of same shape as data with values of mean + 3*std
            print(mean + multiplier*std)
            dummy = np.ones(mags.shape) * (mean + multiplier*std)

            # outliers are defined as values greater than mean + 3*std
            outliers_filtered = np.where(mags > mean + multiplier*std, dummy, mags)
            
            if scalar: 
                data = outliers_filtered
                
            else:
                # get ratio between mags and outliers_filtered
                ratio = outliers_filtered/mags
                # fluff ratio by copying it 3 times
                ratio = ratio.reshape([len(fields), 1, shape[0], shape[1], shape[2]])
                ratio = np.repeat(ratio, 3, axis=1)
                # multiply data by ratio
                data = np.multiply(data, ratio)
                
                
        if standardize:
            mag = np.sqrt((data**2).sum(axis=1))
            self.mean_mag = mag.mean()
            self.st_mag = mag.std()

            # standardize every field 
            if mean_mag == None and st_mag == None:
                data = (data - self.mean_mag) / (self.st_mag + 1)            
            else: 
                data = (data - mean_mag) / (st_mag + 1)    
            
        
        if transform:
            # throw assertion error if scaler is true 
            assert scalar == False, "Cannot transform scalar fields"

            transform_mat = []
            for i in range(data.shape[0]):
                transform_mat.append(helmholtz_hodge_decomp_approx(data[i]))
            transform_mat = np.array(transform_mat)
            data = transform_mat
            self.mags = np.sqrt((data**2).sum(axis=1))
        

        if min_max_scale:
            mag = np.sqrt((data**2).sum(axis=1))
            max = mag.max()
            min = mag.min()
            ## min max scale 
            data = (data - min) / (max - min)
            self.mags = np.sqrt((data**2).sum(axis=1))
        
        
        if lower_filter:
            filter_mat = []
            
            #print(self.shape)

            for i in range(self.shape[0]-1):
                #print(i)
                filter_mat.append(
                    filter(
                        data[i].reshape([1, dim, self.shape[2], self.shape[3], self.shape[4]]),
                        cutoff_low_percentile=85, 
                        cutoff_high_percentile=False,
                        dim = dim).reshape([dim, self.shape[2], self.shape[3], self.shape[4]]))
            filter_mat = np.array(filter_mat)
            data = filter_mat


        if show:
            self.mags = np.sqrt((data**2).sum(axis=1))
            plt.hist(self.mags.flatten(), bins=100)
            plt.show()
        
        
        print("Data shape: ", data.shape)
        print("Data type: ", data.dtype)
        print("-" * 25 + " Preprocessing Info " + "-" * 25)
        print("Helmholtz-Hodge decomposition applied: ", transform)
        print("Lower filter applied: ", lower_filter)
        print("Log scale applied: ", log_scale)
        print("Standardization applied: ", standardize)
        print("Min max scaling applied: ", min_max_scale)
        print("Wrangling outliers applied: ", wrangle_outliers)
        print("-" * 25 + " Data Info " + "-" * 25)
        print("Mean value in dataset: ", data.mean())
        print("Standard deviation in dataset: ", data.std())
        print("Largest value in dataset: ", data.max())
        print("Smallest value in dataset: ", data.min())
        print("Nan values in dataset: ", np.isnan(data).any())
        print("Inf values in dataset: ", np.isinf(data).any()) 

        self.mags = np.sqrt((data**2).sum(axis=1))
        self.max = data.max()
        self.min = data.min()

        #self.shape = data.shape
        self.scalar = scalar
        self.data = data
        
        self.transform = False
        self.augmentation = augmentation       
        self.device = device
        self.standardize = standardize
        self.names = names
        self.min_max_scale = min_max_scale
        self.log_scale = log_scale
        self.wrangle_outliers = wrangle_outliers
        self.lower_filter = lower_filter
        
        
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
            
        else:
            data = self.data[index]

        #return tensor 
        data = torch.tensor(data)

        if not self.augmentation and len(index) == 1:
            channels = 3 
            if self.scalar: channels = 1
            
            data = data.reshape([channels, self.shape[2], self.shape[3], self.shape[4]])
            
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
 