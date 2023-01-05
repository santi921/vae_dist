import torch 
from vae_dist.dataset.fields import pull_fields

# create class for dataset
class FieldDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, augmentation=None):
        self.data = pull_fields(root)
        self.dataraw = self.data
        self.transform = transform
        self.augmentation = augmentation
        self.transform = transform        
        

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

        return data

    def dataset_to_tensor(self):
        self.data = torch.tensor(self.data)
        
    def dataset_to_numpy(self): 
        self.data = self.data.numpy()
        