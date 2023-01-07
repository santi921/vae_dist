import torch 
from vae_dist.dataset.fields import pull_fields

# create class for dataset
class FieldDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, augmentation=None, device='cpu'):
        fields, shape = pull_fields(root)
        data = fields.reshape([len(fields), 3, shape[0], shape[1], shape[2]])
        
        self.shape = shape
        self.data = data
        self.dataraw = self.data
        self.transform = transform
        self.augmentation = augmentation
        self.transform = transform        
        self.device = device
        
        

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
        