from vae_dist.dataset.dataset import FieldDataset
from vae_dist.data.augmentation import Augment
from vae_dist.data.transformation import Transform
from sklearn.decomposition import PCA
import numpy as np 

def test_load_data():
    root = "../data/cpet/"
    dataset_1 = FieldDataset(root, transform=None, augmentation=None)
    print(dataset_1[0].shape)


def test_augment_data():
    root = "../data/cpet/"
    aug = Augment(xy=True, z = True)
    dataset_1 = FieldDataset(root, transform=None, augmentation=aug)
    print(dataset_1[0].shape)


def test_transformations():
    root = "../data/cpet/"
    dataset_vanilla = FieldDataset(root, transform=None, augmentation=None)

    # internally defined pca
    transform_1 = Transform(pca = True)
    dataset_1 = FieldDataset(root, transform=transform_1, augmentation=None)

    # self defined pca
    pca_obj = PCA(n_components=3)
    all_indices = np.arange(0, len(dataset_vanilla))
    full_data = dataset_vanilla[all_indices]
    full_data_flat = full_data.reshape(full_data.shape[0], full_data.shape[1] * full_data.shape[2] * full_data.shape[3] * full_data.shape[4])
    pca_obj.fit(full_data_flat)
    transform_2 = Transform(pca = pca_obj)
    dataset_2 = FieldDataset(root, transform=transform_2, augmentation=None)

    # helmholtz
    transform_3 = Transform(helm = True)
    dataset_3 = FieldDataset(root, transform=transform_3, augmentation=None)

    print(dataset_2[0].shape)
    print(dataset_1[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]].shape)
    print(dataset_3[0].shape)
    print(dataset_3[[0,1,2]].shape)

def test_transformations_and_augmentations():

    root = "../data/cpet/"
    dataset_vanilla = FieldDataset(root, transform=None, augmentation=None)


    # self defined pca
    pca_obj = PCA(n_components=3)
    all_indices = np.arange(0, len(dataset_vanilla))
    full_data = dataset_vanilla[all_indices]
    full_data_flat = full_data.reshape(full_data.shape[0], full_data.shape[1] * full_data.shape[2] * full_data.shape[3] * full_data.shape[4])
    pca_obj.fit(full_data_flat)
    transform_1 = Transform(pca = pca_obj)

    dataset_1 = FieldDataset(root, transform=transform_1, augmentation=None)

    # helmholtz
    transform_2 = Transform(helm = True)
    dataset_2 = FieldDataset(root, transform=transform_2, augmentation=None)

    # call dataset with augmentation and transform
    print(dataset_1[0].shape)
    print(dataset_2[0].shape)
    print(dataset_1[[0,1]].shape)
    print(dataset_2[[0,1]].shape)
    

def scale_data(data):
    #TODO 
    pass

def main():
    print("testing base data loading")
    test_load_data()
    print("testing augmentation")
    test_augment_data()
    print("testing transformations")
    test_transformations()
    print("testing transformations and augmentations")
    test_transformations_and_augmentations()
    

main()