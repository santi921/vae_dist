from vae_dist.dataset.dataset import FieldDataset
from vae_dist.data.transformation import Transform
from sklearn.decomposition import PCA
import numpy as np 
import matplotlib.pyplot as plt 

def get_cum_ratio(pca_obj):
    cumulative_ratio = []
    for ind, i in enumerate(list(pca_obj.explained_variance_ratio_)): 
        if ind == 0: 
            cumulative_ratio.append(i)
        else: 
            cumulative_ratio.append(i + cumulative_ratio[-1])

    return cumulative_ratio

def main():
    root = "../../data/cpet/"
    dataset_vanilla = FieldDataset(root, transform=None, augmentation=None)

    # internally defined pca
    transform_1 = Transform(pca = True)

    # self defined pca
    pca_obj = PCA(n_components=10)
    all_indices = np.arange(0, len(dataset_vanilla))
    full_data = dataset_vanilla[all_indices]
    full_data_flat = full_data.reshape(full_data.shape[0], full_data.shape[1] * full_data.shape[2] * full_data.shape[3] * full_data.shape[4])
    transformed = pca_obj.fit_transform(full_data_flat)

    print(get_cum_ratio(pca_obj))
    plt.scatter(x=transformed[:,0], y=transformed[:,1])
    plt.show()

main()