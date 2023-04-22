from vae_dist.dataset.dataset import FieldDataset
from vae_dist.data.augmentation import Augment
from vae_dist.data.transformation import Transform
from sklearn.decomposition import PCA
import numpy as np


def test_load_data():
    root = "../data/cpet/"
    # dataset_1 = FieldDataset(root, transform=None, augmentation=None, offset=1)
    for standardize in [True, False]:
        for lower_filter in [True, False]:
            for log_scale in [True, False]:
                for min_max_scale in [True, False]:
                    for wrangle_outliers in [True, False]:
                        dataset_1 = FieldDataset(
                            root,
                            transform=False,
                            augmentation=False,
                            standardize=standardize,
                            lower_filter=lower_filter,
                            log_scale=log_scale,
                            min_max_scale=min_max_scale,
                            wrangle_outliers=wrangle_outliers,
                            offset=1,
                        )

                        assert dataset_1[0].shape == (3, 21, 21, 21)
                        # assert dataset_1[0].dtype == np.float32


def main():
    print("testing base data loading")
    test_load_data()


main()
