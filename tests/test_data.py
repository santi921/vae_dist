from vae_dist.dataset.dataset import FieldDataset
from vae_dist.data.augmentation import Augment
from vae_dist.data.transformation import Transform
from sklearn.decomposition import PCA
import numpy as np


class TestTransfer:
    root = "./test_data/"

    def test_all_transforms(self):
        FieldDataset(
            self.root,
            transform=False,
            augmentation=False,
            standardize=True,
            lower_filter=True,
            log_scale=True,
            min_max_scale=True,
            wrangle_outliers=True,
            sparsify=10,
            offset=1,
        )

    def test_standardize(self):
        FieldDataset(
            self.root,
            transform=False,
            augmentation=False,
            standardize=True,
            lower_filter=False,
            log_scale=False,
            min_max_scale=False,
            wrangle_outliers=False,
            sparsify=-1,
            offset=1,
        )

    def test_log(self):
        FieldDataset(
            self.root,
            transform=False,
            augmentation=False,
            standardize=False,
            lower_filter=False,
            log_scale=True,
            min_max_scale=False,
            wrangle_outliers=False,
            sparsify=-1,
            offset=1,
        )

    def test_min_max(self):
        FieldDataset(
            self.root,
            transform=False,
            augmentation=False,
            standardize=False,
            lower_filter=False,
            log_scale=False,
            min_max_scale=True,
            wrangle_outliers=False,
            sparsify=-1,
            offset=1,
        )

    def test_outliers(self):
        FieldDataset(
            self.root,
            transform=False,
            augmentation=False,
            standardize=False,
            lower_filter=False,
            log_scale=False,
            min_max_scale=False,
            wrangle_outliers=True,
            sparsify=-1,
            offset=1,
        )

    def test_offset(self):
        offset = FieldDataset(
            self.root,
            transform=False,
            augmentation=False,
            standardize=False,
            lower_filter=False,
            log_scale=False,
            min_max_scale=False,
            wrangle_outliers=False,
            sparsify=-1,
            offset=1,
        )
        offset_numpy = offset.dataset_to_numpy()

    def test_sparsify(self):
        base = FieldDataset(
            self.root,
            transform=False,
            augmentation=False,
            standardize=True,
            lower_filter=False,
            log_scale=False,
            min_max_scale=False,
            wrangle_outliers=False,
            sparsify=-1,
            offset=1,
        )

        sparsify = FieldDataset(
            self.root,
            transform=False,
            augmentation=False,
            standardize=True,
            lower_filter=False,
            log_scale=False,
            min_max_scale=False,
            wrangle_outliers=False,
            sparsify=3,
            offset=1,
        )

        base_numpy = base.dataset_to_numpy()
        sparsify_numpy = sparsify.dataset_to_numpy()

        assert base_numpy.shape[0] == sparsify_numpy.shape[0]
        assert base_numpy.shape[1] == sparsify_numpy.shape[1]
        assert base_numpy.shape[2] == sparsify_numpy.shape[2] * 3
        assert base_numpy.shape[3] == sparsify_numpy.shape[3] * 3
        assert base_numpy.shape[4] == sparsify_numpy.shape[4] * 3
