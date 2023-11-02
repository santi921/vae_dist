import numpy as np
from torch import Tensor, tensor
from torch.nn import functional as F
from torchmetrics import Metric
import torch


class Metrics_crossentropy(Metric):
    def __init__(self):
        super().__init__()
        # to count the correct predictions
        is_differentiable: bool = True
        higher_is_better: bool = False
        full_state_update: bool = False
        sum_cross: Tensor
        total: Tensor
        self.add_state("sum_kl", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor, reduction: str = "sum") -> None:
        # preds = preds if preds.is_floating_point else preds.float()
        # target = target if target.is_floating_point else target.float()
        sum_cross = F.cross_entropy(preds, target, reduction="sum")
        n_obs = target.numel()
        self.sum_cross += sum_cross
        self.total += n_obs

    def compute(self):
        return self.sum_cross / self.total


def test_equivariance(model, im_size=21):
    """
    Given a model and im_size generate a random input and test equivariance
    """
    x = torch.randn(1, 3, im_size, im_size, im_size)
    x = x.to(model.device)
    model.eval()
    x_90 = torch.rot90(x, 1, [2, 3])
    x_180 = torch.rot90(x, 2, [2, 3])
    x_270 = torch.rot90(x, 3, [2, 3])
    x_flip = torch.flip(x, [2, 3])
    x_z90 = torch.rot90(x, 1, [2, 4])
    x_z180 = torch.rot90(x, 2, [2, 4])
    x_z270 = torch.rot90(x, 3, [2, 4])
    x_zflip = torch.flip(x, [2, 4])
    x_y90 = torch.rot90(x, 1, [3, 4])
    x_y180 = torch.rot90(x, 2, [3, 4])
    x_y270 = torch.rot90(x, 3, [3, 4])
    x_yflip = torch.flip(x, [3, 4])

    test_input_list = []
    test_input_list.append(x)
    test_input_list.append(x_90)
    test_input_list.append(x_180)
    test_input_list.append(x_270)
    test_input_list.append(x_z90)
    test_input_list.append(x_z180)
    test_input_list.append(x_z270)
    test_input_list.append(x_y90)
    test_input_list.append(x_y180)
    test_input_list.append(x_y270)

    output_list = []
    for test_input in test_input_list:
        output_list.append(model(test_input))

    equi_dict = {"equivariant": 1.0, "equi_max_diff": 0.0}
    frac_equivariant = 0.0

    for i in range(len(output_list)):
        for j in range(i + 1, len(output_list)):
            tf = torch.allclose(output_list[i], output_list[j], atol=1e-3, rtol=1e-3)
            max_diff_temp = torch.max(torch.abs(output_list[i] - output_list[j]))
            if max_diff_temp > equi_dict["equi_max_diff"]:
                equi_dict["equi_max_diff"] = max_diff_temp

            if tf:
                frac_equivariant += 1
            else:
                equi_dict["equivariant"] = 0.0

    frac_equivariant = frac_equivariant / (
        len(output_list) * (len(output_list) - 1) / 2
    )
    equi_dict["frac_equivariant"] = frac_equivariant
    return equi_dict


def test_activity(supervised_loader, model):
    """
    get the average distance between the latent representations of the same class vs different classes
    """
    print("testing activity")
    # load all of the data from the loader
    # x, labels = next(iter(supervised_loader))
    x = supervised_loader.data
    labels = supervised_loader.labels

    x = x.to(model.device)
    z = model.latent(x)
    z = z.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    n_samples = z.shape[0]
    n_classes = len(np.unique(labels))
    dict_distances = {}
    for i in range(n_classes):
        for j in range(n_classes):
            dict_distances[(i, j)] = []

    for i in range(n_samples):
        for j in range(n_samples):
            if i != j:
                dict_distances[(labels[i], labels[j])].append(
                    np.linalg.norm(z[i] - z[j])
                )
    # compute the average distance between the latent representations of the same class vs different classes
    dict_distances["mean_same"] = np.mean([dict_distances[i] for i in range(n_classes)])
    dict_distances["mean_diff"] = np.mean(
        [
            dict_distances[(i, j)]
            for i in range(n_classes)
            for j in range(n_classes)
            if i != j
        ]
    )

    return dict_distances
