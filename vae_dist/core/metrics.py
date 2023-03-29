
from torch import Tensor, tensor
from torch.nn import functional as F
from torchmetrics import Metric

class Metrics_crossentropy(Metric):
    def __init__(self):
        super().__init__()
        # to count the correct predictions
        is_differentiable: bool = True
        higher_is_better: bool = False
        full_state_update: bool = False
        sum_cross: Tensor
        total: Tensor
        self.add_state("sum_cross", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")
    def update(self, preds: Tensor, target: Tensor, reduction: str = "sum")-> None:
        #preds = preds if preds.is_floating_point else preds.float()
        #target = target if target.is_floating_point else target.float()
        sum_cross = F.cross_entropy(preds, target, reduction='sum')
        n_obs = target.numel()
        self.sum_cross += sum_cross
        self.total += n_obs
        
    def compute(self):
        return self.sum_cross / self.total