import torch


def stepwise_inverse_huber_loss(
        x: torch.Tensor, 
        y: torch.Tensor, 
        delta1:float=1.0, 
        delta2:float=0.1):
    """
    Computes the stepwise huber loss for a batch of data.
    Takes
        x: torch.Tensor, the input data
        y: torch.Tensor, the predicted data
        delta1: float, the first threshold
        delta2: float, the second threshold
    Returns
        loss: torch.Tensor, the loss
    """
    diff = x - y
    abs_diff = torch.abs(diff)

    # reimplement huber loss
    less_than_delta1 = abs_diff < delta1
    greater_than_delta2 = abs_diff > delta2
    between = less_than_delta1 & greater_than_delta2

    loss = torch.zeros_like(x)
    loss[~less_than_delta1] = diff[~less_than_delta1] ** 2 - 0.5 * delta1**2
    loss[between] = delta2 * (abs_diff[between] - 0.5 * delta2)
    loss[~greater_than_delta2] = abs_diff[~greater_than_delta2] ** 0.5

    return loss.mean()


def inverse_huber(
        x: torch.Tensor, 
        y: torch.Tensor, 
        delta: float=1.0):
    """
    Computes the stepwise huber loss for a batch of data.
    Takes
        x: torch.Tensor, the input data
        y: torch.Tensor, the predicted data
        delta1: float, the first threshold
        delta2: float, the second threshold
    Returns
        loss: torch.Tensor, the loss
    """
    diff = x - y
    abs_diff = torch.abs(diff)

    # reimplement huber loss
    # get the indices of the elements that are less than delta1 and greater than delta2
    less_than_delta = abs_diff < delta
    loss = torch.zeros_like(x)
    loss[~less_than_delta] = delta * (diff[~less_than_delta] ** 2 - 0.5 * delta**2)
    loss[less_than_delta] = abs_diff[less_than_delta]

    return loss.mean()
