import math
from torch import nn
from escnn.nn.init import generalized_he_init, deltaorthonormal_init
from escnn.nn import R3Conv
from vae_dist.core.escnnlayers import R3Upsampling


def xavier_init(model):
    for name, param in model.named_parameters():
        # print(name)
        if (
            name.startswith("fc_mu")
            or name.startswith("fc_var")
            or name.endswith("eps")
        ):
            pass

        elif isinstance(param, R3Conv) or isinstance(param, R3Upsampling):
            generalized_he_init(param, param.basisexpansion)

        elif isinstance(param, nn.BatchNorm3d):
            param.weight.data.fill_(1)
            param.bias.data.zero_()

        elif name.endswith(".bias"):
            param.data.fill_(0)

        else:
            # print(param.shape)
            if len(param.shape) == 1:
                nn.init.uniform_(param, 0, 1)
            else:
                nn.init.xavier_normal_(param)  # this is the only part using the xavier


def kaiming_init(model):
    for name, param in model.named_parameters():
        # print(name)
        if (
            name.startswith("fc_mu")
            or name.startswith("fc_var")
            or name.endswith("eps")
        ):
            pass

        elif isinstance(param, R3Conv) or isinstance(param, R3Upsampling):
            generalized_he_init(param, param.basisexpansion)

        elif isinstance(param, nn.BatchNorm3d):
            param.weight.data.fill_(1)
            param.bias.data.zero_()

        elif name.endswith(".bias"):
            param.data.fill_(0)

        else:
            if len(param.shape) == 1:
                nn.init.uniform_(param, 0, 1)
            else:
                nn.init.kaiming_normal_(
                    param, a=0, mode="fan_in", nonlinearity="leaky_relu"
                )  # this is the only part using the kaiming


def equi_var_init(model):
    for name, param in model.named_parameters():
        # print(name)
        if (
            name.startswith("fc_mu")
            or name.startswith("fc_var")
            or name.endswith("eps")
        ):
            pass

        elif isinstance(param, R3Conv) or isinstance(param, R3Upsampling):
            generalized_he_init(param, param.basisexpansion)

        elif isinstance(param, nn.BatchNorm3d):
            param.weight.data.fill_(1)
            param.bias.data.zero_()

        elif name.endswith(".bias"):
            param.data.fill_(0)

        else:
            if len(param.shape) == 1:
                nn.init.normal_(param, 0, 1)
            else:
                nn.init.normal_(param)
