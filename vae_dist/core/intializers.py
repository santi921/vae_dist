import math
from torch import nn


def xavier_init(model):
    for name, param in model.named_parameters():
        if name.startswith("fc_mu") or name.startswith("fc_var"):
            pass
        elif name.endswith(".bias"):
            param.data.fill_(0)
        else:
            if len(param.shape) == 1:
                nn.init.uniform_(param, 0, 1)
            else:
                nn.init.xavier_normal_(param)


def kaiming_init(model):
    for name, param in model.named_parameters():
        # print(name)

        if name.startswith("fc_mu") or name.startswith("fc_var"):
            pass
        elif name.endswith(".bias"):
            param.data.fill_(0)
        else:
            if len(param.shape) == 1:
                nn.init.uniform_(param, 0, 1)
            else:
                nn.init.kaiming_normal_(
                    param, a=0, mode="fan_in", nonlinearity="leaky_relu"
                )


def equi_var_init(model):
    for name, param in model.named_parameters():
        # print(name)
        if name.startswith("fc_mu") or name.startswith("fc_var"):
            pass

        elif name.endswith(".bias"):
            # print("bias")
            param.data.fill_(0)

        else:
            # print("weight")
            if len(param.shape) == 1:
                nn.init.normal_(param, 0, 1)
            else:
                nn.init.normal_(param)
