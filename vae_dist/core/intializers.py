import math 
from torch import nn 

def xavier_init(model):
    
    for name, param in model.named_parameters():
        if name.endswith(".bias"):
            param.data.fill_(0)
        else:
            if len(param.shape) == 1:
                nn.init.uniform_(param, 0, 1)
            else: 
                nn.init.xavier_uniform_(param)
        
            
def kaiming_init(model):
    
    for name, param in model.named_parameters():
        if name.endswith(".bias"):
            param.data.fill_(0)
        else:
            if len(param.shape) == 1:
                nn.init.uniform_(param, 0, 1)
            else: 
                nn.init.kaiming_normal_(param, a=0, mode='fan_in', nonlinearity='leaky_relu')

            


def equi_var_init(model):
    
    for name, param in model.named_parameters():
        if name.endswith(".bias"):
            param.data.fill_(0)
        else:
            if len(param.shape) == 1:
                nn.init.uniform_(param, 0, 1)
            else: 
                nn.init.equi_var_(param)
        