from escnn import gspaces                                         
from escnn.nn import FieldType, R3Conv, ReLU                                        
import torch                                                      
from vae_dist.dataset.dataset import FieldDataset

def main():                                                                  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #root = "../../data/cpet/"
    #dataset_vanilla = FieldDataset(
    #    root, 
    #    transform=None, 
    #    augmentation=None, 
    #    device=device
    #    )

    r3_act = gspaces.rot3dOnR3(maximum_frequency=3)           

    feat_type_in  = FieldType(r3_act,  3*[r3_act.trivial_repr]) 

    feat_type_out = FieldType(r3_act,  3*[r3_act.trivial_repr])     
                                                                    
    conv = R3Conv(feat_type_in, feat_type_out, kernel_size=5)       
    relu = ReLU(feat_type_out)                                      
                                            
    #x = torch.randn(16, 3, 32, 32)    
    x = torch.randn(1, 3, 21, 21, 21)                                
    x = feat_type_in(x)                                                
    y = relu(conv(x))        
    print(y.shape)                     
main()                             