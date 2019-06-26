import torch
import torch.nn.functional as F
import numpy as np

class BinaryRBM():
    
    # RBM Initialization
    def __init__(self, num_v, num_h):
        """
        Args:
            num_v (int): the number of nodes in the visible layer
            num_h (int): the number of nodes in the hidden layer       
        """

        self.num_v = num_v
        self.num_h = num_h

        # normalization to ensure stability ? 
        self.w = torch.randn(num_h, num_v,dtype=torch.float64) / np.sqrt(num_v)
        self.c = torch.zeros(num_h, dtype=torch.float64).unsqueeze(1) # bias (column) vector for the hidden layer
        self.b = torch.zeros(num_v, dtype=torch.float64).unsqueeze(1) # bias (column) vector for the visible layer


    # Calculation of the free energy F(v)
    def free_energy_func (self,h,v,c,b,w):
        """
        Args:
        h (torch.Tensor): the hidden states
        v (torch.Tensor): the visible states
        c (torch.Tensor): the hidden bias
        b (torch.Tensor): the visible bias

        Returns:
        free_energy (torch.Tensor): the free energy F(v) (c.f. Eq (12))

        """
        # c.f. Eq.(12)
        return -torch.matmul(b.t(),v) - F.softplus(torch.addmm(b,w,v)).sum(0)
