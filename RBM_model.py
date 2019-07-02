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
    def free_energy_func (self, h, v, c, b, w):
        """
        Args:
        h (torch.Tensor): the hidden states
        v (torch.Tensor): the visible states
        c (torch.Tensor): the hidden bias
        b (torch.Tensor): the visible bias
        w (torch.Tensor): the weight matrix

        Returns:
        free_energy (torch.Tensor): the free energy F(v) (c.f. Eq (12))

        """
        # c.f. Eq.(12)
        # .sum(0) represents the summation of different rows
        # F.softplus(x) calculates log(1+exp(x))
        return -torch.matmul(b.t(),v) - F.softplus(torch.addmm(c,w,v)).sum(0)

    def  sample_h_given_v(self, v, c, w):
        """
          Args:
        h (torch.Tensor): the hidden states
        v (torch.Tensor): the visible states
        c (torch.Tensor): the hidden bias
        w (torch.Tensor): the weight matrix
        
        Returns:
        sampled_h (torch.Tensor): the sample h according to Eq.(17). 
                                  It is a column vector that contains 0 or 1. 

        """
        return (torch.addmm(c,w,v).sigmoid_()>torch.rand(self.num_h,1)).float() # c.f. Eq (17)

    def  sample_v_given_h(self, h, b, w):
        """
        Args:
        h (torch.Tensor): the hidden states
        v (torch.Tensor): the visible states
        b (torch.Tensor): the visible bias
        w (torch.Tensor): the weight matrix
        
        Returns:
        sampled_v (torch.Tensor): the sample v according to Eq.(18). 
                                  It is a column vector that contains 0 or 1. 

        """
        return ( torch.addmm(b,w.t(),h.sigmoid_()>torch.rand(self.num_v, 1) ).float() # c.f. Eq (18)


    def block_gibbs_sampling(self, initial_v, num_iter, h, v, c, b, w):
        """
        Args:
        initial_v (torch.Tensor): the initial visible states to start the block gibbs sampling
        num_iter(int): the number of iterations for the gibbs sampling
        h (torch.Tensor): the hidden states
        v (torch.Tensor): the visible states
        c (torch.Tensor): the hidden bias
        b (torch.Tensor): the visible bias
        w (torch.Tensor): the weight matrix

        Returns:
        gibbs_v (torch.Tensor): the sampled visible states

        """
        v = initial_v

        for _ in range(num_iter):
            h = sample_h_given_v(v,c,w)
            v = sample_v_given_h(h,b,w)

        return v

    def free_energy_gradient(self, v, c, w):
        """
        Args:
        v (torch.Tensor): the visible states
        c (torch.Tensor): the hidden bias
        w (torch.Tensor): the weight matrix

        Returns:
        grad_w (torch.Tensor): the gradient of the free energy with respect to w
        grad_b (torch.Tensor): the gradient of the free energy with respect to b
        grad_c (torch.Tensor): the gradient of the free energy with respect to c

        """

        grad_c =  torch.addmm(c.t(),w,v).sigmoid_().sum(dim=1).
        grad_b = -v
        grad_w = torch.malmul(grad_c,v.t())
        
        return grad_w, grad_b, grad_c

    def mini_batch_gradient_func (self, cd_k, h, v, c, b, w):
        
         """
        Args:
        cd_k (int): cd_k mode that is chosen
        h (torch.Tensor): the hidden states
        v (torch.Tensor): the visible states
        c (torch.Tensor): the hidden bias
        b (torch.Tensor): the visible bias
        w (torch.Tensor): the weight matrix

        Returns:
        grad_mini_batch (Torch) average gradient across the mini-batch
        """

        v_k = self.block_gibbs_sampling(initial_v = v, num_iter = cd_k, h, v, c, b, w)
        
        [grad_w_posi, grad_b_posi, grad_c_posi] = free_energy_gradient(v,c,w)

        [grad_w_neg, grad_b_neg, grad_c_neg] = free_energy_gradient(v_k,c,w))

        return .sum()/
        
        v.size(1)


    def train(self, data, max_epochs = 1000, learning_rate = 0.1):

    def generate v

