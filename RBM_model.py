import torch
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import transforms

import matplotlib.pyplot as plt
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
        self.w = torch.randn(num_h, num_v,dtype=torch.float32) / np.sqrt(num_v)
        self.c = torch.zeros(num_h, dtype=torch.float32).unsqueeze(1) # bias (column) vector for the hidden layer
        self.b = torch.zeros(num_v, dtype=torch.float32).unsqueeze(1) # bias (column) vector for the visible layer


    # Calculation of the free energy F(v)
    def free_energy_func (self, v, c, b, w):
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

    def sample_h_given_v(self, v, c, w):
        """
          Args:
        v (torch.Tensor): the visible states
        c (torch.Tensor): the hidden bias
        w (torch.Tensor): the weight matrix
        
        Returns:
        sampled_h (torch.Tensor): the sample h according to Eq.(17). 
                                  It is a column vector that contains 0 or 1. 

        """
     
        return (torch.addmm(c,w,v).sigmoid_()>torch.rand(self.num_h,1)).float() # c.f. Eq (17)

    def sample_v_given_h(self, h, b, w):
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
        return ( torch.addmm(b,w.t(),h).sigmoid_()>torch.rand(self.num_v, 1) ).float() # c.f. Eq (18)


    def block_gibbs_sampling(self, initial_v, num_iter):
        """
        Args:
        initial_v (torch.Tensor): the initial visible states to start the block gibbs sampling
        num_iter(int): the number of iterations for the gibbs sampling
        v (torch.Tensor): the visible states
        c (torch.Tensor): the hidden bias
        b (torch.Tensor): the visible bias
        w (torch.Tensor): the weight matrix

        Returns:
        gibbs_v (torch.Tensor): the sampled visible states

        """
        v = initial_v

        for _ in range(num_iter):
            h = self.sample_h_given_v(v,self.c,self.w)
            v = self.sample_v_given_h(h,self.b,self.w)

        return v

    def free_energy_gradient(self, v, c, w):
        """
        Args:
        v (torch.Tensor): the visible states
        c (torch.Tensor): the hidden bias
        w (torch.Tensor): the weight matrix

        Returns:
        grad_w (torch.Tensor): the average gradient of the free energy with respect to w across all samples
        grad_b (torch.Tensor): the average gradient of the free energy with respect to b across all samples
        grad_c (torch.Tensor): the average gradient of the free energy with respect to c across all samples

        """
        temp = torch.addmm(c,w,v).sigmoid_()

        grad_c =  temp.mean(dim=1).unsqueeze(1)
        grad_b = -v.mean(dim=1).unsqueeze(1)
        grad_w = torch.matmul(temp,v.t())/v.size(1)

        
        return grad_w, grad_b, grad_c

    def mini_batch_gradient_func (self, cd_k, v, c, b, w):
        
        """
        Args:
        cd_k (int): cd_k mode that is chosen
        h (torch.Tensor): the hidden states
        v (torch.Tensor): the visible states
        c (torch.Tensor): the hidden bias
        b (torch.Tensor): the visible bias
        w (torch.Tensor): the weight matrix

        Returns:
        grad_mini_batch (Torch) the average gradient across all samples in the mini-batch
        """
        v_k = self.block_gibbs_sampling(initial_v = v, num_iter = cd_k)
        
        [grad_w_pos, grad_b_pos, grad_c_pos] = self.free_energy_gradient(v,c,w)
        [grad_w_neg, grad_b_neg, grad_c_neg] = self.free_energy_gradient(v_k,c,w)

        return grad_w_pos-grad_w_neg, grad_b_pos-grad_b_neg, grad_c_pos-grad_c_neg # c.f. Eq.(13)
        

    def train(self, dataloader, cd_k, max_epochs = 5, lr = 1):
        """
        Args:
        dataloader: each data samle is a row vector in the matrix data. 
        """
        for iter in range(max_epochs):
            print('Epoch {}'.format(iter))

            for mini_batch_samples in dataloader:
                mini_batch_samples_=mini_batch_samples[0].squeeze(1).view(-1,mini_batch_samples[0].size(0))
                grad_w, grad_b, grad_c = self.mini_batch_gradient_func(cd_k,mini_batch_samples_ , self.c, self.b, self.w)
                
                # update w, b, c
                
                self.w -= lr * grad_w
                self.b -= lr * grad_b
                self.c -= lr * grad_c
            
            print(torch.norm(self.w))
        return self.w, self.b, self.c

def gen_mnist_image(X,num_of_img = 10):

    # Display images as a group
    return np.rollaxis(np.rollaxis(X[0:num_of_img].reshape(num_of_img, -1, 28, 28), 0, 2), 1, 3).reshape(-1, num_of_img * 28)

if __name__ == "__main__":

    model = BinaryRBM(784,100)

    # Load Data
    train_dataset = torchvision.datasets.MNIST("~", train=True, transform=transforms.ToTensor(), download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 64,shuffle = True)
    
    model.train(train_loader,cd_k=1)
    
    # Test 
    i = iter(train_loader).next()

    # Generate new visible states with a random initial visible states via Gibbs sampling the RBM
    v_gen = model.block_gibbs_sampling(initial_v = i[0].view(-1,i[0].size(0)),num_iter = 40)
    
    # Display the images
    plt.imshow(gen_mnist_image(v_gen.t().view(64,1,28,28).numpy().round(),64),cmap = "gray")
    plt.show()

    

