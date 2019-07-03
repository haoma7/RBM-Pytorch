import torch
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import pickle

class BinaryRBM():
    
    # RBM Initialization
    def __init__(self, num_v, num_h):

        """
        Args:
            num_v (int): the number of nodes in the visible layer
            num_h (int): the number of nodes in the hidden layer       
        """

        self.num_v = num_v # the number of visible nodes
        self.num_h = num_h # the number of hidden nodes

        # normalization to ensure stability ? 
        self.w = torch.randn(num_h, num_v, dtype=torch.float32) / np.sqrt(num_v)
        self.b = torch.zeros(num_v, dtype=torch.float32).unsqueeze(1) # bias (column) vector for the visible layer
        self.c = torch.zeros(num_h, dtype=torch.float32).unsqueeze(1) # bias (column) vector for the hidden layer

    

    # Calculation of the free energy F(v)
    def free_energy_func (self, v):
        """
        Args:
        v (torch.Tensor): the visible states

        Returns:
        free_energy (torch.Tensor): the free energy F(v) (c.f. Eq (12))

        """
        # c.f. Eq.(12)
        # .sum(0) represents the summation of different rows
        # F.softplus(x) calculates log(1+exp(x))
  
        return -torch.matmul(self.b.t(),v) - F.softplus(torch.addmm(self.c,self.w,v)).sum(0)

    def sample_h_given_v(self, v):
        """
          Args:
        v (torch.Tensor): the visible states

        
        Returns:
        sampled_h (torch.Tensor): the sample h according to Eq.(17). 
                                  It is a column vector that contains 0 or 1. 

        """
     
        return (torch.addmm(self.c,self.w,v).sigmoid_()>torch.rand(self.num_h,1)).float() # c.f. Eq (17)

    def sample_v_given_h(self, h):
        """
        Args:
        h (torch.Tensor): the hidden states
        
        Returns:
        sampled_v (torch.Tensor): the sample v according to Eq.(18). 
                                  It is a column vector that contains 0 or 1. 

        """
        return ( torch.addmm(self.b, self.w.t(), h).sigmoid_()>torch.rand(self.num_v, 1) ).float() # c.f. Eq (18)


    def block_gibbs_sampling(self, initial_v, num_iter):
        """
        Args:
        initial_v (torch.Tensor): the initial visible states to start the block gibbs sampling
        num_iter(int): the number of iterations for the gibbs sampling
    
        Returns:
        gibbs_v (torch.Tensor): the sampled visible states
        """
        v = initial_v

        for _ in range(num_iter):
            
            h = self.sample_h_given_v(v)
            v = self.sample_v_given_h(h)

        return v

    def free_energy_gradient(self, v):
        """
        Args:
        v (torch.Tensor): the visible states

        Returns:
        grad_w (torch.Tensor): the average gradient of the free energy with respect to w across all samples
        grad_b (torch.Tensor): the average gradient of the free energy with respect to b across all samples
        grad_c (torch.Tensor): the average gradient of the free energy with respect to c across all samples

        """
        temp = -torch.addmm(self.c,self.w,v).sigmoid_()

        grad_c =  temp.mean(dim=1).unsqueeze(1)
        grad_b = - v.mean(dim=1).unsqueeze(1)
        grad_w = torch.matmul(temp,v.t())/v.size(1)

        
        return grad_w, grad_b, grad_c

    def mini_batch_gradient_func (self, v, cd_k):
        
        """
        Args:
        v (torch.Tensor): the visible states
        cd_k (int): cd_k mode that is chosen

        Returns:
        grad_mini_batch (Torch) the average gradient across all samples in the mini-batch
        """
        v_k = self.block_gibbs_sampling(initial_v = v, num_iter = cd_k)
        
        [grad_w_pos, grad_b_pos, grad_c_pos] = self.free_energy_gradient(v)
        [grad_w_neg, grad_b_neg, grad_c_neg] = self.free_energy_gradient(v_k)

        return grad_w_pos - grad_w_neg, grad_b_pos - grad_b_neg, grad_c_pos - grad_c_neg # c.f. Eq.(13)
        

    def train(self, dataloader, cd_k, max_epochs = 10, lr = 0.01):
        """
        Args:
        dataloader: dataloader of the training data 
        cd_k: the contrastive divergence mode
        max_epochs: number of epochs
        lr: the learning rate

        Returns:
        w, b, c: the parameters of the RBM
        """
        for iter in range(max_epochs):

            print('Epoch {}'.format(iter))

            for mini_batch_samples in dataloader:
                mini_batch_samples_ = torch.flatten(mini_batch_samples[0].squeeze(1),start_dim=1).t().round()
                grad_w, grad_b, grad_c = self.mini_batch_gradient_func(mini_batch_samples_, cd_k)
                
                
                # update w, b, c
                self.w -= lr * grad_w
                self.b -= lr * grad_b
                self.c -= lr * grad_c
            # break
            print(torch.norm(self.w))
        
        return self.w, self.b, self.c

def gen_mnist_image(X,num_of_img = 10):

    # Display images as a group
    return np.rollaxis(np.rollaxis(X[0:num_of_img].reshape(num_of_img, -1, 28, 28), 0, 2), 1, 3).reshape(-1, num_of_img * 28)

if __name__ == "__main__":

    model = BinaryRBM(784,128)

    # Load Data
    train_dataset = torchvision.datasets.MNIST("~", train=True, transform=transforms.ToTensor(), download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32,shuffle = True)
    
    # Train
    w, b, c = model.train(train_loader, cd_k = 5)

    with open('objs.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([w.numpy(), b.numpy(),c.numpy()], f)


    # Test 
    # input data for testing 
    i = iter(train_loader).next()

    # Generate new visible states with a random initial visible states via Gibbs sampling the RBM
    v_gen = model.block_gibbs_sampling(initial_v = i[0][0].view(-1,1).round(),num_iter = 1000)

    plt.figure(1)
    plt.subplot(121)
    plt.imshow(i[0][0].squeeze(0).round(),cmap = "gray")

    # Display the images
    plt.subplot(122)
    plt.imshow(v_gen.view(28,28).numpy(),cmap = "gray")
    plt.show()

    

