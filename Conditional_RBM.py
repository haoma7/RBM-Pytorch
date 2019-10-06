import torch
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import pickle
import time

class BinaryRBM():
    
    # RBM Initialization
    def __init__(self, num_v, num_h, num_l, device="CPU"):

        """
        Args:
            num_v (int): the number of nodes in the visible layer
            num_h (int): the number of nodes in the hidden layer 
            device (str): CPU or GPU mode
        """

        self.num_v = num_v # the number of visible nodes
        self.num_h = num_h # the number of hidden nodes
        self.num_l = num_l # the number of label nodes (which are part of the visible layer)

        if device == "GPU" and not torch.cuda.is_available():
            raise ValueError("GPU is not supported")
        elif device == "GPU" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # normalization to ensure stability ? 
        self.w_v = torch.randn(num_h, num_v, device=self.device, dtype=torch.float32) / np.sqrt(num_v)
        self.w_l = torch.randn(num_h, num_l, device=self.device, dtype=torch.float32) / np.sqrt(num_l)

        self.b = torch.zeros(num_v, device=self.device, dtype=torch.float32).unsqueeze(1) # bias (column) vector for the visible layer
        self.c = torch.zeros(num_h, device=self.device, dtype=torch.float32).unsqueeze(1) # bias (column) vector for the hidden layer

    def idx2onehot(self, idx, n):

        assert torch.max(idx).item() < n and idx.dim() == 1
        idx2dim = idx.view(-1,1) # change from 1-dim tensor to 2-dim tensor
        onehot = torch.zeros(idx2dim.size(0),n).scatter_(1,idx2dim,1)

        return onehot



    # Calculation of the free energy F(v)
    def free_energy_func(self, v, l):
        """
        Args:
        v (torch.Tensor): the visible states

        Returns:
        free_energy (torch.Tensor): the free energy F(v) (c.f. Eq (12))

        """
        # c.f. Eq.(12)
        # .sum(0) represents the summation of different rows
        # F.softplus(x) calculates log(1+exp(x))
        
        return -torch.matmul(self.b.t(),v.to(device=self.device)) - F.softplus(torch.addmm(self.c,torch.cat((self.w_v,self.w_l),1),torch.cat((v.to(device=self.device),l.to(device=self.device)),0))).sum(0)

    def sample_h_given_v(self, v, label):
        """
          Args:
        v (torch.Tensor): the visible states

        
        Returns:
        sampled_h (torch.Tensor): the sample h according to Eq.(17). 
                                  It is a column vector that contains 0 or 1. 

        """
        
        return (torch.addmm(self.c, torch.cat((self.w_v,self.w_l),1),torch.cat((v,label),0)).sigmoid_()>torch.rand(self.num_h,1,device=self.device)).float() # c.f. Eq (17)

    def sample_v_given_h(self, h):
        """
        Args:
        h (torch.Tensor): the hidden states
        
        Returns:
        sampled_v (torch.Tensor): the sample v according to Eq.(18). 
                                  It is a column vector that contains 0 or 1. 

        """
        return ( torch.addmm(self.b, self.w_v.t(), h).sigmoid_()>torch.rand(self.num_v, 1,device=self.device) ).float() # c.f. Eq (18)


    def block_gibbs_sampling(self, initial_v, label, num_iter):
        """
        Args:
        initial_v (torch.Tensor): the initial visible states to start the block gibbs sampling
        num_iter(int): the number of iterations for the gibbs sampling
    
        Returns:
        gibbs_v (torch.Tensor): the sampled visible states
        """
        v = initial_v.to(device=self.device)
        label = label.to(device=self.device)
        for _ in range(num_iter):
            
            h = self.sample_h_given_v(v,label)
            v = self.sample_v_given_h(h)

        return v

    def free_energy_gradient(self, v, label = None):
        """
        Args:
        v (torch.Tensor): the visible states

        Returns:
        grad_w (torch.Tensor): the average gradient of the free energy with respect to w across all samples
        grad_b (torch.Tensor): the average gradient of the free energy with respect to b across all samples
        grad_c (torch.Tensor): the average gradient of the free energy with respect to c across all samples

        """
        temp = -torch.addmm(self.c, torch.cat((self.w_v,self.w_l),1),torch.cat((v,label),0)).sigmoid_()

        grad_c =  temp.mean(dim=1).unsqueeze(1)
        grad_b = - v.mean(dim=1).unsqueeze(1)
        grad_w_v = torch.matmul(temp,v.t())/v.size(1)
        grad_w_l = torch.matmul(temp,label.t())/label.size(1)

        # v.size(1)  == label.size(1) == batch_size
        
        return grad_w_v, grad_w_l, grad_b, grad_c

    def mini_batch_gradient_func (self, v, cd_k,labels):
        
        """
        Args:
        v (torch.Tensor): the visible states
        cd_k (int): cd_k mode that is chosen

        Returns:
        grad_mini_batch (Torch) the average gradient across all samples in the mini-batch
        """
        v = v.to(device=self.device) # move to GPU if necessary 
        
        v_k = self.block_gibbs_sampling(initial_v = v, label = labels, num_iter = cd_k)
        
        [grad_w_v_pos,grad_w_l_pos, grad_b_pos, grad_c_pos] = self.free_energy_gradient(v,labels)
        [grad_w_v_neg,grad_w_l_neg, grad_b_neg, grad_c_neg] = self.free_energy_gradient(v_k,labels)

        return grad_w_v_pos - grad_w_v_neg, grad_w_l_pos - grad_w_l_neg, grad_b_pos - grad_b_neg, grad_c_pos - grad_c_neg # c.f. Eq.(13)

    def train(self, dataloader, cd_k, max_epochs = 5, lr = 0.01):
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

                labels = self.idx2onehot(mini_batch_samples[1],10).t()

              
                # we use mini_batch_samples[0] to extract the data since mini_batch_samples[1] is the label. 
                # each column in  mini_batch_samples_ is corresponding to one data sample. 

                grad_w_v, grad_w_l, grad_b, grad_c = self.mini_batch_gradient_func(mini_batch_samples_, cd_k, labels)
                
                
                # update w, b, c
                self.w_v -= lr * grad_w_v
                self.w_l -= lr * grad_w_l
                self.b -= lr * grad_b
                self.c -= lr * grad_c
            # break
        
        return self.w_v, self.w_l, self.b, self.c




if __name__ == "__main__":

    model = BinaryRBM(784,128,10,device="CPU")

    # Load Data
    train_dataset = torchvision.datasets.MNIST("~", train=True, transform=transforms.ToTensor(), download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32,shuffle = True)
    
    # Train
    start_time = time.time()
    w, b, c = model.train(train_loader, cd_k = 5)
    elapsed_time = time.time() - start_time

    with open('objs.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([w.cpu().numpy(), b.cpu().numpy(),c.cpu().numpy()], f)
    
    print("Training Ended. Elapsed Time is {0:.2f}s".format(elapsed_time))



    # Test 

    # retrieve the next image data for testing 
    i = iter(train_loader).next()[0][0]

    # Generate new visible states with a random initial visible states via Gibbs sampling the RBM
    v_gen = model.block_gibbs_sampling(initial_v = i.view(-1,1).round(),num_iter = 1000)

    plt.figure(1)
    plt.subplot(121)
    plt.imshow(i.squeeze(0).round(),cmap = "gray")

    # Display the images
    plt.subplot(122)
    plt.imshow(v_gen.view(28,28).numpy(),cmap = "gray")
    plt.show()

    

