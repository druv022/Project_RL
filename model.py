import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from environment import get_env

#Naive implementation
class MountainNetwork(nn.Module):
    
    def __init__(self, input_size, output_size, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(input_size, num_hidden)
        self.l2 = nn.Linear(num_hidden, 24)
        self.l3 = nn.Linear(24, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        
        # YOUR CODE HERE
        out = self.relu(self.l1(x))
        out = self.relu(self.l2(out))
        out = self.l3(out)
        
        return out

class CartNetwork(nn.Module):

    def __init__(self, input_size, output_size, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(input_size, num_hidden)
        self.l2 = nn.Linear(num_hidden, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        
        # YOUR CODE HERE
        out = self.relu(self.l1(x))
        out = self.l2(out)
        
        return out


#TODO: implement a network based on obeservation. Also use CNNs




if __name__=="__main__":
    # Let's instantiate and test if it works
    num_hidden = 128
    torch.manual_seed(1234)
    env, _size = get_env("Acrobot-v1")
    input_size, output_size = _size
    # Sample a transition
    s = env.reset()
    a = env.action_space.sample()
    s_next, r, done, _ = env.step(a)

    model = QNetwork(input_size, output_size, num_hidden)

    torch.manual_seed(1234)
    test_model = nn.Sequential(
        nn.Linear(input_size, num_hidden), 
        nn.ReLU(), 
        nn.Linear(num_hidden, output_size)
    )

    x = torch.rand(10, input_size)

    # If you do not need backpropagation, wrap the computation in the torch.no_grad() context
    # This saves time and memory, and PyTorch complaints when converting to numpy
    with torch.no_grad():
        assert np.allclose(model(x).numpy(), test_model(x).numpy())