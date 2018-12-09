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
        self.l2 = nn.Linear(num_hidden, 64)
        self.l3 = nn.Linear(64, output_size)
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

class LanderNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        nn.Module.__init__(self)
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
#TODO: implement a network based on obeservation. Also use CNNs