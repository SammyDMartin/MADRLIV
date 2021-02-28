import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



class PolicyGrad(nn.Module):
    def __init__(self,state_space,action_space,hidden_layer_size,gamma):
        super(PolicyGrad, self).__init__()
        self.state_space = state_space
        self.action_space = action_space
       
        self.l1 = nn.Linear(self.state_space, hidden_layer_size, bias=False)
        self.l2 = nn.Linear(hidden_layer_size, self.action_space, bias=False)

        
        self.gamma = gamma
        
        # Episode policy and reward history 
        self.policy_history = Variable(torch.Tensor()) 
        self.reward_episode = []
        # Overall reward and loss history
        #self.reward_history = []
        #self.loss_history = []

    def forward(self, x):    
        model = torch.nn.Sequential(
            self.l1,
            nn.Dropout(p=0.6),
            nn.ReLU(),
            self.l2,
            nn.Softmax(dim=-1)
        )
        return model(x)


#https://github.com/Azulgrana1/pytorch-a3c/blob/master/model.py