import os 
import numpy as np 
import torch as T
import torch.nn as nn # will handle the neural network layers
import torch.nn.functional as F 
import torch.optim as optim

# this model will use LINEAR layers 

"""
Notes for self: 
learning rate is simply the learning rate of the neural network, its a hyper paramater used when training the network 
input_dims is just number of dimensions for the input vector
fc1_dims number of NEURONS in first layer
fc2_dims number of NEURONS in second layer
n_actions is just the idfferent actions that agent can take in the enviornment when evaluating multiple states

"""

class DeepQNetwork(nn.Module):
    def __init__(self, learning_rate, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.learning_rate = learning_rate
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims) # creates first full layer so we map the input dimensions to the neurons in first layer
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims) # neurons from first layer are being mapped to neurons in second layer
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions) # neurons from second layer are being mapped to the different actions this is OUTPUT layer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate) # the parameters() is to get all learnable paramters of network so weights and biases etc...
        # we use the adams algorithm
        # our optimizer is responsible for updating the neural networks paramaters to minimize the loss during training
        self.loss = nn.MSELoss() # used to predict how well the predictive q-learning model matches the target q-values
        # our loss function simply quantifies how well the model is performing with how the predicted q values are compared to target q values
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu' ) # use the gpu is we have it connected otherwise use the cpu



