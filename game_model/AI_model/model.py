import torch
import torch.nn as nn
import torch.nn.functional as F

class SplendidSplendorModel(nn.Module):
    def __init__(self, hidden_layers_width, hidden_layers_num):
        super().__init__()
        self.hidden_num = hidden_layers_num
        self.hidden_width = hidden_layers_width

        self.in_width = (4*(6+5+(3*(5+1+1))))+(4*(5+1))+(3*(4*(5+1+1))) #236
        self.out_width = (4+(3*5)+1+5+1+1+6) #33

        self.in_layer = nn.Linear(in_features = self.in_width, out_features = self.hidden_width)
        self.in_activation = nn.ReLU()
        
        hidden_layer = nn.Sequential(nn.Linear(self.hidden_width,self.hidden_width),nn.ReLU())
        self.hidden_layers = nn.ModuleList([hidden_layer for i in range(self.hidden_num)])
        self.out_layer = nn.Linear(in_features = self.hidden_width, out_features = self.out_width)
        #Todo: create named parallel output linear layers so we can more cleanly handle the outputs

    def init_weights(self):
        #initialize with random noise
        for m in self.modules():
            nn.init.orthogonal_(m.weight, val)
            torch.nn.utils.weight_norm(m)
    
    def forward(self,input):
        output = self.in_activation(self.in_layer(input))
        for layer in enumerate(self.hidden_layers):
            output = layer(output)
        return output