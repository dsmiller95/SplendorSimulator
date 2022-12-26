import torch
import torch.nn as nn
import torch.nn.functional as F

def SplendidSplendorModel(nn.Module):
    def __init__(self, hidden_layers_width, hidden_layers_num):
        super().__init__()
        self.hidden_layers_num = hidden_layers_num
        self.hidden_layers_width = hidden_layers_width

        self.in_width = (12*4)+()


    def init_weights(self):
        for m in self.modules():
            nn.init.orthogonal_(m.weight, val)
            torch.nn.utils.weight_norm(m)
    
    def forward(self,input):
        return input