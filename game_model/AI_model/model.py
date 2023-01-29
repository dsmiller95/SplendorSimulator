import torch
import torch.nn as nn
import torch.nn.functional as F

from utilities.simple_profile import SimpleProfileAggregator

class SplendidSplendorModel(nn.Module):
    def __init__(self, input_len: int, output_len: int, hidden_layers_width: int, hidden_layers_num: int):
        '''Takes input and output objects in a dict form, with the keys being the input/output
        names and the values being 1: length of vectors needed and 2: clamp bounds (output dict only).
        Takes hidden layer parameters to construct an arbitrary multi-layer perceptron'''
        super().__init__()
        self.hidden_num = hidden_layers_num
        self.hidden_width = hidden_layers_width
        self.input_lane = nn.Linear(in_features = input_len, out_features = self.hidden_width)
        self.in_activation = nn.ReLU()
        
        hidden_layer = nn.Sequential(nn.Linear(self.hidden_width,self.hidden_width),nn.ReLU())
        self.hidden_layers = nn.ModuleList([hidden_layer for i in range(self.hidden_num)])

        self.output_lane = nn.Linear(in_features = self.hidden_width, out_features = output_len)

    def init_weights(self):
        #initialize with random noise
        for m in self.modules():
            nn.init.orthogonal_(m.weight, val)
            torch.nn.utils.weight_norm(m)
    
    def forward(self,input: torch.Tensor, profiler: SimpleProfileAggregator = None) -> torch.Tensor:
        '''
        input_dict is layed out the same way as the input_shape_dict, except the values are the
        actual scalar vectors that get passed to the model {'in1':torch.Tensor[n], etc.} 
        '''
        
        output:torch.Tensor = self.input_lane.forward(input)
        if profiler is not None:
            profiler.sample("input lane")

        output = self.in_activation(output)
        for layer in self.hidden_layers:
            output = layer(output)
        
        if profiler is not None:
            profiler.sample("model forward")
        
        output = self.output_lane(output)
        
        if profiler is not None:
            profiler.sample("output lane")

        return output