import torch
import torch.nn as nn
import torch.nn.functional as F

from utilities.simple_profile import SimpleProfile

class SplendidSplendorModel(nn.Module):
    def __init__(self, input_shape_dict, output_shape_dict: dict[str, list[float]], hidden_layers_width, hidden_layers_num):
        '''Takes input and output objects in a dict form, with the keys being the input/output
        names and the values being 1: length of vectors needed and 2: clamp bounds (output dict only).
        Takes hidden layer parameters to construct an arbitrary multi-layer perceptron'''
        super().__init__()
        self.input_shape_dict = input_shape_dict
        self.output_shape_dict = output_shape_dict
        self.hidden_num = hidden_layers_num
        self.hidden_width = hidden_layers_width
        self.input_lanes = nn.ModuleDict()
        for input_key in self.input_shape_dict:
            self.input_lanes[input_key] = nn.Linear(in_features = len(self.input_shape_dict[input_key]), out_features = self.hidden_width)
        self.in_activation = nn.ReLU()
        
        hidden_layer = nn.Sequential(nn.Linear(self.hidden_width,self.hidden_width),nn.ReLU())
        self.hidden_layers = nn.ModuleList([hidden_layer for i in range(self.hidden_num)])

        self.output_lanes = nn.ModuleDict()
        for output_key in self.output_shape_dict:
            self.output_lanes[output_key] = nn.Linear(in_features = self.hidden_width, out_features = len(output_shape_dict[output_key]))

    def init_weights(self):
        #initialize with random noise
        for m in self.modules():
            nn.init.orthogonal_(m.weight, val)
            torch.nn.utils.weight_norm(m)
    
    def forward(self,input_dict: dict[str, torch.Tensor], profiler: SimpleProfile = None) -> dict[str, torch.Tensor]:
        '''input_dict is layed out the same way as the input_shape_dict, except the values are the
        actual scalar vectors that get passed to the model {'in1':torch.Tensor[n], etc.} '''
        
        
        output:torch.Tensor = None
        for key in self.input_lanes:
            if output is None:
                output = self.input_lanes[key](input_dict[key])
            else:
                output += self.input_lanes[key](input_dict[key])
        if profiler is not None:
            profiler.sample_next("input lane concatenation")

        output = self.in_activation(output)
        for layer in self.hidden_layers:
            output = layer(output)
        
        if profiler is not None:
            profiler.sample_next("model forward")
        
        out_dict = {}
        for key in self.output_lanes:
            out_dict[key] = self.output_lanes[key](output)
        
        if profiler is not None:
            profiler.sample_next("output lane evaluation")

        return out_dict


'''
Structure works like this:

      in1  in2  in3  [dictionary of vectors with their own individual lengths]
         \  |  /
          \ | /
         hidden1
            |
         hidden2
            |
         hidden3
           /\
          /  \
       out1  out2 [dictionary of vectors with their own individual lengths]
'''