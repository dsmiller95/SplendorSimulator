import torch
import torch.nn as nn
import torch.nn.functional as F

class SplendidSplendorModel(nn.Module):
    def __init__(self, input_shape_dict, output_shape_dict, hidden_layers_width, hidden_layers_num):
        '''Takes input and output objects in a dict form, with the keys being the input/output
        names and the values being 1: length of vectors needed and 2: clamp bounds if needed.
        Takes hidden layer parameters to construct an arbitrary multi-layer perceptron'''
        super().__init__()
        self.hidden_num = hidden_layers_num
        self.hidden_width = hidden_layers_width
        self.in_width = sum([value[0] for value in input_shape_dict.values()])
        self.out_width = sum([value for value in output_shape_dict.values()])
        self.clamp_vals = [value[1:] for value in input_shape_dict.values()]
        self.input_lanes = nn.ModuleDict()
        for input_key in input_shape_dict:
            self.input_lanes[input_key] = nn.Linear(in_features = input_shape_dict[input_key][0], out_features = self.hidden_width)
        self.in_activation = nn.ReLU()
        
        hidden_layer = nn.Sequential(nn.Linear(self.hidden_width,self.hidden_width),nn.ReLU())
        self.hidden_layers = nn.ModuleList([hidden_layer for i in range(self.hidden_num)])

        self.output_lanes = nn.ModuleDict()
        for output_key in output_shape_dict:
            self.output_lanes[output_key] = nn.Linear(in_features = self.hidden_width, out_features = output_shape_dict[output_key])


    def init_weights(self):
        #initialize with random noise
        for m in self.modules():
            nn.init.orthogonal_(m.weight, val)
            torch.nn.utils.weight_norm(m)
    
    def forward(self,input_dict):
        '''input_dict is layed out the same way as the input_shape_dict, except the values are the
        actual scalar vectors that get passed to the model {'in1':torch.Tensor[n], etc.} '''
        #for i,input in enumerate(input_dict):
        #    outputs[]
        #output = torch.cat([self.input_lanes.value()])
        #combine all input linear layers by summing their hidden_width outputs together
        #inputs = 
        output = torch.stack([self.input_lanes[key](input_dict[key]) for key in self.input_lanes],dim=0)
        output = self.in_activation(output)
        for layer in self.hidden_layers:
            output = layer(output)
        output = [self.output_lanes[key](output) for key in self.output_lanes]
        return output


'''
Structure works like this:

      in1  in2  in3  [these are vectors with their own individual lengths]
         \  |  /
          \ | /
         hidden1     [The hidden layer input size is the ]
            |
         hidden2
            |
         hidden3
           /\
          /  \
       out1  out2
'''