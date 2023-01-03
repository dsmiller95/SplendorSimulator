import torch
import torch.nn as nn
import torch.nn.functional as F

class SplendidSplendorModel(nn.Module):
    def __init__(self,
                 input_shape_dict,
                 hidden_layers_width,
                 hidden_layers_num,
                 Q_width
                 ):
        '''Takes input in dict form, with the keys being
        the input names and the values being the length of vectors needed.
        Hidden layer parameters construct a multi-layer perceptron.
        Output is Q-value vector with size Q_width.'''
        super().__init__()
        self.input_shape_dict = input_shape_dict
        self.output_shape_dict = output_shape_dict
        self.hidden_num = hidden_layers_num
        self.hidden_width = hidden_layers_width
        self.Q_width = Q_width
        self.input_lanes = nn.ModuleDict()
        for input_key in self.input_shape_dict:
            self.input_lanes[input_key] = nn.Linear(in_features = len(self.input_shape_dict[input_key]), out_features = self.hidden_width)
        self.in_activation = nn.ReLU()
        
        hidden_layer = nn.Sequential(nn.Linear(self.hidden_width,self.hidden_width),nn.ReLU())
        self.hidden_layers = nn.ModuleList([hidden_layer for i in range(self.hidden_num)])

        self.Q_layer = nn.Linear(in_features = self.hidden_width,
                                 out_features = self.Q_width)

    def init_weights(self):
        #initialize with random noise
        for m in self.modules():
            nn.init.orthogonal_(m.weight, val)
            torch.nn.utils.weight_norm(m)
    
    def forward(self,input_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        '''input_dict is layed out the same way as the input_shape_dict, except the values are the
        actual scalar vectors that get passed to the model {'in1':torch.Tensor[n], etc.} '''
        
        output:torch.Tensor = None
        for key in self.input_lanes:
            if output is None:
                output = self.input_lanes[key](input_dict[key])
            else:
                output += self.input_lanes[key](input_dict[key])
        output = self.in_activation(output)
        for layer in self.hidden_layers:
            output = layer(output)
        
        output = self.Q_layer(output) #to save memory, we should name 

        return output


'''
Structure works like this:

      in1  in2  in3  [these are vectors with their own individual lengths]
         \  |  /
          \ | /
         hidden1
            |
         hidden2
            |
         hidden3
            |
         Q-layer
            |
         output     [this is our Q-values vector]
'''