import torch
import torch.nn as nn
import torch.nn.functional as F
from utilities.better_param_dict import BetterParamDict

from utilities.simple_profile import SimpleProfileAggregator

class SplendidSplendorModel(nn.Module):
    def __init__(self, input_shape_dict: BetterParamDict[torch.Tensor], output_shape_dict: BetterParamDict[torch.Tensor], hidden_layers_width, hidden_layers_num):
        '''Takes input and output objects in a dict form, with the keys being the input/output
        names and the values being 1: length of vectors needed and 2: clamp bounds (output dict only).
        Takes hidden layer parameters to construct an arbitrary multi-layer perceptron'''
        super().__init__()
        self.input_shape_dict = input_shape_dict
        self.output_shape_dict = output_shape_dict
        self.hidden_num = hidden_layers_num
        self.hidden_width = hidden_layers_width
        self.input_lane = nn.Linear(in_features = len(self.input_shape_dict.get_backing_packed_data()), out_features = self.hidden_width)
        # self.input_lanes = nn.ModuleDict()
        # for input_key in self.input_shape_dict:
        #     self.input_lanes[input_key] = nn.Linear(in_features = len(self.input_shape_dict[input_key]), out_features = self.hidden_width)
        self.in_activation = nn.ReLU()
        
        hidden_layer = nn.Sequential(nn.Linear(self.hidden_width,self.hidden_width),nn.ReLU())
        self.hidden_layers = nn.ModuleList([hidden_layer for i in range(self.hidden_num)])

        self.output_lane = nn.Linear(in_features = self.hidden_width, out_features = len(self.output_shape_dict.get_backing_packed_data()))
        # self.output_lanes = nn.ModuleDict()
        # for output_key in self.output_shape_dict:
        #     self.output_lanes[output_key] = nn.Linear(in_features = self.hidden_width, out_features = len(output_shape_dict[output_key]))

    def init_weights(self):
        #initialize with random noise
        for m in self.modules():
            nn.init.orthogonal_(m.weight, val)
            torch.nn.utils.weight_norm(m)
    
    def forward(self,input_dict: BetterParamDict[torch.Tensor], profiler: SimpleProfileAggregator = None) -> BetterParamDict[torch.Tensor]:
        '''
        input_dict is layed out the same way as the input_shape_dict, except the values are the
        actual scalar vectors that get passed to the model {'in1':torch.Tensor[n], etc.} 
        '''
        
        output:torch.Tensor = self.input_lane.forward(input_dict.get_backing_packed_data())
        if profiler is not None:
            profiler.sample("input lane concatenation")

        output = self.in_activation(output)
        for layer in self.hidden_layers:
            output = layer(output)
        
        if profiler is not None:
            profiler.sample("model forward")
        
        output = self.output_lane(output)
        ## ensure that the output dict has the correct shape, imported from the original shape dictionary
        ## because a tensor-based BetterParamDict does not support resizing
        out_dict = BetterParamDict.reindex_over_new_data(self.output_shape_dict, output)
        
        if profiler is not None:
            profiler.sample("output lane evaluation")

        return out_dict

    def forward_from_dictionary(self,input_dict: dict[str, torch.Tensor], profiler: SimpleProfileAggregator = None) -> dict[str, torch.Tensor]:
        '''input_dict is layed out the same way as the input_shape_dict, except the values are the
        actual scalar vectors that get passed to the model {'in1':torch.Tensor[n], etc.} '''
        
        # create a single tensor-backed list from the dictionary, by using tensor.cat
        input_hunk: torch.Tensor = None
        for input_shape in sorted(self.input_shape_dict.index_dict.items(), key=lambda x: x[1][0]):
            if input_shape[0] not in input_dict:
                raise Exception("input dict must be in same shape as original shape dict")
            expected_len = input_shape[1][1] - input_shape[1][0] 
            input_slice = input_dict[input_shape[0]]

            if input_slice.size(-1) != expected_len:
                raise Exception("input dict must be in same shape as original shape dict. found a tensor of size " + str(input_slice.size(-1)) + ", expected " + str(expected_len))
            
            if input_hunk is None:
                input_hunk = input_slice
            else:
                input_hunk = torch.cat((input_hunk, input_slice), -1)

        output:torch.Tensor = self.input_lane.forward(input_hunk)

        if profiler is not None:
            profiler.sample("input lane concatenation")

        output = self.in_activation(output)
        for layer in self.hidden_layers:
            output = layer(output)
        
        if profiler is not None:
            profiler.sample("model forward")
        
        output = self.output_lane(output)

        out_dict = {}
        for out_key, (start, end) in self.output_shape_dict.index_dict.items():
            indexes = range(start, end)
            select_indexes = torch.Tensor(indexes).int().to(output.get_device())
            out_dict[out_key] = output.index_select(-1, select_indexes)
        
        if profiler is not None:
            profiler.sample("output lane evaluation")

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