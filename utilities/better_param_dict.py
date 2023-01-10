import torch
from typing import Callable, Generic, TypeVar

T = TypeVar('T')

class BetterParamDict(Generic[T]):
    """Packs a bunch of lists keyed by keys in a dict into a single list, keyed by indexes"""

    def __init__(self, data: dict[str, T], empty: T):

        self.input_dict = data
        self._init_from_dict(self.input_dict, empty)
        
        # self.aggregate_list : torch.Tensor = []
        # self.index_dict : dict[str, tuple[int, int]] = {}
        # for key in data:
        #     origin_index = len(self.aggregate_list)
        #     self.aggregate_list += data[key]
        #     end_index = len(self.aggregate_list)
        #     self.index_dict[key] = (origin_index, end_index)
    
    def __getitem__(self, key: str) -> T:
        data_range = self.index_dict[key]
        return self.aggregate_data[data_range[0]:data_range[1]]
    
    def get_backing_packed_data(self) -> T:
        return self.aggregate_data

    def set_backing_packed_data(self,aggregate_tensor: T):
        '''Take in a tensor of the correct shape and reset the dictionary
        values to the values from the tensor'''
        self.aggregate_data = aggregate_tensor
        for key in self.index_dict:
            indices = self.index_dict[key][0],self.index_dict[key][1]
            self.input_dict[key] = self.aggregate_data[indices[0]:indices[1]]
        self._init_from_dict(self.input_dict)
        return self.input_dict

    def _init_from_dict(self,data: dict[str, T], empty: T):
        '''Do the init functions to setup a new dict'''
        self.aggregate_data = empty
        self.index_dict : dict[str, tuple[int, int]] = {}
        origin_index: int = 0
        for key in data:
            origin_index = len(self.aggregate_data)
            self.aggregate_data += data[key]
            end_index = len(self.aggregate_data)
            self.index_dict[key] = (origin_index, end_index)
