import torch
from typing import Callable, Generic, TypeVar

T = TypeVar('T')

class BetterParamDict:
    """Packs a bunch of lists keyed by keys in a dict into a single Tensor, keyed by indexes"""

    def __init__(self, data: dict[str, list[float]]):

        self.input_dict = data
        self._init_from_dict(self.input_dict)
        
        # self.aggregate_list : torch.Tensor = []
        # self.index_dict : dict[str, tuple[int, int]] = {}
        # for key in data:
        #     origin_index = len(self.aggregate_list)
        #     self.aggregate_list += data[key]
        #     end_index = len(self.aggregate_list)
        #     self.index_dict[key] = (origin_index, end_index)
    
    def __getitem__(self, key: str) -> torch.Tensor:
        data_range = self.index_dict[key]
        return self.aggregate_tensor[data_range[0]:data_range[1]]
    
    def get_backing_packed_data(self) -> torch.Tensor:
        return self.aggregate_tensor

    def set_backing_packed_data(self,aggregate_tensor: torch.Tensor):
        '''Take in a tensor of the correct shape and reset the dictionary
        values to the values from the tensor'''
        self.aggregate_tensor = aggregate_tensor
        for key in self.index_dict:
            indices = self.index_dict[key][0],self.index_dict[key][1]
            self.input_dict[key] = self.aggregate_tensor[indices[0]:indices[1]]
        self._init_from_dict(self.input_dict)
        return self.input_dict

    def _init_from_dict(self,data: dict[str, list[float]]):
        '''Do the init functions to setup a new dict'''
        self.aggregate_list : torch.Tensor = []
        self.index_dict : dict[str, tuple[int, int]] = {}
        origin_index: int = 0
        for key in data:
            origin_index = len(self.aggregate_list)
            self.aggregate_list += data[key]
            end_index = len(self.aggregate_list)
            self.index_dict[key] = (origin_index, end_index)
        self.aggregate_tensor = torch.stack(self.aggregate_list)

