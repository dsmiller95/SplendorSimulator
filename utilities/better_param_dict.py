import torch
from typing import Callable, Generic, TypeVar

T = TypeVar('T')

class BetterParamDict:
    """Packs a bunch of lists keyed by keys in a dict into a single Tensor, keyed by indexes"""

    def __init__(self, data: dict[str, list[float]]):
        self.aggregate_list : torch.Tensor = []
        self.index_dict : dict[str, tuple[int, int]] = {}
        for key in data:
            origin_index = len(self.aggregate_list)
            self.aggregate_list += data[key]
            end_index = len(self.aggregate_list)
            self.index_dict[key] = (origin_index, end_index)
    
    def __getitem__(self, key: str) -> torch.Tensor:
        data_range = self.index_dict[key]
        return self.aggregate_list[data_range[0]:data_range[1]]
    
    def get_backing_packed_data(self) -> torch.Tensor:
        return self.aggregate_list