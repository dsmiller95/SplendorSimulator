import torch
from utilities.better_param_dict import BetterParamDict

from utilities.utils import Lazy

def to_hot_from_scalar(scalar: int, length: int) -> list[int]:
    new_list = [0] * length
    new_list[scalar] = 1
    return new_list

def map_all_to_tensors(dict: BetterParamDict[list[float]]) -> BetterParamDict[torch.Tensor]:
    new_tensor = torch.Tensor(
        [0 if x is None else x for x in dict.get_backing_packed_data()] 
        ).to(torch.device('cpu'))
    return BetterParamDict.reindex_over_new_data(dict, new_tensor)