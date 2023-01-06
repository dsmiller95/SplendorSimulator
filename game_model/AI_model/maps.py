import torch

from utilities.utils import Lazy

def to_hot_from_scalar(scalar: int, length: int) -> list[int]:
    new_list = [0] * length
    new_list[scalar] = 1
    return new_list

def map_all_to_tensors(dict: dict[str, list[float]]) -> dict[str, torch.Tensor]:
    return {
        key: torch.Tensor([0 if x is None else x for x in value]).to(torch.device('cpu'))
        for (key, value)
        in dict.items()
    }

