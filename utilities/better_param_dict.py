from __future__ import annotations
import torch
from typing import Callable, Generic, TypeVar

T = TypeVar('T')
J = TypeVar('J')


class BetterParamDict(Generic[T]):
    """Packs a bunch of lists keyed by keys in a dict into a single Tensor, keyed by indexes.
    the generic type must be sliceable, and support concatenation"""

    def __init__(self, empty: T):
        self.aggregate_list : T = empty
        self.index_dict : dict[str, tuple[int, int]] = {}

    @staticmethod
    def map_from_dict(data: dict[str, T], empty: T) -> BetterParamDict[T]:
        output : BetterParamDict[T] = BetterParamDict(empty)
        for key in data:
            output._append_item(key, data[key])
        return output

    @staticmethod
    def reindex_over_new_data(original_dict : BetterParamDict[T], remapped_data: J ) -> BetterParamDict[J]:
        if len(remapped_data) != len(original_dict.aggregate_list):
            raise IndexError("length of remapped data does not match original dictionary length")
        output : BetterParamDict[J] = BetterParamDict(None)
        output.aggregate_list = remapped_data
        output.index_dict = original_dict.index_dict
        return output

    def _append_item(self, key: str, value: T) -> None:
        origin_index = len(self.aggregate_list)
        self.aggregate_list += value
        end_index = len(self.aggregate_list)
        self.index_dict[key] = (origin_index, end_index)

    def keys(self) -> list[str]:
        return list(self.index_dict.keys())

    def __contains__(self, item: str) -> bool:
        return item in self.index_dict

    def __getitem__(self, key: str) -> T:
        data_range = self.index_dict[key]
        return self.aggregate_list[data_range[0]:data_range[1]]
    
    def __setitem__(self, key: str, value: T) -> None:
        if not (key in self.index_dict):
            self._append_item(key, value)
            return
        
        placement = self.index_dict[key]
        orig_len = (placement[1] - placement[0])
        if len(value) != orig_len:
            raise IndexError("length of new data does not match original data length at key, original: " + str(orig_len) + " at " + key + ", new len: " + str(len(value)))
        self.aggregate_list[placement[0]:placement[1]] = value

    
    def get_backing_packed_data(self) -> T:
        return self.aggregate_list