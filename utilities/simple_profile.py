from __future__ import annotations
from typing import Callable, Generic, TypeVar

import time

class SimpleProfile():
    def __init__(self):
        self.start_time = time.time()
        self.samples: dict[str, float] = {}
    
    def sample_next(self, sample_name: str):
        self.samples[sample_name] = time.time() - self.start_time
        self.start_time = time.time()
    
    def __add__(self, other: SimpleProfile) -> SimpleProfile:
        new_profile = SimpleProfile()
        for key, val in self.samples.items():
            new_profile.samples[key] = val
        for key, val in other.samples.items():
            new_profile.samples[key] = val + (new_profile.samples.get(key) or 0)
        return new_profile

    def __truediv__(self, other: int) -> SimpleProfile:
        new_profile = SimpleProfile()
        new_profile.samples = {key: val/other for key, val in self.samples.items()}
        return new_profile
    
    def describe_samples(self) -> str:
        res = ""
        total = 0
        max_width = max([len(x) for x in self.samples.keys()])
        for key, time in self.samples.items():
            res +=  key.rjust(max_width) + ": " + str(round(time * 1000, 2)) + "ms\n"
            total += time
        res += "total".rjust(max_width) + ": " + str(round(total * 1000, 2)) + "ms\n"
        return res
        
