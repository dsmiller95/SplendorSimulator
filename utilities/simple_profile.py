from __future__ import annotations
from typing import Callable, Generic, TypeVar

import time

class SimpleProfile():
    def __init__(self):
        self.start_time = time.time()
        self.samples: dict[str, float] = {}
    
    def sample_next(self, sample_name: str):
        delta_time = time.time() - self.start_time
        if sample_name in self.samples:
            self.samples[sample_name] += delta_time
        else:
            self.samples[sample_name] = delta_time
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
            res +=  key.ljust(max_width) + " : " + '{:,}'.format(round(time * 1000000)).rjust(10) + "ns\n"
            total += time
        res += "total".ljust(max_width) + " : " + '{:,}'.format(round(total * 1000000)).rjust(10) + "ns\n"
        return res
        
class SimpleProfileAggregator():
    active : SimpleProfileAggregator = None
    def __init__(self, description:str, initial_average_size: int):
        self.average_samples: list[SimpleProfile] = []
        self.sample_batch_size = initial_average_size
        self.all_averaged_samples: list[SimpleProfile] = []
        self.max_running_samples = 100
        self.current_sampler: SimpleProfile = None
        self.description = description

    @staticmethod
    def sample_static(sample_description: str):
        if SimpleProfileAggregator.active is None:
            return
        SimpleProfileAggregator.active.sample(sample_description)

    def begin_sample_run(self):
        self.current_sampler = SimpleProfile()
        SimpleProfileAggregator.active = self
    
    def end_sample_run(self):
        self.average_samples.append(self.current_sampler)
        self.current_sampler = None
        SimpleProfileAggregator.active = None
        
        if len(self.average_samples) >= self.sample_batch_size:
            self.all_averaged_samples.append( sum(self.average_samples, SimpleProfile()) / len(self.average_samples))
            self.average_samples.clear()
            avg = sum(self.all_averaged_samples, SimpleProfile()) / len(self.all_averaged_samples)
            if len(self.all_averaged_samples) > self.max_running_samples:
                del self.all_averaged_samples[0:int(self.max_running_samples/4)]
            print(self.description + ":\n" + avg.describe_samples())


    def sample(self, sample_description: str):
        self.current_sampler.sample_next(sample_description)
    