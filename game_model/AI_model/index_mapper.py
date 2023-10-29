import itertools
import operator as op
from functools import reduce
from typing import Generator


'''
This file contains functions to remap n-hot representations into 1-hot representations.
Done via Combinatorics to represent all possible states
'''

class CombinatorialIndexMapping:
    '''
    Used to model the shape of valid choices inside an "N-hot list". includes the length of the list,
    and how many options can be chosen at once. Can be used to map a list of these characteristics into a single number,
    in the range [0, total_options), and back.
    '''
    def __init__(self, option_space_size: int, option_choices: int, allow_pick_multiple: bool = False, allow_pick_less: bool = False):
        self.length = option_space_size
        self.option_choices = option_choices
        self.multiple = allow_pick_multiple
        
        self.options_list = []

        all_option_choice_list = range(0, option_choices + 1) if allow_pick_less else [option_choices]
        for choice_n in all_option_choice_list:
            self.options_list += self._generate_combinations(choice_n)
    
    def total_possible_options(self):
        return len(self.options_list)
    
    def _generate_combinations(self, choice_n: int) -> Generator[list[int], None, None]:
        combo_iterator = itertools.combinations_with_replacement if self.multiple else itertools.combinations
        for combo in combo_iterator(range(self.length), choice_n):
            new_list = [0] * self.length
            for index in combo:
                new_list[index] += 1
            yield new_list


    def map_to_index(self, n_hot_list: list[int]) -> int:
        '''
        map from a list containing n-hot values into the index in a 1-hot list
        '''
        assert len(n_hot_list) == self.length

        potential_indexes = range(len(self.options_list))
        for hot_index, hot_val in enumerate(n_hot_list):
            potential_indexes = [x for x in potential_indexes if self.options_list[x][hot_index] == hot_val]
        if len(potential_indexes) != 1:
            raise Exception("error mapping to index. could not find valid combination")
        return potential_indexes[0]

    def map_from_index(self, combinatorial_index: int) -> list[int]:
        return self.options_list[combinatorial_index]
    
    def get_1_hot_from_n_hot(self, desired_combination_n_hot: list[int]) -> list[int]:
        full_list = [0] * self.total_possible_options()
        combination_index = self.map_to_index(desired_combination_n_hot)
        full_list[combination_index] = 1
        return full_list

def nCr(n, r):
    r = min(r, n-r)
    numerator = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numerator // denom  # or / in Python 2


