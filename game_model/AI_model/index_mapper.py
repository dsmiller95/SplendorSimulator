import operator as op
from functools import reduce


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
    def __init__(self, option_space_size: int, option_choices: int, pick_multiples: bool = False, all_option_numbers: bool = False):

        if (pick_multiples == False and all_option_numbers == True):
            raise Exception("this use case is unsupported")

        self.length = option_space_size
        self.option_choices = option_choices
        self.total_options = 0
        self.multiple = pick_multiples
        self.all_options = all_option_numbers
        all_option_choice_list = range(0, option_choices) if all_option_numbers else [option_choices]
        for choice_n in all_option_choice_list:
            if pick_multiples:
                ## if multiples are allowed, choice space is space ^ number of choices
                self.total_options += pow(option_space_size, choice_n) 
            else:
                self.total_options += nCr(option_space_size, choice_n)
            
    
    def map_to_index(self, n_hot_list: list[int]) -> int:
        '''
        map from a list containing n-hot values into the index in a 1-hot list
        '''
        assert len(n_hot_list) == self.length
        if self.multiple:
            total_n = 0
            working_index = 0
            for i, val in enumerate(n_hot_list):
                while val > 0:
                    val -= 1
                    total_n += 1
                    working_index *= self.length
                    working_index += i
            
            if self.all_options:
                ## offset the index by the total index space of all lesser-n choices, if enabled
                for x in range(0, total_n):
                    working_index += pow(self.length, x)
            else:
                assert total_n == self.option_choices
            return working_index
        else:
            return 0

    def map_from_index(self, combinatorial_index: int) -> list[int]:
        result_list = [0] * self.length

        return result_list

def nCr(n, r):
    r = min(r, n-r)
    numerator = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numerator // denom  # or / in Python 2


