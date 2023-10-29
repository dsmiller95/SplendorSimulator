from game_model.AI_model.index_mapper import CombinatorialIndexMapping
from utilities.better_param_dict import BetterParamDict

def test_n_hot_representation_accurate_size_when_no_multiples():

    representation = CombinatorialIndexMapping(5, 3)
    assert 10 == representation.total_possible_options()
    assert all([sum(x) == 3 for x in representation.options_list])

def test_n_hot_representation_accurate_size_when_multiples():

    representation = CombinatorialIndexMapping(5, 3, allow_pick_multiple=True)
    assert 35 == representation.total_possible_options()
    assert all([sum(x) == 3 for x in representation.options_list])

def test_n_hot_maps_to_and_back_no_multiples():

    representation = CombinatorialIndexMapping(5, 3)

    test_arrays = [
        [1, 0, 0, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 0, 0],
        [1, 0, 1, 0, 1],
    ]

    for test_array in test_arrays:
        number = representation.map_to_index(test_array)
        assert isinstance(number, int)

        round_trip = representation.map_from_index(number)
        assert test_array == round_trip

def test_n_hot_maps_to_and_back_with_multiples():

    representation = CombinatorialIndexMapping(6, 3, allow_pick_multiple=True, allow_pick_less=True)

    test_arrays = [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [2, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 3],
        [0, 0, 2, 0, 1, 0],
        [1, 0, 0, 1, 1, 0],
    ]

    for test_array in test_arrays:
        number = representation.map_to_index(test_array)
        assert isinstance(number, int)

        round_trip = representation.map_from_index(number)
        assert test_array == round_trip