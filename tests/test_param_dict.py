from utilities.better_param_dict import BetterParamDict

def test_param_dict_packs_and_unpacks():
    input_data = {
        "test1" : [1, 0.1, 10],
        "another key" : [],
        "a 1 key" : [42],
    }
    param_dict = BetterParamDict(input_data)

    assert [1, 0.1, 10, 42] == param_dict.get_backing_packed_data()

    assert [1, 0.1, 10] == param_dict["test1"]
    assert [] == param_dict["another key"]
    assert [42] == param_dict["a 1 key"]