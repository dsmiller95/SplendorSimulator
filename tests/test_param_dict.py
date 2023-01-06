from utilities.better_param_dict import BetterParamDict

def test_param_dict_packs_and_unpacks():
    input_data = {
        "test1" : [1, 0.1, 10],
        "another key" : [],
        "a 1 key" : [42],
    }
    param_dict : BetterParamDict[list[float]] = BetterParamDict.map_from_dict(input_data, [])

    assert [1, 0.1, 10, 42] == param_dict.get_backing_packed_data()

    assert [1, 0.1, 10] == param_dict["test1"]
    assert [] == param_dict["another key"]
    assert [42] == param_dict["a 1 key"]

def test_param_dict_builds():
    param_dict : BetterParamDict[list[float]] = BetterParamDict([])
    param_dict["key 1"] = [1, 100, 0.1]
    param_dict["another key"] = [22, 2]
    param_dict["another key"] = [22, 55]
    param_dict["empty key"] = []
    param_dict["another key 2"] = [3, 33, 333]

    assert [1, 100, 0.1, 22, 55, 3, 33, 333] == param_dict.get_backing_packed_data()

    assert [1, 100, 0.1] == param_dict["key 1"]
    assert [22, 55] == param_dict["another key"]
    assert [] == param_dict["empty key"]
    assert [3, 33, 333] == param_dict["another key 2"]
