

def test_imports_splendor_simulation():
    import splendor_simulation
    added_one = splendor_simulation.add_one_as_string(1)
    assert added_one.something == "2", "1 plus one should be 2"
    assert added_one.another == 1, "1 should be 1"



def test_success():
    assert 1 == 1, "1 should be 1"