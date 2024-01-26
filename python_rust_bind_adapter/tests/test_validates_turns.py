import splendor_simulation

import test_data as test_data;


def test_cannot_construct_pick_three_of_same():
    Ruby = splendor_simulation.SplendorResourceType.Ruby
    Sapphire = splendor_simulation.SplendorResourceType.Sapphire

    # expect throw:
    throws = False
    try:
        turn = splendor_simulation.SplendorTurn.new_take_three(Ruby, Ruby, Sapphire)
    except:
        throws = True
        pass
    assert throws, "should throw when constructing turn. turn: " + str(turn)


def test_cannot_construct_pick_three_with_gold():
    Ruby = splendor_simulation.SplendorResourceType.Ruby
    Gold = splendor_simulation.SplendorResourceType.Gold

    # expect throw:
    throws = False
    try:
        turn = splendor_simulation.SplendorTurn.new_take_three(Ruby, Ruby, Gold)
    except:
        throws = True
        pass
    assert throws, "should throw when constructing turn. turn: " + str(turn)
    


def test_can_construct_pick_three_of_different():
    Ruby = splendor_simulation.SplendorResourceType.Ruby
    Sapphire = splendor_simulation.SplendorResourceType.Sapphire
    Emerald = splendor_simulation.SplendorResourceType.Emerald

    turn = splendor_simulation.SplendorTurn.new_take_three(Ruby, Sapphire, Emerald)
