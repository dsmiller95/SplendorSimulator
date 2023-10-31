import splendor_simulation

import test_data as test_data;


def test_construct_game_and_takes_pick_three():
    config = splendor_simulation.SplendorConfig.parse_config_csv(test_data.test_config_raw)
    game = splendor_simulation.SplendorGame(config, 4)
    assert game.turn_n == 0, "turn should be 0"
    assert game.active_player_index == 0, "active player should be 0"

    Ruby = splendor_simulation.SplendorResourceType.Ruby
    Sapphire = splendor_simulation.SplendorResourceType.Sapphire
    Emerald = splendor_simulation.SplendorResourceType.Emerald
    Diamond = splendor_simulation.SplendorResourceType.Diamond
    Onyx = splendor_simulation.SplendorResourceType.Onyx
    Gold = splendor_simulation.SplendorResourceType.Gold

    turn = splendor_simulation.SplendorTurn.new_take_three(Ruby, Emerald, Sapphire)
    game.take_turn(turn)

    assert game.turn_n == 1, "turn should be 1"
    assert game.active_player_index == 1, "active player should be 1"
    assert game.active_player().points == 0, "active player should have 0 points"
    assert game.active_player().resources[Ruby] == 1, "active player should have 1 Ruby"
    assert game.active_player().resources[Sapphire] == 1, "active player should have 1 Sapphire"
    assert game.active_player().resources[Emerald] == 1, "active player should have 1 Emerald"
    assert game.active_player().resources[Diamond] == 0, "active player should have 0 Diamond"
    assert game.active_player().resources[Onyx] == 0, "active player should have 0 Onyx"
    assert game.active_player().resources[Gold] == 0, "active player should have 0 Gold"



def test_when_taking_invalid_turn__throws():
    config = splendor_simulation.SplendorConfig.parse_config_csv(test_data.test_config_raw)
    game = splendor_simulation.SplendorGame(config, 4)
    turn = splendor_simulation.SplendorTurn.new_buy_card_on_board(2, 1)

    # expect throw:
    throws = False
    try:
        game.take_turn(turn)
    except:
        throws = True
        pass
    assert throws, "should have thrown."

def test_when_picked_many_tokens__can_purchase_card():
    config = splendor_simulation.SplendorConfig.parse_config_csv(test_data.test_config_raw)
    game = splendor_simulation.SplendorGame(config, 1)
    assert game.turn_n == 0, "turn should be 0"
    assert game.active_player_index == 0, "active player should be 0"

    Ruby = splendor_simulation.SplendorResourceType.Ruby
    Sapphire = splendor_simulation.SplendorResourceType.Sapphire
    Emerald = splendor_simulation.SplendorResourceType.Emerald
    Diamond = splendor_simulation.SplendorResourceType.Diamond
    Onyx = splendor_simulation.SplendorResourceType.Onyx
    Gold = splendor_simulation.SplendorResourceType.Gold

    location_of_card_15 = next(i for i,v in enumerate(game.get_card_row(2)) if v != None and v.id == 15)

    turn_1 = splendor_simulation.SplendorTurn.new_take_two(Emerald)
    turn_2 = splendor_simulation.SplendorTurn.new_take_two(Sapphire)
    turn_3 = splendor_simulation.SplendorTurn.new_buy_card_on_board(2, location_of_card_15)

    game.take_turn(turn_1)
    assert game.turn_n == 1, "turn should be 1"
    assert game.active_player_index == 1, "active player should be 0"
    game.take_turn(turn_2)
    game.take_turn(turn_3)
    