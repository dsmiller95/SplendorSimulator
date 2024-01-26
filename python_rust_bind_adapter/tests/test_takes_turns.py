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
    turn_result = game.take_turn(turn)

    assert game.turn_n == 1, "turn should be 1"
    assert game.active_player_index == 1, "active player should be 1"
    changed_player = game.get_player_at(0)
    assert changed_player.points == 0, "active player should have 0 points"
    assert changed_player.resources[Ruby] == 1, "active player should have 1 Ruby"
    assert changed_player.resources[Sapphire] == 1, "active player should have 1 Sapphire"
    assert changed_player.resources[Emerald] == 1, "active player should have 1 Emerald"
    assert changed_player.resources[Diamond] == 0, "active player should have 0 Diamond"
    assert changed_player.resources[Onyx] == 0, "active player should have 0 Onyx"
    assert changed_player.resources[Gold] == 0, "active player should have 0 Gold"



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
    extra_config_row = "\n420,Regular,2,0,2,2,0,0,0,0,0,1,0,1"
    config = splendor_simulation.SplendorConfig.parse_config_csv(test_data.test_config_raw + extra_config_row)
    game = splendor_simulation.SplendorGame(config, 2)
    assert game.turn_n == 0, "turn should be 0"
    assert game.active_player_index == 0, "active player should be 0"

    Ruby = splendor_simulation.SplendorResourceType.Ruby
    Sapphire = splendor_simulation.SplendorResourceType.Sapphire
    Emerald = splendor_simulation.SplendorResourceType.Emerald
    Diamond = splendor_simulation.SplendorResourceType.Diamond
    Onyx = splendor_simulation.SplendorResourceType.Onyx
    Gold = splendor_simulation.SplendorResourceType.Gold

    location_of_card = next(i for i,v in enumerate(game.get_card_row(2)) if v != None and v.id == 420)
    assert location_of_card != None, "card should be in row 2"

    noop_turn = splendor_simulation.SplendorTurn.new_take_three(Ruby, Diamond, Onyx)
    turn_1 = splendor_simulation.SplendorTurn.new_take_two(Emerald)
    turn_2 = splendor_simulation.SplendorTurn.new_take_two(Sapphire)
    turn_3 = splendor_simulation.SplendorTurn.new_buy_card_on_board(2, location_of_card + 1)

    assert game.active_player.points == 0, "active player should have 0 points"
    
    assert game.take_turn(turn_1) == "Turn success: Success, next player: 1", "turn should be successful"
    assert game.take_turn(noop_turn) == "Turn success: Success, next player: 0", "noop turn should be successful"
    assert game.turn_n == 2, "turn count should be 2"
    assert game.active_player_index == 0, "active player should be 0"
    assert game.active_player.resources[Emerald] == 2, "active player should have 2 Emerald"
    assert game.take_turn(turn_2) == "Turn success: Success, next player: 1", "turn should be successful"
    assert game.take_turn(noop_turn) == "Turn success: Success, next player: 0", "noop turn should be successful"
    assert game.turn_n == 4, "turn count should be 4"
    assert game.active_player_index == 0, "active player should be 0"
    assert game.active_player.resources[Sapphire] == 2, "active player should have 2 Sapphire"
    assert game.take_turn(turn_3) == "Turn success: Success, next player: 1", "turn should be successful"
    assert game.take_turn(noop_turn) == "Turn success: Success, next player: 0", "noop turn should be successful"

    assert game.active_player.points == 1, "active player should have 1 points"
    