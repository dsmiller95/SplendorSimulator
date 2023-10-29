import splendor_simulation

import test_data as test_data;

def test_parses_config_data():
    config = splendor_simulation.SplendorConfig.parse_config_csv(test_data.test_config_raw)
    assert len(config.cards) == 7, "there should be 10 cards"
    
    assert config.cards[0].id == 11, "first card should have id 11"
    assert config.cards[6].id == 17, "last card should have id 17"

    assert config.cards[0].costs[splendor_simulation.SplendorResourceType.Ruby] == 0, "first card should cost 0 Ruby"
    assert config.cards[0].costs[splendor_simulation.SplendorResourceType.Sapphire] == 1, "first card should cost 0 Sapphire"

    assert len(config.nobles) == 3, "there should be 3 nobles"
    assert config.nobles[0].id == 1, "first noble should have id 1"
    assert config.nobles[2].id == 3, "last noble should have id 3"
    assert config.nobles[0].points == 3, "first noble should have 3 points"
    assert config.nobles[0].costs[splendor_simulation.SplendorResourceType.Ruby] == 3, "first noble should cost 3 Ruby"
    assert config.nobles[0].costs[splendor_simulation.SplendorResourceType.Diamond] == 0, "first noble should cost 3 Diamond"


def test_constructs_inspectable_game():
    config = splendor_simulation.SplendorConfig.parse_config_csv(test_data.test_config_raw)
    game = splendor_simulation.SplendorGame(config, 4)
    assert game.turn_n == 0, "turn should be 0"
    assert game.active_player_index == 0, "active player should be 0"
    assert game.active_player().points == 0, "active player should have 0 points"
    assert game.active_player().resources[splendor_simulation.SplendorResourceType.Ruby] == 0, "active player should have 0 rubies"

    assert game.bank[splendor_simulation.SplendorResourceType.Ruby] == 5, "bank should have 5 rubies"
    assert game.bank[splendor_simulation.SplendorResourceType.Sapphire] == 5, "bank should have 5 Sapphires"
    assert game.bank[splendor_simulation.SplendorResourceType.Emerald] == 5, "bank should have 5 Emeralds"
    assert game.bank[splendor_simulation.SplendorResourceType.Diamond] == 5, "bank should have 5 Diamonds"
    assert game.bank[splendor_simulation.SplendorResourceType.Onyx] == 5, "bank should have 5 Onyx"
    assert game.bank[splendor_simulation.SplendorResourceType.Gold] == 5, "bank should have 5 Gold"

    assert len(game.cards_by_level[0]) == 4, "there should be 4 level 1 cards"
    assert len(game.cards_by_level[2]) == 4, "there should be 4 level 3 cards"

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

    location_of_card_15 = game.cards_by_level[1].index(lambda x: x.id == 15)

    turn_1 = splendor_simulation.SplendorTurn.new_take_two(Emerald)
    turn_2 = splendor_simulation.SplendorTurn.new_take_two(Sapphire)
    turn_3 = splendor_simulation.SplendorTurn.new_buy_card_on_board(2, location_of_card_15)

    game.take_turn(turn_1)
    assert game.turn_n == 1, "turn should be 1"
    assert game.active_player_index == 1, "active player should be 0"
    game.take_turn(turn_2)
    game.take_turn(turn_3)
    