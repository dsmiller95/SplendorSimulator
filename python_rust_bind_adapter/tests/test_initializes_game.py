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
    assert game.active_player.points == 0, "active player should have 0 points"
    assert game.active_player.resources[splendor_simulation.SplendorResourceType.Ruby] == 0, "active player should have 0 rubies"

    assert game.bank[splendor_simulation.SplendorResourceType.Ruby] == 7, "bank should have 5 rubies"
    assert game.bank[splendor_simulation.SplendorResourceType.Sapphire] == 7, "bank should have 5 Sapphires"
    assert game.bank[splendor_simulation.SplendorResourceType.Emerald] == 7, "bank should have 5 Emeralds"
    assert game.bank[splendor_simulation.SplendorResourceType.Diamond] == 7, "bank should have 5 Diamonds"
    assert game.bank[splendor_simulation.SplendorResourceType.Onyx] == 7, "bank should have 5 Onyx"
    assert game.bank[splendor_simulation.SplendorResourceType.Gold] == 5, "bank should have 5 Gold"

    card_row_one = game.get_card_row(1)
    assert len(card_row_one) == 4, "there should be 4 level 1 card slots"
    assert len([x for x in card_row_one if x != None]) == 3, "there should be 3 level 1 cards"
    card_row_three = game.get_card_row(3)
    assert len(card_row_three) == 4, "there should be 4 level 3 card slots"
    assert len([x for x in card_row_three if x != None]) == 1, "there should be 1 level 3 cards"


def test_when_different_seed__different_cards():
    config = splendor_simulation.SplendorConfig.parse_config_csv(test_data.test_config_raw)
    game_base = splendor_simulation.SplendorGame(config, 4, 1)
    game_diff_seed = splendor_simulation.SplendorGame(config, 4, 2)
    game_same_seed = splendor_simulation.SplendorGame(config, 4, 1)

    base_card_row = game_base.get_card_row(1)
    diff_card_row = game_diff_seed.get_card_row(1)
    same_card_row = game_same_seed.get_card_row(1)
    first_card_in_row = next(i for i, x in enumerate(base_card_row) if x != None)
    assert base_card_row[first_card_in_row] != None, "row 1 should have one non-none card"
    assert base_card_row[first_card_in_row].id != diff_card_row[first_card_in_row].id, "first card in row 1 should be different when different seed"
    assert base_card_row[first_card_in_row].id == same_card_row[first_card_in_row].id, "first card in row 1 should be same when same seed"
