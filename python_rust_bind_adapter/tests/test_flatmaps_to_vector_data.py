import splendor_simulation

import test_data as test_data;


def test_game_state_maps_to_correct_size():
    config = splendor_simulation.SplendorConfig.parse_config_csv(test_data.test_config_raw)
    game = splendor_simulation.SplendorGame(config, 4, hash(test_game_state_maps_to_correct_size.__name__))
    game_vector = game.get_packed_state_array()

    assert len(game_vector) == 512, "game vector should be 512 long"

def test_game_state_provides_correct_index_mapping():
    config = splendor_simulation.SplendorConfig.parse_config_csv(test_data.test_config_raw)
    game = splendor_simulation.SplendorGame(config, 4, hash(test_game_state_maps_to_correct_size.__name__))
    game_vector_indexes = game.get_packed_state_array_indexes()
    
    expected_keys = [
        "player_0_temp_resources",
        "player_0_perm_resources",
        "player_0_points",
        "player_0_reserved_card_0_costs",
        "player_0_reserved_card_0_returns",
        "player_0_reserved_card_0_points",
        "player_0_reserved_card_1_costs",
        "player_0_reserved_card_1_returns",
        "player_0_reserved_card_1_points",
        "player_0_reserved_card_2_costs",
        "player_0_reserved_card_2_returns",
        "player_0_reserved_card_2_points",
        "player_1_temp_resources",
        "player_1_perm_resources",
        "player_1_points",
        "player_1_reserved_card_0_costs",
        "player_1_reserved_card_0_returns",
        "player_1_reserved_card_0_points",
        "player_1_reserved_card_1_costs",
        "player_1_reserved_card_1_returns",
        "player_1_reserved_card_1_points",
        "player_1_reserved_card_2_costs",
        "player_1_reserved_card_2_returns",
        "player_1_reserved_card_2_points",
        "player_2_temp_resources",
        "player_2_perm_resources",
        "player_2_points",
        "player_2_reserved_card_0_costs",
        "player_2_reserved_card_0_returns",
        "player_2_reserved_card_0_points",
        "player_2_reserved_card_1_costs",
        "player_2_reserved_card_1_returns",
        "player_2_reserved_card_1_points",
        "player_2_reserved_card_2_costs",
        "player_2_reserved_card_2_returns",
        "player_2_reserved_card_2_points",
        "player_3_temp_resources",
        "player_3_perm_resources",
        "player_3_points",
        "player_3_reserved_card_0_costs",
        "player_3_reserved_card_0_returns",
        "player_3_reserved_card_0_points",
        "player_3_reserved_card_1_costs",
        "player_3_reserved_card_1_returns",
        "player_3_reserved_card_1_points",
        "player_3_reserved_card_2_costs",
        "player_3_reserved_card_2_returns",
        "player_3_reserved_card_2_points",
        "board_noble_0_costs",
        "board_noble_0_points",
        "board_noble_1_costs",
        "board_noble_1_points",
        "board_noble_2_costs",
        "board_noble_2_points",
        "board_noble_3_costs",
        "board_noble_3_points",
        "board_noble_4_costs",
        "board_noble_4_points",
        "tier_0_open_card_0_costs",
        "tier_0_open_card_0_returns",
        "tier_0_open_card_0_points",
        "tier_0_open_card_1_costs",
        "tier_0_open_card_1_returns",
        "tier_0_open_card_1_points",
        "tier_0_open_card_2_costs",
        "tier_0_open_card_2_returns",
        "tier_0_open_card_2_points",
        "tier_0_open_card_3_costs",
        "tier_0_open_card_3_returns",
        "tier_0_open_card_3_points",
        "tier_0_hidden_card_costs",
        "tier_0_hidden_card_returns",
        "tier_0_hidden_card_points",
        "tier_0_points",
        "tier_1_open_card_0_costs",
        "tier_1_open_card_0_returns",
        "tier_1_open_card_0_points",
        "tier_1_open_card_1_costs",
        "tier_1_open_card_1_returns",
        "tier_1_open_card_1_points",
        "tier_1_open_card_2_costs",
        "tier_1_open_card_2_returns",
        "tier_1_open_card_2_points",
        "tier_1_open_card_3_costs",
        "tier_1_open_card_3_returns",
        "tier_1_open_card_3_points",
        "tier_1_hidden_card_costs",
        "tier_1_hidden_card_returns",
        "tier_1_hidden_card_points",
        "tier_1_points",
        "tier_2_open_card_0_costs",
        "tier_2_open_card_0_returns",
        "tier_2_open_card_0_points",
        "tier_2_open_card_1_costs",
        "tier_2_open_card_1_returns",
        "tier_2_open_card_1_points",
        "tier_2_open_card_2_costs",
        "tier_2_open_card_2_returns",
        "tier_2_open_card_2_points",
        "tier_2_open_card_3_costs",
        "tier_2_open_card_3_returns",
        "tier_2_open_card_3_points",
        "tier_2_hidden_card_costs",
        "tier_2_hidden_card_returns",
        "tier_2_hidden_card_points",
        "tier_2_points",
        "resources"
    ]

    for expected in expected_keys:
        assert expected in game_vector_indexes

