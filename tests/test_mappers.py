
from game_model.AI_model.maps import map_from_AI_output
from game_model.game import Game
from game_model.game_runner import step_game
from tests.test_helpers import assert_banks, test_config


def test_fit_best_pick_three():
    game = Game(2, test_config, force_shuffle=False)
    output_vector = \
        [1, 0, 0, 0] + \
        ([0]*5*3) + \
        [0] + \
        [.1, .2, .3, .4, .5] + \
        [0] + [0] + \
        [0] * 6
    game.available_resources = [1, 1, 0, 1, 1, 4]
    mapped_action = map_from_AI_output(output_vector, game, game.players[0])
    assert not (mapped_action is None)

    assert_banks([0, 0, 0, 0, 0], game.players[0].resource_tokens)

    step_result = step_game(game, mapped_action)
    assert step_result == None, step_result

    assert_banks([0, 1, 0, 1, 1], game.players[0].resource_tokens)
    assert_banks([1, 0, 0, 0, 0, 4], game.available_resources)
