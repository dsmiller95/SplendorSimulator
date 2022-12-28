
from game_model.AI_model.action_output import ActionOutput
from game_model.AI_model.maps import map_from_AI_output
from game_model.game import Game
from game_model.game_runner import step_game
from tests.test_helpers import assert_banks, test_config
from game_model.turn import Turn,Action_Type


def test_fit_best_pick_three():
    game = Game(2, test_config, force_shuffle=False)
    action_out = ActionOutput()
    action_out.action_choice = [1, 0, 0, 0]
    action_out.resource_token_draw = [.1, .2, .3, .4, .5]

    game.available_resources = [1, 1, 0, 1, 1, 4]
    mapped_action = map_from_AI_output(action_out, game, game.players[0])
    assert not (mapped_action is None)

    assert_banks([0, 0, 0, 0, 0], game.players[0].resource_tokens)

    step_result = step_game(game, mapped_action)
    assert step_result == None, step_result

    assert_banks([0, 1, 0, 1, 1], game.players[0].resource_tokens)
    assert_banks([1, 0, 0, 0, 0, 4], game.available_resources)


def test_fit_best_pick_three_invalid_then_pick_two():
    game = Game(2, test_config, force_shuffle=False)
    action_out = ActionOutput()
    action_out.action_choice = [1, .1, 0, 0]
    action_out.resource_token_draw = [.1, .2, .3, .4, .5]

    game.available_resources = [0, 4, 0, 0, 1, 4]
    mapped_action = map_from_AI_output(action_out, game, game.players[0])
    assert not (mapped_action is None)
    assert mapped_action.action_type == Action_Type.TAKE_TWO

    assert_banks([0, 0, 0, 0, 0], game.players[0].resource_tokens)

    step_result = step_game(game, mapped_action)
    assert step_result == None, step_result

    assert_banks([0, 2, 0, 0, 0, 0], game.players[0].resource_tokens)
    assert_banks([0, 2, 0, 0, 1, 4], game.available_resources)

def test_fit_pick_two():
    game = Game(2, test_config, force_shuffle=False)

    action_out = ActionOutput()
    action_out.action_choice = [0, 1, 0, 0]
    action_out.resource_token_draw = [.1, .2, .3, .4, .5]

    game.available_resources = [0, 4, 0, 0, 1, 4]
    mapped_action = map_from_AI_output(action_out, game, game.players[0])
    assert not (mapped_action is None)
    assert mapped_action.action_type == Action_Type.TAKE_TWO

    assert_banks([0, 0, 0, 0, 0], game.players[0].resource_tokens)

    step_result = step_game(game, mapped_action)
    assert step_result == None, step_result

    assert_banks([0, 2, 0, 0, 0, 0], game.players[0].resource_tokens)
    assert_banks([0, 2, 0, 0, 1, 4], game.available_resources)

def test_fit_pick_two_invalid_then_pick_three():
    game = Game(2, test_config, force_shuffle=False)

    action_out = ActionOutput()
    action_out.action_choice = [.1, 1, 0, 0]
    action_out.resource_token_draw = [.1, .2, .3, .4, .5]
    
    game.available_resources = [0, 3, 1, 0, 1, 4]
    mapped_action = map_from_AI_output(action_out, game, game.players[0])
    assert not (mapped_action is None)
    assert mapped_action.action_type == Action_Type.TAKE_THREE_UNIQUE

    assert_banks([0, 0, 0, 0, 0], game.players[0].resource_tokens)

    step_result = step_game(game, mapped_action)
    assert step_result == None, step_result

    assert_banks([0, 1, 1, 0, 1, 0], game.players[0].resource_tokens)
    assert_banks([0, 2, 0, 0, 0, 4], game.available_resources)
