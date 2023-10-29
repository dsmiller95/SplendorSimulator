import torch
from game_model.AI_model.action_output import ActionOutput, pick_three_choices, discard_choices
from game_model.game import Game
from game_model.game_runner import step_game
from game_model.resource_types import ResourceType
from tests.test_helpers import assert_banks, test_config
from game_model.turn import Turn,Action_Type


def test_fit_best_pick_three():
    game = Game(2, test_config, force_shuffle=False)
    action_out = ActionOutput()
    action_out.action_choice = torch.Tensor([1, 0, 0, 0])

    action_out.pick_three_choice = torch.Tensor(pick_three_choices.get_1_hot_from_n_hot([0, 1, 0, 1, 1]))

    game.available_resources = [1, 1, 0, 1, 1, 4]
    mapped_action = ActionOutput._map_internal(action_out, game, game.players[0])[0]
    assert not (mapped_action is None)

    assert_banks([0, 0, 0, 0, 0], game.players[0].resource_tokens)

    step_result = step_game(game, mapped_action)
    assert step_result == None, step_result

    assert_banks([0, 1, 0, 1, 1], game.players[0].resource_tokens)
    assert_banks([1, 0, 0, 0, 0, 4], game.available_resources)


def test_fit_best_pick_three_invalid_then_pick_two():
    game = Game(2, test_config, force_shuffle=False)
    action_out = ActionOutput()

    action_out.action_choice = torch.Tensor([1, .1, 0, 0])
    action_out.pick_two_choice = torch.Tensor([.1, .2, .3, .4, .5])

    game.available_resources = [0, 4, 0, 0, 1, 4]
    mapped_action = ActionOutput._map_internal(action_out, game, game.players[0])[0]
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
    action_out.action_choice = torch.Tensor([0, 1, 0, 0])
    action_out.resource_token_draw = torch.Tensor([.1, .2, .3, .4, .5])

    game.available_resources = [0, 4, 0, 0, 1, 4]
    mapped_action = ActionOutput._map_internal(action_out, game, game.players[0])[0]
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
    action_out.action_choice = torch.Tensor([.1, 1, 0, 0])
    action_out.resource_token_draw = torch.Tensor([.1, .2, .3, .4, .5])
    
    game.available_resources = [0, 3, 1, 0, 1, 4]
    mapped_action = ActionOutput._map_internal(action_out, game, game.players[0])[0]
    assert not (mapped_action is None)
    assert mapped_action.action_type == Action_Type.TAKE_THREE_UNIQUE

    assert_banks([0, 0, 0, 0, 0], game.players[0].resource_tokens)

    step_result = step_game(game, mapped_action)
    assert step_result == None, step_result

    assert_banks([0, 1, 1, 0, 1, 0], game.players[0].resource_tokens)
    assert_banks([0, 2, 0, 0, 0, 4], game.available_resources)


def test_fit_buy_second_choice_card():
    game = Game(2, test_config, force_shuffle=False)

    action_out = ActionOutput()
    action_out.action_choice = torch.Tensor([0, 1, 2, 0])
    action_out.card_buy = torch.Tensor([
        0, 2, 1, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
    ])
    action_out.resource_token_draw = torch.Tensor([.1, .2, .3, .4, .5])
    
    game.available_resources = [0, 3, 1, 0, 1, 4]
    game.players[0].resource_tokens = [0, 1, 2, 1, 0, 0]
    mapped_action = ActionOutput._map_internal(action_out, game, game.players[0])[0]
    assert not (mapped_action is None)
    assert mapped_action.action_type == Action_Type.BUY_CARD

    assert_banks([0, 1, 2, 1, 0], game.players[0].resource_tokens)

    step_result = step_game(game, mapped_action)
    assert step_result == None, step_result

    assert_banks([0, 0, 0, 0, 0, 0], game.players[0].resource_tokens)
    assert_banks([0, 0, 1, 0, 0], game.players[0].resource_persistent)
    assert_banks([0, 4, 3, 1, 1, 4], game.available_resources)

def test_fit_buy_reserved_card():
    game = Game(2, test_config, force_shuffle=False)
    step_game(game, Turn(Action_Type.RESERVE_CARD, card_index=2))
    step_game(game, Turn(Action_Type.NOOP))
    step_game(game, Turn(Action_Type.TAKE_THREE_UNIQUE, resources_desired=[1, 1, 1, 0, 0]))
    step_game(game, Turn(Action_Type.NOOP))

    action_out = ActionOutput()
    action_out.action_choice = torch.Tensor([0, 1, 2, 0])
    action_out.card_buy = torch.Tensor([
        0, 2, 1, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
    ])
    action_out.reserve_buy = torch.Tensor([10, 20, 0])
    action_out.resource_token_draw = torch.Tensor([.1, .2, .3, .4, .5])
    
    assert not (game.players[0].reserved_cards[0] is None)
    assert game.players[0].reserved_cards[0].returns == ResourceType.EMERALD

    mapped_action = ActionOutput._map_internal(action_out, game, game.players[0])[0]
    assert not (mapped_action is None)
    assert mapped_action.action_type == Action_Type.BUY_CARD

    assert_banks([1, 1, 1, 0, 0, 1], game.players[0].resource_tokens)
    assert_banks([3, 3, 3, 4, 4, 4], game.available_resources)

    step_result = step_game(game, mapped_action)
    assert step_result == None, step_result

    assert_banks([0, 0, 0, 0, 0, 0], game.players[0].resource_tokens)
    assert_banks([0, 1, 0, 0, 0], game.players[0].resource_persistent)
    assert_banks([4, 4, 4, 4, 4, 5], game.available_resources)


def test_fit_buy_card_fails_take_3_instead():
    game = Game(2, test_config, force_shuffle=False)

    action_out = ActionOutput()
    action_out.action_choice = torch.Tensor([.1, 1, 2, 0])
    action_out.card_buy = torch.Tensor([
        0, 2, 1, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
    ])
    action_out.resource_token_draw = torch.Tensor([.1, .2, .3, .4, .5])
    
    game.available_resources = [0, 3, 1, 0, 1, 4]
    game.players[0].resource_tokens = [0, 1, 1, 1, 0, 0]
    mapped_action = ActionOutput._map_internal(action_out, game, game.players[0])[0]
    assert not (mapped_action is None)
    assert mapped_action.action_type == Action_Type.TAKE_THREE_UNIQUE

    assert_banks([0, 1, 1, 1, 0], game.players[0].resource_tokens)

    step_result = step_game(game, mapped_action)
    assert step_result == None, step_result

    assert_banks([0, 2, 2, 1, 1, 0], game.players[0].resource_tokens)
    assert_banks([0, 0, 0, 0, 0], game.players[0].resource_persistent)
    assert_banks([0, 2, 0, 0, 0, 4], game.available_resources)


def test_fit_reserve_card_topdeck():
    game = Game(2, test_config, force_shuffle=False)

    action_out = ActionOutput()
    action_out.action_choice = torch.Tensor([.1, 1, 2, 3])
    action_out.card_buy = torch.Tensor([
        10, 2, 1, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
    ])
    action_out.reserve_buy = torch.Tensor([10, 20, 0])
    action_out.resource_token_draw = torch.Tensor([.1, .2, .3, .4, .5])
    
    game.players[0].resource_tokens = [0, 1, 1, 1, 0, 0]
    mapped_action = ActionOutput._map_internal(action_out, game, game.players[0])[0]
    assert not (mapped_action is None)
    assert mapped_action.action_type == Action_Type.RESERVE_CARD

    assert_banks([0, 1, 1, 1, 0, 0], game.players[0].resource_tokens)

    step_result = step_game(game, mapped_action)
    assert step_result == None, step_result

    assert_banks([0, 1, 1, 1, 0, 1], game.players[0].resource_tokens)    
    assert_banks([0, 0, 0, 0, 0], game.players[0].resource_persistent)

    
    assert not (game.players[0].reserved_cards[0] is None)
    assert game.players[0].reserved_cards[0].returns == ResourceType.ONYX


def test_fit_reserve_card_second_choice():
    game = Game(2, test_config, force_shuffle=False)
    game.take_card_by_index(1)
    game.take_card_by_index(1)

    action_out = ActionOutput()
    action_out.action_choice = torch.Tensor([.1, 1, 2, 3])
    action_out.card_buy = torch.Tensor([
        0, 2, 1, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
    ])
    action_out.reserve_buy = torch.Tensor([10, 20, 0])
    action_out.resource_token_draw = torch.Tensor([.1, .2, .3, .4, .5])
    
    game.players[0].resource_tokens = [0, 1, 1, 1, 0, 0]
    mapped_action = ActionOutput._map_internal(action_out, game, game.players[0])[0]
    assert not (mapped_action is None)
    assert mapped_action.action_type == Action_Type.RESERVE_CARD

    assert_banks([0, 1, 1, 1, 0, 0], game.players[0].resource_tokens)

    step_result = step_game(game, mapped_action)
    assert step_result == None, step_result

    assert_banks([0, 1, 1, 1, 0, 1], game.players[0].resource_tokens)    
    assert_banks([0, 0, 0, 0, 0], game.players[0].resource_persistent)

    
    assert not (game.players[0].reserved_cards[0] is None)
    assert game.players[0].reserved_cards[0].returns == ResourceType.EMERALD

def test_fit_reserve_card_fail_full_reserve_take_3_instead():
    game = Game(2, test_config, force_shuffle=False)
    step_game(game, Turn(Action_Type.RESERVE_CARD, card_index=2+5))
    step_game(game, Turn(Action_Type.NOOP))
    step_game(game, Turn(Action_Type.RESERVE_CARD, card_index=3+5))
    step_game(game, Turn(Action_Type.NOOP))
    step_game(game, Turn(Action_Type.RESERVE_CARD, card_index=4+5))
    step_game(game, Turn(Action_Type.NOOP))

    action_out = ActionOutput()
    action_out.action_choice = torch.Tensor([1, 0.1, 0.1, 3])
    action_out.card_buy = torch.Tensor([
        0, 2, 1, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
    ])
    action_out.reserve_buy = torch.Tensor([10, 20, 0])
    action_out.pick_three_choice = torch.Tensor(pick_three_choices.get_1_hot_from_n_hot([0, 0, 1, 1, 1]))

    mapped_action = ActionOutput._map_internal(action_out, game, game.players[0])[0]
    assert not (mapped_action is None)
    assert mapped_action.action_type == Action_Type.TAKE_THREE_UNIQUE

    assert_banks([0, 0, 0, 0, 0, 3], game.players[0].resource_tokens)
    assert_banks([4, 4, 4, 4, 4, 2], game.available_resources)

    step_result = step_game(game, mapped_action)
    assert step_result == None, step_result

    assert_banks([4, 4, 3, 3, 3, 2], game.available_resources)
    assert_banks([0, 0, 1, 1, 1, 3], game.players[0].resource_tokens)    
    assert_banks([0, 0, 0, 0, 0], game.players[0].resource_persistent)

    assert len(game.players[0].reserved_cards) == 3


def test_fit_pick_3_discard_3():
    game = Game(2, test_config, force_shuffle=False)
    action_out = ActionOutput()
    action_out.action_choice = torch.Tensor([1, 0, 0, 0])

    action_out.pick_three_choice       = torch.Tensor(pick_three_choices.get_1_hot_from_n_hot([0, 0, 1, 1, 1]))
    action_out.discard_combination_choice = torch.Tensor(discard_choices.get_1_hot_from_n_hot([1, 0, 1, 0, 1, 0]))

    game.available_resources = [1, 1, 1, 1, 1, 4]
    game.players[0].resource_tokens = [1, 2, 2, 2, 2, 1]

    mapped_action = ActionOutput._map_internal(action_out, game, game.players[0])[0]
    assert not (mapped_action is None)
    assert mapped_action.action_type == Action_Type.TAKE_THREE_UNIQUE


    step_result = step_game(game, mapped_action)
    assert step_result == None, step_result

    assert_banks([0, 2, 2, 3, 2, 1], game.players[0].resource_tokens)
    assert_banks([2, 1, 1, 0, 1, 4], game.available_resources)


def test_fit_pick_3_discard_other_3():
    game = Game(2, test_config, force_shuffle=False)
    action_out = ActionOutput()
    action_out.action_choice = torch.Tensor([1, 0, 0, 0])
    action_out.resource_token_draw = torch.Tensor([.1, .2, .3, .4, .5])
    action_out.discard_amounts = torch.Tensor([.9, 1.1, 0.02, .01, 2.2, .7])

    action_out.pick_three_choice       = torch.Tensor(pick_three_choices.get_1_hot_from_n_hot([0, 0, 1, 1, 1]))
    action_out.discard_combination_choice = torch.Tensor(discard_choices.get_1_hot_from_n_hot([1, 1, 0, 0, 0, 1]))

    game.players[0].resource_tokens = [1, 2, 2, 2, 2, 1]
    game.available_resources = [1, 1, 1, 1, 1, 4]

    mapped_action = ActionOutput._map_internal(action_out, game, game.players[0])[0]
    assert not (mapped_action is None)
    assert mapped_action.action_type == Action_Type.TAKE_THREE_UNIQUE

    step_result = step_game(game, mapped_action)
    assert step_result == None, step_result

    assert_banks([0, 1, 3, 3, 3, 0], game.players[0].resource_tokens)
    assert_banks([2, 2, 0, 0, 0, 5], game.available_resources)

