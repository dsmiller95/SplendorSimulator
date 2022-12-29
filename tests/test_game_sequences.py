from game_data.game_config_data import GameConfigData
from game_model.actor import Actor
from game_model.card import Card
from game_model.game import Game
from game_model.noble import Noble
from game_model.resource_types import ResourceType
from game_model.turn import Turn, Action_Type
from game_model.game_runner import step_game
import random
from tests.test_helpers import assert_banks, assert_noble_claimed, assert_noble_n, assert_points, test_config

from utilities.print_utils import stringify_resources

def test_game_unshuffled():
    game = Game(2, test_config, force_shuffle=False)
    for i, tier in enumerate(game.open_cards):
        for j, card in enumerate(tier):
            assert card.tier == i
            ## due to the test data setup, every resource type is sequential
            assert card.returns.value == j
        assert len(game.remaining_cards_by_level[i]) == 1
        assert game.remaining_cards_by_level[i][0].returns == ResourceType.ONYX

def test_game_shuffled():
    random.seed(222)
    game = Game(2, test_config, force_shuffle=True)
    
    for i, tier in enumerate(game.open_cards):
        for j, card in enumerate(tier):
            assert card.tier == i
            ## due to the test data setup, every resource type is sequential
            if card.returns.value != j:
                return
    assert False

def test_game_gather_and_purchase():
    game = Game(2, test_config, force_shuffle=False)
    turns = [
        Turn(Action_Type.TAKE_TWO, [2, 0, 0, 0, 0]),
        Turn(Action_Type.TAKE_TWO, [0, 0, 2, 0, 0]),

        Turn(Action_Type.TAKE_THREE_UNIQUE, [1, 1, 0, 0, 1]),
        Turn(Action_Type.TAKE_THREE_UNIQUE, [0, 1, 1, 1, 0]),

        Turn(Action_Type.BUY_CARD, card_index=1),
        Turn(Action_Type.BUY_CARD, card_index=3),
    ]

    for turn in turns:
        step_result = step_game(game, turn)
        assert step_result == None, step_result
    
    assert_banks([3, 4, 3, 4, 4, 5], game.available_resources)
    
    assert_banks([1, 0, 0, 0, 0], game.players[0].resource_tokens)
    assert_banks([1, 0, 0, 0, 0], game.players[0].resource_persistent)
    assert game.players[0].sum_points == 1
    
    assert_banks([0, 0, 1, 0, 0], game.players[1].resource_tokens)
    assert_banks([0, 0, 1, 0, 0], game.players[1].resource_persistent)
    assert game.players[1].sum_points == 0

def test_attempt_take_two_when_less():
    game = Game(2, test_config, force_shuffle=False)
    turns = [
        Turn(Action_Type.TAKE_TWO, [2, 0, 0, 0, 0]),
        Turn(Action_Type.TAKE_TWO, [2, 0, 0, 0, 0]),
    ]
    
    step_result = step_game(game, turns[0])
    assert step_result == None, step_result
    step_result = step_game(game, turns[0])
    assert step_result == "cannot take two from bank with less than 4 available", step_result

def test_attempt_take_more_than_maximum_token_capacity():
    game = Game(2, test_config, force_shuffle=False)
    single_player_turns = [
        Turn(Action_Type.TAKE_TWO, [2, 0, 0, 0, 0]),
        Turn(Action_Type.TAKE_THREE_UNIQUE, [1, 1, 1, 0, 0]),
        Turn(Action_Type.TAKE_THREE_UNIQUE, [0, 1, 1, 1, 0]),
    ]
    
    for turn in single_player_turns:
        step_result = step_game(game, turn)
        assert step_result == None, step_result
        step_result = step_game(game, Turn(Action_Type.NOOP))
        assert step_result == None, step_result

    step_result = step_game(game, Turn(Action_Type.TAKE_THREE_UNIQUE, [0, 1, 1, 0, 1]))
    assert step_result == None, step_result

    assert_banks([2, 3, 3, 1, 1, 0], game.players[0].resource_tokens)

def test_attempt_buy_topdeck_card():
    game = Game(2, test_config, force_shuffle=False)
    single_player_turns = [
        Turn(Action_Type.TAKE_TWO, [0, 2, 0, 0, 0]),
        Turn(Action_Type.TAKE_THREE_UNIQUE, [1, 1, 1, 0, 0]),
    ]
    
    for turn in single_player_turns:
        step_result = step_game(game, turn)
        assert step_result == None, step_result
        step_result = step_game(game, Turn(Action_Type.NOOP))
        assert step_result == None, step_result

    step_result = step_game(game, Turn(Action_Type.BUY_CARD, card_index=0))
    assert step_result == "Cannot buy card from top deck", step_result

def test_attempt_buy_empty_card_after_clearing_deck():
    game = Game(2, test_config, force_shuffle=False)
    single_player_turns = [
        Turn(Action_Type.TAKE_TWO, [0, 0, 2, 0, 0]),
        Turn(Action_Type.TAKE_THREE_UNIQUE, [0, 1, 0, 1, 1]),
        Turn(Action_Type.BUY_CARD, card_index=3),
        Turn(Action_Type.TAKE_THREE_UNIQUE, [1, 0, 0, 1, 1]),
        Turn(Action_Type.BUY_CARD, card_index=3),
    ]
    
    for turn in single_player_turns:
        step_result = step_game(game, turn)
        assert step_result == None, step_result
        step_result = step_game(game, Turn(Action_Type.NOOP))
        assert step_result == None, step_result

    step_result = step_game(game, Turn(Action_Type.BUY_CARD, card_index=3))
    assert step_result == "no card in reserve at that index, or no card available in game due to card exhaustion", step_result

    step_result = step_game(game, Turn(Action_Type.RESERVE_CARD, card_index=3))
    assert step_result == "no card in reserve at that index, or no card available in game due to card exhaustion", step_result

    step_result = step_game(game, Turn(Action_Type.RESERVE_CARD, card_index=0))
    assert step_result == "no card in reserve at that index, or no card available in game due to card exhaustion", step_result

def run_turns(single_player_turns, game: Game):
    turn_n = 0
    for turn in single_player_turns:
        turn_n += 1
        try:
            if not isinstance(turn, Turn) :
                turn(game)
                continue
            step_result = step_game(game, turn)
            assert step_result == None, step_result
            step_result = step_game(game, Turn(Action_Type.NOOP))
            assert step_result == None, step_result
        except:
            print("on turn " + str(turn_n))
            raise

def test_clone_game_branch_verify():
    game = Game(2, test_config, force_shuffle=False)
    single_player_turns = [
        Turn(Action_Type.TAKE_TWO, [0, 2, 0, 0, 0]),
        Turn(Action_Type.TAKE_THREE_UNIQUE, [1, 1, 1, 0, 0]),
    ]

    run_turns(single_player_turns, game)

    branched_verify = [
        [
            Turn(Action_Type.TAKE_THREE_UNIQUE, [1, 1, 1, 0, 0]),
            lambda cur_game : assert_banks([2, 4, 2, 0, 0, 0], cur_game.players[0].resource_tokens),
        ],
        [
            Turn(Action_Type.TAKE_THREE_UNIQUE, [0, 0, 1, 1, 1]),
            lambda cur_game: assert_banks([1, 3, 2, 1, 1, 0], cur_game.players[0].resource_tokens),
        ],
        [
            Turn(Action_Type.BUY_CARD, card_index=2),
            lambda cur_game: assert_banks([0, 1, 0, 0, 0, 0], cur_game.players[0].resource_tokens),
        ],
        [
            Turn(Action_Type.RESERVE_CARD, card_index=2),
            lambda cur_game: assert_banks([1, 3, 1, 0, 0, 1], cur_game.players[0].resource_tokens),
        ],
    ]
    for i, verify_list in enumerate(branched_verify):
        cloned_game = game.clone()
        try:
            run_turns(verify_list, cloned_game)
        except:
            print("on verify branch " + str(i))
            raise
    


def test_game_gather_and_achieve_nobles():
    game = Game(2, test_config, force_shuffle=False)
    single_player_turns = [
        Turn(Action_Type.TAKE_TWO, [0, 0, 2, 0, 0]),
        Turn(Action_Type.TAKE_THREE_UNIQUE, [0, 1, 0, 1, 1]),
        Turn(Action_Type.BUY_CARD, card_index=3), ## sapphire
        lambda game: assert_banks([0, 0, 0, 0, 1, 0], game.players[0].resource_tokens),
        Turn(Action_Type.TAKE_THREE_UNIQUE, [1, 0, 0, 1, 1]),
        Turn(Action_Type.BUY_CARD, card_index=3), ## onyx
        lambda game: assert_banks([0, 0, 0, 0, 0, 0], game.players[0].resource_tokens),
        Turn(Action_Type.TAKE_TWO, [0, 0, 0, 2, 0]),
        Turn(Action_Type.BUY_CARD, card_index=4), ## diamond
        lambda game: assert_banks([0, 0, 0, 0, 0, 0], game.players[0].resource_tokens),
        Turn(Action_Type.TAKE_TWO, [0, 2, 0, 0, 0]),
        Turn(Action_Type.TAKE_THREE_UNIQUE, [1, 1, 1, 0, 0]),
        Turn(Action_Type.BUY_CARD, card_index=2), ## emerald
        lambda game: assert_banks([0, 1, 1, 0, 0, 0], game.players[0].resource_tokens),
        lambda game: assert_banks([0, 1, 1, 1, 1], game.players[0].resource_persistent),

        Turn(Action_Type.TAKE_THREE_UNIQUE, [1, 1, 1, 0, 0]),
        Turn(Action_Type.BUY_CARD, card_index=5 + 3), ## sapphire
        lambda game: assert_banks([1, 1, 0, 0, 0, 0], game.players[0].resource_tokens),

        Turn(Action_Type.TAKE_THREE_UNIQUE, [1, 1, 0, 1, 0]),
        Turn(Action_Type.BUY_CARD, card_index=5 + 2), ## emerald
        lambda game: assert_banks([0, 0, 0, 1, 0, 0], game.players[0].resource_tokens),
        
        ## turn 22
        Turn(Action_Type.TAKE_THREE_UNIQUE, [1, 0, 0, 1, 1]),
        Turn(Action_Type.BUY_CARD, card_index=5 + 4), ## diamond
        lambda game: assert_banks([1, 0, 0, 0, 1, 0], game.players[0].resource_tokens),

        ## turn 25
        Turn(Action_Type.TAKE_THREE_UNIQUE, [1, 1, 0, 0, 1]),
        Turn(Action_Type.BUY_CARD, card_index=5 + 3), ## onyx
        lambda game: assert_banks([1, 1, 0, 0, 0, 0], game.players[0].resource_tokens),

        lambda game: assert_banks([0, 2, 2, 2, 2], game.players[0].resource_persistent),
        lambda game: assert_points(game.players[0], 10),

        ## turn 30
        Turn(Action_Type.TAKE_THREE_UNIQUE, [1, 1, 0, 0, 1]),
        Turn(Action_Type.BUY_CARD, card_index=10 + 2), ## emerald
        lambda game: assert_banks([0, 0, 0, 0, 1, 0], game.players[0].resource_tokens),

        Turn(Action_Type.TAKE_THREE_UNIQUE, [1, 0, 0, 1, 1]),
        Turn(Action_Type.BUY_CARD, card_index=10 + 2), ## onyx
        lambda game: assert_banks([0, 0, 0, 1, 0, 0], game.players[0].resource_tokens),
        
        Turn(Action_Type.TAKE_THREE_UNIQUE, [1, 0, 1, 1, 0]),
        Turn(Action_Type.BUY_CARD, card_index=10 + 4), ## diamond
        lambda game: assert_banks([0, 0, 1, 0, 0, 0], game.players[0].resource_tokens),

        ## turn 39
        lambda game: assert_banks([0, 3, 2, 3, 3], game.players[0].resource_persistent),
        lambda game: assert_points(game.players[0], 22),
        lambda game: assert_noble_n(game.players[0], 0),
        ## Turn 42
        Turn(Action_Type.TAKE_THREE_UNIQUE, [1, 1, 1, 0, 0]),
    ]

    run_turns(single_player_turns, game)

    branched_verify = [
        [
            lambda game: assert_points(game.players[0], 22),
            Turn(Action_Type.BUY_CARD, card_index=10 + 3), ## sapphire
            
            lambda game: assert_banks([0, 3, 3, 3, 3], game.players[0].resource_persistent),
            lambda game: assert_noble_n(game.players[0], 1),
            lambda game: assert_points(game.players[0], 22 + 4 + 4), ## claimed noble 2
            lambda game: assert_noble_claimed(game.players[0], 17),

            Turn(Action_Type.TAKE_THREE_UNIQUE, [1, 1, 1, 0, 0]),
            lambda game: assert_noble_n(game.players[0], 2),
            lambda game: assert_points(game.players[0], 22 + 4 + 4 + 5), ## claimed noble 3
            lambda game: assert_noble_claimed(game.players[0], 18),
        ],
        [
            lambda game: assert_points(game.players[0], 22),
            Turn(Action_Type.BUY_CARD, card_index=10 + 3, noble_preference=2), ## sapphire
            
            lambda game: assert_banks([0, 3, 3, 3, 3], game.players[0].resource_persistent),
            lambda game: assert_noble_n(game.players[0], 1),
            lambda game: assert_points(game.players[0], 22 + 4 + 5), ## claimed noble 3
            lambda game: assert_noble_claimed(game.players[0], 18),

            Turn(Action_Type.TAKE_THREE_UNIQUE, [1, 1, 1, 0, 0]),
            lambda game: assert_noble_n(game.players[0], 2),
            lambda game: assert_points(game.players[0], 22 + 4 + 5 + 4), ## claimed noble 2
            lambda game: assert_noble_claimed(game.players[0], 17),
        ]
    ]
    for i, verify_list in enumerate(branched_verify):
        cloned_game = game.clone()
        try:
            run_turns(verify_list, cloned_game)
        except:
            print("on verify branch " + str(i))
            raise
    
def test_discard_tokens_when_max_reached():
    game = Game(2, test_config, force_shuffle=False)
    single_player_turns = [
        Turn(Action_Type.TAKE_THREE_UNIQUE, [0, 0, 1, 1, 1]),
        Turn(Action_Type.TAKE_THREE_UNIQUE, [1, 1, 1, 0, 0]),
        Turn(Action_Type.TAKE_THREE_UNIQUE, [1, 1, 1, 0, 0]),
        lambda cur_game : assert_banks([2, 2, 3, 1, 1, 0], cur_game.players[0].resource_tokens),
    ]

    run_turns(single_player_turns, game)

    branched_verify = [
        [
            Turn(Action_Type.TAKE_THREE_UNIQUE, [0, 0, 1, 1, 1], discard_preference_levels=[0.4, 0.09, 0.3, 0.07, 0.06, 0.05]),
            lambda cur_game : assert_banks([1, 2, 3, 2, 2, 0], cur_game.players[0].resource_tokens),
        ],
        [
            Turn(Action_Type.TAKE_THREE_UNIQUE, [0, 0, 1, 1, 1], discard_preference_levels=[1.4, 0.09, 0.08, 0.07, 0.06, 0.05]),
            lambda cur_game : assert_banks([0, 2, 4, 2, 2, 0], cur_game.players[0].resource_tokens),
        ],
        [
            Turn(Action_Type.TAKE_THREE_UNIQUE, [0, 0, 1, 1, 1], discard_preference_levels=[0.02, 0.09, 0.08, 0.2, 0.3, 0.4]),
            lambda cur_game : assert_banks([2, 2, 4, 1, 1, 0], cur_game.players[0].resource_tokens),
        ],
        [
            Turn(Action_Type.TAKE_THREE_UNIQUE, [0, 0, 1, 1, 1], discard_preference_levels=[.06, .3, .04, .03, .02, .4]),
            lambda cur_game : assert_banks([1, 1, 4, 2, 2, 0], cur_game.players[0].resource_tokens),
        ],
        [
            Turn(Action_Type.TAKE_THREE_UNIQUE, [0, 0, 1, 1, 1], discard_preference_levels=[.06, .05, .04, .03, .02, 1.4]),
            lambda cur_game : assert_banks([1, 1, 4, 2, 2, 0], cur_game.players[0].resource_tokens),
        ],
    ]
    for i, verify_list in enumerate(branched_verify):
        cloned_game = game.clone()
        try:
            run_turns(verify_list, cloned_game)
        except:
            print("on verify branch " + str(i))
            raise
    