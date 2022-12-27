from game_data.game_config_data import GameConfigData
from game_model.card import Card
from game_model.game import Game
from game_model.noble import Noble
from game_model.resource_types import ResourceType
from game_model.turn import Turn, Action_Type
from game_model.game_runner import step_game
import random

from utilities.print_utils import stringify_resources

def capital_case(x):
    return x.capitalize()

def test_capital_case():
    assert capital_case('semaphore') == 'Semaphore'

test_config = GameConfigData(
        cards=[
            Card(0, [1, 2, 1, 0, 0], ResourceType.RUBY, 1, 1),
            Card(0, [0, 1, 2, 1, 0], ResourceType.EMERALD, 1, 1),
            Card(0, [0, 0, 1, 2, 1], ResourceType.SAPPHIRE, 0, 1),
            Card(0, [1, 0, 0, 1, 2], ResourceType.DIAMOND, 1, 1),
            Card(0, [2, 1, 0, 0, 1], ResourceType.ONYX, 0, 1),
            
            Card(1, [2, 3, 1, 0, 0], ResourceType.RUBY, 2, 1),
            Card(1, [0, 2, 3, 1, 0], ResourceType.EMERALD, 2, 1),
            Card(1, [0, 0, 2, 3, 1], ResourceType.SAPPHIRE, 2, 1),
            Card(1, [1, 0, 0, 2, 3], ResourceType.DIAMOND, 2, 1),
            Card(1, [3, 1, 0, 0, 2], ResourceType.ONYX, 2, 1),

            Card(2, [2, 4, 1, 1, 0], ResourceType.RUBY, 4, 1),
            Card(2, [0, 2, 4, 1, 1], ResourceType.EMERALD, 4, 1),
            Card(2, [1, 0, 2, 4, 1], ResourceType.SAPPHIRE, 4, 1),
            Card(2, [1, 1, 0, 2, 4], ResourceType.DIAMOND, 4, 1),
            Card(2, [4, 1, 1, 0, 2], ResourceType.ONYX, 4, 1),
        ],
        nobles=[
            Noble([3, 3, 3, 0, 0], 3),
            Noble([0, 3, 3, 3, 0], 4),
            Noble([0, 0, 3, 3, 3], 5),
        ]
    )

def test_game_unshuffled():
    game = Game(2, test_config, force_shuffle=False)
    for i, tier in enumerate(game.open_cards):
        for j, card in enumerate(tier):
            assert card.tier == i
            ## due to the test data setup, every resource type is sequential
            assert card.reward.value == j

def test_game_shuffled():
    random.seed(222)
    game = Game(2, test_config, force_shuffle=True)
    
    for i, tier in enumerate(game.open_cards):
        for j, card in enumerate(tier):
            assert card.tier == i
            ## due to the test data setup, every resource type is sequential
            if card.reward.value != j:
                return
    assert False


def assert_banks(expected: list[int], actual: list[int]):
    for i, x in enumerate(expected):
        assert actual[i] == x, "Expected bank of " + stringify_resources(expected) + ", got " + stringify_resources(actual)

def test_game_gather_and_purchase():
    game = Game(2, test_config, force_shuffle=False)
    turns = [
        Turn(Action_Type.TAKE_THREE_UNIQUE, [1, 1, 1, 0, 0]),
        Turn(Action_Type.TAKE_THREE_UNIQUE, [0, 0, 1, 1, 1]),

        Turn(Action_Type.TAKE_THREE_UNIQUE, [1, 1, 1, 0, 0]),
        Turn(Action_Type.TAKE_THREE_UNIQUE, [0, 0, 1, 1, 1]),

        Turn(Action_Type.BUY_CARD, card_index=1),
        Turn(Action_Type.BUY_CARD, card_index=3),
    ]

    for turn in turns:
        step_result = step_game(game, turn)
        assert step_result == None, step_result
    
    assert_banks([3, 4, 2, 4, 3, 5], game.available_resources)
    
    assert_banks([1, 0, 1, 0, 0], game.players[0].resource_tokens)
    assert_banks([1, 0, 0, 0, 0], game.players[0].resource_persistent)
    assert game.players[0].sum_points == 1
    
    assert_banks([0, 0, 1, 0, 1], game.players[1].resource_tokens)
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