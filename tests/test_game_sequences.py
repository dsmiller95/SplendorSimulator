from game_data.game_config_data import GameConfigData
from game_model.card import Card
from game_model.game import Game
from game_model.noble import Noble
from game_model.resource_types import ResourceType
import random

def capital_case(x):
    return x.capitalize()

def test_capital_case():
    assert capital_case('semaphore') == 'Semaphore'

test_config = GameConfigData(
        cards=[
            Card(0, [1, 2, 1, 0, 0], ResourceType.RUBY, 0, 1),
            Card(0, [0, 1, 2, 1, 0], ResourceType.EMERALD, 0, 1),
            Card(0, [0, 0, 1, 2, 1], ResourceType.SAPPHIRE, 0, 1),
            Card(0, [1, 0, 0, 1, 2], ResourceType.DIAMOND, 0, 1),
            Card(0, [2, 1, 0, 0, 1], ResourceType.ONYX, 0, 1),
            
            Card(1, [2, 3, 1, 0, 0], ResourceType.RUBY, 1, 1),
            Card(1, [0, 2, 3, 1, 0], ResourceType.EMERALD, 1, 1),
            Card(1, [0, 0, 2, 3, 1], ResourceType.SAPPHIRE, 1, 1),
            Card(1, [1, 0, 0, 2, 3], ResourceType.DIAMOND, 1, 1),
            Card(1, [3, 1, 0, 0, 2], ResourceType.ONYX, 1, 1),

            Card(2, [2, 4, 1, 1, 0], ResourceType.RUBY, 3, 1),
            Card(2, [0, 2, 4, 1, 1], ResourceType.EMERALD, 3, 1),
            Card(2, [1, 0, 2, 4, 1], ResourceType.SAPPHIRE, 3, 1),
            Card(2, [1, 1, 0, 2, 4], ResourceType.DIAMOND, 3, 1),
            Card(2, [4, 1, 1, 0, 2], ResourceType.ONYX, 3, 1),
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


# def test_game_gather_and_purchase():
#     game = Game(2, test_config, force_shuffle=False)
#     turns = [
#         Turn()
#     ]
#     for i, tier in enumerate(game.open_cards):
#         for j, card in enumerate(tier):
#             assert card.tier == i
#             ## due to the test data setup, every resource type is sequential
#             assert card.reward.value == j