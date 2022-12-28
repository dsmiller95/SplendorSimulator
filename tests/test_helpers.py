
from game_data.game_config_data import GameConfigData
from game_model.actor import Actor
from game_model.card import Card
from game_model.noble import Noble
from game_model.resource_types import ResourceType
from utilities.print_utils import stringify_resources


test_config = GameConfigData(
        cards=[
            Card(0, [2, 1, 0, 0, 1], ResourceType.RUBY, 1, 1),
            Card(0, [1, 2, 1, 0, 0], ResourceType.EMERALD, 1, 2),
            Card(0, [0, 1, 2, 1, 0], ResourceType.SAPPHIRE, 0, 3),
            Card(0, [0, 0, 1, 2, 1], ResourceType.DIAMOND, 1, 4),
            Card(0, [1, 0, 0, 1, 2], ResourceType.ONYX, 0, 5),
            
            Card(1, [3, 1, 0, 0, 2], ResourceType.RUBY, 2, 6),
            Card(1, [2, 3, 1, 0, 0], ResourceType.EMERALD, 2, 7),
            Card(1, [0, 2, 3, 1, 0], ResourceType.SAPPHIRE, 2, 8),
            Card(1, [0, 0, 2, 3, 1], ResourceType.DIAMOND, 2, 9),
            Card(1, [1, 0, 0, 2, 3], ResourceType.ONYX, 2, 10),

            Card(2, [4, 1, 1, 0, 2], ResourceType.RUBY, 4, 11),
            Card(2, [2, 4, 1, 1, 0], ResourceType.EMERALD, 4, 12),
            Card(2, [0, 2, 4, 1, 1], ResourceType.SAPPHIRE, 4, 13),
            Card(2, [1, 0, 2, 4, 1], ResourceType.DIAMOND, 4, 14),
            Card(2, [1, 1, 0, 2, 4], ResourceType.ONYX, 4, 15),
        ],
        nobles=[
            Noble([3, 3, 3, 0, 0], 3, 16),
            Noble([0, 3, 3, 3, 0], 4, 17),
            Noble([0, 0, 3, 3, 3], 5, 18),
        ]
    )

def assert_banks(expected: list[int], actual: list[int]):
    for i, x in enumerate(expected):
        assert actual[i] == x, "Expected bank of " + stringify_resources(expected) + ", got " + stringify_resources(actual)

def assert_points(player: Actor, points: int):
    assert player.sum_points == points
def assert_noble_n(player: Actor, noble_num: int):
    assert len(player.claimed_nobles) == noble_num
def assert_noble_claimed(player: Actor, noble_id: int):
    assert noble_id in [x.id for x in player.claimed_nobles if not (x is None)]
