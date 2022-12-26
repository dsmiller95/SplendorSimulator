from game_data.game_config_data import GameConfigData
from game_model.action import Turn, Action_Type
from game_model.game import Game
from game_model.resource_types import ResourceType


print("hello there")

game_config = GameConfigData.read_file("./game_data/cards.csv")
game = Game(player_count=3, game_config=game_config)
print(game.describe_common_state())
first_action = Turn(
    action_type=Action_Type.TAKE_TWO,
    resources=[ResourceType.DIAMOND, ResourceType.ONYX, ResourceType.EMERALD]
    )
game.step_game(first_action)