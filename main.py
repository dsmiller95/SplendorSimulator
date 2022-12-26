from game_data.game_config_data import GameConfigData
from game_model.game_runner import step_game
from game_model.turn import Turn, Action_Type
from game_model.game import Game

print("hello there")

game_config = GameConfigData.read_file("./game_data/cards.csv")
game = Game(player_count=3, game_config=game_config)

print(game.describe_common_state())
first_action = Turn(
    action_type=Action_Type.TAKE_THREE_UNIQUE,
    resources=[0, 1, 1, 0, 1]
    )
step_game(game, first_action)
print(game.describe_common_state())