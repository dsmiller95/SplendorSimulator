
from game_data.game_config_data import GameConfigData
from game_model.game import Game

print("hello there")

from game_server_interface.game_server import app, game_data

game_config = GameConfigData.read_file("./game_data/cards.csv")
game = Game(player_count=4, game_config=game_config)

