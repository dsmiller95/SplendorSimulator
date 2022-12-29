import torch
from game_data.game_config_data import GameConfigData
from game_model.game import Game
from game_model.AI_model.model import SplendidSplendorModel
from game_model.AI_model.gamestate_input import GamestateInputVector
from game_model.AI_model.action_output import ActionOutput

game = Game()

n_hidden_layers = 3
hidden_layer_width = 20
input_shape_dict = None
output_shape_dict = None
model = SplendidSplendorModel()