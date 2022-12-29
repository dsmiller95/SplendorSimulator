#import torch
from game_data.game_config_data import GameConfigData
from game_model.game import Game
from game_model.AI_model.model import SplendidSplendorModel
from game_model.AI_model.gamestate_input import GamestateInputVector
from game_model.AI_model.maps import map_to_AI_input
from game_model.AI_model.action_output import ActionOutput
from game_model.AI_model.maps import map_from_AI_output

def train():
    game_config = GameConfigData.read_file("./game_data/cards.csv")
    game = Game(player_count=4, game_config=game_config)

    AI_input = map_to_AI_input(game)

    n_hidden_layers = 5
    hidden_layer_width = 100
    input_shape_dict = AI_input
    output_shape_dict = ActionOutput().in_dict_form()
    model = SplendidSplendorModel(input_shape_dict,output_shape_dict,hidden_layer_width,n_hidden_layers)
    forward_result = model.forward(AI_input)
    internal_len = len(forward_result['action_choice'])

    print(forward_result)