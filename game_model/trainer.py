import torch
from game_data.game_config_data import GameConfigData
from game_model.game import Game
from game_model.AI_model.model import SplendidSplendorModel
from game_model.AI_model.gamestate_input import GamestateInputVector
from game_model.AI_model.maps import map_to_AI_input
from game_model.AI_model.action_output import ActionOutput
from game_model.AI_model.maps import map_from_AI_output
from game_model.game_runner import step_game
from game_model.turn import Action_Type, Turn

def train():
    game_config = GameConfigData.read_file("./game_data/cards.csv")
    game = Game(player_count=4, game_config=game_config)

    AI_input = map_to_AI_input(game)

    n_hidden_layers = 5
    hidden_layer_width = 100
    input_shape_dict = AI_input
    output_shape_dict = ActionOutput().in_dict_form()
    model = SplendidSplendorModel(input_shape_dict,output_shape_dict,hidden_layer_width,n_hidden_layers)

    for tries in range(20):
        AI_input = map_to_AI_input(game)
        forward_result = model.forward(AI_input)

        next_action = _get_next_action_from_forward_result(forward_result, game)
        if not isinstance(next_action, Turn):
            print(next_action)
            return
        print("next ai action:")
        next_player = game.get_current_player()
        print(next_action.describe_state(game, next_player))
        step_status = step_game(game, next_action)
        print("taken by player " + str(game.get_current_player_index()))
        print(next_player.describe_state())

        if not (step_status is None):
            print("ai turns failed with: " + step_status)
            return
    print("done with game?")
    print(game.describe_common_state())
    print("=====================================================")
    
def _get_next_action_from_forward_result(forward: dict[str, torch.Tensor], game: Game) -> Turn | str:
        next_action = ActionOutput()
        next_action.map_dict_into_self(forward)
        return map_from_AI_output(next_action, game, game.get_current_player())

def train():
    pass

def reward():
    pass

def play_single_game():
    pass