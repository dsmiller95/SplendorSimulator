import torch
import random
from game_data.game_config_data import GameConfigData
from game_model.game import Game
from game_model.AI_model.model import SplendidSplendorModel
from game_model.AI_model.gamestate_input import GamestateInputVector
from game_model.AI_model.maps import map_to_AI_input
from game_model.AI_model.action_output import ActionOutput
from game_model.AI_model.maps import map_from_AI_output
from game_model.game_runner import step_game
from game_model.turn import Action_Type, Turn


learning_rate = 0.001 
gamma = 0.9 #discount factor, how much it cares about future reward vs current reward
            #(0: only current, 1: current and all future states)
memory_length = 1000 #number of rounds to play of the game
target_network_update_rate = 10 #number of rounds to play before copying weights from main to target network

# Load game configuration data
game_config = GameConfigData.read_file("./game_data/cards.csv")

# Map game state to AI input
ai_input = map_to_AI_input(game)

# Create model
input_shape_dict = ai_input
output_shape_dict = ActionOutput().in_dict_form()
model = SplendidSplendorModel(input_shape_dict, output_shape_dict, 100, 3)
target_model = SplendidSplendorModel(input_shape_dict, output_shape_dict, 100, 3)

# Define loss function and optimizer
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Instantiate memory
# Each memory instance contains the game state, the turn object, the Q prediction, and the reward
replay_memory: list[list[Game,torch.ModuleDict,float]] = []



while len(replay_memory < memory_length):

    # Create game model
    game = Game(player_count=4, game_config=game_config)
    won = False
    while game not won:
        # Use target network to actually play the game
        # Play move
        # Store game state in memory
        # Store Q-value dict in memory
        # Get reward for action
        # Store reward in memory
        if game.get_current_player().sum_points >= 15

#non-batching
for i,turn in enumerate(replay_memory):
    #main network is updated every step, its weights directly updated by the backwards pass
    #target network is updated less often, its weights copied directly from the main net
    pass
