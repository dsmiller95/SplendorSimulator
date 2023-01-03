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


n_games = 1
learning_rate = 0.001
gamma = 0.9

# Load game configuration data
game_config = GameConfigData.read_file("./game_data/cards.csv")

# Create game model
game = Game(player_count=4, game_config=game_config)
# Map game state to AI input
ai_input = map_to_AI_input(game)

# Create model
input_shape_dict = ai_input
output_shape_dict = ActionOutput().in_dict_form()
model = SplendidSplendorModel(input_shape_dict, output_shape_dict, 100, 3)


# Define loss function and optimizer
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#for n epochs
    #while game not won
        #for each turn
            #for each player
                #play move
                #store game state in memory
                #store Q-value vector in memory
                #get action mask
                #mask Q-value vector
                #turn masked Q-value vector into an action choice
                #get reward for action
                #store reward in memory

    #non-batching
    #for each turn
        #for each player
            #get loss from predicted reward and actual reward
            #backprop loss, update weights


'''
                                                                     Q
                                                                   ┌───┐
     gradient flows    ──►                                action?──┤8.6│
                                                                   │2.3│
gradient can't flow    -/►                            sub-action?──┤1.9│
                                                                   │9.8│
                                                      sub-action?──┤3.4│
                                                                   │2.2│
                                                                   └───┘




       ┌─/►rules─────────/────┐
       │                      │
       │                 Q    ▼
       │               ┌───┐ ┌─┐ ┌───┐
       │               │8.6│ │1│ │8.6│
       │               │2.3│ │1│ │2.3│
  gamestate1-/►model──►│1.9│.│1│=│1.9│
                       │9.8│ │1│ │9.8│
                       │3.4│ │1│ │3.4│
                       │2.2│ │1│ │2.2│
                       └───┘ └─┘ └───┘

       ┌─/►rules─────────/────┐
       │                      │
       │                 Q    ▼
       │               ┌───┐ ┌─┐ ┌───┐
       │               │0.3│ │0│ │0.0│
       │               │6.4│ │1│ │6.4│
  gamestate2-/►model──►│1.3│.│0│=│0.0│
                       │4.3│ │0│ │0.0│
                       │5.6│ │1│ │5.6│
                       │8.9│ │1│ │8.9│
                       └───┘ └─┘ └───┘
'''