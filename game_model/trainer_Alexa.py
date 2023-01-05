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

def train():
    learning_rate = 0.001 
    gamma = 0.9 #discount factor, how much it cares about future reward vs current reward
                #(0: only current, 1: current and all future states)
    memory_length = 1000 #number of rounds to play of the game
    target_network_update_rate = 10 #number of rounds to play before copying weights from main to target network

    # Load game configuration data
    game_config = GameConfigData.read_file("./game_data/cards.csv")

    # Create models
    # Map game state to AI input
    game = Game(player_count=4, game_config=game_config)
    ai_input = map_to_AI_input(game)
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
    turn_memory:        list[Game,torch.ModuleDict,float]  = [None,None,None]

    while len(replay_memory) < memory_length:

        # Create game model
        game = Game(player_count=4, game_config=game_config)
        won = False
        while not won and len(replay_memory) < memory_length:
            ''' Use target network to play the games'''
            
            # Map game state to AI input
            ai_input = map_to_AI_input(game)

            # Get model's predicted action
            Q = target_model.forward(ai_input) #Q values == expected reward for an action taken
            next_action = _get_next_action_from_forward_result(Q, game) #this should pick the highest Q-valued action
            
            # Store game state in memory
            turn_memory[0] = game
            # Store Q-value dict in memory
            turn_memory[1] = Q
            
            

            original_fitness = game.get_current_player().get_fitness()
            # Play move
            step_game(game, next_action)

            fitness_delta = game.get_current_player().get_fitness() - original_fitness

            step_status = None # Win/Lose condition descriptor
            if game.get_current_player().has_won():
                step_status = "victory"
                won = True

            # Get reward for action
            reward = _get_reward(game, step_status, fitness_delta)

            # Store reward in memory
            turn_memory[2] = reward

            #Store turn in replay memory
            replay_memory.append(turn_memory)


                

    #non-batching
    for i,turn in enumerate(replay_memory):
        #main network is updated every step, its weights directly updated by the backwards pass
        #target network is updated less often, its weights copied directly from the main net
        pass


def _get_next_action_from_forward_result(forward: dict[str, torch.Tensor], game: Game) -> Turn | str:
    """Get the next action from the model's forward pass result."""
    next_action = ActionOutput()
    next_action.map_dict_into_self(forward)
    return map_from_AI_output(next_action, game, game.get_current_player())

def _get_reward(game: Game, step_status: str, fitness_delta: float) -> float:
    """Determine the reward for the current game state."""
    # If the game ended, return a large negative or positive reward
    if step_status == "game_over":
        return -100.0
    if step_status == "victory":
        return 100.0

    # Otherwise, return a small positive reward for making progress based on fitness
    return fitness_delta