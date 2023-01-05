import torch
import random
from sys import getsizeof
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
    epsilon = 0.9 #how often to pick the maximum-Q-valued action
    memory_length = 1000000 #number of rounds to play of the game
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
            
            # Store game state in memory
            turn_memory[0] = game

            # Map game state to AI input
            ai_input = map_to_AI_input(game)

            # Get model's predicted action
            Q = target_model.forward(ai_input) #Q values == expected reward for an action taken
            
            # Store Q-value dict in memory
            turn_memory[1] = Q

            # Apply epsilon greedy function to somewhat randomize the action picks for exploration
            Q = _epsilon_greedy(Q,epsilon)

            # Pick the highest Q-valued action that works in the game
            next_action = _get_next_action_from_forward_result(Q, game) 
        
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
            
            print(len(replay_memory),getsizeof(replay_memory))


                

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

def _epsilon_greedy(Q: torch.ModuleDict, epsilon: float):
    '''The epsilon greedy algorithm is supposed to choose the max Q-valued
    action with a probability of epsilon. Otherwise, it will randomly choose
    another possible action. We're going to do this by either allowing the
    Q values to go undisturbed to the action mapper, or by swapping the max
    Q value for a particular action to another position, so that the action
    mapper will pick that action instead.'''
    
    for choice_type in Q:
        choices = Q[choice_type]
        if random.uniform(0,1) > epsilon and len(choices) > 1:
            maxIndex = choices.argmax()

            index_clash = True
            while index_clash:
                swapIndex = random.randint(0, choices.size(0)-1)
                if swapIndex != maxIndex:
                    index_clash = False
            
            choices_copy = choices.clone()
            #swapVal = choices[swapIndex]
            choices_copy[swapIndex] = choices[maxIndex]
            choices_copy[maxIndex] = choices[swapIndex]
            #choices[maxIndex] = swapVal
            Q[choice_type] = choices_copy
    return Q
            