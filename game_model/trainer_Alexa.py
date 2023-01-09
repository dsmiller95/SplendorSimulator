import torch
from torch.utils.data import DataLoader
import random
from game_data.game_config_data import GameConfigData
from game_model.game import Game
from game_model.AI_model.model import SplendidSplendorModel
from game_model.AI_model.gamestate_input import GamestateInputVector
from game_model.AI_model.maps import map_to_AI_input
from game_model.AI_model.action_output import ActionOutput
from game_model.AI_model.maps import map_from_AI_output
from game_model.AI_model.dataloader import BellmanEquationDataSet
from game_model.game_runner import step_game
from game_model.turn import Action_Type, Turn
from game_model.replay_memory import ReplayMemoryEntry
        

def train():
    learning_rate = 0.001 
    gamma = 0.9 #discount factor, how much it cares about future reward vs current reward
                #(0: only current, 1: current and all future states)
    epsilon = 0.9 #how often to pick the maximum-Q-valued action
    memory_length = 1000      #number of rounds to play of the game (not absolute, it will finish a game)
    target_network_update_rate = 10 #number of rounds to play before copying weights from main to target network
    batch_size: int = 3

    # Load game configuration data
    game_config = GameConfigData.read_file("./game_data/cards.csv")

    # Create models
    # Map game state to AI input
    game = Game(player_count=4, game_config=game_config)
    ai_input = map_to_AI_input(game)
    input_shape_dict = ai_input
    output_shape_dict = ActionOutput().in_dict_form()
    target_model = SplendidSplendorModel(input_shape_dict, output_shape_dict, 100, 3)


    def play(target_model):
        # Instantiate memory
        replay_memory: list[ReplayMemoryEntry] = []

        while len(replay_memory) < memory_length:

            # Create game model
            game = Game(player_count=4, game_config=game_config)
            won = False
            while not (won and game.active_index == 0):
                ''' Use target network to play the games'''

                # Map game state to AI input
                ai_input = map_to_AI_input(game)
                
                # Store game state in memory
                player_dict_mem = {'game_state':None,'q_prediction':None,'next_turn_game_state':None,'reward':None,'is_last_turn':None}
                player_dict_mem['game_state'] = ai_input
                #player_mem = ReplayMemoryEntry(ai_input)
                
                #save this game to the last turn of this player's memory
                turns_since_last = game.get_player_num() - 1
                if len(replay_memory) >= turns_since_last:
                    replay_memory[-turns_since_last]['next_turn_game_state'] = ai_input
                    #replay_memory[-turns_since_last].next_turn_game_state = ai_input
                
                # Get model's predicted action
                Q = target_model.forward(ai_input) #Q values == expected reward for an action taken
                
                # Store Q-value dict in memory
                player_dict_mem['q_prediction'] = Q
                #player_mem.q_prediction = Q

                # Apply epsilon greedy function to somewhat randomize the action picks for exploration
                Q = _epsilon_greedy(Q,epsilon)

                # Pick the highest Q-valued action that works in the game
                next_action = _get_next_action_from_forward_result(Q, game) 
                
                original_fitness = game.get_current_player().get_fitness()
                # Play move
                step_status = step_game(game, next_action)
                if not (step_status is None):
                    raise Exception("invalid game step generated, " + step_status)

                fitness_delta = game.get_current_player().get_fitness() - original_fitness

                if game.get_current_player().has_won():
                    step_status = "victory"
                    won = True
                # Get reward for action
                reward = _get_reward(game, step_status, fitness_delta)

                # Store reward in memory
                player_dict_mem['reward'] = torch.as_tensor(reward)
                #player_mem.reward = reward

                player_dict_mem['is_last_turn'] = torch.as_tensor(int(0))
                #Store turn in replay memory
                replay_memory.append(player_dict_mem)
                #replay_memory.append(player_mem)

                print(len(replay_memory))

            ending_state = map_to_AI_input(game)
            for player_index in range(game.get_num_players()):
                if replay_memory[-player_index]['next_turn_game_state'] is None:
                    replay_memory[-player_index]['next_turn_game_state'] = ending_state
                replay_memory[-player_index]['is_last_turn'] = torch.as_tensor(int(1))
                #last_turn_player = replay_memory[-player_index]
                #if last_turn_player.next_turn_game_state is None:
                #    last_turn_player.next_turn_game_state = ending_state
                #last_turn_player.is_last_turn = True

        return replay_memory
        
    def learn(target_model,replay_memory):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = target_model

        # Define loss function and optimizer
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Transfer all the data to the GPU for blazing fast train speed
        if device == torch.device("cuda"):
            for i,turn in enumerate(replay_memory):
                for key in turn['game_state']:
                    replay_memory[i]['game_state'][key] = replay_memory[i]['game_state'][key].to(device)
                for key in turn['q_prediction']:
                    replay_memory[i]['q_prediction'][key] = replay_memory[i]['q_prediction'][key].to(device)
                for key in turn['next_turn_game_state']:
                    replay_memory[i]['next_turn_game_state'][key] = replay_memory[i]['next_turn_game_state'][key].to(device)
                replay_memory[i]['reward'] = replay_memory[i]['reward'].to(device)
                replay_memory[i]['is_last_turn'] = replay_memory[i]['is_last_turn'].to(device)

        model = model.to(device)
        target_model = target_model.to(device)
        dataset = BellmanEquationDataSet(replay_memory,device)
        dataloader = DataLoader(dataset,batch_size,shuffle=False,num_workers=0)

        # Base the target model update rate on how many turns it takes to win
        target_network_update_rate = _avg_turns_to_win(replay_memory)

        for iteration,batch in enumerate(dataloader):
            Q_dicts = model(batch[0])
            print(Q_dicts)
            
            #main network is updated every step, its weights directly updated by the backwards pass
            #target network is updated less often, its weights copied directly from the main net

    for epoch in range(2):
        replay_memory = play(target_model)
        learn(target_model,replay_memory)
        target_model.to(torch.device('cpu'))

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

def _epsilon_greedy(Q: dict[str, torch.Tensor], epsilon: float):
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
            choices_copy[swapIndex] = choices[maxIndex]
            choices_copy[maxIndex] = choices[swapIndex]
            Q[choice_type] = choices_copy
    return Q
            
def _avg_turns_to_win(replay_memory) -> int:
    total_len = len(replay_memory)
    last_round_count = sum([1 if x['is_last_turn'] == 1 else 0 for x in replay_memory])
    return round(total_len/last_round_count)