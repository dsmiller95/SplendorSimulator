import torch
from torch.utils.data import DataLoader
import random
from math import ceil
from game_data.game_config_data import GameConfigData
from game_model.AI_model.reward import Reward
from game_model.game import Game
from game_model.AI_model.model import SplendidSplendorModel
from game_model.AI_model.gamestate_input import GamestateInputVector
from game_model.AI_model.action_output import ActionOutput
from game_model.AI_model.dataloader import BellmanEquationDataSet
from game_model.game_runner import step_game
from game_model.turn import Action_Type, Turn
from game_model.replay_memory import ReplayMemoryEntry
        

# Hyperparameters
learning_rate: float = 0.00001 
gamma: float = 0.9 #discount factor, how much it cares about future reward vs current reward
            #(0: only current, 1: current and all future states)
epsilon: float = 0.95 #how often to pick the maximum-Q-valued action
memory_length: int = 10000      #number of rounds to play of the game (not absolute, it will finish a game)
batch_size: int = 500
epochs: int = 100 #how many play->learn cycles to run

def train():
    # Load game configuration data
    game_config = GameConfigData.read_file("./game_data/cards.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create models
    # Map game state to AI input
    game = Game(player_count=4, game_config=game_config)
    ai_input = GamestateInputVector.map_to_AI_input(game)
    input_shape_dict = ai_input
    output_shape_dict = ActionOutput().in_dict_form()
    target_model = SplendidSplendorModel(input_shape_dict, output_shape_dict, 1024, 10)
    target_model = target_model.to(device)


    def play(target_model) -> list[ReplayMemoryEntry]:
        # Instantiate memory
        replay_memory: list[ReplayMemoryEntry] = []
        while len(replay_memory) < memory_length:
            len_left_in_replay: int = memory_length - len(replay_memory)
            replay_memory += play_single_game(target_model,len_left_in_replay)
        return replay_memory
    
    def play_single_game(target_model,len_left_in_replay: int) -> list[ReplayMemoryEntry]:
        replay_memory: list[ReplayMemoryEntry] = []
        game = Game(player_count=4, game_config=game_config)
        won = False
        while not (won and game.active_index == 0):
            ''' Use target network to play the games'''

            # Map game state to AI input
            ai_input = GamestateInputVector.map_to_AI_input(game)
            
            # Store game state in memory
            player_mem = ReplayMemoryEntry(ai_input)
            
            #save this game to the last turn of this player's memory
            turns_since_last = game.get_player_num() - 1
            if len(replay_memory) >= turns_since_last:
                replay_memory[-turns_since_last].next_turn_game_state = ai_input
            
            # Get model's predicted action
            ai_input = {key:ai_input[key].to(device) for key in ai_input}
            target_model.eval()
            with torch.no_grad(): #no need to save gradients since we're not backpropping, this saves a lot of time/memory
                Q = target_model.forward(ai_input) #Q values == expected reward for an action taken
                Q = {key:Q[key].to(torch.device('cpu')) for key in Q}

            # Apply epsilon greedy function to somewhat randomize the action picks for exploration
            Q = _epsilon_greedy(Q,epsilon)

            # Pick the highest Q-valued action that works in the game
            (next_action, chosen_Action) = _get_next_action_from_forward_result(Q, game) 

            player_mem.taken_action = chosen_Action

            original_fitness = Reward(game)
            # Play move
            step_status = step_game(game, next_action)
            if not (step_status is None):
                raise Exception("invalid game step generated, " + step_status)

            # Get reward from state transition, and convert to dict form 
            reward = Reward(game).base_reward - original_fitness.base_reward
            reward_as_dict = {choice:(reward * player_mem.taken_action[choice]) for choice in player_mem.taken_action}

            # Store reward in memory
            player_mem.reward_new = reward_as_dict


            if game.get_current_player().qualifies_to_win():
                won = True

            #Store turn in replay memory
            replay_memory.append(player_mem)

            if len(replay_memory) == len_left_in_replay:
                break
        
        ending_state = GamestateInputVector.map_to_AI_input(game)
        for player_index in range(game.get_num_players()):
            last_turn_player = replay_memory[-player_index]
            if last_turn_player.next_turn_game_state is None:
                last_turn_player.next_turn_game_state = ending_state
            last_turn_player.is_last_turn = torch.as_tensor(int(1))
        print("Played a full game. Working replay mem length: " + str(len(replay_memory)))
        return replay_memory

    def learn(target_model,replay_memory: list[ReplayMemoryEntry]):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = target_model

        # Base the target model update rate on how many turns it takes to win
        target_network_update_rate: int = _avg_turns_to_win(replay_memory)

        model.train()

        # Define loss function and optimizer
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Transfer all the data to the GPU for blazing fast train speed
        if device == torch.device("cuda"):
            for turn in replay_memory:
                for key in turn.game_state:
                    turn.game_state[key] = turn.game_state[key].to(device)
                for key in turn.taken_action:
                    turn.taken_action[key] = turn.taken_action[key].to(device)
                for key in turn.next_turn_game_state:
                    turn.next_turn_game_state[key] = turn.next_turn_game_state[key].to(device)
                for key in turn.reward_new:
                    turn.reward_new[key] = turn.reward_new[key].to(device)
                turn.is_last_turn = turn.is_last_turn.to(device)

        model = model.to(device)
        target_model = target_model.to(device)
        dataset = BellmanEquationDataSet(replay_memory,device)
        dataloader = DataLoader(dataset,batch_size=target_network_update_rate,shuffle=True,num_workers=0)

        for iteration,batch in enumerate(dataloader):
            Q_dicts = model(batch[0]) ## dict of tensors of size batch x orig size
            next_Q_dicts = target_model(batch[1])
            rewards = batch[2]
            is_last_turns = batch[3]
            error = {}
            optimizer.zero_grad()
            avg_loss: float = 0.0
            iter_count: int = 0
            for i,key in enumerate(Q_dicts):
                target = target_Q(Q_dicts[key],next_Q_dicts[key],rewards[key],gamma,is_last_turns)
                loss = loss_fn(Q_dicts[key],target)
                loss.backward(retain_graph=True) #propagate the loss through the net, saving the graph because we do this for every key
                iter_count += _avg_turns_to_win #this is also the batch size, so we always get the loss divided by the number of samples
                avg_loss += loss.cpu().item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1000.0) #clip the gradients to avoid exploding gradient problem
            optimizer.step() #update the weights
            print(avg_loss/iter_count)
            
            #main network is updated every step, its weights directly updated by the backwards pass
            #target network is updated less often, its weights copied directly from the main net
            
            # if (iteration+1) % ceil(target_network_update_rate / batch_size) == 0:
            #     print("updating target model")
            #     target_model = model
            target_model = model #changed this to just run the batch size as the average win period

        return target_model

    for epoch in range(epochs):
        replay_memory = play(target_model)
        target_model = learn(target_model,replay_memory)

def _get_next_action_from_forward_result(forward: dict[str, torch.Tensor], game: Game) -> tuple[Turn | str, dict[str, torch.Tensor]]:
    """Get the next action from the model's forward pass result."""
    return ActionOutput.map_from_AI_output(forward, game, game.get_current_player())

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
            
def _avg_turns_to_win(replay_memory: list[ReplayMemoryEntry]) -> int:
    total_len = len(replay_memory)
    last_round_count = sum([1 if x.is_last_turn == 1 else 0 for x in replay_memory])
    return round(total_len/last_round_count)

def target_Q(Q_vals:torch.Tensor, #[batch_size, action_space_len]
         next_Q_vals:torch.Tensor, #[batch_size, action_space_len]
         reward:torch.Tensor, #[batch_size, action_space_len]
         gamma:float,
         is_last_turn:torch.Tensor) -> torch.Tensor:
    '''This function operates on a single action-space (key) in the
    Q dictionary'''
    assert reward.size() == Q_vals.size()

    #flip the 1's and 0's so the last_turn designator becomes a 0
    is_last_turn = (~is_last_turn.bool()).int() 
    # is_last_turn functions as an on-off switch for the next state Q values
    max_next_reward = is_last_turn * torch.max(next_Q_vals.detach()) #detach because we don't want gradients from the next state
    max_next_reward = max_next_reward.unsqueeze(1)
    discounted_next_reward_estimation = reward + (gamma * max_next_reward)

    return discounted_next_reward_estimation