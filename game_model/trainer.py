import random
from math import ceil
from os.path import exists
import threading
from typing import Callable
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from yaml import safe_load as load
from torch.utils.tensorboard import SummaryWriter
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
        

# Default hyperparameters
settings: dict[str,float|int] = {}
settings['learning_rate']: float = 0.0001 
settings['gamma']: float = 0.99 #discount factor, how much it cares about future reward vs current reward
            #(0: only current, 1: current and all future states)
settings['epsilon']: float = 0.5 #how often to pick the maximum-Q-valued action
settings['memory_length']: int = 10000      #number of rounds to play of the game
settings['batch_size_multiplier']: float = 1.0 #batch size is set to the average number of turns per game multiplied by this factor
settings['max_batch_size']: int = 1000 #so that we don't run out of memory accidentally
settings['epochs']: int = 1 #how many play->learn cycles to run
settings['hidden_layer_width'] = 128 #I like to keep things like linear layer widths at multiples of 2 for faster GPU processing
settings['n_hidden_layers'] = 3

# Rewards: [use this reward?, value of this reward]
settings['tokens_held'] = [False,1.0]
settings['cards_held'] = [False,1.0]
settings['points'] = [True,5.0]
settings['win_lose'] = [True,200]
settings['length_of_game'] = [True,-0.1]


# Overwrite with user-defined parameters if they exist
if exists('game_model/AI_model/train_settings.yaml'):
    settings = load(open('game_model/AI_model/train_settings.yaml','r'))

def train(on_game_changed : Callable[[Game, Turn], None], game_data_lock: threading.Lock):
    # Load game configuration data
    game_config = GameConfigData.read_file("./game_data/cards.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(flush_secs=15) #tensorboard writer
    # Keeps track of the training steps for tensorboard
    step_tracker: dict[str,int] = {'epoch':0,'play_loop_iters':0,'learn_loop_iters':0,'total_learn_iters':0}
    
    # Create models
    # Map game state to AI input
    game = Game(player_count=4, game_config=game_config)
    input_shape_dict = GamestateInputVector.map_to_AI_input(game)
    output_shape_dict = ActionOutput().in_dict_form()

    
    target_model = SplendidSplendorModel(input_shape_dict, output_shape_dict, settings['hidden_layer_width'], settings['n_hidden_layers'])
    if exists('AI_model/SplendidSplendor-model.pkl'):
        target_model.load_state_dict(torch.load('game_model/AI_model/SplendidSplendor-model.pkl',
                                         map_location='cpu'))
    target_model = target_model.to(device) 

    def play(target_model) -> list[ReplayMemoryEntry]:
        # Instantiate memory
        replay_memory: list[ReplayMemoryEntry] = []
        while len(replay_memory) < settings['memory_length']:
            len_left_in_replay: int = settings['memory_length'] - len(replay_memory)
            replay_memory += play_single_game(target_model,len_left_in_replay)
        return replay_memory
    
    def play_single_game(target_model,len_left_in_replay: int) -> list[ReplayMemoryEntry]:
        replay_memory: list[ReplayMemoryEntry] = []
        game_data_lock.acquire()
        try:
            game = Game(player_count=random.randint(2,4), game_config=game_config)
            on_game_changed(game, None)
            next_player_index = game.active_index
        finally:
            game_data_lock.release()
        won = False
        while not (won and next_player_index == 0):
            ''' Use target network to play the games'''

            # Map game state to AI input
            game_data_lock.acquire()
            try:
                ai_input = GamestateInputVector.map_to_AI_input(game)
                turns_since_last = game.get_player_num() - 1
            finally:
                game_data_lock.release()
            
            # Store game state in memory
            player_mem = ReplayMemoryEntry(ai_input)
            #save this game to the last turn of this player's memory
            if len(replay_memory) >= turns_since_last:
                replay_memory[-turns_since_last].next_turn_game_state = ai_input
            
            # Get model's predicted action
            ai_input = {key:ai_input[key].to(device) for key in ai_input}
            target_model.eval()
            with torch.no_grad(): #no need to save gradients since we're not backpropping, this saves a lot of time/memory
                Q = target_model.forward(ai_input) #Q values == expected reward for an action taken
                Q = {key:Q[key].to(torch.device('cpu')) for key in Q}

            # Apply epsilon greedy function to somewhat randomize the action picks for exploration
            Q = _epsilon_greedy(Q,settings['epsilon'])

            game_data_lock.acquire()
            try:
                # Pick the highest Q-valued action that works in the game
                (next_action, chosen_Action) = _get_next_action_from_forward_result(Q, game) 

                player_mem.taken_action = {key:value.detach() for key,value in chosen_Action.items()} #detach to not waste memory
                player_mem.num_players = game.get_num_players()

                # Play move
                step_status = step_game(game, next_action)
                on_game_changed(game, next_action)
                next_player_index = game.active_index
                if not (step_status is None):
                    raise Exception("invalid game step generated, " + step_status)

                # Get reward from state transition, and convert to dict form 
                reward = Reward(game,game.get_current_player_index(),settings).all_rewards()# - original_fitness.base_reward
                reward_dict = {choice:(reward * player_mem.taken_action[choice]) for choice in player_mem.taken_action}

                # Store reward in memory
                player_mem.reward_new = reward_dict


                if game.get_current_player().qualifies_to_win():
                    won = True
            finally:
                game_data_lock.release()

            #Store turn in replay memory
            replay_memory.append(player_mem)

            if (len(replay_memory) == len_left_in_replay) or (len(replay_memory) >= 1000): #games shouldn't last longer than 1000 turns
                break
        
        game_data_lock.acquire()
        try:
            ending_state = GamestateInputVector.map_to_AI_input(game)
            total_players = game.get_num_players()
        finally:
            game_data_lock.release()
        
        for player_index in range(total_players):
            last_turn_player = replay_memory[-player_index]
            if last_turn_player.next_turn_game_state is None:
                last_turn_player.next_turn_game_state = ending_state
            if won == True: #if a game overruns the replay length, this will eval false
                last_turn_player.is_last_turn = torch.as_tensor(int(1))
                reward = Reward(game,player_index,settings).all_rewards()
                reward_dict = {choice:(reward * last_turn_player.taken_action[choice]) for choice in last_turn_player.taken_action}
                last_turn_player.reward_new = reward_dict
        
        return replay_memory

    def learn(target_model,replay_memory: list[ReplayMemoryEntry]):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = target_model

        # Base the target model update rate on how many turns it takes to win
        target_network_update_rate: int = _avg_turns_to_win(replay_memory)
        writer.add_scalar('Avg turns to win (epoch)',target_network_update_rate,step_tracker['epoch'])

        model.train()

        # Define loss function and optimizer
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'])
        scheduler = CosineAnnealingWarmRestarts(optimizer,
                                                T_0=2,
                                                T_mult=2,
                                                eta_min=1e-12,
                                                last_epoch=-1,
                                                verbose=False)
            
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
        batch_size: float = min(ceil(target_network_update_rate*settings['batch_size_multiplier']),settings['max_batch_size'])
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=0)

        for iteration,batch in enumerate(dataloader):
            Q_dicts = model(batch[0]) ## dict of tensors of size batch x orig size
            next_Q_dicts = target_model(batch[1])
            rewards = batch[2]
            is_last_turns = batch[3]
            optimizer.zero_grad()
            avg_loss: float = 0.0
            batch_len: int = int(batch[0]['player_0_temp_resources'].size()[0]) #this is also the batch size, so we always get the loss divided by the number of samples
            for key in Q_dicts:
                target = target_Q(Q_dicts[key],next_Q_dicts[key],rewards[key],settings['gamma'],is_last_turns)
                loss = loss_fn(target,Q_dicts[key])
                loss.backward(retain_graph=True) #propagate the loss through the net, saving the graph because we do this for every key
                avg_loss += loss.detach().item()/batch_len
                writer.add_scalar('Loss (iter)/'+key,loss.detach().item()/batch_len,step_tracker["total_learn_iters"])
            
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1000.0) #clip the gradients to avoid exploding gradient problem
            scheduler.step(step_tracker['total_learn_iters']) #update the weights
            writer.add_scalar('Learning rate (iter)', scheduler._last_lr[0], step_tracker['total_learn_iters'])

            n_keys = len(Q_dicts)
            writer.add_scalar('Key-averaged loss (iter)', avg_loss/n_keys,step_tracker["total_learn_iters"])
            
            #main network is updated every step, its weights directly updated by the backwards pass
            #target network is updated less often, its weights copied directly from the main net
            
            # if (iteration+1) % ceil(target_network_update_rate / batch_size) == 0:
            #     print("updating target model")
            #     target_model = model
            target_model = model #changed this to just run the batch size as the average win period
            step_tracker["learn_loop_iters"] += 1
            step_tracker["total_learn_iters"] += 1

        step_tracker["learn_loop_iters"] = 0
        return target_model

    for epoch in range(settings['epochs']):
        replay_memory = play(target_model)
        target_model = learn(target_model,replay_memory)
        step_tracker['epoch'] += 1
        torch.save(target_model.state_dict(), 'game_model/AI_model/SplendidSplendor-model.pkl')

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
    last_round_count: float = sum([(1.0/x.num_players) if x.is_last_turn == 1 else 0 for x in replay_memory])

    # something produces a divide-by-0 error but I can't reproduce it consistently, figure out later
    try:
        return ceil(total_len/last_round_count)
    except:
        return(settings['memory_length'])

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
    max_next_reward = max_next_reward.unsqueeze(1) #add an outer batch dimension to the tensor (broadcasting requirements)

    # The central update function. Reward describes player reward at (state,action). Gamma describes the discount towards
    # future actions vs. current action reward. The max_next_reward describes the model's best prediction of the total reward
    # it will be able to acheive over the converging series of SUM[now -> infinity/end](discount*(action->reward)).
    # All put together, what this means is that we add this action's reward to the predicted total reward. This gives us
    # our target Q value, which we can send off to the loss function, where the Q value will be compared to this target 
    # Q value from which we get a loss value.
    
    discounted_reward_estimate = reward + (gamma * max_next_reward)

    return discounted_reward_estimate