import random
from math import ceil
from copy import deepcopy
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
from utilities.better_param_dict import BetterParamDict

from utilities.simple_profile import SimpleProfileAggregator

# Default hyperparameters
settings: dict[str,float|int] = {}
settings['learning_rate']: float = 0.0001 
settings['gamma']: float = 0.99 #discount factor, how much it cares about future reward vs current reward
            #(0: only current, 1: current and all future states)
settings['epsilon']: float = 0.5 #how often to pick the maximum-Q-valued action
settings['memory_length']: int = 10000      #number of rounds to play of the game
settings['batch_size']: int = 1000 #so that we don't run out of memory accidentally
settings['reps_per_play_sess']: int = 10 #how many times to train over the same replay memory buffer
settings['epochs']: int = 1 #how many play->learn cycles to run
settings['hidden_layer_width'] = 128 #I like to keep things like linear layer widths at multiples of 2 for faster GPU processing
settings['n_hidden_layers'] = 3

# Rewards: [use this reward?, value of this reward]
settings['tokens_held'] = [False,1.0]
settings['cards_held'] = [False,1.0]
settings['points'] = [True,5.0]
settings['win_lose'] = [True,200]
settings['length_of_game'] = [True,-0.1]

settings['force_cpu_turns'] = False
settings['force_cpu_learning'] = False
settings['randomize_player_num'] = True


# Overwrite with user-defined parameters if they exist
if exists('game_model/AI_model/train_settings.yaml'):
    settings = load(open('game_model/AI_model/train_settings.yaml','r'))

turn_profiler = SimpleProfileAggregator("turn generation time analysis", 1000)
learn_profiler = SimpleProfileAggregator("learning time analysis", 10)

def train(on_game_changed : Callable[[Game, Turn], None]):
    # Load game configuration data
    game_config = GameConfigData.read_file("./game_data/cards.csv")
    device = torch.device("cuda" if torch.cuda.is_available() and not settings['force_cpu_turns'] else "cpu")
    learning_device = torch.device("cuda" if torch.cuda.is_available() and not settings['force_cpu_learning'] else "cpu")
    writer = SummaryWriter(flush_secs=15) #tensorboard writer
    # Keeps track of the training steps for tensorboard
    step_tracker: dict[str,int] = {'epoch':0,'play_loop_iters':0,'learn_loop_iters':0,'total_learn_iters':0}
    
    # Create models
    # Map game state to AI input
    game = Game(player_count=4, game_config=game_config)
    input_shape_dict = GamestateInputVector.map_to_AI_input(game)
    output_shape_dict = ActionOutput().in_dict_form()

    
    target_model = SplendidSplendorModel(
        input_shape_dict.get_backing_len(),
        output_shape_dict.get_backing_len(), 
        settings['hidden_layer_width'], 
        settings['n_hidden_layers'])
    #if exists('AI_model/SplendidSplendor-model.pkl'):
    #    target_model.load_state_dict(torch.load('game_model/AI_model/SplendidSplendor-model.pkl',
    #                                     map_location='cpu'))
    target_model = target_model.to(device) 

    play_stats: dict[str, float | list[float]] = {}

    def play(target_model: SplendidSplendorModel) -> list[ReplayMemoryEntry]:
        stats_tracker = {}
        stats_tracker['mandatory discarded tokens'] = 0
        stats_tracker['optional discarded tokens'] = 0
        stats_tracker['hand tokens'] = 0
        stats_tracker['reserved cards'] = 0
        stats_tracker['actions'] = {
            Action_Type.BUY_CARD: 0,
            Action_Type.RESERVE_CARD: 0,
            Action_Type.TAKE_THREE_UNIQUE: 0,
            Action_Type.TAKE_TWO: 0,
            Action_Type.NOOP: 0,
        }
        stats_tracker['total_play_rounds'] = 0

        # Instantiate memory
        target_model = target_model.to(device) 
        replay_memory: list[ReplayMemoryEntry] = []
        while len(replay_memory) < settings['memory_length']:
            len_left_in_replay: int = settings['memory_length'] - len(replay_memory)
            ## at least 16 turns. there was an edge case in our end-game code, making the assumption that at least one full loop of play had completed
            len_left_in_replay = max(len_left_in_replay, 16)
            replay_memory += play_single_game(target_model, len_left_in_replay, stats_tracker)
        

        for key in stats_tracker:
            if key == 'actions' or key == 'total_play_rounds':
                continue
            play_stats[key] = stats_tracker[key] / stats_tracker['total_play_rounds']
        for key in stats_tracker['actions']:
            play_stats["action taken/" + key.name] = stats_tracker['actions'][key] / stats_tracker['total_play_rounds']
        
        return replay_memory
    
    def play_single_game(target_model: SplendidSplendorModel,len_left_in_replay: int, statistic_tracker: dict) -> list[ReplayMemoryEntry]:
        replay_memory: list[ReplayMemoryEntry] = []
        
        player_count = random.randint(2,4) if settings['randomize_player_num'] else 4
        game = Game(player_count=player_count, game_config=game_config)
        on_game_changed(game, None)
        next_player_index = game.active_index
            
        won = False
        while not (won and next_player_index == 0):
            ''' Use target network to play the games'''
            turn_profiler.begin_sample_run()
            # Map game state to AI input
            ai_input = GamestateInputVector.map_to_AI_input(game, input_shape_dict)
            turns_since_last = game.get_player_num() - 1
            
            # Store game state in memory
            player_mem = ReplayMemoryEntry(ai_input)
            #save this game to the last turn of this player's memory
            if len(replay_memory) >= turns_since_last:
                replay_memory[-turns_since_last].next_turn_game_state = ai_input

            turn_profiler.sample("input mapping")
            # Get model's predicted action
            ai_input = ai_input.remap(lambda x: x.to(device))
            turn_profiler.sample("device mapping in")

            target_model.eval()
            turn_profiler.sample("model eval")
            with torch.no_grad(): #no need to save gradients since we're not backpropping, this saves a lot of time/memory
                unshaped_Q = target_model.forward(ai_input.get_backing_packed_data(), turn_profiler) #Q values == expected reward for an action taken
                Q = BetterParamDict.reindex_over_new_data(output_shape_dict, unshaped_Q)
                Q = Q.remap(lambda x: x.to(torch.device('cpu')))
            turn_profiler.sample("device mapping out")

            # Apply epsilon greedy function to somewhat randomize the action picks for exploration
            Q = _epsilon_greedy(Q,settings['epsilon'])
            turn_profiler.sample("q greedy")

            # Get the reward at initial state
            init_reward = Reward(game,game.get_current_player_index(),settings).all_rewards()

            # Pick the highest Q-valued action that works in the game
            (next_action, chosen_Action) = _get_next_action_from_forward_result(Q, game)
            statistic_tracker['actions'][next_action.action_type] += 1
            statistic_tracker['total_play_rounds'] += 1

            player_mem.taken_action = chosen_Action.remap(lambda x: x.detach()) #detach to not waste memory
            player_mem.num_players = game.get_num_players()
            turn_profiler.sample("output mapping to action")

            # Play move
            current_player = game.get_current_player()
            step_status = step_game(game, next_action)
            turn_profiler.sample("game step")
            on_game_changed(game, next_action)

            statistic_tracker['mandatory discarded tokens'] += next_action.last_discarded_mandatory
            statistic_tracker['optional discarded tokens'] += next_action.last_discarded_optional
            statistic_tracker['hand tokens'] += current_player.total_tokens()
            statistic_tracker['reserved cards'] += sum([0 if x is None else 1 for x in current_player.reserved_cards])

            next_player_index = game.active_index
            if not (step_status is None):
                raise Exception("invalid game step generated, " + step_status)

            # Get reward from state transition, and convert to dict form 
            next_reward = Reward(game,game.get_current_player_index(),settings).all_rewards()
            transition_reward = next_reward - init_reward
            reward_dict = player_mem.taken_action.remap(lambda x: transition_reward * x)

            # Store reward in memory
            player_mem.reward_new = reward_dict
            turn_profiler.sample("post-game stats/reward")


            if game.get_current_player().qualifies_to_win():
                won = True

            #Store turn in replay memory
            replay_memory.append(player_mem)
            turn_profiler.end_sample_run()

            if (len(replay_memory) == len_left_in_replay) or (len(replay_memory) >= 1000): #games shouldn't last longer than 1000 turns
                break
        
        ending_state = GamestateInputVector.map_to_AI_input(game, input_shape_dict)
        total_players = game.get_num_players()
        
        for player_index in range(total_players):
            last_turn_player = replay_memory[-player_index]
            if last_turn_player.next_turn_game_state is None:
                last_turn_player.next_turn_game_state = ending_state
            if won == True: #if a game overruns the replay length, this will eval false
                last_turn_player.is_last_turn = torch.as_tensor(int(1))
                reward = Reward(game,player_index,settings).all_rewards()
                reward_dict = last_turn_player.taken_action.remap(lambda x: reward * x)
                last_turn_player.reward_new = reward_dict
        
        return replay_memory

    def learn(target_model: SplendidSplendorModel,replay_memory: list[ReplayMemoryEntry]):
        # Transfer params of target model to a learner model and set training mode
        model = deepcopy(target_model)
        model.train()

        #Put both models on desired training device
        target_model = target_model.to(learning_device)
        model = model.to(learning_device)
        

        writer.add_scalar('Avg turns to win (epoch)',_avg_turns_to_win(replay_memory),step_tracker['epoch'])
        for key in play_stats:
            writer.add_scalar('Gameplay (epoch)/' + key,play_stats[key],step_tracker['epoch'])

        # Define loss function and optimizer
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'])
        scheduler = CosineAnnealingWarmRestarts(optimizer,
                                                T_0=2,
                                                T_mult=2,
                                                eta_min=1e-12,
                                                last_epoch=-1,
                                                verbose=False)
        scheduler.step(step_tracker["epoch"]) #updates the scheduler to the current epoch "step"
            
        # Transfer all the data to the GPU for blazing fast train speed
        if learning_device == torch.device("cuda"):
            for turn in replay_memory:
                turn.game_state = turn.game_state.remap(lambda x: x.to(learning_device))
                turn.taken_action = turn.taken_action.remap(lambda x: x.to(learning_device))
                turn.next_turn_game_state = turn.next_turn_game_state.remap(lambda x: x.to(learning_device))
                turn.reward_new = turn.reward_new.remap(lambda x: x.to(learning_device))
                turn.is_last_turn = turn.is_last_turn.to(learning_device)

        # Set up dataset
        dataset = BellmanEquationDataSet(replay_memory,learning_device)
        
        for i in range(settings['reps_per_play_sess']):
            # Instantiate dataloader for each epoch
            dataloader = DataLoader(dataset,
                                    batch_size=settings['batch_size'],
                                    shuffle=True,
                                    num_workers=0)

            for iteration,batch in enumerate(dataloader):
                current_game_states : torch.Tensor = batch[0] ## dict of tensors of size batch x orig size
                next_game_states : torch.Tensor = batch[1]
                rewards : torch.Tensor = batch[2]
                is_last_turns: torch.Tensor = batch[3]

                learn_profiler.begin_sample_run()

                Q_batch = model.forward(current_game_states, learn_profiler)
                next_Q_batch = target_model.forward(next_game_states, learn_profiler)
                # Warning: this modifies next_Q_dicts in-place. next_Q_dicts is equal to target
                target_batch = target_Q(next_Q_batch,rewards,settings['gamma'],is_last_turns, output_shape_dict.index_dict)

                optimizer.zero_grad()

                learn_profiler.sample("target Q eval")

                loss: torch.Tensor = loss_fn(Q_batch,target_batch)
                loss.backward() #propagate the loss through the net

                batch_len: int = int(current_game_states.size()[0])
                loss_amount = loss.detach().item()/batch_len
                writer.add_scalar('net loss (iter)', loss_amount,step_tracker["total_learn_iters"])

                learn_profiler.sample("loss function")

                optimizer.step() #update the weights

                learn_profiler.sample("optimizer step")

                #torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0) #clip the gradients to avoid exploding gradient problem 
                
                # Overwrite the target model with the (hopefully) more better model
                target_model = deepcopy(model)

                learn_profiler.sample("copy model")

                step_tracker["learn_loop_iters"] += 1
                step_tracker["total_learn_iters"] += 1
                
                learn_profiler.end_sample_run()
                
            writer.add_scalar('Learning rate (epoch)', scheduler._last_lr[0], step_tracker['epoch'])
        step_tracker["learn_loop_iters"] = 0
        return target_model

    for epoch in range(settings['epochs']):
        replay_memory = play(target_model)
        target_model = learn(target_model,replay_memory)
        step_tracker['epoch'] += 1
        torch.save(target_model.state_dict(), 'game_model/AI_model/SplendidSplendor-model.pkl')

def _get_next_action_from_forward_result(forward: BetterParamDict[torch.Tensor], game: Game) -> tuple[Turn | str, BetterParamDict[torch.Tensor]]:
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

def _epsilon_greedy(Q: BetterParamDict[torch.Tensor], epsilon: float):
    '''The epsilon greedy algorithm is supposed to choose the max Q-valued
    action with a probability of 1-epsilon. Otherwise, it will randomly choose
    another possible action with probability epsilon. We're going to do this by either allowing the
    Q values to go undisturbed to the action mapper, or by swapping the max
    Q value for a particular action to another position, so that the action
    mapper will pick that action instead.'''
    
    for choice_type in Q:
        choices = Q[choice_type]
        if random.uniform(0,1) < epsilon and len(choices) > 1:
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

def target_Q(next_Q_batch: torch.Tensor, ## a 2D tensor, <batch dim> x <output size>
         reward_batch: torch.Tensor,     ## a 2D tensor, <batch dim> x <output size>
         gamma:float,
         is_last_turn:torch.Tensor,
         action_output_shape: dict[str, tuple[int, int]]) -> torch.Tensor:
    '''This function operates on a single action-space (key) in the
    Q dictionary'''
    
    #flip the 1's and 0's so the last_turn designator becomes a 0
    is_last_turn = (~is_last_turn.bool()).int() 

    for key in action_output_shape:
        action_range = action_output_shape[key]

        ## <batch size> x <action_range>
        next_Q_slice = next_Q_batch[:,action_range[0]:action_range[1]]
        # is_last_turn functions as an on-off switch for the next state Q values
        max_next_reward = is_last_turn * torch.max(next_Q_slice.detach()) #detach because we don't want gradients from the next state
        ## <batch size> x 1
        max_next_reward = max_next_reward.unsqueeze(1) #add an outer batch dimension to the tensor (broadcasting requirements)

        # The central update function. Reward describes player reward at (state,action). Gamma describes the discount towards
        # future actions vs. current action reward. The max_next_reward describes the model's best prediction of the total reward
        # it will be able to achieve through the whole converging series of SUM[i: now->endgame]( (discount^i) * (reward[i]) ).
        # All put together, what this means is that we add this action's reward to the predicted total reward. This gives us
        # our target estimated Q value, which we can send off to the loss function, where the Q value will be compared to this target 
        # Q value from which we get a loss value.

        ## <batch size> x <action_range>
        reward_slice = reward_batch[:,action_range[0]:action_range[1]]
        ## <batch size> x <action_range>
        target_result = reward_slice + (gamma * max_next_reward)
        next_Q_batch[:,action_range[0]:action_range[1]] = target_result

    return next_Q_batch
