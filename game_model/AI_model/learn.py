from copy import deepcopy
from math import ceil
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from game_model.AI_model.dataloader import BellmanEquationDataSet
from game_model.AI_model.model import SplendidSplendorModel
from game_model.AI_model.action_output import ActionOutput
from game_model.replay_memory import ReplayMemoryEntry

class Learner:
    '''Class for training the model'''
    def __init__(self, target_model: SplendidSplendorModel, replay_memory: list[ReplayMemoryEntry],settings: dict, writer: SummaryWriter,step_tracker:dict[str,int]):
        self.target_model = target_model
        self.replay_mem = replay_memory
        self.settings = settings
        self.writer = writer
        self.step_tracker = step_tracker

        # Transfer params of target model to a learner model and set training mode
        self.model = deepcopy(self.target_model)
        self.model.train()

        #Put both models on desired training device
        self.learn_device = torch.device("cuda" if torch.cuda.is_available() and settings['learn_device'] == "cuda" else "cpu")
        self.target_model = self.target_model.to(self.learn_device)
        self.model = self.model.to(self.learn_device)

        # Make sure replay memory ends up where it's needed
        for turn in self.replay_mem:
            turn.game_state = turn.game_state.remap(lambda x: x.to(self.learn_device))
            turn.taken_action = turn.taken_action.remap(lambda x: x.to(self.learn_device))
            turn.next_turn_game_state = turn.next_turn_game_state.remap(lambda x: x.to(self.learn_device))
            turn.reward = turn.reward.remap(lambda x: x.to(self.learn_device))
            turn.is_last_turn = turn.is_last_turn.to(self.learn_device)
    
    def learn(self) -> SplendidSplendorModel:
        
        # Define loss function and optimizer
        loss_fn = torch.nn.HuberLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.settings['learning_rate'])
        scheduler = CosineAnnealingWarmRestarts(optimizer,
                                                T_0=2,
                                                T_mult=2,
                                                eta_min=1e-12,
                                                last_epoch=-1,
                                                verbose=False)

        #TODO: get a step tracker working
        scheduler.step(self.step_tracker['epoch']) #updates the scheduler to the current epoch "step"

        # Set up dataset
        dataset = BellmanEquationDataSet(self.replay_mem,self.learn_device)

        for i in range(self.settings['reps_per_play_sess']):
            # Instantiate dataloader for each epoch
            dataloader = DataLoader(dataset,
                                    batch_size=self.settings['batch_size'],
                                    shuffle=True,
                                    num_workers=0)

            for iteration,batch in enumerate(dataloader):
                current_game_states : torch.Tensor = batch[0] ## dict of tensors of size batch x orig size
                next_game_states : torch.Tensor = batch[1]
                rewards : torch.Tensor = batch[2]
                is_last_turns: torch.Tensor = batch[3]

                Q_batch = self.model.forward(current_game_states)
                next_Q_batch = self.target_model.forward(next_game_states)
                # Warning: this modifies next_Q_batch in-place. next_Q_batch is equal to target_batch
                output_shape_dict = ActionOutput().in_dict_form()
                target_batch = self._target_Q(next_Q_batch,rewards,self.settings['gamma'],is_last_turns, output_shape_dict.index_dict)

                optimizer.zero_grad()

                loss: torch.Tensor = loss_fn(Q_batch,target_batch)
                loss.backward() #propagate the loss through the net

                batch_len: int = int(current_game_states.size()[0])
                loss_amount = loss.detach().item()/batch_len
                self.writer.add_scalar('net loss (iter)', loss_amount,self.step_tracker["total_learn_iters"])


                optimizer.step() #update the weights

                #torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0) #clip the gradients to avoid exploding gradient problem 
                
                # Overwrite the target model with the (hopefully) more better model
                self.target_model = deepcopy(self.model)

                self.step_tracker["learn_loop_iters"] += 1
                self.step_tracker["total_learn_iters"] += 1

            self.writer.add_scalar('Learning rate (epoch)', scheduler._last_lr[0], self.step_tracker['epoch'])
        self.step_tracker["learn_loop_iters"] = 0

        return self.target_model

    def _target_Q(self,
                 next_Q_batch: torch.Tensor, ## a 2D tensor, <batch dim> x <output size>
                 reward_batch: torch.Tensor, ## a 2D tensor, <batch dim> x <output size>
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