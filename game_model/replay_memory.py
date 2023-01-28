import torch

from utilities.better_param_dict import BetterParamDict
class ReplayMemoryEntry:
    '''Each memory instance contains the game state, the Q prediction, and the reward'''
    def __init__(
        self, 
        game_state: BetterParamDict[torch.Tensor]
        ):
        self.game_state : BetterParamDict[torch.Tensor] = game_state
        self.next_turn_game_state: BetterParamDict[torch.Tensor] = None
        # Indicates if this is the last turn which was taken by this player in a game
        self.is_last_turn: torch.Tensor = torch.as_tensor(int(0))

        ## dict of 1-hot representation of action which was actually taken, as sub-actions
        ##  after action coercion process chooses a valid action based on game state
        self.taken_action: BetterParamDict[torch.Tensor] = None
        self.reward_new: BetterParamDict[torch.Tensor] = None ## dict of reward. equals reward scalar multiplied by chosen_action
        
        self.num_players: int = None





