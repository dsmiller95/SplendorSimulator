import torch

from utilities.better_param_dict import BetterParamDict
class ReplayMemoryEntry:
    '''Each memory instance contains the game state, next game state,
    Q prediction, state transition reward, action mask, last turn indicator,
    and number of players in the game'''
    def __init__(self, game_state: BetterParamDict[torch.Tensor]):
        self.game_state : BetterParamDict[torch.Tensor] = game_state
        self.next_turn_game_state: BetterParamDict[torch.Tensor] = None
        # Indicates if this is the last turn which was taken by this player in a game
        self.is_last_turn: torch.Tensor = torch.as_tensor(int(0))

        # dict of 1-hot representation of action which was actually taken, as sub-actions
        self.taken_action: BetterParamDict[torch.Tensor] = None

        # reward equals float Reward multiplied by taken_action to get a masked reward
        self.reward: BetterParamDict[torch.Tensor] = None 
        
        # some information about the game being played in this turn
        self.num_players: int = None
        self.player_type: str = None





