import torch
class ReplayMemoryEntry:
    '''Each memory instance contains the game state, the Q prediction, and the reward'''
    def __init__(
        self, 
        game_state: dict[str, torch.Tensor]
        ):
        self.game_state : dict[str, torch.Tensor] = game_state
        self.next_turn_game_state: dict[str, torch.Tensor] = {} 
        # Indicates if this is the last turn which was taken by this player in a game
        self.is_last_turn: torch.Tensor = torch.as_tensor(int(0))

        ## dict of 1-hot representation of action which was actually taken, as sub-actions
        ##  after action coercion process chooses a valid action based on game state
        self.taken_action: dict[str, torch.Tensor] = {} 
        self.reward_new: dict[str, torch.Tensor] = {} ## dict of reward. equals reward scalar multiplied by chosen_action

        #### depreciated ####
        self.q_prediction: dict[str, torch.Tensor] = {}
        ## scalar reward. depreciated.
        self.reward: torch.Tensor = torch.as_tensor(int(-1))
        self.q_prediction: dict[str, torch.Tensor] = {}





