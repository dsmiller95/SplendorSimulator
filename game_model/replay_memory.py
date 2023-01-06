import torch
class ReplayMemoryEntry:
    '''Each memory instance contains the game state, the Q prediction, and the reward'''
    def __init__(
        self, 
        game_state: dict[str, torch.Tensor]
        ):
        self.game_state : dict[str, torch.Tensor] = game_state
        self.q_prediction: dict[str, torch.Tensor] = {}
        self.next_turn_game_state: dict[str, torch.Tensor] = {}
        self.reward: float = -1
        # Indicates if this is the last turn which was taken by this player in a game
        self.is_last_turn: bool = False