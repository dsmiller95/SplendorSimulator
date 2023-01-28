import torch
from torch.utils.data import Dataset
from game_model.replay_memory import ReplayMemoryEntry


class BellmanEquationDataSet(Dataset):
    def __init__(self, input_data: list[ReplayMemoryEntry], device):
        self.input_data = input_data
        self.device = device
        
    def __getitem__(self, index: int):
        current_game_state = self.input_data[index].game_state.map_to_dict()
        next_game_state = self.input_data[index].next_turn_game_state.map_to_dict()
        reward = self.input_data[index].reward_new.map_to_dict()
        is_last_turn = self.input_data[index].is_last_turn
        output = [current_game_state,next_game_state,reward,is_last_turn]
        return output
    
    def __len__(self):
        return len(self.input_data)