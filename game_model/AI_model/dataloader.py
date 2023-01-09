import torch
from torch.utils.data import Dataset
from game_model.replay_memory import ReplayMemoryEntry


class BellmanEquationDataSet(Dataset):
    def __init__(self, input_data, device):
        self.input_data = input_data
        self.device = device
        
    def __getitem__(self, index):
        current_game_state = self.input_data[index]['game_state']
        next_game_state = self.input_data[index]['next_turn_game_state']
        reward = self.input_data[index]['reward']
        is_last_turn = self.input_data[index]['is_last_turn']
        return [current_game_state,next_game_state,reward,is_last_turn]
    
    def __len__(self):
        print(len(self.input_data))
        return len(self.input_data)