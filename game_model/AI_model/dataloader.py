import torch
from torch.utils.data import Dataset
from game_model.replay_memory import ReplayMemoryEntry


class BellmanEquationDataSet(Dataset):
    def __init__(self, input_data, device):
        self.input_data = input_data
        self.device = device
        
    def __getitem__(self, index):
        current_game_state = self.input_data[index]['game_state']
        next_game_state = self.input_data[index]['next_turn_game_state'].to(device)
        reward = self.input_data[index]['reward'].to(device)
        return current_game_state,next_game_state,reward
    
    def __len__(self):
        print(len(self.input_data))
        return len(self.input_data)