import torch
from torch.utils.data import Dataset
from game_model.replay_memory import ReplayMemoryEntry


class BellmanEquationDataSet(Dataset):
    def __init__(self, data:ReplayMemoryEntry):
        self.data = data
        
    def __getitem__(self, index):
        print(index)
        #print(self.data[index].game_state)
        current_game_state = self.data[index].game_state
        if not self.data[index].is_last_turn:
            next_game_state = self.data[index].next_turn_game_state
        else:
            next_game_state = None
        reward = self.data[index].reward
        return current_game_state,next_game_state,reward
    
    def __len__(self):
        return len(self.data)