import torch
class ActionOutput:
    def __init__(self):
        self.action_choice: list[float] = [0] * 4
        self.card_buy: list[float] = [0] * 15
        self.reserve_buy: list[float]= [0] * 3
        self.resource_token_draw: list[float] = [0] * 5
        
        self.noble_choice: list[float] = [0] * 5
        self.discard_choice: list[float] = [0]
        self.discard_amounts: list[float] = [0] * 6
    
    def in_dict_form(self):
        action_output_dict : dict[str, list[float]] = {}
        action_output_dict['action_choice'] = self.action_choice
        action_output_dict['card_buy'] = self.card_buy
        action_output_dict['reserve_buy'] = self.reserve_buy
        action_output_dict['resource_token_draw'] = self.resource_token_draw
        action_output_dict['noble_choice'] = self.noble_choice
        action_output_dict['discard_choice'] = self.discard_choice
        action_output_dict['discard_amounts'] = self.discard_amounts

        return action_output_dict

    def map_dict_into_self(self, into_dict: dict[str, torch.Tensor]):
        self.action_choice = into_dict['action_choice']
        self.card_buy = into_dict['card_buy']
        self.reserve_buy = into_dict['reserve_buy']
        self.resource_token_draw = into_dict['resource_token_draw']
        self.noble_choice = into_dict['noble_choice']
        self.discard_choice = into_dict['discard_choice']
        self.discard_amounts = into_dict['discard_amounts']