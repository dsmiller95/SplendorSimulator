class ActionOutput:
    def __init__(self):
        self.action_choice: list[float] = [0] * 4
        self.action_choice_clamp_range = 0,1
        self.card_buy: list[float] = [0] * 15
        self.card_buy_clamp_range = 0,1
        self.reserve_buy: list[float]= [0] * 3
        self.reserve_buy_clamp_range = 0,1
        self.resource_token_draw: list[float] = [0] * 5
        self.resource_token_draw_clamp_range = 0,1
        
        self.noble_choice: list[float] = [0] * 5
        self.noble_choice_clamp_range = 0,1
        self.discard_choice: list[float] = [0]
        self.discard_choice_clamp_range = 0,1
        self.discard_amounts: list[float] = [0] * 6
        self.discard_amounts_clamp_range = 0,7
    
    def in_dict_form(self):
        action_output_dict:dict[str, list[float]] = {}
        action_output_dict['action_choice'] = self.action_choice: list[float] = [0]*4
        action_output_dict[''] = self.action_choice_clamp_range = [0,1]
        action_output_dict[''] = self.card_buy: list[float] = [0]*15
        self.card_buy_clamp_range = [0,1]
        self.reserve_buy: list[float]= [0]*3
        self.reserve_buy_clamp_range = [0,1]
        self.resource_token_draw: list[float] = [0]*5
        self.resource_token_draw_clamp_range = [0,1]
        
        self.noble_choice: list[float] = [0]*5
        self.noble_choice_clamp_range = [0,1]
        self.discard_choice: list[float] = [0]
        self.discard_choice_clamp_range = [0,1]
        self.discard_amounts: list[float] = [0]*6
        self.discard_amounts_clamp_range = [0,7]
