import torch

from utilities.better_param_dict import BetterParamDict
class ActionOutput:
    def __init__(self):
        self.mapped_properties : BetterParamDict[list[float]] = BetterParamDict([])

        self.mapped_properties["action_choice"] = [0] * 4
        self.action_choice_clamp_range = [0,1]
        self.mapped_properties["card_buy"] = [0] * 15
        self.card_buy_clamp_range = [0,1]
        self.mapped_properties["reserve_buy"]= [0] * 3
        self.reserve_buy_clamp_range = [0,1]
        self.mapped_properties["resource_token_draw"] = [0] * 5
        self.resource_token_draw_clamp_range = [0,1]
        
        self.mapped_properties["noble_choice"] = [0] * 5
        self.noble_choice_clamp_range = [0,1]
        self.mapped_properties["discard_choice"] = [0]
        self.discard_choice_clamp_range = [0,1]
        self.mapped_properties["discard_amounts"] = [0] * 6
        self.discard_amounts_clamp_range = [0,7]

    def __setattr__(self, name: str, value):
        '''Try to set the attribute in the mapped_properties ParamDict first, then default to normal implementation'''
        try:
            my_props = object.__getattribute__(self, "mapped_properties")
            if name in my_props:
                my_props[name] = value
                return
        except AttributeError:
            pass
        
        object.__setattr__(self, name, value)
    
    def __getattribute__(self, name: str):
        '''Try to get the attribute from the mapped_properties ParamDict first, then default to normal implementation'''
        try:
            my_props = object.__getattribute__(self, "mapped_properties")
            if name in my_props:
                return my_props[name]
        except AttributeError:
            pass
        
        return object.__getattribute__(self, name)

    def get_data_length(self) -> int:
        return len(self.mapped_properties.get_backing_packed_data())

    def map_tensor_into_self(self, into_tensor: torch.Tensor):
        mapped_list = list(into_tensor)
        self.mapped_properties = BetterParamDict.reindex_over_new_data(self.mapped_properties, mapped_list)
