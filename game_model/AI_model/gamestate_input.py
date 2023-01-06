import torch
from game_model.AI_model.maps import map_all_to_tensors, to_hot_from_scalar

from game_model.game import Game

from utilities.better_param_dict import BetterParamDict

def flat_map_group(map_list: list, prefix: str, into_dict: BetterParamDict[list[float]]):
    for i, item in enumerate(map_list):
        item.flat_map_into(prefix + str(i), into_dict)

class GamestateInputVector:
    def __init__(self):
        self.players = [PlayerVector() for x in range(0, 4)]
        self.nobles = [NobleVector() for x in range(5)]
        self.resources = [None] * 6
        self.tiers = [RowVector() for x in range(3)]
    
    def flat_map(self, prefix: str = "", into_dict: BetterParamDict[list[float]] = None) -> BetterParamDict[list[float]]:
        if into_dict is None:
            into_dict = BetterParamDict([])
        flat_map_group(self.players, prefix + "player_", into_dict)
        flat_map_group(self.nobles, prefix + "board_noble_", into_dict)
        flat_map_group(self.tiers, prefix + "tier_", into_dict)
        into_dict[prefix + "resources"] = self.resources
        return into_dict
    
    @staticmethod
    def map_to_AI_input(game_state: Game) -> dict[str, torch.Tensor]:
        '''
        Maps the game state into a dictionary of tensors, for use by the AI model
        '''
        input_vect_model = GamestateInputVector()

        input_vect_model.resources = game_state.available_resources
        
        for i,player in enumerate(game_state.get_players_in_immediate_turn_priority()):
            player_vect = input_vect_model.players[i]
            player_vect.temp_resources = player.resource_tokens
            player_vect.perm_resources = player.resource_persistent
            player_vect.points = [player.sum_points]

            for j,card in enumerate(player.reserved_cards):
                if card is None:
                    continue
                player_vect.reserved_cards[j].costs = card.costs
                player_vect.reserved_cards[j].returns = to_hot_from_scalar(card.returns.value, 5)
                player_vect.reserved_cards[j].points = [card.points]


        for i,noble in enumerate(game_state.active_nobles):
            if noble is None:
                continue
            noble_vect = input_vect_model.nobles[i]
            
            noble_vect.costs = noble.costs
            noble_vect.points = [noble.points]

        for i,tier in enumerate(game_state.open_cards):
            tier_vect = input_vect_model.tiers[i]
            hidden_card = game_state.get_card_by_index(i * 5)
            if hidden_card is not None:
                tier_vect.hidden_card.costs = hidden_card.costs
                tier_vect.hidden_card.returns = to_hot_from_scalar(hidden_card.returns.value, 5)
                tier_vect.hidden_card.points = [hidden_card.points]
            for j,card in enumerate(tier):
                if card is None:
                    continue
                card_vect = tier_vect.open_cards[j]
                card_vect.costs = card.costs
                card_vect.returns = to_hot_from_scalar(card.returns.value, 5)
                card_vect.points = [card.points]
        

        flat_mapped_values = input_vect_model.flat_map()
        return map_all_to_tensors(flat_mapped_values)

        

class CardVector:
    def __init__(self):
        self.costs = [None]*5
        self.returns = [None]*5
        self.points = [None]
    def flat_map_into(self, prefix: str, into_dict: BetterParamDict[list[float]]):
        into_dict[prefix + "_costs"] = self.costs
        into_dict[prefix + "_returns"] = self.returns
        into_dict[prefix + "_points"] = self.points

class NobleVector:
    def __init__(self):
        self.costs = [None]*5
        self.points = [None]
    def flat_map_into(self, prefix: str, into_dict: BetterParamDict[list[float]]):
        into_dict[prefix + "_costs"] = self.costs
        into_dict[prefix + "_points"] = self.points

class RowVector:
    def __init__(self):
        self.hidden_card = CardVector()
        self.open_cards = [CardVector() for x in range(4)]
        self.points = [None]
    def flat_map_into(self, prefix: str, into_dict: BetterParamDict[list[float]]):
        flat_map_group(self.open_cards, prefix + "_open_card_", into_dict)
        self.hidden_card.flat_map_into(prefix + "_hidden_card", into_dict)
        into_dict[prefix + "_points"] = self.points

class PlayerVector:
    def __init__(self):
        self.temp_resources = [None]*6
        self.perm_resources = [None]*5
        self.points = [None]
        self.reserved_cards = [CardVector() for x in range(3)]

    def flat_map_into(self, prefix: str, into_dict: BetterParamDict[list[float]]):
        into_dict[prefix + "_temp_resources"] = self.temp_resources
        into_dict[prefix + "_perm_resources"] = self.perm_resources
        into_dict[prefix + "_points"] = self.points
        flat_map_group(self.reserved_cards, prefix + "_reserved_card_", into_dict)