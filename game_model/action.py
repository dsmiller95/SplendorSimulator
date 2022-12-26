from enum import Enum
from game_model.actor import Actor
from game_model.card import Card
from game_model.game import Game

from game_model.resource_types import ResourceType

class Action_Type(Enum):
    TAKE_THREE_UNIQUE = 1,
    TAKE_TWO = 2,
    BUY_CARD = 3,
    RESERVE_CARD = 4

"""
Card indexes:
[0... tiers * open_cards_per_tier) : board reserve
[(tiers * open_cards_per_tier)...+max_reserved_cards) : card reserve
"""

class Turn:
    def __init__(
        self,
        action_type: Action_Type,
        resources: list[int] = None,
        card_index: int = None):
        self.action_type = action_type
        self.resources = resources
        self.card_index = card_index
    
    def validate(self, game_state: Game, player: Actor) -> bool:
        ## Validating resource taking moves
        total_buy = 0
        if self.action_type == Action_Type.TAKE_THREE_UNIQUE:
            for idx, resource in enumerate(self.resources):
                if resource <= 0:
                    continue
                total_buy += resource
                if resource > 1:
                    ## cannot pick more than one from each resource on a unique take
                    return False
                if game_state.available_resources[idx] < self.resources:
                    ## cannot take more resources than available in bank
                    return False
                
            if total_buy > 3:
                ## cannot take more than 3 total on unique take
                return False
        if self.action_type == Action_Type.TAKE_TWO:
            for idx, resource in enumerate(self.resources):
                if resource <= 0:
                    continue
                if(total_buy > 0):
                    ## cannot pick more than one resource type on a Take Two
                    return False
                total_buy += resource
                if resource > 2:
                    ## cannot pick more than one on a unique take
                    return False
                if game_state.available_resources[idx] < self.resources:
                    ## cannot take more resources than available in bank
                    return False
        if total_buy + player.total_tokens() > game_state.config.max_resource_tokens:
            ## cannot take tokens which would increase player bank above limit
            return False
        
        is_reserved_card = self.card_index >= game_state.config.total_available_cards()
        reserved_card_index = self.card_index - game_state.config.total_available_cards()

        ## Validate card purchasing
        if self.action_type == Action_Type.BUY_CARD:
            target_card : Card = None
            if is_reserved_card:
                target_card = player.get_reserved_card(reserved_card_index)
            else:
                target_card = game_state.get_card_by_index(self.card_index)
            if target_card is None:
                ## no card in reserve at that index, or no card available in game due to card exhaustion
                return False
            if not player.can_purchase(target_card):
                return False
        
        if self.action_type == Action_Type.RESERVE_CARD:
            if is_reserved_card:
                ## cannot reserve already reserved card
                return False
            if not player.can_reserve_another():
                return False
            target_card : Card = game_state.get_card_by_index(self.card_index)
            if target_card is None:
                ## no card in reserve at that index, or no card available in game due to card exhaustion
                return False
            
        
        return True