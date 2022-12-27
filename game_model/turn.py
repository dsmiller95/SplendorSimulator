from enum import IntEnum
from game_model.actor import Actor
from game_model.card import Card
from game_model.game import Game

from game_model.resource_types import ResourceType
from game_model.turn_actions import add_reserved_card, purchase_card
from utilities.print_utils import stringify_resources

class Action_Type(IntEnum):
    TAKE_THREE_UNIQUE = 1,
    TAKE_TWO = 2,
    BUY_CARD = 3,
    RESERVE_CARD = 4,
    NOOP = 5 ## reserved for testing, player passes their turn

"""
Card indexes:
0: tier 1 top of deck
2: tier 1 2nd revealed card
5: tier 2 top of deck
9: tier 2 4th revealed card
[0... tiers * (open_cards_per_tier + 1)) : board reserve
[(tiers * (open_cards_per_tier + 1)...+max_reserved_cards) : card reserve
"""

class Turn:
    def __init__(
        self,
        action_type: Action_Type,
        resources: list[int] = None,
        card_index: int = None):
        self.action_type = action_type
        if not(resources is None) and len(resources) < 5:
            raise "only 5 valid resources to grab, 6 provided"
        self.resources = resources
        self.card_index = card_index
    
    def describe_state(self, game_state: Game, player: Actor) -> str:
        result = ""
        result += self.action_type.name + ": "
        if self.action_type == Action_Type.TAKE_THREE_UNIQUE or self.action_type == Action_Type.TAKE_TWO:
            result += stringify_resources(self.resources, ignore_empty=True)
        elif self.action_type == Action_Type.BUY_CARD or self.action_type == Action_Type.RESERVE_CARD:
            is_reserved_card = self.card_index >= game_state.config.total_available_card_indexes()
            reserved_card_index = self.card_index - game_state.config.total_available_card_indexes()
            if is_reserved_card:
                result += "reserved card " + str(reserved_card_index+1) + " [" + player.get_reserved_card(reserved_card_index).describe_state() + "]"
            else:
                (tier, select) = game_state.get_tier_and_selected(self.card_index)
                is_topdeck = select == 0

                result += "tier " + str(tier)
                if is_topdeck:
                    result += ", topdeck"
                else:
                    result += " card " + str(self.card_index) + ": [" + game_state.get_card_by_index(self.card_index).describe_state() + "]"
        elif self.action_type == Action_Type.NOOP:
            result = result[0:-3]
        return result

    def validate(self, game_state: Game, player: Actor) -> str:
        ## passing turn is always valid, for testing purposes
        if self.action_type == Action_Type.NOOP:
            return None
        ## Validating resource taking moves
        if (self.action_type == Action_Type.TAKE_THREE_UNIQUE or
            self.action_type == Action_Type.TAKE_TWO):
            total_buy = 0
            if self.action_type == Action_Type.TAKE_THREE_UNIQUE:
                for idx, resource in enumerate(self.resources):
                    if resource <= 0:
                        continue
                    total_buy += resource
                    if resource > 1:
                        return "cannot pick more than one from each resource on a unique take"
                    if game_state.available_resources[idx] < self.resources[idx]:
                        return "cannot take more resources than available in bank"
                    
                if total_buy > 3:
                    return "cannot take more than 3 total on unique take"
            if self.action_type == Action_Type.TAKE_TWO:
                for idx, resource in enumerate(self.resources):
                    if resource <= 0:
                        continue
                    if(total_buy > 0):
                        return "cannot pick more than one resource type on a Take Two"
                    total_buy += resource
                    if resource > 2:
                        ## 
                        return "cannot pick more than one on a unique take"
                    if game_state.available_resources[idx] < 4:
                        ## 
                        return "cannot take two from bank with less than 4 available"
            if total_buy + player.total_tokens() > game_state.config.max_resource_tokens:
                ## 
                return "cannot take tokens which would increase player bank above limit"
        else:
            is_reserved_card = self.card_index >= game_state.config.total_available_card_indexes()
            reserved_card_index = self.card_index - game_state.config.total_available_card_indexes()

            ## Validate card purchasing
            if self.action_type == Action_Type.BUY_CARD:
                target_card : Card = None
                if is_reserved_card:
                    target_card = player.get_reserved_card(reserved_card_index)
                else:
                    if game_state.is_top_deck_index(self.card_index) :
                        return "Cannot buy card from top deck"
                    target_card = game_state.get_card_by_index(self.card_index)
                if target_card is None:
                    ## 
                    return "no card in reserve at that index, or no card available in game due to card exhaustion"
                if not player.can_purchase(target_card):
                    return "not enough funds to purchase this card"
            
            if self.action_type == Action_Type.RESERVE_CARD:
                if is_reserved_card:
                    ## 
                    return "cannot reserve already reserved card"
                if not player.can_reserve_another():
                    return "maximum number of cards already reserved"
                target_card : Card = game_state.get_card_by_index(self.card_index)
                if target_card is None:
                    ## 
                    return "no card in reserve at that index, or no card available in game due to card exhaustion"
                
            
            return None
    

    """
    execute this action, modifying the game and player state
    be sure to validate this action against the game state and player before executing it
    """
    def execute(self, game_state: Game, player: Actor):
        if self.action_type == Action_Type.NOOP:
            return
        
        if (self.action_type == Action_Type.TAKE_THREE_UNIQUE or
            self.action_type == Action_Type.TAKE_TWO):
            for idx, amount in enumerate(self.resources):
                game_state.available_resources[idx] -= amount
                player.resource_tokens[idx] += amount
            return
        
        is_reserved_card = self.card_index >= game_state.config.total_available_card_indexes()
        reserved_card_index = self.card_index - game_state.config.total_available_card_indexes()

        if self.action_type == Action_Type.BUY_CARD:
            target_card : Card = None
            if is_reserved_card:
                target_card = player.take_reserved_card(reserved_card_index)
            else:
                target_card = game_state.take_card_by_index(self.card_index)
            purchase_card(player, game_state, target_card)
        elif self.action_type == Action_Type.RESERVE_CARD:
            target_card = game_state.take_card_by_index(self.card_index)
            add_reserved_card(player, game_state, target_card)