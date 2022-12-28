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

default_discard_pref = [ResourceType.RUBY, ResourceType.EMERALD, ResourceType.SAPPHIRE, ResourceType.DIAMOND, ResourceType.ONYX, ResourceType.GOLD]

class Turn:
    def __init__(
        self,
        action_type: Action_Type,
        resources: list[int] = None,
        card_index: int = None,
        noble_preference: float = 0, ## which noble to pick. if more than one noble available, will pick the one closest to this index
        discard_preference: list[ResourceType] = default_discard_pref ## ordered list of which resource to discard if maximum is reached
        ):
        self.action_type = action_type
        if not(resources is None) and len(resources) < 5:
            raise "only 5 valid resources to grab, 6 provided"
        self.resources = resources
        self.card_index = card_index
        self.noble_preference = noble_preference
        self.discard_preference = discard_preference
    
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
        self._execute_primary(game_state, player)
        self._discard_down(game_state, player)
        self._reward_nobles(game_state, player)
    
    def _execute_primary(self, game_state: Game, player: Actor):
        if self.action_type == Action_Type.NOOP:
            return
        
        if (self.action_type == Action_Type.TAKE_THREE_UNIQUE or
            self.action_type == Action_Type.TAKE_TWO):
            for idx, amount in enumerate(self.resources):
                game_state.give_tokens_to_player(player, idx, amount)
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
    
    def _reward_nobles(self, game_state: Game, player: Actor):
        valid_noble_indexes = [i for i, x in enumerate(game_state.active_nobles) if x.satisfied_by(player.resource_persistent)]
        min_noble_dist = 100000
        chosen_noble_index = None
        for noble_index in valid_noble_indexes:
            dist = abs(noble_index - self.noble_preference)
            if dist < min_noble_dist:
                min_noble_dist = dist
                chosen_noble_index = noble_index
        
        if chosen_noble_index is None:
            return
        chosen_noble = game_state.active_nobles.pop(chosen_noble_index)
        player.claimed_nobles.append(chosen_noble)
        player.sum_points += chosen_noble.points

    """
    discard resources in order of the discard preferences in the turn object
    a single token of the first preference will be discarded, then a single token of the next preference, so on until total tokens are below 10
    we will loop through the preferences as many times as is needed to get below 10
    if any loop through the preferences does not result in a discarded token, then the preferences will be replaced with numeric-order: ruby up to gold. this is the default.
    """
    def _discard_down(self, game_state: Game, player: Actor):
        total_tokens = sum(player.resource_tokens)
        if total_tokens <= 10:
            return

        total_tokens_discarded_in_round = 0
        
        current_preference = self.discard_preference

        for try_num in range(0, 6):
            for next_discard in current_preference:
                if player.resource_tokens[next_discard.value] > 0:
                    total_tokens_discarded_in_round += 1
                    total_tokens -= 1
                    game_state.give_tokens_to_player(player, next_discard.value, -1)
                    if total_tokens <= 10:
                        return

            if total_tokens_discarded_in_round <= 0:
                current_preference = default_discard_pref
            total_tokens_discarded_in_round = 0

        raise RuntimeError("could not discard down enough. error in discard down algorithm")
