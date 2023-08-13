from enum import IntEnum
from game_model.actor import Actor
from game_model.card import Card
from game_model.game import Game
from game_data.game_config_data import GameConfigData
from game_model.resource_types import ResourceType
from game_model.turn_actions import add_reserved_card, purchase_card
from utilities.print_utils import stringify_resources

class Action_Type(IntEnum):
    TAKE_THREE_UNIQUE = 0,
    TAKE_TWO = 1,
    BUY_CARD = 2,
    RESERVE_CARD = 3,
    NOOP = 4 ## reserved for testing, player passes their turn

"""                                  vectorize-------------------------> Ihidden | I   I   I   I ---,
Card indexes:                                                      ,----------------<---------------'
0: tier 1 top of deck                                              '--> IIhidden | II  II  II  II --,
2: tier 1 2nd revealed card                                        ,----------------<---------------'
5: tier 2 top of deck                                              '-> IIIhidden | III III III III -,
9: tier 2 4th revealed card                                        ,----------------<---------------'
[0... tiers * (open_cards_per_tier + 1)) : board reserve           '-----> rsrvd1 rsrvd2 rsrvd3
[(tiers * (open_cards_per_tier + 1)...+max_reserved_cards) : card reserve
"""

default_discard_pref = [.06, .05, .04, .03, .02, .01]

game_config = game_config = GameConfigData.read_file("./game_data/cards.csv")

class Turn:
    def __init__(
        self,
        action_type: Action_Type,
        resources_desired: list[int] = None,
        card_index: int = None, #which card to pick
        noble_preference: float = 0, ## which noble to pick. if more than one noble available, will pick the one closest to this index
        ## which tokens to discard. if necessary, will be used to determine which tokens to discard first when we over-inventory
        ## values in [0, 0.5) will not automatically discard. the highest value 
        discard_preference_levels: list[float] = default_discard_pref,
        ):
        
        self.action_type = action_type
        if not(resources_desired is None) and len(resources_desired) < 5:
            raise "only 5 valid resources to grab, 6 provided"
        self.resources_desired = resources_desired
        self.card_index = card_index
        self.noble_preference = noble_preference

        self.set_discard_preferences(discard_preference_levels)
        
        self.last_discarded_mandatory = 0
        self.last_discarded_optional = 0

    def set_discard_preferences(self, new_preferences: list[int]):
        if len(new_preferences) < 6:
            raise RuntimeError("discard preferences must be length 6, matching number of resource types. got " + str(len(new_preferences)))
        self._discard_commands = [(ResourceType(i), x) for i, x in enumerate(new_preferences)]
        self._discard_commands.sort(key= lambda x: x[1], reverse=True)
    
    def describe_state(self, game_state: Game, player: Actor) -> str:
        result = ""
        result += self.action_type.name + ": "
        if self.action_type == Action_Type.TAKE_THREE_UNIQUE or self.action_type == Action_Type.TAKE_TWO:
            result += stringify_resources(self.resources_desired, ignore_empty=True)
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
        '''Check that a turn is valid, if it's not, return the reason that it isn't'''
        ## passing turn is always valid, for testing purposes
        if self.action_type == Action_Type.NOOP:
            return None
        ## Validating resource taking moves
        if (self.action_type == Action_Type.TAKE_THREE_UNIQUE or
            self.action_type == Action_Type.TAKE_TWO):
            if self.resources_desired is None:
                return "no resource preferences provided for a take n action"
            total_buy = 0
            if self.action_type == Action_Type.TAKE_THREE_UNIQUE:
                for idx, resource in enumerate(self.resources_desired):
                    if resource <= 0:
                        continue
                    total_buy += resource
                    if resource > 1:
                        return "cannot pick more than one from each resource on a unique take"
                    if game_state.available_resources[idx] < self.resources_desired[idx]:
                        return "cannot take more resources than available in bank"
                    
                if total_buy > 3:
                    return "cannot take more than 3 total on unique take"
            if self.action_type == Action_Type.TAKE_TWO:
                for idx, resource in enumerate(self.resources_desired):
                    if resource <= 0:
                        continue
                    if total_buy > 0:
                        return "cannot pick more than one resource type on a Take Two"
                    total_buy += resource
                    if resource > 2:
                        ## 
                        return "cannot pick more than one on a unique take"
                    if game_state.available_resources[idx] < 4:
                        ## 
                        return "cannot take two from bank with less than 4 available"
        else:
            if self.card_index is None:
                return "no card index provided for a pick card action"
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
            for idx, amount in enumerate(self.resources_desired):
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

    def _discard_down(self, game_state: Game, player: Actor):
        # first discard outright if values round to >1

        #remove its ability to voluntarily discard, but keep its ability to choose what will be discarded if it goes over 7 tokens
        '''
        self.last_discarded_optional = 0
        for discard_command in self._discard_commands:
           true_amount = min(player.resource_tokens[discard_command[0].value], round(discard_command[1]))
           if true_amount <= 0:
               continue
           game_state.give_tokens_to_player(player, discard_command[0].value, -true_amount)
           self.last_discarded_optional += true_amount
        '''
        
        total_tokens = sum(player.resource_tokens)
        if total_tokens <= game_config.max_resource_tokens:
            return

        self.last_discarded_mandatory = 0
        for try_num in range(5):
            for next_discard in self._discard_commands:
                if player.resource_tokens[next_discard[0].value] > 0:
                    total_tokens -= 1
                    game_state.give_tokens_to_player(player, next_discard[0].value, -1)
                    self.last_discarded_mandatory += 1
                    if total_tokens <= game_config.max_resource_tokens:
                        return

        raise RuntimeError("could not discard down enough. error in discard down algorithm")
    
    def as_serializable_data(self) -> dict:
        return {
            "type" : self.action_type.name,
            "discarded_optional": self.last_discarded_optional,
            "discarded_mandatory": self.last_discarded_mandatory,
            "resource_desired": self.resources_desired,
            "card_index": self.card_index,
            "noble_preference": self.noble_preference,
            "discard_preference": self._discard_commands
        }



