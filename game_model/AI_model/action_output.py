from __future__ import annotations
import torch
from game_model.actor import Actor
from game_model.game import Game
from game_model.card import Card
from game_model.resource_types import ResourceType
from game_model.turn import Turn,Action_Type
from utilities.utils import Lazy
from utilities.subsamples import clone_shallow, pad_list


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
        action_output_dict : dict[str, torch.Tensor] = {}
        action_output_dict['action_choice'] = torch.Tensor(self.action_choice)
        action_output_dict['card_buy'] = torch.Tensor(self.card_buy)
        action_output_dict['reserve_buy'] = torch.Tensor(self.reserve_buy)
        action_output_dict['resource_token_draw'] = torch.Tensor(self.resource_token_draw)
        action_output_dict['noble_choice'] = torch.Tensor(self.noble_choice)
        action_output_dict['discard_choice'] = torch.Tensor(self.discard_choice)
        action_output_dict['discard_amounts'] = torch.Tensor(self.discard_amounts)

        return action_output_dict

    def map_dict_into_self(self, into_dict: dict[str, torch.Tensor]):
        self.action_choice = into_dict['action_choice'].tolist()
        self.card_buy = into_dict['card_buy'].tolist()
        self.reserve_buy = into_dict['reserve_buy'].tolist()
        self.resource_token_draw = into_dict['resource_token_draw'].tolist()
        self.noble_choice = into_dict['noble_choice'].tolist()
        self.discard_choice = into_dict['discard_choice'].tolist()
        self.discard_amounts = into_dict['discard_amounts'].tolist()
    
    def map_from_AI_output(action_output: ActionOutput,game:Game,player:Actor) -> Turn:
        '''
        TODO: map to a action tensor dictionary. should also return a tensor dictionary which represents the chosen action
        '''
        #Fit the AI output to valid game states
        fit_check = False
        turn: Turn = None
        
        #behavior: first it will try to validate the most preferred action, then the second most, etc.
        action_attempts = 0

        prioritized_resource_preferences : Lazy[list[ResourceType]] = Lazy(
            lambda: [ResourceType(i) for i, x in sorted(enumerate(action_output.resource_token_draw), key = lambda tup: tup[1], reverse=True)]
        )
        prioritized_card_indexes : Lazy[list[ResourceType]] = Lazy(
            lambda: [i for i, x in sorted(enumerate(action_output.card_buy + action_output.reserve_buy), key = lambda tup: tup[1], reverse=True)]
        )
        
        action = clone_shallow(action_output.action_choice)
        while fit_check == False and action_attempts < 5:
            best_action_index = action.index(max(action))
            action_num = best_action_index #find most preferred action
            action[best_action_index] = -10000000 #means it won't select this action again
            action_type = Action_Type(action_num)
            turn = Turn(action_type)

            if action_type==Action_Type.TAKE_THREE_UNIQUE:
                best_pick = _find_best_pick_three(prioritized_resource_preferences.val(), game.available_resources)
                if not (best_pick is None):
                    turn.resources_desired = best_pick
                    fit_check = True

            elif action_type==Action_Type.TAKE_TWO:
                best_pick = _find_best_pick_two(prioritized_resource_preferences.val(), game.available_resources)
                if not (best_pick is None):
                    turn.resources_desired = best_pick
                    fit_check = True

            elif action_type==Action_Type.BUY_CARD:
                best_pick = _find_best_card_buy(prioritized_card_indexes.val(), player, game)
                if not (best_pick is None):
                    turn.card_index = best_pick
                    fit_check = True

            elif action_type==Action_Type.RESERVE_CARD:
                best_pick = _find_best_card_to_reserve(prioritized_card_indexes.val(), player, game)
                if not (best_pick is None):
                    turn.card_index = best_pick
                    fit_check = True

            action_attempts+=1
        
        #taking noble goes here
        # TODO: is hack. but kinda mostly will work
        turn.noble_preference = max(enumerate(action_output.noble_choice))[0]
        
        # discarding tokens goes here
        turn.set_discard_preferences(action_output.discard_amounts)
        if action_attempts >= 5:
            ## for training, may be best to provide a noop when the game state prohibits any other actions
            return Turn(Action_Type.NOOP) 
        validate_msg = turn.validate(game,player)
        if validate_msg != None:
            return "Something went wrong and the AI->game mapper couldn't coerce a valid state. tried " + str(action_attempts) + " times. " + validate_msg
        
        return turn
    

def _find_best_pick_three(sorted_resource_prefs: list[ResourceType], available_resources: list[int]) -> list[float]:
    selected_num = 0
    output_selections = [0] * 5
    for next_pref in sorted_resource_prefs:
        if available_resources[next_pref.value] > 0:
            selected_num += 1
            output_selections[next_pref.value] = 1
            if selected_num >= 3:
                return output_selections
    return None

def _find_best_pick_two(sorted_resource_prefs: list[ResourceType], available_resources: list[int]) -> list[float]:
    output_selections = [0] * 5
    for next_pref in sorted_resource_prefs:
        if available_resources[next_pref.value] >= 4:
            output_selections[next_pref.value] = 2
            return output_selections
    return None

def _find_best_card_buy(
    prioritized_card_indexes: list[int], 
    player: Actor,
    game: Game) -> int:
    total_cards = game.config.total_available_card_indexes()
    for next_index in prioritized_card_indexes:
        card_target : Card = None
        if next_index >= total_cards: ## is reserved card index
            card_target = player.get_reserved_card(next_index - total_cards)
        else:
            if game.is_top_deck_index(next_index):
                continue
            card_target = game.get_card_by_index(next_index)
        if card_target is None:
            continue
        if player.can_purchase(card_target):
            return next_index
    return None

def _find_best_card_to_reserve(
    prioritized_card_indexes: list[int], 
    player: Actor,
    game: Game
    ) -> int:
    if not player.can_reserve_another():
        return None
    total_cards = game.config.total_available_card_indexes()
    for next_index in prioritized_card_indexes:
        card_target : Card = None
        if next_index >= total_cards: ## is reserved card index
            continue
        
        card_target = game.get_card_by_index(next_index)
        if card_target is None:
            continue
        return next_index
    return None
