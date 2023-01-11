from __future__ import annotations
import torch
from game_model.AI_model.index_mapper import CombinatorialIndexMapping
from game_model.actor import Actor
from game_model.game import Game
from game_model.card import Card
from game_model.resource_types import ResourceType
from game_model.turn import Turn,Action_Type
from utilities.utils import Lazy
from utilities.subsamples import clone_shallow, pad_list

pick_three_choices = CombinatorialIndexMapping(5, 3, allow_pick_multiple=False, allow_pick_less=False)
discard_choices = CombinatorialIndexMapping(6, 3, allow_pick_multiple=True, allow_pick_less=True)

class ActionOutput:
    def __init__(self):
        self.action_choice: torch.Tensor = torch.Tensor([0] * 4)
        self.card_pref: torch.Tensor = torch.Tensor([0] * 15) #this will be used for both card buying and reserving
        self.reserve_buy: torch.Tensor= torch.Tensor([0] * 3) #this is specifically for buying cards from reserve
        self.noble_choice: torch.Tensor = torch.Tensor([0] * 5)
        
        self.pick_three_choice: torch.Tensor = torch.Tensor([0] * pick_three_choices.total_possible_options())
        self.pick_two_choice: torch.Tensor = torch.Tensor([0] * 5)
        self.discard_combination_choice: torch.Tensor = torch.Tensor([0] * discard_choices.total_possible_options())
    
    def in_dict_form(self):
        action_output_dict : dict[str, torch.Tensor] = {}
        action_output_dict['action_choice'] = torch.Tensor(self.action_choice)
        action_output_dict['card_buy'] = torch.Tensor(self.card_pref)
        action_output_dict['reserve_buy'] = torch.Tensor(self.reserve_buy)
        action_output_dict['noble_choice'] = torch.Tensor(self.noble_choice)

        action_output_dict['pick_three_choice'] = torch.Tensor(self.pick_three_choice)
        action_output_dict['pick_two_choice'] = torch.Tensor(self.pick_two_choice)
        action_output_dict['discard_combination_choice'] = torch.Tensor(self.discard_combination_choice)

        return action_output_dict

    def map_dict_into_self(self, into_dict: dict[str, torch.Tensor]):
        self.action_choice = into_dict['action_choice']
        self.card_pref = into_dict['card_buy']
        self.reserve_buy = into_dict['reserve_buy']
        self.noble_choice = into_dict['noble_choice']

        self.pick_three_choice = into_dict['pick_three_choice']
        self.pick_two_choice = into_dict['pick_two_choice']
        self.discard_combination_choice = into_dict['discard_combination_choice']
    
    @staticmethod
    def map_from_AI_output(forward_result: dict[str, torch.Tensor],game:Game,player:Actor) -> tuple[Turn | str, dict[str, torch.Tensor]]:
        '''
        TODO: map to a action tensor dictionary. should also return a tensor dictionary which represents the chosen action
        '''

        action_output = ActionOutput()
        action_output.map_dict_into_self(forward_result)
        turn_index_tuple = ActionOutput._map_internal(action_output, game, player)
        if turn_index_tuple[1] == None: # This happens when _map_internal fails to find a valid action and returns NOOP
            turn,_ = turn_index_tuple
            zero_dict = ActionOutput().in_dict_form() #return an ActionOutput with all 0's because no action was taken
        else:
            (turn, taken_action_indexes) = turn_index_tuple
            zero_dict = ActionOutput().in_dict_form()
            for key in taken_action_indexes:
                chosen_index = taken_action_indexes[key]
                zero_dict[key][chosen_index] = 1 #put a 1 at the location of the action that was taken
            
        
        return (turn, zero_dict)
    @staticmethod
    def _map_internal(action_output: ActionOutput,game:Game,player:Actor) -> tuple[Turn | str, dict[str, int]]:

        #Fit the AI output to valid game states
        fit_check: bool = False
        turn: Turn = None
        
        #behavior: first it will try to validate the most preferred action, then the second most, etc.
        action_attempts: int = 0

        prioritized_card_indexes : Lazy[list[ResourceType]] = Lazy(
            lambda: [i for i, x in sorted(enumerate(action_output.card_pref.tolist() + action_output.reserve_buy.tolist()), key = lambda tup: tup[1], reverse=True)]
        )
        
        chosen_action_indexes: dict[str, int] = {}
        action = clone_shallow(action_output.action_choice.tolist())
        while fit_check == False and action_attempts < 5:
            best_action_index = action.index(max(action))
            action[best_action_index] = -10000000 #means it won't select this action again
            action_type = Action_Type(best_action_index)
            chosen_action_indexes['action_choice'] = best_action_index
            turn = Turn(action_type)
            # Diagnosis printout
            # print(action_type,'\n',
            #       'pick_three_choice',[f'{val:.2f}' for val in action_output.pick_three_choice.tolist()],'\n',
            #       'pick_two_choice',[f'{val:.2f}' for val in action_output.pick_two_choice.tolist()],'\n',
            #       'card_buy_choice',[f'{val:.2f}' for val in action_output.card_pref.tolist()],'\n',
            #       'reserve_choice',[f'{val:.2f}' for val in action_output.reserve_buy.tolist()],'\n',
            #       'available_resources',game.available_resources,'\n\n')
            if action_type==Action_Type.TAKE_THREE_UNIQUE:
                best_pick = _find_best_pick_three_from_tensor(action_output.pick_three_choice, game.available_resources)
                if not (best_pick is None):
                    chosen_action_indexes['pick_three_choice'] = best_pick
                    turn.resources_desired = pick_three_choices.map_from_index(best_pick)
                    fit_check = True

            elif action_type==Action_Type.TAKE_TWO:
                best_pick = _find_best_pick_two_from_tensor(action_output.pick_two_choice, game.available_resources)
                if not (best_pick is None):
                    chosen_action_indexes['pick_two_choice'] = best_pick
                    pick = [0] * 5
                    pick[best_pick] = 2
                    turn.resources_desired = pick
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

            if fit_check == True and turn.card_index != None:
                total_cards = game.config.total_available_card_indexes()
                if turn.card_index >= total_cards:
                    chosen_action_indexes['reserve_buy'] = turn.card_index - total_cards
                else:
                    chosen_action_indexes['card_buy'] = turn.card_index - total_cards
                pass

            action_attempts+=1
        
        #taking noble goes here
        # TODO: is hack. but kinda mostly will work.
        # TODO: should we only set the "taken action" tensor value iff the noble is actually rewarded after selection?
        turn.noble_preference = max(enumerate(action_output.noble_choice.tolist()))[0]
        chosen_action_indexes['noble_choice'] = turn.noble_preference 
        
        # discarding tokens goes here
        discard_choice = _find_valid_discard_options(action_output.discard_combination_choice, player.resource_tokens, turn)
        if discard_choice is not None:
            chosen_action_indexes['discard_combination_choice'] = discard_choice
            turn.set_discard_preferences(discard_choices.map_from_index(discard_choice))

        if action_attempts >= 5:
            ## for training, may be best to provide a noop when the game state prohibits any other actions
            return (Turn(Action_Type.NOOP),None)
        validate_msg = turn.validate(game,player)
        if validate_msg != None:
            return "Something went wrong and the AI->game mapper couldn't coerce a valid state. tried " + str(action_attempts) + " times. " + validate_msg
        
        return (turn, chosen_action_indexes)

def _find_valid_discard_options(q_vector_discard_pref: torch.Tensor, available_resources_to_player: list[int], current_turn: Turn) -> int:
    if current_turn.resources_desired != None:
        available_resources_to_player = clone_shallow(available_resources_to_player)
        for i, x in enumerate(current_turn.resources_desired):
            available_resources_to_player[i] += x
    
    sorted, indices = q_vector_discard_pref.sort(descending=True)
    for next_pref in range(len(q_vector_discard_pref)):
        index = indices[next_pref]
        discarded = discard_choices.map_from_index(index)
        if not any([available_resources_to_player[i] < x for i, x in enumerate(discarded)]):
            return index
    return None

def _find_best_pick_three_from_tensor(q_vector_choice_pref: torch.Tensor, available_resources: list[int] ) -> int:
    '''
    finds a valid pick three action with the highest q-value and returns the index of that value
    '''
    if sum([1 if x > 0 else 0 for x in available_resources]) < 3:
        return None
    sorted, indices = q_vector_choice_pref.sort(descending=True)
    for next_pref in range(len(q_vector_choice_pref)):
        index = indices[next_pref]
        pick_choice = pick_three_choices.map_from_index(index)
        if not any([available_resources[i] < x for i, x in enumerate(pick_choice)]):
            return index
    return None

def _find_best_pick_two_from_tensor(q_vector_choice_pref: torch.Tensor, available_resources: list[int] ) -> int:
    '''
    finds a valid pick two action with the highest q-value and returns the index of that value
    '''
    if not any([x > 4 for x in available_resources]):
        return None
    sorted, indices = q_vector_choice_pref.sort(descending=True)
    for next_pref in range(len(q_vector_choice_pref)):
        index = indices[next_pref] ## index here happens to be the literal resource index, because we only need to pick 1
        if available_resources[index] > 4:
            return index
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
