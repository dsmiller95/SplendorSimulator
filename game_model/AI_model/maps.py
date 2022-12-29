from game_model.AI_model.gamestate_input import GamestateInputVector
from game_model.card import Card
from game_model.game import Game
from game_model.actor import Actor
from game_model.resource_types import ResourceType
from game_model.turn import Turn,Action_Type
from utilities.subsamples import clone_shallow, pad_list
from itertools import chain
from game_model.AI_model.action_output import ActionOutput

from utilities.utils import Lazy

def to_hot_from_scalar(scalar: int, length: int) -> list[int]:
    return [1 if scalar == i else 0 for i in range(length)]

def map_to_AI_input(game_state: Game) -> dict[str, list[float]]:
    input_vect_model = GamestateInputVector()
    input_vect_flattened : dict = {}

    input_vect_model.resources = game_state.available_resources
    
    for i,player in enumerate(game_state.get_players_in_immediate_turn_priority()):
        if player is None:
            input_vect_flattened['player_'+str(i)+'_temp_resources'] = [None]*6
            input_vect_flattened['player_'+str(i)+'_perm_resources'] = [None]*5
            input_vect_flattened['player_'+str(i)+'_points'] = [None]
            for j in range(len(player.reserved_cards)):
                input_vect_flattened['player_'+str(i)+'_reserved_card_'+str(j)+'_costs'] = [None]*5
                input_vect_flattened['player_'+str(i)+'_reserved_card_'+str(j)+'_returns'] = [None]*5
                input_vect_flattened['player_'+str(i)+'_reserved_card_'+str(j)+'_points'] = [None]
            continue

        player_vect = input_vect_model.players[i]
        player_vect.temp_resources = player.resource_tokens
        input_vect_flattened['player_'+str(i)+'_temp_resources'] = player_vect.temp_resources
        player_vect.perm_resources = player.resource_persistent
        input_vect_flattened['player_'+str(i)+'_perm_resources'] = player_vect.perm_resources
        player_vect.points = [player.sum_points]
        input_vect_flattened['player_'+str(i)+'_points'] = player_vect.points

        for j,card in enumerate(player.reserved_cards):
            if card is None:
                input_vect_flattened['player_'+str(i)+'_reserved_card_'+str(j)+'_costs'] = [None]*5
                input_vect_flattened['player_'+str(i)+'_reserved_card_'+str(j)+'_returns'] = [None]*5
                input_vect_flattened['player_'+str(i)+'_reserved_card_'+str(j)+'_points'] = [None]
                continue
            player_vect.reserved_cards[j].costs = card.costs
            input_vect_flattened['player_'+str(i)+'_reserved_card_'+str(j)+'_costs'] = player_vect.reserved_cards[j].costs
            player_vect.reserved_cards[j].returns = to_hot_from_scalar(card.returns.value, 5)
            input_vect_flattened['player_'+str(i)+'_reserved_card_'+str(j)+'_returns'] = player_vect.reserved_cards[j].returns
            player_vect.reserved_cards[j].points = [card.points]
            input_vect_flattened['player_'+str(i)+'_reserved_card_'+str(j)+'_points'] = player_vect.reserved_cards[j].points


    noble_shape = 5+1
    for i,noble in enumerate(game_state.active_nobles):
        if noble is None:
            input_vect_flattened['board_noble_'+str(i)+'costs'] = [None]*5
            input_vect_flattened['board_noble_'+str(i)+'points'] = [None]
            continue
        noble_vect = input_vect_model.nobles[i]
        
        noble_vect.costs = noble.costs
        input_vect_flattened['board_noble_'+str(i)+'_costs'] = noble_vect.costs
        noble_vect.points = [noble.points]
        input_vect_flattened['board_noble_'+str(i)+'_points'] = noble_vect.points

    for i,tier in enumerate(game_state.open_cards):
        tier_vect = input_vect_model.tiers[i]
        hidden_card = game_state.get_card_by_index(i * 5)
        tier_vect.hidden_card.costs = hidden_card.costs
        tier_vect.hidden_card.returns = to_hot_from_scalar(hidden_card.returns.value, 5)
        tier_vect.hidden_card.points = [hidden_card.points]
        input_vect_flattened['tier_'+str(i)+'_hidden_card_costs'] = tier_vect.hidden_card.costs
        input_vect_flattened['tier_'+str(i)+'_hidden_card_returns'] = tier_vect.hidden_card.returns
        input_vect_flattened['tier_'+str(i)+'_hidden_card_points'] = tier_vect.hidden_card.points
        for j,card in enumerate(tier):
            card_vect = tier_vect.open_cards[j]
            card_vect.costs = card.costs
            input_vect_flattened['tier_'+str(i)+'_open_card_'+str(j)+'costs'] = card_vect.costs
            card_vect.returns = to_hot_from_scalar(card.returns.value, 5)
            input_vect_flattened['tier_'+str(i)+'_open_card_'+str(j)+'_returns'] = card_vect.returns
            card_vect.points = [card.points]
            input_vect_flattened['tier_'+str(i)+'_open_card_'+str(j)+'points'] = card_vect.points
    

    return input_vect_model.flat_map()

def map_from_AI_output(action_output: ActionOutput,game:Game,player:Actor):
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
    while fit_check == False and action_attempts < 4:
        best_action_index = action.index(max(action))
        action_num = best_action_index #find most preferred action
        action[best_action_index] = 0 #means it won't select this action again
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
    action_output.noble_choice = max(enumerate(action_output.noble_choice))[0]
    
    # discarding tokens goes here
    turn.set_discard_preferences(action_output.discard_amounts)

    if turn.validate(game,player) != None:
        return "Something went wrong and the AI->game mapper couldn't coerce a valid state"
    
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