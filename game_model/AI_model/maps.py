from game_model.game import Game
from game_model.actor import Actor
from game_model.resource_types import ResourceType
from game_model.turn import Turn,Action_Type
from utilities.subsamples import clone_shallow, pad_list
from itertools import chain
from game_model.AI_model.action_output import ActionOutput

from utilities.utils import Lazy

class VectorBuilder:
    def __init__(self, vect_len:int):
        self.vector = [None]*vect_len
        self.locked = False

    def put(self,idx,sublist):
        '''insert the sublist into the list at specified location'''
        if self.locked:
            raise 'you can\'t change this anymore'
        if type(sublist)==int:
            sublist = [sublist]
        self.vector[idx:idx+len(sublist)] = sublist
        return list
    
    def return_vector(self):
        self.locked = True
        return self.vector



def map_to_AI_input(game_state: Game):
     
    input_vector = VectorBuilder(236)
    #populate a player vector so we can rotate it
    #into the right position to have the current player
    # on the top of the list always, so that the "view"
    #the model has of the game is always from the same
    #relative perspective
    reserved_card_shape = (5+5+1)
    player_shape = 6+5+(3*reserved_card_shape)
    player_vector = VectorBuilder(4*player_shape)
    for i,player in enumerate(game_state.players + [None] * (4 - len(game_state.players)) ):
        if player is None:
            player_vector.put((i*player_shape)+0, [0] * player_shape)
            continue

        player_vector.put((i*player_shape)+0,player.resource_tokens)
        player_vector.put((i*player_shape)+6,player.resource_persistent)

        for j,card in enumerate(player.reserved_cards):
            offset = (i*player_shape)+11
            if card == None:
                player_vector.put((offset+(j*reserved_card_shape))+0,[0]*5)
                player_vector.put((offset+(j*reserved_card_shape))+5,[0]*5)
                player_vector.put((offset+(j*reserved_card_shape))+10,0)
            else:
                player_vector.put((offset+(j*reserved_card_shape))+0,card.costs)
                player_vector.put((offset+(j*reserved_card_shape))+5,[1 if card.reward.value == i else 0 for i in range(0, 5)])
                player_vector.put((offset+(j*reserved_card_shape))+10,card.points)
    
    player_num = game_state.get_current_player_index()

    player_vector_rotated = player_vector.return_vector()[player_num*player_shape:] + player_vector.return_vector()[:player_num*player_shape]
    input_vector.put(0,player_vector_rotated)

    noble_shape = 5+1
    for i,noble in enumerate(pad_list(game_state.active_nobles, 5)):
        if noble is None:
            input_vector.put((player_shape*4)+(i*noble_shape),[0] * noble_shape)
            continue
        input_vector.put((player_shape*4)+(i*noble_shape),noble.costs)
        input_vector.put((player_shape*4)+(i*noble_shape+5),noble.points)
    
    for i,tier in enumerate(game_state.open_cards):
        for j,card in enumerate(tier):
            card_size = 5+1+1
            tier_size = 4*card_size
            offset = (player_shape*4)+(noble_shape*5)+(i*tier_size)+(j*card_size)
            input_vector.put(offset,card.costs) 
            input_vector.put(offset+5,card.reward.value)
            input_vector.put(offset+6,card.points)

    return input_vector.return_vector()

def map_from_AI_output(action_output: ActionOutput,game:Game,player:Actor):
    #Fit the AI output to valid game states
    fit_check = False
    turn: Turn = None
    
    #behavior: first it will try to validate the most preferred action, then the second most, etc.
    action_attempts = 0

    prioritized_resource_preferences : Lazy[list[ResourceType]] = Lazy(
        lambda: [ResourceType(x[0]) for x in sorted(enumerate(action_output.resource_token_draw), key = lambda x: x[1], reverse=True)]
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
            if not best_pick is None:
                turn.resources_desired = best_pick
                fit_check = True

        elif action_type==Action_Type.TAKE_TWO:
            best_pick = _find_best_pick_two(prioritized_resource_preferences.val(), game.available_resources)
            if not best_pick is None:
                turn.resources_desired = best_pick
                fit_check = True

        elif action_type==Action_Type.BUY_CARD:
            visible_cards = [card for i, card in enumerate(action_output.card_buy) if i % 5 != 0]
            visible_cards = list(chain(*visible_cards)) #flatten to 1d list
            indices_by_dislike = sorted(range(len(visible_cards)),
                                key = lambda sub: visible_cards[sub])
            indices_by_desire = indices_by_dislike.reverse() #is this worth an extra variable for clarity?
            for possible_buy in indices_by_desire:
                buy_index = (1*(possible_buy//4)+possible_buy)
                turn.card_index(buy_index)
                if turn.validate(game,player) == None:  
                    fit_check = True

        elif action_type==Action_Type.RESERVE_CARD:
            reserved_cards = [card for i, card in enumerate(action_output.card_buy) if i % 5 == 0]
            reserved_cards = list(chain(*reserved_cards)) #flatten to 1d list
            indices_by_dislike = sorted(range(len(reserved_cards)),
                                key = lambda sub: reserved_cards[sub])
            indices_by_desire = indices_by_dislike.reverse() #is this worth an extra variable for clarity?
            for possible_buy in indices_by_desire:
                buy_index = (4*5)+possible_buy
                turn.card_index(buy_index)
                if turn.validate(game,player) == None:  
                    fit_check = True

        action_attempts+=1
    
    #taking noble goes here
    
    #discarding tokens goes here

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
