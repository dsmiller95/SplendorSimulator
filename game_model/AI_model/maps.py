from game_model.game import Game
from game_model.actor import Actor
from game_model.turn import Turn,Action_Type
from utilities.subsamples import pad_list
from itertools import chain

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

def map_from_AI_output(output_vector: list[float],game:Game,player:Actor):

    #Deconstruct AI output into components
    action = list[float] = [None]*4
    for i in range(4):
        action.append(output_vector.pop(0))
    tiers: list[list[float]] = [[None]*5]*3
    for tier in range(3):
        cards = []
        for card in range(5):
            cards.append(output_vector.pop(0))
        tiers[tier] = cards
    purchase_reserve = output_vector.pop(0)
    resource_draw: list[float] = []*5
    for i in range(5):
        resource_draw.append(output_vector.pop(0))
    noble_choice = output_vector.pop(0)
    discard_resources = output_vector.pop(0) #Todo: clamp between 0 and 7 (max number of possible tokens)
    for i in range(6):
        discard_numbers.append(output_vector.pop(0))

    #Fit the AI output to valid game states
    fit_check = False
    turn = Turn()
    
    #behavior: first it will try the closest action to the float output in the 0-4 space
    #next it will try the second closes, then third, then 4th
    tries = 0

    while fit_check == False and tries <4:

        action_num = action.index(max(action))
        action_num[action.index(max(action))] = 0 #means it won't select this action again
        action_type = Action_Type[action_num]

        if action_type==Action_Type.TAKE_THREE_UNIQUE:
            if sum(round(resource_draw)) == 3 and 2 not in round(resource_draw): #the only valid outcome
                turn.resources_desired = round(resource_draw)
            else:
                #Normalize the array to 0-1, assign 1's to the three highest values, 0 to the others
                resource_draw_normalized = [elem/max(resource_draw) for elem in resource_draw]
                three_highest_indices = sorted(range(len(resource_draw_normalized)),
                                        key = lambda sub: resource_draw_normalized[sub])[-3:]
                resource_draw = [1 if i in three_high_indices else 0 for i,elem in enumerate(test_list_normalized)]
                turn.resources_desired = round(resource_draw)
            if turn.validate(game,player) == None:  
                fit_check = True

        elif action_type==Action_Type.TAKE_TWO:
            if sum(round(resource_draw)) == 2 and 1 not in round(resource_draw):
                turn.resources_desired = round(resource_draw)
            else:
                #Normalize the array to 0-1, assign 1's to the two highest values, 0 to the others
                resource_draw_normalized = [elem/max(resource_draw) for elem in resource_draw]
                three_highest_indices = sorted(range(len(resource_draw_normalized)),
                                        key = lambda sub: resource_draw_normalized[sub])[-2:]
                resource_draw = [1 if i in three_high_indices else 0 for i,elem in enumerate(test_list_normalized)]
                turn.resources_desired = round(resource_draw)
            if turn.validate(game,player) == None:  
                fit_check = True

        elif action_type==Action_Type.BUY_CARD:
            visible_cards = [cards[1:] for cards in tiers]
            visible_cards = list(chain(*visible_cards)) #flatten to 1d list
            indices_by_dislike = sorted(range(len(resource_draw_normalized)),
                                key = lambda sub: resource_draw_normalized[sub])
            indices_by_desire = indices_by_dislike.reverse() #is this worth an extra variable for clarity?
            for possible_buy in indices_by_desire:

                tiers_temp = list[list[float]] = [[0]*5]*3
                #assign a 1 to the location where the card the AI wants to buy is
                tiers_temp[(int(possible_buy/4)*1)+possible_buy] = 1
                turn.card_index(tiers_temp)
                if turn.validate(game,player) == None:
                    break
            if turn.validate(game,player) == None:  
                fit_check = True
        elif action_type==Action_Type.RESERVE_CARD:
            if turn.validate(game,player) == None:  
                fit_check = True
            pass

        tries+=1
        
    if turn.validate(game,player) != None:
        return "Something went wrong and the AI->game mapper couldn't coerce a valid state"
    
    return turn