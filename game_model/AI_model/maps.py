from game_model.game import Game
from game_model.turn import Turn
from utilities.subsamples import pad_list

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

def map_from_AI_output(output_vector: list[float],game,actor):

    action = output_vector.pop(0)
    tiers: list[list[float]] = [[None]*5]*3
    for tier in range(3):
        cards = []
        for card in range(4):
            cards.append(output_vector.pop(0))
        tiers[tier] = cards
    purchase_reserve = output_vector.pop(0)
    triplet: list[float] = []
    for i in range(3):
        triplet.append(output_vector.pop(0))
    doublet = output_vector.pop(0)
    noble_choice = output_vector.pop(0)
    discard_tokens = output_vector.pop(0)
    print(action,tiers,purchase_reserve,triplet,doublet,draw_noble,noble_choice,discard_tokens)
    
    for i in range(6):
        discard_numbers.append(output_vector.pop(0))

    #Tries to fit the AI output to valid game states
    passed_fit_check = False
    #while not passed_fit_check():
        
    #turn_actions = Turn()
    
    return None