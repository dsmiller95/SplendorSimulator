from game_model.turn import Turn

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



def map_to_AI_input(game_state):
     
    input_vector = VectorBuilder(236)
    #populate a player vector so we can rotate it
    #into the right position to have the current player
    # on the top of the list always, so that the "view"
    #the model has of the game is always from the same
    #relative perspective
    reserved_card_shape = (5+5+1)
    player_shape = 6+5+(3*reserved_card_shape)
    player_vector = VectorBuilder(4*player_shape)
    for i,player in enumerate(game_state.players):
        player_vector.put((i*player_shape)+0,player.resource_tokens)
        player_vector.put((i*player_shape)+6,player.resource_persistent)

        for j,card in enumerate(player.reserved_cards):
            offset = (i*player_shape)+11
            if card == None:
                player_vector.put((offset+(j*reserved_card_shape))+0,[0]*5)
                player_vector.put((offset+(j*reserved_card_shape))+5,[0]*5)
                player_vector.put((offset+(j*reserved_card_shape))+10,0)
            else:
                player_vector.put((offset+(j*reserved_card_shape))+0,card.cost)
                player_vector.put((offset+(j*reserved_card_shape))+5,card.reward)
                player_vector.put((offset+(j*reserved_card_shape))+10,card.points)
    
    player_num = game_state.get_current_player_index()

    player_vector_rotated = player_vector.return_vector()[player_num*player_shape:] + player_vector.return_vector()[:player_num*player_shape]
    input_vector.put(0,player_vector_rotated)

    noble_shape = 5+1
    for i,noble in enumerate(game_state.active_nobles):
    
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

def map_from_AI_output(output_vector):

    action = output_vector[0]
    tiers = []
    for tier in range(3):
        tiers.append(output_vector[1+(4*tier):1+(4*tier)+4])
    purchase_reserve = output_vector[13]
    triplet = output_vector[14:17]
    doublet = output_vector[17]
    draw_noble = output_vector[18]
    noble_choice = output_vector[19]
    discard_tokens = output_vector[20]
    discard_numbers = output_vector[21::]

    #Code that tries to fit the AI output to valid game states goes here
    #turn_actions = Turn()
    
    return None