
#init settings
#load the game
#load the model
#init game memory (how to store Q-values with game memory?)

#for n epochs
    #while game not won
        #for each turn
            #for each player
                #play move
                #store game state in memory
                #store Q-value vector in memory
                #get action mask
                #mask Q-value vector
                #turn masked Q-value vector into an action choice
                #get reward for action
                #store reward in memory

    #non-batching
    #for each turn
        #for each player
            #get loss from predicted reward and actual reward
            #backprop loss, update weights

    #batching
    #something with a torch dataloader class?