class GamestateInput_vector:
    def __init__(self):
        self.player1 = Player_vector
        self.player2 = Player_vector
        self.player3 = Player_vector
        self.player4 = Player_vector
        self.nobles = Nobles_row_vector
        self.resources = [None]*6
        self.tier1 = Row_vector
        self.tier2 = Row_vector
        self.tier3 = Row_vector

class Card_vector:
    def __init__(self):
        self.costs = [None]*5
        self.returns = [None]*5
        self.points = [None]

class Noble_vector:
    def __init__(self):
        self.costs = [None]*5
        self.points = [None]

class Nobles_row_vector:
    def __init__(self):
        self.noble1 = Noble_vector
        self.noble2 = Noble_vector
        self.noble3 = Noble_vector
        self.noble4 = Noble_vector
        self.noble5 = Noble_vector
class Row_vector:
    def __init__(self):
        self.hidden_card = Card
        self.card1 = Card
        self.card2 = Card
        self.card3 = Card
        self.card4 = Card

class Reserved_Cards_vector:
    def __init__(self):
        self.Card1 = Card
        self.Card2 = Card
        self.Card3 = Card

class Player_vector:
    def __init__(self):
        self.temp_resources = [None]*6
        self.perm_resources = [None]*6
        self.points = [None]
        self.reserved_cards = Reserved_Cards