class GamestateInputVector:
    def __init__(self):
        self.player1 = PlayerVector()
        self.player2 = PlayerVector()
        self.player3 = PlayerVector()
        self.player4 = PlayerVector()
        self.nobles = NoblesRowVector()
        self.resources = [None]*6
        self.tier1 = RowVector()
        self.tier2 = RowVector()
        self.tier3 = RowVector()

class CardVector:
    def __init__(self):
        self.costs = [None]*5
        self.returns = [None]*5
        self.points = [None]

class NobleVector:
    def __init__(self):
        self.costs = [None]*5
        self.points = [None]

class NoblesRowVector:
    def __init__(self):
        self.noble1 = NobleVector()
        self.noble2 = NobleVector()
        self.noble3 = NobleVector()
        self.noble4 = NobleVector()
        self.noble5 = NobleVector()
class RowVector:
    def __init__(self):
        self.hidden_card = CardVector()
        self.card1 = CardVector()
        self.card2 = CardVector()
        self.card3 = CardVector()
        self.card4 = CardVector()

class ReservedCardsVector:
    def __init__(self):
        self.Card1 = CardVector()
        self.Card2 = CardVector()
        self.Card3 = CardVector()

class PlayerVector:
    def __init__(self):
        self.temp_resources = [None]*6
        self.perm_resources = [None]*6
        self.points = [None]
        self.reserved_cards = ReservedCardsVector()