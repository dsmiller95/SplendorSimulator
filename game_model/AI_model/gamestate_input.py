class GamestateInputVector:
    def __init__(self):
        self.players = [PlayerVector() for x in range(0, 4)]
        self.nobles = [NobleVector() for x in range(5)]
        self.resources = ResourcesVector
        self.tiers = [RowVector() for x in range(3)]

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

class ResourcesVector:
    def __init__(self):
        self.resources = [None]*5
class RowVector:
    def __init__(self):
        self.hidden_card = CardVector()
        self.open_cards = [CardVector() for x in range(4)]

class PlayerVector:
    def __init__(self):
        self.temp_resources = [None]*6
        self.perm_resources = [None]*6
        self.points = [None]
        self.reserved_cards = [CardVector() for x in range(3)]