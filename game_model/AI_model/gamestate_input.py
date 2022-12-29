

def flat_map_group(map_list: list, prefix: str, into_dict: dict[str, list[int]]):
    for i, item in enumerate(map_list):
        item.flat_map_into(prefix + str(i), into_dict)

class GamestateInputVector:
    def __init__(self):
        self.players = [PlayerVector() for x in range(0, 4)]
        self.nobles = [NobleVector() for x in range(5)]
        self.resources = [None] * 5
        self.tiers = [RowVector() for x in range(3)]
    
    def flat_map(self, prefix: str = "") -> dict[str, list[float]]:
        into_dict: dict[str, list[float]] = {}
        flat_map_group(self.players, prefix + "player_", into_dict)
        flat_map_group(self.nobles, prefix + "noble_", into_dict)
        flat_map_group(self.tiers, prefix + "tier_", into_dict)
        into_dict[prefix + "resources"] = self.resources
        return into_dict
        

class CardVector:
    def __init__(self):
        self.costs = [None]*5
        self.returns = [None]*5
        self.points = [None]
    def flat_map_into(self, prefix: str, into_dict: dict[str, list[int]]):
        into_dict[prefix + "_costs"] = self.costs
        into_dict[prefix + "_returns"] = self.returns
        into_dict[prefix + "_points"] = self.points

class NobleVector:
    def __init__(self):
        self.costs = [None]*5
        self.points = [None]
    def flat_map_into(self, prefix: str, into_dict: dict[str, list[int]]):
        into_dict[prefix + "_costs"] = self.costs
        into_dict[prefix + "_points"] = self.points

class RowVector:
    def __init__(self):
        self.hidden_card = CardVector()
        self.open_cards = [CardVector() for x in range(4)]
        self.points = [None]
    def flat_map_into(self, prefix: str, into_dict: dict[str, list[int]]):
        flat_map_group(self.open_cards, prefix + "_card_", into_dict)
        self.hidden_card.flat_map_into(prefix + "_hidden_card", into_dict)
        into_dict[prefix + "_points"] = self.points

class PlayerVector:
    def __init__(self):
        self.temp_resources = [None]*6
        self.perm_resources = [None]*6
        self.points = [None]
        self.reserved_cards = [CardVector() for x in range(3)]

    def flat_map_into(self, prefix: str, into_dict: dict[str, list[int]]):
        into_dict[prefix + "_temp_resources"] = self.temp_resources
        into_dict[prefix + "_perm_resources"] = self.perm_resources
        into_dict[prefix + "_points"] = self.points
        flat_map_group(self.reserved_cards, prefix + "_reserved_card_", into_dict)