from game_model.action import Turn
from game_model.card import Card


class Actor:
    def __init__(self):
        self.reserved_cards : list[Card] = [None, None, None]
        self.resource_tokens : list[int] = [0, 0, 0, 0, 0, 0]
        self.purchased_cards : list[Card] = []
        self.resource_persistent: list[int] = [0, 0, 0, 0, 0]
        pass
    
    def can_reserve_another(self):
        return any(elem is None for elem in self.reserved_cards)

    def execute_action(self, action : Turn):
        raise "not implemented"
    
    def total_tokens(self) -> int:
        return sum(self.resource_tokens)
    
    def get_reserved_card(self, card_index: int) -> Card:
        return self.reserved_cards[card_index]

    def can_purchase(self, card: Card) -> bool:
        wildcard_bank = self.resource_tokens[-1]
        for idx, cost in enumerate(card.costs):
            remaining = cost - (self.resource_persistent[idx] + self.resource_tokens[idx])
            if remaining <= 0:
                continue
            wildcard_bank = wildcard_bank - remaining
            if wildcard_bank <= 0:
                return False
        return True
