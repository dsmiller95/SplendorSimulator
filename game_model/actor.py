from __future__ import annotations
from game_model.card import Card
from game_model.noble import Noble
from game_model.resource_types import ResourceType
from utilities.print_utils import stringify_resources

class Actor:
    def __init__(self):
        self.claimed_nobles : list[Noble] = []
        self.reserved_cards : list[Card] = [None, None, None]
        self.resource_tokens : list[int] = [0, 0, 0, 0, 0, 0]
        self.purchased_cards : list[Card] = []
        self.resource_persistent: list[int] = [0, 0, 0, 0, 0]
        self.sum_points = 0
        pass
    
    def can_reserve_another(self):
        return None in self.reserved_cards
    def has_reserved_cards(self):
        for card in self.reserved_cards:
            if not(card is None):
                return True
        return False

    def total_tokens(self) -> int:
        return sum(self.resource_tokens)
    
    def get_reserved_card(self, card_index: int) -> Card:
        return self.reserved_cards[card_index]
    
    def take_reserved_card(self, card_index: int) -> Card:
        if self.reserved_cards[card_index] is None:
            raise "reserved card is already clear at index " + str(card_index)
        removed_card = self.reserved_cards[card_index]
        self.reserved_cards[card_index] = None
        return removed_card

    def can_purchase(self, card: Card) -> bool:
        wildcard_bank = self.resource_tokens[ResourceType.GOLD.value]
        for idx, cost in enumerate(card.costs):
            remaining = cost - (self.resource_persistent[idx] + self.resource_tokens[idx])
            if remaining <= 0:
                continue
            wildcard_bank = wildcard_bank - remaining
            if wildcard_bank < 0:
                return False
        return True
    
    def clone(self) -> Actor:
        new = Actor()
        new.claimed_nobles = self.claimed_nobles[0:]
        new.reserved_cards = self.reserved_cards[0:]
        new.resource_tokens = self.resource_tokens[0:]
        new.purchased_cards = self.purchased_cards[0:]


    def describe_state(self) -> str:
        result = ""
        result += "Points: " + str(self.sum_points) + "\n"
        result += "Tokens: " + stringify_resources(self.resource_tokens, ignore_empty=True) + "\n"
        result += "Cards : " + stringify_resources(self.resource_persistent, ignore_empty=True) + "\n"
        result += "Reserved Cards: \n"
        for card in self.reserved_cards:
            if card is None:
                continue
            result += card.describe_state() + "\n"
        return result
    

