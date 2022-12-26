from __future__ import annotations
from game_data.game_config_data import GameConfigData
from game_model.actor import Actor
from random import sample, shuffle
from game_model.card import Card
from game_model.noble import Noble
from utilities.print_utils import stringify_resources
from utilities.subsamples import draw_n

class Game:
    def __init__(self, player_count: int, game_config: GameConfigData):

        if player_count <= 1 or player_count > 4:
            raise "Invalid player number, must have 2, 3, or 4 players"

        self.players = [Actor() for x in range(0, player_count)]
        self.active_index = 0
        self.config = game_config
        
        self.active_nobles : list[Noble]
        self.remaining_cards_by_level : list[list[Card]] = [[] for x in range(0, game_config.tiers)]
        self.open_cards : list[list[Card]] = [[] for x in range(0, game_config.tiers)]

        res_base : int
        match player_count:
            case 2:
                res_base = 4
            case 3:
                res_base = 5
            case 4:
                res_base = 7
        self.available_resources : list[int] = [res_base, res_base, res_base, res_base, res_base, 5]
        
        ## Nobles
        self.active_nobles = sample(game_config.nobles, player_count + 1)

        ## Cards
        for card in game_config.cards:
            self.remaining_cards_by_level[card.tier].append(card)
        
        ## TODO: assert proper card size
        for card_tier in self.remaining_cards_by_level:
            shuffle(card_tier)

        for idx, cards in enumerate(self.remaining_cards_by_level):
            self.open_cards[idx] = draw_n(cards, game_config.open_cards_per_tier)
    
    """
    card index orders from tier 1 to tier 3, left to right. total of 12 in base game.
    """
    def get_card_by_index(self, card_index: int) -> Card:
        tier = card_index // self.config.tiers
        selected_card = card_index % self.config.open_cards_per_tier
        return self.open_cards[tier][selected_card]
    
    def take_card_by_index(self, card_index: int) -> Card:
        tier = card_index // self.config.tiers
        selected_card = card_index % self.config.open_cards_per_tier
        taken_card = self.open_cards[tier][selected_card]
        if len(self.remaining_cards_by_level[tier]) <= 0:
            self.open_cards[tier][selected_card] = None
            return taken_card
        self.open_cards[tier][selected_card] = self.remaining_cards_by_level[tier].pop()
        return taken_card


    def get_player(self, player_index: int) -> Actor:
        raise self.players[player_index]

    def get_current_player_index(self) -> int:
        return self.active_index
    
    def describe_common_state(self) -> str:
        result_str = ""
        result_str += "Nobles:\n"
        for noble in self.active_nobles:
            result_str += noble.describe_self() + "\n"
        result_str += "\n"
        for tier, cards in reversed(list(enumerate(self.open_cards))):
            result_str += "Tier " + str(tier + 1) + ":\n"
            for card in cards:
                result_str += card.describe_state() + "\n"
        result_str += "\nBank:\n" + stringify_resources(self.available_resources) + "\n"

        for idx, player in enumerate(self.players):
            result_str += "\n-----Player " + str(idx + 1) + "-----\n" + player.describe_state()
        return result_str

    def clone(self) -> Game:
        raise "not implemented"
    