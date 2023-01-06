from __future__ import annotations
from game_data.game_config_data import GameConfigData
from game_model.actor import Actor
from random import sample, shuffle
from game_model.card import Card
from game_model.noble import Noble
from utilities.print_utils import stringify_resources
from utilities.subsamples import clone_shallow, clone_two_deep, draw_n

class Game:
    def __init__(self, player_count: int, game_config: GameConfigData, force_shuffle: bool = True):

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
        nobles = game_config.nobles[0:]
        if force_shuffle:
            shuffle(nobles)
        self.active_nobles = nobles[0:player_count + 1]

        ## Cards
        for card in game_config.cards:
            self.remaining_cards_by_level[card.tier].append(card)
        
        if force_shuffle:
            for card_tier in self.remaining_cards_by_level:
                shuffle(card_tier)

        for idx, cards in enumerate(self.remaining_cards_by_level):
            self.open_cards[idx] = draw_n(cards, game_config.open_cards_per_tier)
    
    def clone(self) -> Game:
        new = Game(len(self.players), self.config, force_shuffle=False)
        new.players = [x.clone() for x in self.players]
        new.active_index = self.active_index
        new.config = self.config
        new.active_nobles = clone_shallow(self.active_nobles)
        new.remaining_cards_by_level = clone_two_deep(self.remaining_cards_by_level)
        new.open_cards = clone_two_deep(self.open_cards)
        new.available_resources = clone_shallow(self.available_resources)
        new.active_nobles = clone_shallow(self.active_nobles)
        return new

    def get_tier_and_selected(self, card_index: int) -> tuple[int, int]:
        tier_width = self.config.open_cards_per_tier + 1
        tier = card_index // tier_width
        selected_card = card_index % tier_width
        return (tier, selected_card)

    def is_top_deck_index(self, card_index: int) -> bool :
        (tier, card) = self.get_tier_and_selected(card_index)
        return card <= 0

    """
    card index orders from tier 1 to tier 3, left to right. total of 12 in base game.
    """
    def get_card_by_index(self, card_index: int) -> Card:
        (tier, selected_card) = self.get_tier_and_selected(card_index)

        if selected_card == 0:
            if len(self.remaining_cards_by_level[tier]) <= 0:
                return None
            return self.remaining_cards_by_level[tier][0]
        return self.open_cards[tier][selected_card - 1]
    
    def take_card_by_index(self, card_index: int) -> Card:
        (tier, selected_card) = self.get_tier_and_selected(card_index)

        if selected_card == 0:
            if len(self.remaining_cards_by_level[tier]) <= 0:
                return None
            return self.remaining_cards_by_level[tier].pop(0)

        taken_card = self.open_cards[tier][selected_card - 1]
        if len(self.remaining_cards_by_level[tier]) <= 0:
            self.open_cards[tier][selected_card - 1] = None
            return taken_card
        self.open_cards[tier][selected_card - 1] = self.remaining_cards_by_level[tier].pop(0)
        return taken_card

    def give_tokens_to_player(self, player: Actor, token_index: int, token_num: int = 1):
            player.resource_tokens[token_index] += token_num
            self.available_resources[token_index] -= token_num

    def get_player(self, player_index: int) -> Actor:
        return self.players[player_index]

    def get_current_player_index(self) -> int:
        return self.active_index
    
    def get_players_in_immediate_turn_priority(self) -> list[Actor]:
        result = clone_shallow(self.players)
        result = result[self.active_index:] + result[:self.active_index]
        return result
    
    def get_player_num(self) -> int:
        return len(self.players)

    def get_current_player(self) -> Actor:
        return self.players[self.active_index]
    
    def get_num_players(self) -> int:
        return len(self.players)
    
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

    