from __future__ import annotations
from game_data.game_config_data import GameConfigData
from game_model.action import Action
from game_model.actor import Actor
from random import sample, shuffle
from game_model.card import Card
from game_model.noble import Noble
from utilities.subsamples import draw_n


class Game:
    def __init__(self, player_count: int, game_config: GameConfigData):
        self.players = [Actor() for x in range(0, player_count)]
        self.active_index = 0
        self.config = game_config
        
        self.active_nobles : list[Noble]
        self.remaining_cards_by_level : list[list[Card]] = [[], [], []]
        self.open_cards : list[list[Card]] = [[], [], []]
        
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
                result_str += card.describe_self() + "\n"
        return result_str;



    def clone(self) -> Game:
        raise "not implemented"
    
    def step_game(self, action: Action):
        action.validate()
        next = self.players[self.active_index]
        next.execute_action(action)
        self.active_index = (self.active_index + 1) % len(self.players)