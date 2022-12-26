from __future__ import annotations
from game_model.action import Action
from game_model.actor import Actor


class Game:
    def __init__(self, player_count: int):
        self.players = [Actor() for x in range(0, player_count)]
        self.active_index = 0
    
    def get_player(self, player_index: int) -> Actor:
        raise "not implemented"

    def get_current_player_index(self) -> int:
        return self.active_index
    
    def clone(self) -> Game:
        raise "not implemented"
    
    def step_game(self, action: Action):
        action.validate()
        next = self.players[self.active_index]
        next.execute_action(action)
        self.active_index = (self.active_index + 1) % len(self.players)