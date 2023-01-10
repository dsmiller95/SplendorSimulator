
from game_model.game import Game


class Reward:
    def __init__(self, game_state: Game):
        current_player = game_state.get_current_player()
        self.base_reward = current_player.get_fitness()
        if current_player.qualifies_to_win() and not any([x for x in game_state.players if x.sum_points > current_player.sum_points]):
            self.base_reward += 200