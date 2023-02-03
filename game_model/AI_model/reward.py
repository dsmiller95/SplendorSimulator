from game_model.game import Game

class Reward:
    def __init__(self, game_state: Game, player_index: int, settings: dict):
        self.current_player = game_state.get_player(player_index)
        self.game_state = game_state
        self.settings = settings
        
    def all_rewards(self) -> float:
        reward: float = 0.0
        reward += self.tokens_held_reward()
        reward += self.cards_held_reward()
        reward += self.points_reward()
        reward += self.win_lose_reward()
        reward += self.length_of_game_reward()
        return reward

    def tokens_held_reward(self) -> float:
        reward_mul = self._get_reward_from_setting(self.settings['tokens_held'])
        reward: float = 0.0
        for i in range(5):
            reward += self.current_player.resource_tokens[i] * reward_mul
        ## gold token is worth 1.5 regular resource tokens
        reward += self.current_player.resource_tokens[5] * 1.5 *reward_mul
        return reward

    def cards_held_reward(self) -> float:
        reward_mul = self._get_reward_from_setting(self.settings['cards_held'])
        reward: float = 0.0
        for i in range(5):
            reward += self.current_player.resource_persistent[i] * reward_mul
        return reward
    
    def points_reward(self) -> float:
        reward_mul = self._get_reward_from_setting(self.settings['points'])
        reward = self.current_player.sum_points * reward_mul
        return reward

    def win_lose_reward(self) -> float:
        # Determine win/lose reward/punishment
        if self.current_player.qualifies_to_win() and not any([player for player in self.game_state.players if player.sum_points > self.current_player.sum_points]):
            return self._get_reward_from_setting(self.settings['win_lose'])
        elif any([player.qualifies_to_win() for player in self.game_state.players if player is not self.current_player]):
            return -1.0 * self._get_reward_from_setting(self.settings['win_lose'])
        else:
            return 0.0

    def length_of_game_reward(self) -> float:
        reward_mul = self._get_reward_from_setting(self.settings['length_of_game'])
        return reward_mul * float(self.game_state.turn_n)

    def _get_reward_from_setting(self,setting: list) -> float:
        return (setting[1] if setting[0] else 0.0)