import random
from typing import Callable
from game_data.game_config_data import GameConfigData
from game_model.game import Game
from game_model.game_runner import step_game
from game_model.turn import Turn


class TournamentRunner:
    def __init__(self,
        participants: dict[str, Callable[[Game], Turn]],
        game_config: GameConfigData):
        self.participants = participants
        self.game_config = game_config
        self.tourney_record : list[dict] = []

    def run_tourney(
        self,
        games: int,
        participant_to_evaluate: str):
        for game_n in range(games):
            player_n = random.randint(2, 4)
            all_ais = list(self.participants.keys())
            selected_players = [participant_to_evaluate] + [all_ais[random.randint(0, len(all_ais) - 1)] for x in range(player_n - 1)]
            # make sure the player under test never plays a full game against only itself
            if all([x == participant_to_evaluate for x in selected_players]):
                all_ais.remove(participant_to_evaluate)
                selected_players[0] = all_ais[random.randint(0, len(all_ais) - 1)]
            
            random.shuffle(selected_players)
            scores = self._run_game(selected_players)
            self.tourney_record.append({
                "scores": scores,
                "players": selected_players,
                "won_index": scores.index(max(scores))
            })
    

    def raw_win_ratios(self):
        """
        Return the win ratios for every AI in the tourney. a win ratio is calculated as:
        <# of games participated in> / <# of games this AI won>
        Note there may be inconsistencies here, as an AI could play multiple players in the same game.
        If the AI won, then this would count as one won game, and 0 lost games.
        """
        result = {}
        for ai_name in self.participants.keys():
            participated_games = [x for x in self.tourney_record if ai_name in x["players"]]
            won_games = [x for x in participated_games if ai_name == x["players"][x["won_index"]]]
            win_ratio = len(won_games) / len(participated_games)
            result[ai_name] = win_ratio
        return result


    def _run_game(
        self,
        selected_participants: list[str]) -> list[int]:
        player_count = len(selected_participants)
        """
        Runs one tournament game, with the listed participants in turn order.
        Returns the final scores of all players in the game
        """
        game = Game(player_count, self.game_config)
        won = False
        
        while not won:
            for player_index in range(player_count):
                participant_name = selected_participants[player_index]
                participant = self.participants[participant_name]
                turn = participant(game)
                acting_player = game.get_current_player()
                step_result = step_game(game, turn)
                if not (step_result is None):
                    print("ERROR: invalid game step generated when running tourney with AI of name '" + participant_name + "' : " + step_result)
                if acting_player.qualifies_to_win():
                    won = True
        
        return [x.sum_points for x in game.players]



