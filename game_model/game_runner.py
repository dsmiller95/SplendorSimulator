from game_model.game import Game
from game_model.turn import Turn

def step_game(game: Game, action: Turn):
    next = game.players[game.active_index]
    action.validate(game, next)
    action.execute(game, next)
    game.active_index = (game.active_index + 1) % len(game.players)