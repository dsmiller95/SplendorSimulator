from game_model.game import Game
from game_model.turn import Turn

def step_game(game: Game, action: Turn) -> str:
    next = game.players[game.active_index]
    validate_message = action.validate(game, next)
    if not (validate_message is None):
        return validate_message
    action.execute(game, next)
    game.turn_n += 1
    game.active_index = (game.active_index + 1) % len(game.players)
    return None