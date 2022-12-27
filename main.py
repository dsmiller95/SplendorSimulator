from game_data.game_config_data import GameConfigData
from game_model.game_runner import step_game
from game_model.turn import Turn, Action_Type
from game_model.game import Game
from input_parser.action_parser import get_action_from_user

print("hello there")

game_config = GameConfigData.read_file("./game_data/cards.csv")
game = Game(player_count=3, game_config=game_config)

while True:
    print(game.describe_common_state())
    print("player " + str(game.get_current_player_index() + 1) + "'s turn!")
    next_action: Turn = None
    while next_action is None:
        next_action = get_action_from_user(game.get_player(game.get_current_player_index()), game)
        if next_action == False:
            print("exiting")
            exit(0)
    
    print(step_game(game, next_action))