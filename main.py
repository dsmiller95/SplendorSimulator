import threading
from game_data.game_config_data import GameConfigData
from game_model.AI_model.gamestate_input import GamestateInputVector
from game_model.game_runner import step_game
from game_model.turn import Turn, Action_Type
from game_model.game import Game
from game_model.trainer import train
from input_parser.action_parser import get_action_from_user
import sys
import random

print("hello there")

from game_server_interface.game_server import app, game_data

host_name = "127.0.0.1"
port = 5000
if __name__ == "__main__":
    spawn_thread = threading.Thread(target=lambda: app.run(host=host_name, port=port, debug=True, use_reloader=False))
    spawn_thread.setDaemon(True)
    spawn_thread.start()

game_config = GameConfigData.read_file("./game_data/cards.csv")
game = Game(player_count=2, game_config=game_config)

def set_game(game: Game) -> None:
    game_data.game = game

if len(sys.argv) <= 1 or sys.argv[1] == "train":
    train(set_game, game_data.lock_object)
    exit(0)
elif sys.argv[1] == "play":
    set_game(game)
    while True:
        random.seed(1337)
        print(game.describe_common_state())
        # print(GamestateInputVector.map_to_AI_input(game))
        print("player " + str(game.get_current_player_index() + 1) + "'s turn!")
        next_action: Turn = None
        while next_action is None:
            next_action = get_action_from_user(game.get_player(game.get_current_player_index()), game)
            if next_action == False:
                print("exiting")
                exit(0)
        
        print(step_game(game, next_action))

