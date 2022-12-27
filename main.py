from game_data.game_config_data import GameConfigData
from game_model.game_runner import step_game
from game_model.turn import Turn, Action_Type
from game_model.game import Game
from input_parser.action_parser import get_action_from_user
import random

print("hello there")

random.seed(1337)
game_config = GameConfigData.read_file("./game_data/cards.csv")
game = Game(player_count=2, game_config=game_config)

test_actions = [
    Turn(Action_Type.TAKE_TWO, [2, 0, 0, 0, 0]),
    Turn(Action_Type.RESERVE_CARD, card_index=1),

    Turn(Action_Type.TAKE_THREE_UNIQUE, [1, 1, 0, 0, 1]),
    Turn(Action_Type.TAKE_THREE_UNIQUE, [0, 1, 1, 0, 1]),

    Turn(Action_Type.BUY_CARD, card_index=4),
    Turn(Action_Type.TAKE_THREE_UNIQUE, [1, 1, 1, 0, 0]),
    
    Turn(Action_Type.TAKE_THREE_UNIQUE, [1, 1, 1, 0, 0]),
    Turn(Action_Type.BUY_CARD, card_index=15 + 0),

    Turn(Action_Type.RESERVE_CARD, card_index=2),
    Turn(Action_Type.TAKE_THREE_UNIQUE, [0, 1, 1, 0, 1]),

    Turn(Action_Type.BUY_CARD, card_index=3),
    Turn(Action_Type.BUY_CARD, card_index=1),
    
    Turn(Action_Type.TAKE_THREE_UNIQUE, [0, 1, 1, 0, 1]),
]

turn_num = 0
for turn in test_actions:
    print("=================================TURN " + str(turn_num) + "=================================")
    print(game.describe_common_state())
    turn_num += 1
    print("player " + str(game.get_current_player_index() + 1) + "'s turn!")
    print(turn.describe_state(game, game.get_current_player()))
    step_result = step_game(game, turn)
    if not (step_result is None):
        print("GAME ERR")
        print(step_result)
        exit(0)
exit(0)



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