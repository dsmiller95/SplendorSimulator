import torch
from statistics import mean, median, stdev
from game_model.AI_model.action_output import ActionOutput
from game_model.AI_model.gamestate_input import GamestateInputVector
from game_model.game import Game
from game_model.game_runner import step_game
from game_model.turn import Action_Type, Turn
from tests.test_game_sequences import run_turns
from tests.test_helpers import assert_banks, assert_noble_claimed, assert_noble_n, assert_points, test_config

import time

def test_all_action_performance_by_segment():

    turn_objs = [ActionOutput() for x in range(4)]

    #pick 2
    turn_objs[0].action_choice = torch.Tensor([0, 1, 0, 0])
    turn_objs[0].resource_token_draw = torch.Tensor([0, 1, 0, 0, 0])
    #pick 3
    turn_objs[1].action_choice = torch.Tensor([1, 0, 0, 0])
    turn_objs[1].resource_token_draw = torch.Tensor([1, 1, 1, 0, 0])
    # reserve card
    turn_objs[2].action_choice = torch.Tensor([0, 0, 0, 1])
    turn_objs[2].card_buy[2] = 1
    # buy card in reserve
    turn_objs[3].action_choice = torch.Tensor([0, 0, 1, 0])
    turn_objs[3].reserve_buy[1] = 1

    map_to_action_durations = []
    step_game_duration = []
    map_to_ai_input_duration = []

    for sample_n in range(100):
        game = Game(2, test_config, force_shuffle=False)
        for i, turn_obj in enumerate(turn_objs):
            start = time.perf_counter()
            (next_action, choice_dict) = ActionOutput._map_internal(turn_obj, game, game.get_current_player())
            map_to_action_durations.append(time.perf_counter() - start)

            start = time.perf_counter()
            step_status = step_game(game, next_action)
            step_game_duration.append(time.perf_counter() - start)
            
            assert step_status is None, step_status

            start = time.perf_counter()
            ai_input = GamestateInputVector.map_to_AI_input(game)
            map_to_ai_input_duration.append(time.perf_counter() - start)

    
    stats = {
        "map_to_action": {
            "avg (ms)": mean(map_to_action_durations) * 1000,
            "stdev (ms)": stdev(map_to_action_durations) * 1000
        },
        "step_game": {
            "avg (ms)": mean(step_game_duration) * 1000,
            "stdev (ms)": stdev(step_game_duration) * 1000
        },
        "map_to_ai_input": {
            "avg (ms)": mean(map_to_ai_input_duration) * 1000,
            "stdev (ms)": stdev(map_to_ai_input_duration) * 1000
        },
    }
    
    print(stats)

