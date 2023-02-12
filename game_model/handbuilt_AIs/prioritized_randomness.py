
import math
import random
from game_model.card import Card
from game_model.game import Game
from game_model.turn import Action_Type, Turn

class PrioritizedRandomnessAI:
    """
    Prioritizes point-giving actions above others, but each sub-action is randomized.
    This ai will first try to find a card to buy, and buy the first one found if it can buy.
    If that fails, the AI will try to choose a random pick 3 action.
    If that fails, the AI will try to choose a random pick 2 action.
    If that fails, the AI will try to reserve a card
    """
    def __init__(self):
        pass

    def next_turn(self, game: Game) -> Turn:
        player = game.get_current_player()

        total_cards = game.config.total_available_card_indexes()
        # try to buy a reserved card
        for reserved_index in range(len(player.reserved_cards)):
            card = player.reserved_cards[reserved_index]
            if player.can_purchase(card):
                return Turn(
                    Action_Type.BUY_CARD,
                    card_index=reserved_index + total_cards)
        
        # find a card to buy from the open tiers
        for card_index in range(0, game.config.total_available_card_indexes()):
            if game.is_top_deck_index(card_index):
                continue
            card_target = game.get_card_by_index(card_index)
            if player.can_purchase(card_target):
                return Turn(
                    Action_Type.BUY_CARD,
                    card_index=card_index)
        
        # try to pick random 3
        available_resources = game.available_resources[:5]
        try:
            picked_three = pick_random_subset([x > 0 for x in available_resources], 3)
            return Turn(
                Action_Type.TAKE_THREE_UNIQUE,
                resources_desired=[1 if x else 0 for x in picked_three]
            )
        except ValueError:
            pass # there may not be three to pick. if not, continue on
        
        ## Try to pick random 2
        try:
            picked_two = pick_random_subset([x >= 4 for x in available_resources], 1)
            return Turn(
                Action_Type.TAKE_TWO,
                resources_desired=[2 if x else 0 for x in picked_two]
            )
        except ValueError:
            pass # there may not be a valid pick two. if not, continue on

        ## Try to reserve a card
        if player.can_reserve_another():
            valid_card_indexes = [i for i, x in range(0, total_cards) if game.get_card_by_index(x) is not None]
            if len(valid_card_indexes) > 0:
                picked_index = math.floor(random.random() * len(valid_card_indexes))
                return Turn(
                    Action_Type.RESERVE_CARD,
                    card_index=valid_card_indexes[picked_index]
                )
        
        ## Try to find a valid pick less than three action
        pick_three_choice = [1 if x > 0 else 0 for x in available_resources]
        if sum(pick_three_choice) <= 0:
            return Turn(
                Action_Type.NOOP
            )
        if sum(pick_three_choice) > 3:
            print("ERROR: prioritized randomness AI defaulted to an invalid pick 3 option")
        return Turn(
            Action_Type.TAKE_THREE_UNIQUE,
            resources_desired=pick_three_choice
        )

def pick_random_subset(flag_list: list[bool], pick_n: int) -> list[bool]:
    valid_indexes = [i for i, x in enumerate(flag_list) if x]
    picked_indexes = random.sample(valid_indexes, pick_n)
    result_list = [False] * len(flag_list)
    for i in picked_indexes:
        result_list[i] = True
    return result_list