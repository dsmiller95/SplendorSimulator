
import math
import random
from game_model.card import Card
from game_model.game import Game
from game_model.resource_types import ResourceType
from game_model.turn import Action_Type, Turn
from utilities.utils import max_index

class TargetedPickAI:
    """
    Prioritizes point-giving actions above others, but each sub-action is randomized.
    This ai will first try to find a card to buy, and buy the first one found if it can buy.
        during this process the AI will keep track of which card is closest to our ability to purchase as determined by a multi-dimensional distance formula
    If no card can be bought, the AI will pick tokens based on what it needs to buy the easiest to purchase card
    If that fails, the AI will try to reserve a card which is easiest to buy
    If that fails, the AI will pick as many unique tokens as it can (pick 3, but less than 3)
    If that fails, NOOP 
    """
    def __init__(self):
        pass

    def next_turn(self, game: Game) -> Turn:
        player = game.get_current_player()

        total_cards = game.config.total_available_card_indexes()

        all_cards = \
            [(x + total_cards, player.reserved_cards[x]) for x in range(len(player.reserved_cards))] + \
            [(x, game.get_card_by_index(x)) for x in range(total_cards) if not game.is_top_deck_index(x)]
        all_cards = [x for x in all_cards if x[1] is not None]

        wildcards = player.resource_tokens[ResourceType.GOLD.value]
        purchasing_power = [x + y for x, y in zip(player.resource_tokens[:5], player.resource_persistent)]

        best_choice = all_cards[0]
        best_choice_distance = 100000
        best_reserve_choice = all_cards[0]
        best_reserve_choice_distance = 100000
        for choice in all_cards:
            distance = self._resource_distance(choice[1].costs, purchasing_power, wildcards)
            if distance < best_choice_distance:
                best_choice_distance = distance
                best_choice = choice
                if best_choice_distance <= 0:
                    return Turn(
                        Action_Type.BUY_CARD,
                        card_index=best_choice[0]
                    )
            if choice[0] < total_cards and distance < best_reserve_choice_distance:
                best_reserve_choice_distance = distance
                best_reserve_choice = choice


        target_cost_delta = self._cost_subtraction(best_choice[1].costs, purchasing_power)
        available_resources = game.available_resources[:5]

        # a list of all costs by resource index, in order from highest cost delta to smallest
        target_costs_prioritized = [(x, ResourceType(i)) for i, x in enumerate(target_cost_delta)]
        target_costs_prioritized.sort(key=lambda x: x[0], reverse=True)


        # try to find the best buy 3 choice
        buy_three_choice = [0] * 5
        total_buy_three_choices = 0
        for cost_next in target_costs_prioritized:
            resource_index = cost_next[1].value
            if available_resources[resource_index] <= 0:
                continue
            buy_three_choice[resource_index] = 1
            total_buy_three_choices += 1
            if total_buy_three_choices >= 3:
                break
        
        
        buy_two_choice = [0] * 5
        # a buy 3 choice which matched all 3 wasn't found.
        # try to find a buy 2 choice
        for cost_next in target_costs_prioritized:
            resource_index = cost_next[1].value
            if available_resources[resource_index] < 4:
                continue
            buy_two_choice[resource_index] = 2
            break
        
        buy_three_cost_delta = self._resource_distance(target_cost_delta, buy_three_choice, wildcards)
        buy_two_cost_delta = self._resource_distance(target_cost_delta, buy_two_choice, wildcards)
        if buy_two_cost_delta < buy_three_cost_delta: # less is better, minimize cost distance
            return Turn(
                Action_Type.TAKE_TWO,
                resources_desired=buy_two_choice
            )
        if sum(buy_three_choice) == 3:
            return Turn(
                Action_Type.TAKE_THREE_UNIQUE,
                resources_desired=buy_three_choice
            )
        
        ## Try to reserve a card, pick the one which is easiest to buy
        if player.can_reserve_another():
            return Turn(
                Action_Type.RESERVE_CARD,
                card_index=best_reserve_choice[0]
            )
        
        # return the best pick-3 action, which at this point will be
        # picking less than 3 items
        return Turn(
            Action_Type.TAKE_THREE_UNIQUE,
            resources_desired=buy_three_choice
        )

    def _cost_subtraction(self, cost: list[int], purchased: list[int]) -> list[int]:
        return [max(0, cost_resource - available_resource) for cost_resource, available_resource in zip(cost, purchased)]

    def _resource_distance(self, cost: list[int], purchase_power: list[int], wildcards: int) -> float:
        additional_cost = self._cost_subtraction(cost, purchase_power)
        for _ in range(wildcards):
            i = max_index(additional_cost)
            additional_cost[i] = max(additional_cost[i] - 1, 0)
        return math.sqrt(sum([x * x for x in additional_cost]))