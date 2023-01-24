from game_model.actor import Actor
from game_model.card import Card
from game_model.game import Game
from game_model.resource_types import ResourceType


def add_reserved_card(player: Actor, game: Game, card: Card) -> bool:
    for idx, x in enumerate(player.reserved_cards):
        if x is None:
            player.reserved_cards[idx] = card
            if game.available_resources[ResourceType.GOLD.value] > 0:
                player.resource_tokens[ResourceType.GOLD.value] += 1
                game.available_resources[ResourceType.GOLD.value] -= 1
            return True
    return False


def purchase_card(player: Actor, game: Game, card: Card):
    for idx, cost in enumerate(card.costs):
        remaining = cost - player.resource_persistent[idx]
        if remaining <= 0:
            continue
        remaining = spend_down(player, game, idx, remaining)
        if remaining <= 0:
            continue
        ## 5 is the wildcard index
        remaining = spend_down(player, game, ResourceType.GOLD.value, remaining)
        if remaining > 0:
            raise "not enough spend. move is invalid"
    player.resource_persistent[card.returns.value] += 1
    player.purchased_cards.append(card)
    player.sum_points += card.points
    

"""
spend resources from resource_index out of our bank,
reducing the remaining_spend down to 0
"""
def spend_down(player: Actor, game: Game, resource_index: int, remaining_spend: int) -> int :
    spent_tokens = min(remaining_spend, player.resource_tokens[resource_index])
    remaining_spend -= spent_tokens
    player.resource_tokens[resource_index] -= spent_tokens
    game.available_resources[resource_index] += spent_tokens
    return remaining_spend
