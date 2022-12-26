
from typing import TypeVar

T = TypeVar('T')      # Declare type variable

def draw_n(deck: list[T], draw_num: int) -> list[T]:
    drawn = deck[0:draw_num]
    deck = deck[draw_num:]
    return drawn


def draw_one(deck: list[T]) -> T:
    return deck.pop(0)