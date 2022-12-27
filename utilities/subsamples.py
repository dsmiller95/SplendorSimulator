
from typing import TypeVar

T = TypeVar('T')      # Declare type variable

def draw_n(deck: list[T], draw_num: int) -> list[T]:
    drawn = deck[:draw_num]
    del deck[:draw_num]
    return drawn


def draw_one(deck: list[T]) -> T:
    return deck.pop(0)

def parse_int(string: str, default: int) -> int:
    try:
        return int(string)
    except ValueError:
        return default

def parse_all_int(strings: list[str], default: int) -> list[int]:
    return [parse_int(x, default) for x in strings]

def clone_shallow(target: list[T]) -> list[T]:
    return target[0:]

def clone_two_deep(target: list[list[T]]) -> list[list[T]]:
    return [clone_shallow(x) for x in target]