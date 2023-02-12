from typing import Callable, Generic, TypeVar

T = TypeVar('T')

class Lazy(Generic[T]):
    def __init__(self, evaluator: Callable[[], T]):
        self._eval = evaluator
        self._value : T = None
    
    def val(self):
        if self._value is None:
            self._value = self._eval()
        return self._value

def max_index(list: list) -> int:
    if len(list) <= 0:
        raise ValueError("list must have at least one item")
    
    max_val = list[0]
    max_index = 0
    for i in range(1, len(list)):
        if list[i] > max_val:
            max_val = list[i]
            max_index = i
    return max_index