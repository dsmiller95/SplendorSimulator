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