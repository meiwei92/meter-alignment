import jsonpickle
from typing import *


def default(value: Any, default_value: Any, comparison_value: Any = None, mapper: Callable = lambda x: x):
    if default_value == comparison_value:
        raise ValueError("Calling this function with the same value for default and comparison value "
                         "makes this function call obsolete!")
    if mapper is None:
        raise ValueError("Mapping function must not be None!")

    return mapper(value) if value != comparison_value else default_value


def relu(value: float) -> float:
    if value > 0:
        return value
    else:
        return 0.0


class JsonReprObject(object):

    def __repr__(self) -> str:
        return super().__repr__()
