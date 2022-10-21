import functools
import random
from os import environ
from random import sample
from sys import path


def my_ver_long_function_is_here(
    first_parameter, second_parameter, third_parameter, fourth_parameter
):
    pass

def square_num(x: float) -> float:
    """
    x: float
    """
    return x**2

def cube_num(x: float) -> float:
    """
    x: float

    ```
    #The below 4 lines is enveloped by ``` and this makes the code viewable on pdoc UI
    if x == 0:
        return 0
    else:
        return x**3
    ```
    """
    return x**3
