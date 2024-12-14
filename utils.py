import torch as t
import time
from torch import tensor
from termcolor import colored
from typing import Callable
from functools import wraps



red = lambda text: colored(text, "red")
green = lambda text: colored(text, "green")

def assert_shape_equal(actual: t.Tensor, expected: t.Tensor) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(f"expected shape={expected.shape}, got {actual.shape}")

def report(test_func: Callable) -> Callable:
    name = f"{test_func.__module__}.{test_func.__name__}"

    @wraps(test_func)
    def wrapper(*args, **kwargs):
        return run_and_report(test_func, name, *args, **kwargs)

    return wrapper

def run_and_report(test_func: Callable, name: str, *test_func_args, **test_func_kwargs):
    start = time.time()
    out = test_func(*test_func_args, **test_func_kwargs)
    elapsed = time.time() - start
    print(green(f"{name} passed in {elapsed:.2f}s."))
    return out

def allclose(actual: t.Tensor, expected: t.Tensor, rtol: float = 1e-4) -> None:
    assert_shape_equal(actual, expected)
    left = (actual - expected).abs()
    right = rtol * expected.abs()
    num_wrong = (left > right).sum().item()
    if num_wrong > 0:
        print(red(f"Test failed. Max absolute deviation: {left.max()}"))
        print(red(f"Actual:\n{actual}\nExpected:\n{expected}"))
        raise AssertionError(f"allclose failed with {num_wrong} / {left.nelement()} entries outside tolerance")

def allclose_atol(actual: t.Tensor, expected: t.Tensor, atol: float) -> None:
    assert_shape_equal(actual, expected)
    left = (actual - expected).abs()
    num_wrong = (left > atol).sum().item()
    if num_wrong > 0:
        print(red(f"Test failed. Max absolute deviation: {left.max()}"))
        print(red(f"Actual:\n{actual}\nExpected:\n{expected}"))
        raise AssertionError(f"allclose failed with {num_wrong} / {left.nelement()} entries outside tolerance")


def allclose_scalar(actual: float, expected: float, rtol: float = 1e-4) -> None:
    left = abs(actual - expected)
    right = rtol * abs(expected)
    wrong = left > right
    if wrong:
        raise AssertionError(f"Test failed. Absolute deviation: {left}\nActual:\n{actual}\nExpected:\n{expected}")


def allclose_scalar_atol(actual: float, expected: float, atol: float) -> None:
    left = abs(actual - expected)
    wrong = left > atol
    if wrong:
        raise AssertionError(f"Test failed. Absolute deviation: {left}\nActual:\n{actual}\nExpected:\n{expected}")