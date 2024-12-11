import os
import torch as t
import time
from torch import tensor
from termcolor import colored
from typing import Callable
from functools import wraps


red = lambda text: colored(text, "red")
green = lambda text: colored(text, "green")
DEBUG_TOLERANCES = os.getenv("DEBUG_TOLERANCES")

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
    elif DEBUG_TOLERANCES:
        print(green(f"Test passed with max absolute deviation of {left.max()}"))


def allclose_atol(actual: t.Tensor, expected: t.Tensor, atol: float) -> None:
    assert_shape_equal(actual, expected)
    left = (actual - expected).abs()
    num_wrong = (left > atol).sum().item()
    if num_wrong > 0:
        print(red(f"Test failed. Max absolute deviation: {left.max()}"))
        print(red(f"Actual:\n{actual}\nExpected:\n{expected}"))
        raise AssertionError(f"allclose failed with {num_wrong} / {left.nelement()} entries outside tolerance")
    elif DEBUG_TOLERANCES:
        print(green(f"Test passed with max absolute deviation of {left.max()}"))


def allclose_scalar(actual: float, expected: float, rtol: float = 1e-4) -> None:
    left = abs(actual - expected)
    right = rtol * abs(expected)
    wrong = left > right
    if wrong:
        raise AssertionError(f"Test failed. Absolute deviation: {left}\nActual:\n{actual}\nExpected:\n{expected}")
    elif DEBUG_TOLERANCES:
        print(green(f"Test passed with absolute deviation of {left}"))


def allclose_scalar_atol(actual: float, expected: float, atol: float) -> None:
    left = abs(actual - expected)
    wrong = left > atol
    if wrong:
        raise AssertionError(f"Test failed. Absolute deviation: {left}\nActual:\n{actual}\nExpected:\n{expected}")
    elif DEBUG_TOLERANCES:
        print(green(f"Test passed with absolute deviation of {left}"))

@report
def test_linear_schedule(linear_schedule):
    expected = tensor([0, 4, 8, 12, 16, 20])
    actual = linear_schedule(6, 0, 20)
    allclose(actual, expected)


@report
def test_q_eq2(q_eq2, linear_schedule, x):
    steps = 25000
    xt = q_eq2(x, steps, linear_schedule(max_steps=steps))

    # After many steps, the image should look just like a Gaussian
    # with mean 0 and std 1
    allclose_scalar_atol(xt.mean(), 0.0, atol=0.1)
    allclose_scalar(xt.std(), 1.0, rtol=0.20)


@report
def test_q_eq4(q_eq4, linear_schedule, x):
    steps = 25000
    xt = q_eq4(x, steps, linear_schedule(max_steps=steps))

    # After many steps, the image should look just like a Gaussian
    # with mean 0 and std 1
    allclose_scalar_atol(xt.mean(), 0.0, atol=0.1)
    allclose_scalar(xt.std(), 1.0, rtol=0.20)


@report
def test_noise_schedule(NoiseSchedule):
    sch = NoiseSchedule(3, "cpu", 1, 3)

    actual = sch.beta(t.tensor([2, 2, 1]))
    expected = t.tensor([3, 3, 2])
    allclose(actual, expected)

    actual = sch.alpha(t.tensor([0, 1, 2]))
    expected = t.tensor([0.0, -1.0, -2.0])
    allclose(actual, expected)

    actual = sch.alpha_bar(2)
    allclose_scalar_atol(actual, 0.0, 0.001)

    keys = sorted(sch.state_dict().keys())
    assert keys == ["alpha_bars", "alphas", "betas"], "Must register buffers so these are saved!"


@report
def test_noise_img(noise_img, NoiseSchedule, gradient_images, normalize_img):
    max_steps = 25_000
    noise_schedule = NoiseSchedule(max_steps=max_steps, device="cpu")
    img = gradient_images(3, (3, 1024, 1024))
    _, _, noised = noise_img(normalize_img(img), noise_schedule, max_steps=max_steps)

    # After many steps, the mean should approach 0 
    actual = noised.mean((-1, -2, -3)) # ignore batch dimension 
    expected = t.zeros_like(actual)
    allclose_atol(actual, expected, 0.1)

    # After many steps, the std should approach 1
    actual = noised.std((-1, -2, -3)) # ignore batch dimension 
    expected = t.ones_like(actual)
    allclose(actual, expected, 0.1)


@report
def test_reconstruct(denorm, img):
    allclose(denorm, img)


@report
def test_tiny_diffuser(TinyDiffuser, TinyDiffuserConfig):
    B, C, H, W = 7, 8, 9, 10
    max_steps = 100
    d_model = 16

    model_config = TinyDiffuserConfig((C, H, W), d_model, max_steps)
    model = TinyDiffuser(model_config)
    imgs = t.randn((B, C, H, W))
    n_steps = t.randint(0, max_steps, (B,))
    out = model(imgs, n_steps)
    assert out.shape == (B, C, H, W)
