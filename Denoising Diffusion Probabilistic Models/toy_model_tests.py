import sys; sys.path.append('..')
from torch import tensor
from utils import report, allclose, allclose_scalar_atol, allclose_scalar, allclose_atol

@report
def test_variance_schedule(variance_schedule):
    expected = tensor([0, 4, 8, 12, 16, 20])
    actual = variance_schedule(6, 0, 20)
    allclose(actual, expected)


@report
def test_q_eq2(q_eq2, variance_schedule, x):
    steps = 25000
    xt = q_eq2(x, steps, variance_schedule(max_steps=steps))

    # After many steps, the image should look just like a Gaussian
    # with mean 0 and std 1
    allclose_scalar_atol(xt.mean(), 0.0, atol=0.1)
    allclose_scalar(xt.std(), 1.0, rtol=0.20)


@report
def test_q_eq4(q_eq4, variance_schedule, x):
    steps = 25000
    xt = q_eq4(x, steps, variance_schedule(max_steps=steps))

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
    max_steps = 30_000
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
def test_toy_diffuser(ToyDiffuser, ToyDiffuserConfig):
    B, C, H, W = 7, 8, 9, 10
    max_steps = 100
    d_model = 16

    model_config = ToyDiffuserConfig((C, H, W), d_model, max_steps)
    model = ToyDiffuser(model_config)
    imgs = t.randn((B, C, H, W))
    n_steps = t.randint(0, max_steps, (B,))
    out = model(imgs, n_steps)
    assert out.shape == (B, C, H, W)
