from __future__ import annotations
from typing import (
    TypedDict,
    TypeAlias,
    TYPE_CHECKING,
    cast,
    Sequence,
    Literal,
    Callable,
)
from random import Random

Value: TypeAlias = float  # | int | str

if TYPE_CHECKING:
    from typing import NotRequired

Distribution: TypeAlias = Callable[[Random], float]


class UniformParams(TypedDict):
    name: Literal["uniform"]
    a: NotRequired[float]
    b: NotRequired[float]


class TruncnormParams(TypedDict):
    name: Literal["truncnorm"]
    a: NotRequired[float]
    b: NotRequired[float]
    mu: NotRequired[float]
    sigma: NotRequired[float]
    delta: NotRequired[float]


class ConstantParams(TypedDict):
    name: Literal["constant"]
    value: NotRequired[float]


class RandrangeParams(TypedDict):
    name: Literal["randrange"]
    start: NotRequired[int]
    stop: NotRequired[int]


class ChoiceParams(TypedDict):
    name: Literal["choice"]
    values: NotRequired[list[Value] | list[tuple[Value, float]]]


DistributionParams: TypeAlias = (
    UniformParams
    | TruncnormParams
    | ConstantParams
    | RandrangeParams
    | ChoiceParams
    | float
)


def create_distribution(
    kwargs: DistributionParams,
) -> Distribution:
    """
    .. list-table:: Available Distributions
       :widths: 5 10 10
       :header-rows: 1

       * - Name
         - Parameters
         - Distribution
       * - uniform
         - a=0, b=0.5
         - .. image:: _static/transformations/uniform_000_050.svg
              :width: 75%
       * - uniform
         - a=-0.75, b=0.75
         - .. image:: _static/transformations/uniform_075_075.svg
              :width: 75%
       * - truncnorm
         - mu=0.0, sigma=0.75, delta=1.0
         - .. image:: _static/transformations/truncnorm_000_075_100.svg
              :width: 75%
       * - truncnorm
         - mu=0.4, sigma=0.3, delta=1.0
         - .. image:: _static/transformations/truncnorm_040_030_100.svg
              :width: 75%
       * - truncnorm
         - mu=0.0, sigma=0.5, a=0.0, b=1.0
         - .. image:: _static/transformations/truncnorm_000_050_000_100.svg
              :width: 75%
       * - constant
         - value=0.25
         - .. image:: _static/transformations/constant_025.svg
              :width: 75%
       * - randrange
         - start=-1, stop=2
         - .. image:: _static/transformations/randrange_-1_2.svg
              :width: 75%
       * - choice
         - values = [-0.1, 0.1, 1]
         - .. image:: _static/transformations/choice.svg
              :width: 75%
       * - choice
         - values = [(-0.1, 30), (0.1, 60), (1, 10)]
         - .. image:: _static/transformations/choice_with_probability.svg
              :width: 75%

    :param name: distribution name in ['uniform', 'truncnorm', 'constant', 'randrange']
    """
    if isinstance(kwargs, (float, int)):
        kwargs = {"name": "constant", "value": kwargs}
    match kwargs["name"]:
        case "uniform":

            def _uniform(
                random: Random,
                a: float = kwargs.pop("a"),
                b: float = kwargs.pop("b"),
            ) -> float:
                return random.uniform(a, b)

            ret = _uniform

        case "truncnorm":
            from .truncnorm import truncnorm

            mu = kwargs.pop("mu", 0.0)
            if "delta" in kwargs:
                assert "a" not in kwargs and "b" not in kwargs
                delta = kwargs.pop("delta")
                a, b = mu - delta, mu + delta
            else:
                a, b = kwargs.pop("a"), kwargs.pop("b")
            f = truncnorm(mu, kwargs.pop("sigma", 1.0), a, b)

            def _truncnorm(
                random: Random, f: Callable[[float], float] = f
            ) -> float:
                return f(random.random())

            ret = _truncnorm

        case "constant":
            value = kwargs.pop("value")

            def _constant(random: Random, value: float = value) -> float:
                return value

            ret = _constant

        case "randrange":

            def _randrange(
                random: Random,
                start: int = kwargs.pop("start"),
                stop: int = kwargs.pop("stop"),
            ):
                return random.randrange(start, stop)

            ret = _randrange
        case "choice":
            from itertools import accumulate

            values = kwargs.pop("values")
            assert values, "no value to choose from"
            population: list[Value]
            if isinstance(values[0], Sequence):
                population, weights = zip(*values)
                cum_weights = list(accumulate(weights))
            else:
                population = cast(list[Value], values)
                cum_weights = None

            def _choice(random: Random):
                return random.choices(population, cum_weights=cum_weights)[0]

            ret = _choice
        case _:
            raise ValueError(f"Unsupported distribution: {kwargs['name']}")
    if len(kwargs) != 1:
        raise ValueError(f"Can't create distribution from kwargs: {kwargs}")
    return ret
