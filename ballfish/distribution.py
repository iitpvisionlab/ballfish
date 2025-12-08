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


class Require:
    """
    A way to limit :func:`create_distribution` expected values
    """

    def __init__(
        self,
        name: str,
        minimum: float | None = None,
        maximum: float | None = None,
        is_int: bool = False,
    ) -> None:
        self._name = name
        self._minimum = minimum
        self._maximum = maximum
        self._is_int = is_int

    def __call__(self, minimum: float, maximum: float, is_int: bool) -> None:
        """
        Raises `ValueError` when validation fails
        """
        if self._minimum is not None and minimum < self._minimum:
            raise ValueError(
                f"{self._name}'s minimum ({minimum}) "
                f"is below the required minimum ({self._minimum})"
            )
        if self._maximum is not None and maximum > self._maximum:
            raise ValueError(
                f"{self._name}'s maximum ({maximum}) "
                f"exceeds the required maximum ({self._maximum})"
            )
        if self._is_int and not is_int:
            raise ValueError(f"{self._name}'s value is required to be integer")


def create_distribution(
    kwargs: DistributionParams, require: Require | None = None
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

    :param require: optional :class:`Require` class that will ensure
                    that distribution is in required range

    Example
    -------
    .. code-block::

        >>> ballfish.create_distribution({"name": "truncnorm", "a": -0.25, "b": 0.25})
        <function create_distribution.<locals>._truncnorm at 0x7feb7e166b60>
    """
    if isinstance(kwargs, (float, int)):
        kwargs = {"name": "constant", "value": kwargs}
    else:
        kwargs = kwargs.copy()
    match kwargs["name"]:
        case "uniform":
            minimum, maximum, is_int = kwargs["a"], kwargs["b"], False

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
            minimum, maximum, is_int = a, b, False
            f = truncnorm(mu, kwargs.pop("sigma", 1.0), a, b)

            def _truncnorm(
                random: Random, f: Callable[[float], float] = f
            ) -> float:
                return f(random.random())

            ret = _truncnorm

        case "constant":
            value = kwargs.pop("value")
            minimum, maximum, is_int = value, value, isinstance(value, int)

            def _constant(random: Random, value: float = value) -> float:
                return value

            ret = _constant

        case "randrange":
            minimum, maximum, is_int = kwargs["start"], kwargs["stop"], True

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

            minimum, maximum, is_int = (
                min(population),
                max(population),
                all([isinstance(val, int) for val in population]),
            )

            def _choice(random: Random):
                return random.choices(population, cum_weights=cum_weights)[0]

            ret = _choice
        case _:
            raise ValueError(f"Unsupported distribution: {kwargs['name']}")
    if len(kwargs) != 1:
        raise ValueError(f"Can't create distribution from kwargs: {kwargs}")
    if require is not None:
        require(minimum, maximum, is_int)
    return ret
