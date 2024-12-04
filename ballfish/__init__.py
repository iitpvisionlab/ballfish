from __future__ import annotations
from typing import Callable, Sequence, Iterator, TYPE_CHECKING
from random import Random

if TYPE_CHECKING:
    from .transformation import Transformation, Datum, Args


def _prepare(
    operations: Sequence[Args],
) -> Iterator[tuple[float, Transformation]]:
    from .transformation import create

    for operation in operations:
        assert isinstance(operation, dict), operation
        probability: float = operation.get("probability", 1.0)
        if probability <= 0.0:
            continue
        inst = create(operation)
        yield probability, inst


def create_augmentation(
    operations: Sequence[Args],
) -> Callable[[Datum, Random], Datum]:
    transformations = list(_prepare(operations))

    def augment(datum: Datum, random: Random) -> Datum:
        for probability, transformation in transformations:
            if probability >= 1.0 or probability >= random.random():
                datum = transformation(datum, random)
        return datum

    return augment
