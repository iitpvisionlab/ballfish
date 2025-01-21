from __future__ import annotations
from typing import Callable, Sequence, Iterator
from random import Random
from .transformation import Transformation, Datum, Args, Quad

__version__ = "0.3.1"


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
    """
    Main function to create augmentation function.
    """
    transformations = list(_prepare(operations))

    def augment(datum: Datum, random: Random) -> Datum:
        for probability, transformation in transformations:
            if probability >= 1.0 or probability >= random.random():
                datum = transformation(datum, random)
        return datum

    return augment
