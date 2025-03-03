from __future__ import annotations
from typing import Literal
from random import Random
from unittest import TestCase
from unittest.mock import Mock, patch
import ballfish
import torch
from ballfish import (
    Transformation,
    DistributionParams,
    create_distribution,
    Datum,
    create_augmentation,
)
from ballfish import transformation


class SetValue(Transformation):
    name = "set"

    class Args(transformation.ArgDict):
        name: Literal["set"]
        value: DistributionParams

    def __init__(self, value: DistributionParams) -> None:
        self._value = create_distribution(value)

    def __call__(self, datum: Datum, random: Random) -> Datum:
        assert datum.image is not None
        datum.image[:] = self._value(random)
        return datum


transformation.TransformationArgs |= SetValue.Args
ballfish.TransformationArgs = transformation.TransformationArgs


class FakeRandom:
    def __init__(self, values: list[float]) -> None:
        self.random = Mock(side_effect=values)


class Probability(TestCase):
    def test_fifty_fifty(self):
        augmentation = create_augmentation(
            [
                {"probability": 0.5, "name": "set", "value": 1},
                {"probability": 0.5, "name": "set", "value": 2},
            ]
        )

        with patch.object(Random, "random", side_effect=[0, 1]):
            datum = augmentation(Datum(image=torch.zeros(1)), Random())
        self.assertEqual(datum.image.item(), 1.0)

        with patch.object(Random, "random", side_effect=[1, 0]):
            datum = augmentation(Datum(image=torch.zeros(1)), Random())
        self.assertEqual(datum.image.item(), 2.0)
