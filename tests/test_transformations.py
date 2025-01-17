from __future__ import annotations
from typing import ClassVar, Literal, cast
from unittest import TestCase
from ballfish import create_augmentation, Datum
from random import Random
import torch
from unittest.mock import patch


class SequentiolChoices(Random):
    def __init__(self):
        self._i = 0

    def choices(self, population: list[float], cum_weights: None):
        assert cum_weights is None
        ret = population[self._i % len(population)]
        self._i += 1
        return (ret,)


class Base:  # hide from unittest
    class OperationTransformTest(TestCase):
        name: ClassVar[Literal["multiply", "divide", "add", "pow"]]
        value_name: ClassVar[Literal["value", "factor", "pow"]]

        @staticmethod
        def op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError

        def test_per_tensor(self):
            result = create_augmentation(
                [{"name": self.name, self.value_name: 2.0, "per": "tensor"}]
            )(Datum(image=torch.ones(2, 3, 2, 2)), Random())
            reference = torch.full((2, 3, 2, 2), self.op(1.0, 2.0))
            self.assertTrue(torch.all(result.image == reference))

        def test_per_batch(self):
            result = create_augmentation(
                [
                    {
                        "name": self.name,
                        self.value_name: {
                            "name": "choice",
                            "values": [1.1, 1.2],
                        },
                        "per": "batch",
                    }
                ]
            )(Datum(image=torch.ones(2, 3, 2, 2)), SequentiolChoices())
            assert result.image is not None
            reference = self.op(
                torch.ones(2, 3, 2, 2),
                torch.tensor([1.1, 1.2])[..., None, None, None],
            )
            self.assertTrue(torch.all(result.image == reference))

        def test_per_channel(self):
            result = create_augmentation(
                [
                    {
                        "name": self.name,
                        self.value_name: {
                            "name": "choice",
                            "values": [1.1, 1.2, 1.3],
                        },
                        "per": "channel",
                    }
                ]
            )(Datum(image=torch.ones(2, 3, 2, 2)), SequentiolChoices())
            assert result.image is not None
            reference = self.op(
                torch.ones(2, 3, 2, 2),
                torch.tensor([1.1, 1.2, 1.3])[..., None, None],
            )
            self.assertTrue(torch.all(result.image == reference))

        def test_different_distributions(self):
            result = create_augmentation(
                [{"name": self.name, self.value_name: [1.1, 1.2, 1.3]}]
            )(Datum(image=torch.ones(2, 3, 2, 2)), cast(Random, None))
            assert result.image is not None
            reference = self.op(
                torch.ones(2, 3, 2, 2),
                torch.tensor([1.1, 1.2, 1.3])[..., None, None],
            )
            self.assertTrue(torch.all(result.image == reference))


class MultipyTest(Base.OperationTransformTest):
    name = "multiply"
    value_name = "factor"

    @staticmethod
    def op(a: torch.Tensor, b: torch.Tensor):
        return a * b


class AddTest(Base.OperationTransformTest):
    name = "add"
    value_name = "value"

    @staticmethod
    def op(a: torch.Tensor, b: torch.Tensor):
        return a + b


class DivideTest(Base.OperationTransformTest):
    name = "divide"
    value_name = "value"

    @staticmethod
    def op(a: torch.Tensor, b: torch.Tensor):
        return a / b


class PowTest(Base.OperationTransformTest):
    name = "pow"
    value_name = "pow"

    @staticmethod
    def op(a: torch.Tensor, b: torch.Tensor):
        return a**b


class NoiseTest(TestCase):
    @patch("torch.randn_like")
    def test_homoscedastic(self, fake_randn_like):
        fake_randn_like.return_value = torch.ones(2, 3, 2, 2)
        image = torch.arange(2 * 3 * 2 * 2, dtype=torch.float64).reshape(
            2, 3, 2, 2
        )
        result = create_augmentation([{"name": "noise", "std": 2.0}])(
            Datum(image=image),
            cast(Random, None),
        )
        reference = torch.arange(2 * 3 * 2 * 2).reshape(2, 3, 2, 2) + 2.0
        self.assertTrue(torch.all(result.image == reference))

    @patch("torch.randn_like")
    def test_heteroscedastic(self, fake_randn_like):
        fake_randn_like.return_value = torch.ones(2, 3, 2, 2)
        image = torch.arange(2 * 3 * 2 * 2, dtype=torch.float64).reshape(
            2, 3, 2, 2
        )
        result = create_augmentation(
            [{"name": "noise", "std": 2.0, "type": "heteroscedastic"}]
        )(
            Datum(image=image),
            cast(Random, None),
        )
        reference = torch.arange(2 * 3 * 2 * 2).reshape(2, 3, 2, 2) * 3.0
        self.assertTrue(torch.all(result.image == reference))
