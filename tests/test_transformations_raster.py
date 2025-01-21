from __future__ import annotations
from typing import ClassVar, Literal, cast
from unittest import TestCase
from ballfish import create_augmentation, Datum
from random import Random
import torch
import torch.nn.functional as F
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


class RasterizeTest(TestCase):
    image = (
        F.pad(
            torch.tensor(
                [
                    [1, 1, 1],
                    [1, 0, 0],
                    [1, 1, 1],
                    [1, 0, 0],
                    [1, 0, 0],
                ],
                dtype=torch.float32,
            ),
            (2, 2, 2, 2),
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )

    def test_padding_border(self):
        result = create_augmentation(
            [
                {"name": "rotate", "angle_deg": 45.0},
                {"name": "rasterize", "padding_mode": "border"},
            ]
        )(
            Datum(source=self.image + 1.0),
            cast(Random, None),
        )
        reference = torch.tensor(
            [
                [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
                [1.0000, 1.0000, 1.1005, 1.0000, 1.0000, 1.0000, 1.0000],
                [1.0000, 1.1716, 1.8787, 1.2426, 1.0000, 1.0000, 1.0000],
                [1.1005, 1.8787, 1.4142, 1.2929, 1.5858, 1.0000, 1.0000],
                [1.0000, 1.5858, 1.7929, 2.0000, 1.2929, 1.0000, 1.0000],
                [1.0000, 1.0000, 1.5858, 1.7929, 1.0000, 1.0000, 1.0000],
                [1.0000, 1.0000, 1.0000, 1.5858, 1.6213, 1.0000, 1.0000],
                [1.0000, 1.0000, 1.0000, 1.0000, 1.1005, 1.0000, 1.0000],
                [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
            ]
        )
        assert result.image is not None
        torch.testing.assert_close(
            result.image[0, 0], reference, rtol=1e-5, atol=1e-3
        )

    def test_padding_zeros(self):
        result = create_augmentation(
            [
                {"name": "rotate", "angle_deg": 45.0},
                {"name": "rasterize", "padding_mode": "zeros"},
            ]
        )(
            Datum(source=self.image + 1.0),
            cast(Random, None),
        )
        reference = torch.tensor(
            [
                [0.0503, 0.7574, 1.0000, 1.0000, 0.4645, 0.0000, 0.0000],
                [0.7574, 1.0000, 1.1005, 1.0000, 1.0000, 0.4645, 0.0000],
                [1.0000, 1.1716, 1.8787, 1.2426, 1.0000, 1.0000, 0.4645],
                [1.1005, 1.8787, 1.4142, 1.2929, 1.5858, 1.0000, 1.0000],
                [1.0000, 1.5858, 1.7929, 2.0000, 1.2929, 1.0000, 1.0000],
                [1.0000, 1.0000, 1.5858, 1.7929, 1.0000, 1.0000, 1.0000],
                [0.4645, 1.0000, 1.0000, 1.5858, 1.6213, 1.0000, 1.0000],
                [0.0000, 0.4645, 1.0000, 1.0000, 1.1005, 1.0000, 0.7574],
                [0.0000, 0.0000, 0.4645, 1.0000, 1.0000, 0.7574, 0.0503],
            ]
        )
        assert result.image is not None
        torch.testing.assert_close(
            result.image[0, 0], reference, rtol=1e-5, atol=1e-3
        )
