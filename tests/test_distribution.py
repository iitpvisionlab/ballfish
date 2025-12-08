from __future__ import annotations
from unittest import TestCase
from random import Random

from ballfish.distribution import create_distribution, Require
import numpy as np
import math


def calculate_correlation(xs: list[float], ys: list[float]) -> float:
    mean1 = sum(xs) / len(xs)
    mean2 = sum(ys) / len(ys)
    numerator = sum((x - mean1) * (y - mean2) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - mean1) ** 2 for x in xs) * sum((y - mean2) ** 2 for y in ys)
    )
    return numerator / denominator


class GaussTest(TestCase):
    def test_standard(self):
        f = create_distribution({"name": "truncnorm", "a": -1, "b": 1})
        random = Random(13)
        distribution = [f(random) for _ in range(1000)]
        hist = np.histogram(distribution, bins=10)[0]
        correlation = calculate_correlation(
            hist, [77, 98, 110, 99, 132, 120, 108, 90, 102, 64]
        )
        self.assertGreater(correlation, 0.95)
        self.assertEqual(sum(hist), 1000)
        self.assertGreater(min(distribution), -1)
        self.assertLess(max(distribution), 1)

    def test_standard_mean_1(self):
        f = create_distribution(
            {"name": "truncnorm", "a": -1, "b": 1, "mu": 1}
        )
        random = Random(13)
        distribution = [f(random) for _ in range(1000)]
        hist = np.histogram(distribution, bins=10)[0]
        correlation = calculate_correlation(
            hist, [25, 41, 59, 76, 92, 102, 151, 143, 150, 161]
        )
        self.assertGreater(correlation, 0.95)
        self.assertEqual(sum(hist), 1000)
        self.assertGreater(min(distribution), -1)
        self.assertLess(max(distribution), 1)


class UniformTest(TestCase):
    def test_standard(self):
        f = create_distribution({"name": "uniform", "a": -1, "b": 1})
        random = Random(13)
        distribution = [f(random) for _ in range(1000)]
        hist = np.histogram(distribution, bins=10)[0]
        self.assertLess(np.std(hist), 12)
        self.assertEqual(sum(hist), 1000)
        self.assertGreater(min(distribution), -1)
        self.assertLess(max(distribution), 1)

    def test_shifted(self):
        f = create_distribution({"name": "uniform", "a": 10, "b": 20})
        random = Random(13)
        distribution = [f(random) for _ in range(1000)]
        hist = np.histogram(distribution, bins=10)[0]
        self.assertLess(np.std(hist), 12)
        self.assertEqual(sum(hist), 1000)
        self.assertGreater(min(distribution), 10)
        self.assertLess(max(distribution), 20)


class ConstantTest(TestCase):
    def test_42(self):
        f = create_distribution({"name": "constant", "value": 42})
        random = Random(13)
        distribution = [f(random) for _ in range(100)]
        self.assertEqual(distribution, [42] * 100)


class RandrangeTest(TestCase):
    def test_default(self):
        f = create_distribution({"name": "randrange", "start": -1, "stop": 42})
        random = Random(13)
        distribution = [f(random) for _ in range(1000)]
        self.assertEqual(set(distribution), set(range(-1, 42)))


class ReuqirementsTest(TestCase):
    def test_minimum(self):
        with self.assertRaises(ValueError) as e:
            create_distribution(0.0, Require("val1", minimum=1.0))
        self.assertEqual(
            str(e.exception),
            "val1's minimum (0.0) is below the required minimum (1.0)",
        )

    def test_maximum(self):
        with self.assertRaises(ValueError) as e:
            create_distribution(1.0, Require("val2", maximum=0.0))
        self.assertEqual(
            str(e.exception),
            "val2's maximum (1.0) exceeds the required maximum (0.0)",
        )

    def test_is_int(self):
        with self.assertRaises(ValueError) as e:
            create_distribution(3.14, Require("val3", is_int=True))
        self.assertEqual(
            str(e.exception),
            "val3's value is required to be integer",
        )
