from __future__ import annotations
from unittest import TestCase
from random import Random
from ballfish import create_augmentation, Datum, Quad
from torch import zeros


def create_test_datum() -> Datum:
    image = zeros(1, 1, 8, 16)
    image[0, 0, 2:6:2, 2:14] = 1.0
    return Datum(source=image)


class AlmostEqualForQuads(TestCase):
    def assertQuadAlmostEqual(self, a: Quad, b: Quad) -> None:
        assert len(a) == 4 and len(b) == 4
        msg = f"{a} != {b}"
        for pt1, pt2 in zip(a, b):
            self.assertAlmostEqual(pt1[0], pt2[0], msg=msg)
            self.assertAlmostEqual(pt1[1], pt2[1], msg=msg)


class Projective1ptTransformationTest(TestCase):
    def test_no_shift(self):
        projective1pt = create_augmentation(
            [
                {
                    "name": "projective1pt",
                    "x": {"name": "constant", "value": 0},
                    "y": {"name": "constant", "value": 0},
                }
            ]
        )
        random = Random(13)
        res = projective1pt(create_test_datum(), random)
        self.assertEqual(res.quads, [((0, 0), (16, 0), (16.0, 8), (0, 8))])

    def test_shift_x(self):
        projective1pt = create_augmentation(
            [
                {
                    "name": "projective1pt",
                    "x": {"name": "constant", "value": 1 / 16},
                    "y": {"name": "constant", "value": 0},
                }
            ]
        )
        random = Random(13)
        res = projective1pt(create_test_datum(), random)
        self.assertEqual(res.quads, [((0, 0), (16, 0), (17.0, 8), (0, 8))])

    def test_shift_y(self):
        projective1pt = create_augmentation(
            [
                {
                    "name": "projective1pt",
                    "x": {"name": "constant", "value": 0},
                    "y": {"name": "constant", "value": 1 / 16},
                }
            ]
        )
        random = Random(13)
        res = projective1pt(create_test_datum(), random)
        self.assertEqual(res.quads, [((0, 0), (16, 0), (16.0, 9), (0, 8))])


class Projective4ptTransformationTest(TestCase):
    def test_no_shift(self):
        projective4pt = create_augmentation(
            [
                {
                    "name": "projective4pt",
                    "x": {"name": "constant", "value": 0},
                    "y": {"name": "constant", "value": 0},
                }
            ]
        )
        random = Random(13)
        res = projective4pt(create_test_datum(), random)
        self.assertEqual(res.quads, [((0, 0), (16, 0), (16.0, 8), (0, 8))])

    def test_shift_x(self):
        projective4pt = create_augmentation(
            [
                {
                    "name": "projective4pt",
                    "x": {"name": "constant", "value": 1 / 16},
                    "y": {"name": "constant", "value": 0},
                }
            ]
        )
        random = Random(13)
        res = projective4pt(create_test_datum(), random)
        self.assertEqual(res.quads, [((1, 0), (17, 0), (17.0, 8), (1, 8))])

    def test_shift_y(self):
        projective4pt = create_augmentation(
            [
                {
                    "name": "projective4pt",
                    "x": {"name": "constant", "value": 0},
                    "y": {"name": "constant", "value": 1 / 16},
                }
            ]
        )
        random = Random(13)
        res = projective4pt(create_test_datum(), random)
        self.assertEqual(res.quads, [((0, 1), (16, 1), (16.0, 9), (0, 9))])


class FlipTest(TestCase):
    def test_horizontal(self):
        flip_h = create_augmentation(
            [{"name": "flip", "direction": "horizontal"}]
        )
        random = Random(13)
        res = flip_h(create_test_datum(), random)
        self.assertEqual(res.quads, [((16, 0), (0, 0), (0, 8), (16, 8))])

    def test_vertical(self):
        flip_v = create_augmentation(
            [{"name": "flip", "direction": "vertical"}]
        )
        random = Random(13)
        res = flip_v(create_test_datum(), random)
        self.assertEqual(res.quads, [((16, 8), (0, 8), (0, 0), (16, 0))])

    def test_primary_diagonal(self):
        flip_pd = create_augmentation(
            [{"name": "flip", "direction": "primary_diagonal"}]
        )
        random = Random(13)
        res = flip_pd(create_test_datum(), random)
        self.assertEqual(res.quads, [((0, 0), (0, 8), (16, 8), (16, 0))])

    def test_secondary_diagonal(self):
        flip_sd = create_augmentation(
            [{"name": "flip", "direction": "secondary_diagonal"}]
        )
        random = Random(13)
        res = flip_sd(create_test_datum(), random)
        self.assertEqual(res.quads, [((16, 8), (16, 0), (0, 0), (0, 8))])


class PaddingsAdditionTest(TestCase):
    def test_none(self):
        paddings = create_augmentation(
            [
                {
                    "name": "paddings_addition",
                    "top": {"name": "constant", "value": 0},
                    "right": {"name": "constant", "value": 0},
                    "bottom": {"name": "constant", "value": 0},
                    "left": {"name": "constant", "value": 0},
                }
            ]
        )
        random = Random(13)
        res = paddings(create_test_datum(), random)
        self.assertEqual(res.quads, [((0, 0), (16, 0), (16, 8), (0, 8))])

    def test_top(self):
        paddings = create_augmentation(
            [
                {
                    "name": "paddings_addition",
                    "top": {"name": "constant", "value": 1 / 8},
                    "right": {"name": "constant", "value": 0},
                    "bottom": {"name": "constant", "value": 0},
                    "left": {"name": "constant", "value": 0},
                }
            ]
        )
        random = Random(13)
        res = paddings(create_test_datum(), random)
        self.assertEqual(res.quads, [((0, 1), (16, 1), (16, 8), (0, 8))])

    def right(self):
        paddings = create_augmentation(
            [
                {
                    "name": "paddings_addition",
                    "top": {"name": "constant", "value": 0},
                    "right": {"name": "constant", "value": 1 / 16},
                    "bottom": {"name": "constant", "value": 0},
                    "left": {"name": "constant", "value": 0},
                }
            ]
        )
        random = Random(13)
        res = paddings(create_test_datum(), random)
        self.assertEqual(res.quads, [((0, 0), (17, 0), (17, 8), (0, 8))])

    def bottom(self):
        paddings = create_augmentation(
            [
                {
                    "name": "paddings_addition",
                    "top": {"name": "constant", "value": 0},
                    "right": {"name": "constant", "value": 0},
                    "bottom": {"name": "constant", "value": 1 / 8},
                    "left": {"name": "constant", "value": 0},
                }
            ]
        )
        random = Random(13)
        res = paddings(create_test_datum(), random)
        self.assertEqual(res.quads, [((0, 0), (16, 0), (16, 9), (0, 9))])

    def left(self):
        paddings = create_augmentation(
            [
                {
                    "name": "paddings_addition",
                    "top": {"name": "constant", "value": 0},
                    "right": {"name": "constant", "value": 0},
                    "bottom": {"name": "constant", "value": 0},
                    "left": {"name": "constant", "value": 1 / 16},
                }
            ]
        )
        random = Random(13)
        res = paddings(create_test_datum(), random)
        self.assertEqual(res.quads, [((-1, 0), (16, 0), (16, 8), (-1, 8))])


class ProjectivePaddingsAdditionTest(TestCase):
    def test_none(self):
        paddings = create_augmentation(
            [
                {
                    "name": "projective_paddings_addition",
                    "top": {"name": "constant", "value": 0},
                    "right": {"name": "constant", "value": 0},
                    "bottom": {"name": "constant", "value": 0},
                    "left": {"name": "constant", "value": 0},
                }
            ]
        )
        random = Random(13)
        res = paddings(create_test_datum(), random)
        self.assertEqual(res.quads, [((0, 0), (16, 0), (16, 8), (0, 8))])

    def test_top(self):
        paddings = create_augmentation(
            [
                {
                    "name": "projective_paddings_addition",
                    "top": {"name": "constant", "value": 1 / 8},
                    "right": {"name": "constant", "value": 0},
                    "bottom": {"name": "constant", "value": 0},
                    "left": {"name": "constant", "value": 0},
                }
            ]
        )
        random = Random(13)
        res = paddings(create_test_datum(), random)
        self.assertEqual(res.quads, [((0, 1), (16, 1), (16, 8), (0, 8))])

    def right(self):
        paddings = create_augmentation(
            [
                {
                    "name": "projective_paddings_addition",
                    "top": {"name": "constant", "value": 0},
                    "right": {"name": "constant", "value": 1 / 16},
                    "bottom": {"name": "constant", "value": 0},
                    "left": {"name": "constant", "value": 0},
                }
            ]
        )
        random = Random(13)
        res = paddings(create_test_datum(), random)
        self.assertEqual(res.quads, [((0, 0), (17, 0), (17, 8), (0, 8))])

    def bottom(self):
        paddings = create_augmentation(
            [
                {
                    "name": "projective_paddings_addition",
                    "top": {"name": "constant", "value": 0},
                    "right": {"name": "constant", "value": 0},
                    "bottom": {"name": "constant", "value": 1 / 8},
                    "left": {"name": "constant", "value": 0},
                }
            ]
        )
        random = Random(13)
        res = paddings(create_test_datum(), random)
        self.assertEqual(res.quads, [((0, 0), (16, 0), (16, 9), (0, 9))])

    def left(self):
        paddings = create_augmentation(
            [
                {
                    "name": "projective_paddings_addition",
                    "top": {"name": "constant", "value": 0},
                    "right": {"name": "constant", "value": 0},
                    "bottom": {"name": "constant", "value": 0},
                    "left": {"name": "constant", "value": 1 / 16},
                }
            ]
        )
        random = Random(13)
        res = paddings(create_test_datum(), random)
        self.assertEqual(res.quads, [((-1, 0), (16, 0), (16, 8), (-1, 8))])


class RotateTest(AlmostEqualForQuads, TestCase):
    def test_none(self):
        rotate = create_augmentation(
            [{"name": "rotate", "angle_deg": {"name": "constant", "value": 0}}]
        )
        random = Random(13)
        res = rotate(create_test_datum(), random)
        self.assertEqual(res.quads, [((0, 0), (16, 0), (16, 8), (0, 8))])

    def test_90(self):
        rotate = create_augmentation(
            [
                {
                    "name": "rotate",
                    "angle_deg": {"name": "constant", "value": 180},
                }
            ]
        )
        random = Random(13)
        res = rotate(create_test_datum(), random)
        self.assertQuadAlmostEqual(
            res.quads[0], ((16, 8), (0, 8), (0, 0), (16, 0))
        )


class ProjectiveShiftTest(TestCase):
    def test_none(self):
        shift = create_augmentation([{"name": "projective_shift"}])
        random = Random(13)
        res = shift(create_test_datum(), random)
        self.assertEqual(res.quads, [((0, 0), (16, 0), (16, 8), (0, 8))])

    def test_x(self):
        shift = create_augmentation(
            [
                {
                    "name": "projective_shift",
                    "x": {"name": "constant", "value": 1 / 16},
                }
            ]
        )
        random = Random(13)
        res = shift(create_test_datum(), random)
        self.assertEqual(res.quads, [((1, 0), (17, 0), (17, 8), (1, 8))])

    def test_y(self):
        shift = create_augmentation(
            [
                {
                    "name": "projective_shift",
                    "y": {"name": "constant", "value": 1 / 8},
                }
            ]
        )
        random = Random(13)
        res = shift(create_test_datum(), random)
        self.assertEqual(res.quads, [((0, 1), (16, 1), (16, 9), (0, 9))])


class ScaleTest(TestCase):
    def test_none(self):
        scale = create_augmentation(
            [{"name": "scale", "factor": {"name": "constant", "value": 1}}]
        )
        random = Random(13)
        res = scale(create_test_datum(), random)
        self.assertEqual(res.quads, [((0, 0), (16, 0), (16, 8), (0, 8))])

    def test_scale_down(self):
        scale = create_augmentation(
            [{"name": "scale", "factor": {"name": "constant", "value": 1 / 2}}]
        )
        random = Random(13)
        res = scale(create_test_datum(), random)
        self.assertEqual(res.quads, [((4, 2), (12, 2), (12, 6), (4, 6))])
